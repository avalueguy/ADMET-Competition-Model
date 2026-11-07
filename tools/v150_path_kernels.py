"""Unfolded non-circular molecular kernels plus target-excluded assay kernels."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='3'
import hashlib,json,sys,time
from pathlib import Path
import joblib,numpy as np,torch
from scipy.special import expit
from sklearn.svm import SVC,SVR
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v146_current_baseline as v
import v147_path_features as features
import v148_experiment_runner as runner
import v139_classification_kernels as chemistry
b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v150'
CONFIGS=tuple(dict(view=view,C=c) for view in ('sum','product') for c in (1.,10.))
METHOD='New unfolded atom-pair, topological torsion, RDKit path generalized Tanimoto plus 2 circular count views, physicochemical and target-excluded assay kernels. SVC/SVR margin learning.'
def sig():return hashlib.sha256((b.sha(__file__)+v.sig()+features.sig()+b.sha(runner.__file__)+chemistry.sig()).encode()).hexdigest()
def key(c):return c['view']+'_C'+str(c['C'])
def matrices(d,k,part):
    fit,q=v.split(d,k,part);rows=np.r_[fit,q];n=len(fit);chem=np.zeros((len(rows),n),np.float64)
    for raw in d['path_fingerprints']:
        x=raw[d['mi'][rows]].astype(np.float64);x.data=np.log1p(x.data);ref=x[:n];rn=np.asarray(ref.multiply(ref).sum(1)).ravel();qn=np.asarray(x.multiply(x).sum(1)).ravel()
        for lo in range(0,len(rows),256):
            dot=(x[lo:lo+256]@ref.T).toarray();sim=np.clip(dot/np.maximum(qn[lo:lo+len(dot),None]+rn[None]-dot,1e-10),0,1)
            chem[lo:lo+len(dot)]+=.8/3*(.5*sim+.5*sim**2)
    for start in (0,2048):
        x=torch.as_tensor(d['views']['multiview'][d['mi'][rows],start:start+2048],device='cuda');ref=x[:n];rn=ref.square().sum(1)
        for lo in range(0,len(rows),256):
            a=x[lo:lo+256];dot=a@ref.T;sim=(dot/(a.square().sum(1)[:,None]+rn[None]-dot).clamp_min(1e-8)).clamp(0,1);chem[lo:lo+len(a)]+=.1*(.5*sim+.5*sim**2).cpu().numpy()
    aux,cols,source,_,_=v.old.old.old.assays.assays(d,k,part);assert k not in cols and np.all(d['tid'][source]!=k)
    assay=chemistry.rbf(aux[rows].astype(float),n)
    physical=chemistry.rbf(np.c_[d['views']['physchem'][d['mi'][rows]],d['conformer_features'][d['mi'][rows]],d['pharmacophore'][d['mi'][rows]]].astype(float),n)
    value={'sum':.7*chem+.2*assay+.1*physical,'product':(.85*chem+.15*physical)*(.4+.6*assay)}
    torch.cuda.empty_cache();return fit,q,{name:((z[:n]+z[:n].T)*.5,z[n:]) for name,z in value.items()}
def family(d,k,part,configs):
    folder=OUT/'models'/d['tasks'][k]/part;folder.mkdir(parents=True,exist_ok=True);missing=[];result={}
    for c in configs:
        p=folder/(key(c)+'.npz')
        if p.exists():
            z=np.load(p);assert str(z['signature'])==sig();np.testing.assert_array_equal(z['query'],v.split(d,k,part)[1]);result[key(c)]=z['p'].copy()
        else:missing.append(c)
    if missing:
        start=time.monotonic();fit,q,mm=matrices(d,k,part);m=d['metrics'][k];y=d['y'][fit];cls=m in ('AUROC','AUPRC');center=float(np.median(y)) if m=='MAE' else 0.;scale=max(float(np.subtract(*np.quantile(y,[.75,.25]))),.05) if m=='MAE' else 1.
        target=y if cls else (y-center)/scale if m=='MAE' else r83.ranknorm(y)
        for c in missing:
            x,qx=mm[c['view']];model=SVC(C=c['C'],kernel='precomputed',tol=1e-4,cache_size=1024) if cls else SVR(C=c['C'],epsilon=.04,kernel='precomputed',tol=1e-4,cache_size=1024)
            model.fit(x,target);assert model.fit_status_==0
            p=expit(model.decision_function(qx)) if cls else model.predict(qx)*scale+center
            if d['tasks'][k]=='ppbr_az':p=p.clip(0,100)
            mp=folder/(key(c)+'.joblib');joblib.dump(model,mp,compress=3);replay=joblib.load(mp);rp=expit(replay.decision_function(qx)) if cls else replay.predict(qx)*scale+center
            if d['tasks'][k]=='ppbr_az':rp=rp.clip(0,100)
            np.testing.assert_array_equal(p,rp);np.savez_compressed(folder/(key(c)+'.npz'),signature=sig(),fit=fit,query=q,p=p,center=center,scale=scale);result[key(c)]=p
        print(json.dumps(dict(stage='train',task=d['tasks'][k],part=part,models=len(missing),seconds=round(time.monotonic()-start))),flush=True)
    return result
def predict(d,k,part,c):
    # One matrix construction for every development C/view combination.
    return family(d,k,part,CONFIGS if part in ('fold0','fold1') else [c])[key(c)]
def main():
    torch.set_num_threads(3);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    d=features.load(v.load());runner.run(sys.modules[__name__],d)
if __name__=='__main__':main()
