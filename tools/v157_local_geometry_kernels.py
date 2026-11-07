"""Density-normalized molecular geometry with query-structure importance weights."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import hashlib,json,sys,time
from pathlib import Path
import joblib,numpy as np,torch
from scipy.special import expit
from sklearn.svm import SVC,SVR
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v146_current_baseline as v
import v147_path_features as features
import v148_experiment_runner as runner
import v150_path_kernels as path
b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v157'
CONFIGS=tuple(dict(view=view,C=c) for view in ('local','shift') for c in (1.,10.))
METHOD='Adaptive local scaling of new path/circular/assay product kernel. PSD sample-normalized RBF via feature-space squared distances. Query-structure importance weighting corrects covariate shift without query labels. C1/10, fixed blend .2/.45; strict paired scaffold confirmation.'

def sig():return hashlib.sha256((b.sha(__file__)+v.sig()+features.sig()+path.sig()+b.sha(runner.__file__)).encode()).hexdigest()
def key(c):return c['view']+'_C'+str(c['C'])

def matrices(d,k,part):
    # Include query-query similarities to infer geometry, without reading labels.
    fit,q=v.split(d,k,part);rows=np.r_[fit,q];n=len(fit);N=len(rows);chem=np.zeros((N,N),np.float64)
    for raw in d['path_fingerprints']:
        x=raw[d['mi'][rows]].astype(float);x.data=np.log1p(x.data);norm=np.asarray(x.multiply(x).sum(1)).ravel()
        for lo in range(0,N,256):
            dot=(x[lo:lo+256]@x.T).toarray();sim=np.clip(dot/np.maximum(norm[lo:lo+len(dot),None]+norm[None]-dot,1e-10),0,1);chem[lo:lo+len(dot)]+=(sim+sim**2)/6
    # Rescale each point's implicit feature vector by its local bandwidth.
    # Then apply an ordinary RBF to squared feature-space distances.
    dist=np.maximum(2-2*chem,0);tmp=dist[:,:n].copy();tmp[np.arange(n),np.arange(n)]=np.inf
    bandwidth=np.partition(tmp,min(15,n-1),axis=1)[:,min(15,n-1)].clip(.05,2)
    diag=np.diag(chem)/bandwidth;scaled=np.maximum(diag[:,None]+diag[None]-2*chem/np.sqrt(bandwidth[:,None]*bandwidth[None]),0);local=np.exp(-scaled)
    aux,cols,source,_,_=v.old.old.old.assays.assays(d,k,part);assert k not in cols and np.all(d['tid'][source]!=k)
    assay=path.chemistry.rbf(aux[rows].astype(float),n)
    physical=path.chemistry.rbf(np.c_[d['views']['physchem'][d['mi'][rows]],d['conformer_features'][d['mi'][rows]],d['pharmacophore'][d['mi'][rows]]].astype(float),n)
    matrix=.7*local[:,:n]+.2*assay+.1*physical;matrix[:n]=(matrix[:n]+matrix[:n].T)*.5
    # Clipped mean similarity ratio, no label- or outcome-dependent weighting.
    ref=np.mean(chem[:n,:n],axis=1).clip(1e-6);query=np.mean(chem[:n,n:],axis=1).clip(1e-6)
    weights=(query/ref).clip(.4,2.5);weights/=weights.mean()
    return fit,q,matrix[:n],matrix[n:],weights,bandwidth,source

def family(d,k,part,configs):
    folder=OUT/'models'/d['tasks'][k]/part;folder.mkdir(parents=True,exist_ok=True);result={};missing=[]
    for c in configs:
        file=folder/(key(c)+'.npz')
        if file.exists():
            z=np.load(file);assert str(z['signature'])==sig();np.testing.assert_array_equal(z['query'],v.split(d,k,part)[1]);result[key(c)]=z['p'].copy()
        else:missing.append(c)
    if missing:
        start=time.monotonic();fit,q,x,qx,w,bw,source=matrices(d,k,part);m=d['metrics'][k];y=d['y'][fit];cls=m in ('AUROC','AUPRC');center=float(np.median(y)) if m=='MAE' else 0.;scale=max(float(np.subtract(*np.quantile(y,[.75,.25]))),.05) if m=='MAE' else 1.
        target=y if cls else (y-center)/scale if m=='MAE' else r83.ranknorm(y)
        for c in missing:
            model=SVC(C=c['C'],kernel='precomputed',tol=1e-4,cache_size=1024) if cls else SVR(C=c['C'],epsilon=.04,kernel='precomputed',tol=1e-4,cache_size=1024)
            model.fit(x,target,sample_weight=w if c['view']=='shift' else np.ones(len(fit)));assert model.fit_status_==0
            p=expit(model.decision_function(qx)) if cls else model.predict(qx)*scale+center
            if d['tasks'][k]=='ppbr_az':p=p.clip(0,100)
            mp=folder/(key(c)+'.joblib');joblib.dump(model,mp,compress=3);replay=joblib.load(mp);rp=expit(replay.decision_function(qx)) if cls else replay.predict(qx)*scale+center
            if d['tasks'][k]=='ppbr_az':rp=rp.clip(0,100)
            np.testing.assert_array_equal(p,rp);np.savez_compressed(folder/(key(c)+'.npz'),signature=sig(),fit=fit,query=q,p=p,center=center,scale=scale,weights=w,bandwidth=bw);result[key(c)]=p
        print(json.dumps(dict(stage='train',task=d['tasks'][k],part=part,models=len(missing),seconds=round(time.monotonic()-start))),flush=True)
    return result

def predict(d,k,part,c):return family(d,k,part,CONFIGS if part in ('fold0','fold1') else [c])[key(c)]
def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');runner.run(sys.modules[__name__],features.load(v.load()))
if __name__=='__main__':main()
