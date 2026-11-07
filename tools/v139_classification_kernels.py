"""Joint molecular and assay kernels for every classification endpoint."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
import hashlib,json,sys,time
from pathlib import Path
import joblib,numpy as np,torch
from sklearn.svm import SVC
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v131_current_controls as v
b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v139';WEIGHTS=(.35,.75)
CONFIGS=tuple(dict(view=view,C=c) for view in ('sum','product') for c in (1.,10.))
def sig():return hashlib.sha256((b.sha(__file__)+v.sig()).encode()).hexdigest()
def key(c):return c['view']+'_C'+str(c['C'])
def rbf(raw,n):
    fit=raw[:n];median=np.array([np.median(z[np.isfinite(z)]) if np.isfinite(z).any() else 0. for z in fit.T]);scale=np.array([np.std(z[np.isfinite(z)]) if np.isfinite(z).any() else 1. for z in fit.T]).clip(.05)
    z=np.nan_to_num((raw-median)/scale).clip(-5,5);active=z[:n].std(0)>.01;z=torch.as_tensor(z[:,active],device='cuda',dtype=torch.float32);ref=z[:n]
    result=np.empty((len(raw),n),np.float64);rn=ref.square().sum(1)
    for lo in range(0,len(raw),256):
        x=z[lo:lo+256];dist=(x.square().sum(1)[:,None]+rn[None]-2*x@ref.T).clamp_min(0)/max(1,z.shape[1]);result[lo:lo+len(x)]=(.5*torch.exp(-dist/.5)+.5*torch.exp(-dist/2)).cpu().numpy()
    return result
def matrices(d,k,part):
    fit,q=v.split(d,k,part);rows=np.r_[fit,q];n=len(fit);raw=d['views']['multiview'];chem=np.zeros((len(rows),n),np.float64)
    for start in range(0,12288,2048):
        x=torch.as_tensor(raw[d['mi'][rows],start:start+2048],device='cuda');ref=x[:n];rn=ref.square().sum(1)
        for lo in range(0,len(rows),256):
            a=x[lo:lo+256];dot=a@ref.T;tan=(dot/(a.square().sum(1)[:,None]+rn[None]-dot).clamp_min(1e-8)).clamp(0,1)
            chem[lo:lo+len(a)]+=(.5*tan+.5*tan.square()).cpu().numpy()/6
    aux,cols,source,_,_=v.old.old.assays.assays(d,k,part);assert k not in cols and np.all(d['tid'][source]!=k)
    assay=rbf(aux[rows].astype(float),n)
    physical=rbf(np.c_[d['views']['physchem'][d['mi'][rows]],d['conformer_features'][d['mi'][rows]],d['pharmacophore'][d['mi'][rows]]].astype(float),n)
    value={'sum':.65*chem+.2*assay+.15*physical,'product':(.8*chem+.2*physical)*(.4+.6*assay)}
    torch.cuda.empty_cache();return fit,q,{name:((z[:n]+z[:n].T)*.5,z[n:]) for name,z in value.items()}
def family(d,k,part,configs):
    folder=OUT/'models'/d['tasks'][k]/part;folder.mkdir(parents=True,exist_ok=True);missing=[];result={}
    for c in configs:
        file=folder/(key(c)+'.npz')
        if not file.exists():missing.append(c)
        else:
            z=np.load(file);assert str(z['signature'])==sig();np.testing.assert_array_equal(z['query'],v.split(d,k,part)[1]);result[key(c)]=z['p'].copy()
    if missing:
        start=time.monotonic();fit,q,mm=matrices(d,k,part);y=d['y'][fit];m=d['metrics'][k]
        center=float(np.median(y)) if m=='MAE' else 0.;scale=max(float(np.subtract(*np.quantile(y,[.75,.25]))),.05) if m=='MAE' else 1.
        target=y
        for c in missing:
            x,qx=mm[c['view']];model=SVC(C=c['C'],kernel='precomputed',tol=1e-4,cache_size=1024);model.fit(x,target);assert model.fit_status_==0
            p=expit(model.decision_function(qx));mp=folder/(key(c)+'.joblib');joblib.dump(model,mp,compress=3)
            replay=joblib.load(mp);np.testing.assert_allclose(p,expit(replay.decision_function(qx)),atol=1e-12,rtol=0)
            if d['tasks'][k]=='ppbr_az':p=p.clip(0,100)
            np.savez_compressed(folder/(key(c)+'.npz'),signature=sig(),fit=fit,query=q,p=p,center=center,scale=scale);result[key(c)]=p
        print(json.dumps(dict(stage='train',task=d['tasks'][k],part=part,models=len(missing),seconds=round(time.monotonic()-start))),flush=True)
    return result
def predict(d,k,part,c):return family(d,k,part,[c])[key(c)]
def record(d,k,part,c,w):
    q=v.split(d,k,part)[1];m=d['metrics'][k];a=v.baseline(d,k,part);z=predict(d,k,part,c);p=v.mix(m,a,z,w)
    return dict(part=part,rows=len(q),baseline=r83.score(m,d['y'][q],a),expert=r83.score(m,d['y'][q],z),candidate=r83.score(m,d['y'][q],p),gain=r83.gain(m,d['y'][q],a,p))
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    d=v.load();tasks=[t for t in v.tasks(d) if d['metrics'][d['tasks'].index(t)] in ('AUROC','AUPRC')];OUT.mkdir(exist_ok=True)
    doc=dict(signature=sig(),tasks=tasks,configs=CONFIGS,weights=WEIGHTS,method='Exact SVC with six count Tanimoto, augmented physicochemical RBF and target-excluded official assay RBF kernels.',
        selection='Freeze maxminimum fold0/1 config and blend; all4 gain positive, mean exceeds threshold; no fallback.',baseline='v130 controls; historical OOF caveats apply.',test_labels_used=False,external_data=False)
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(doc))
    else:b.save(path,doc)
    development=[];selected=[]
    for task in tasks:
        k=d['tasks'].index(task)
        for part in ('fold0','fold1'):family(d,k,part,CONFIGS)
        choices=[]
        for c in CONFIGS:
            for w in WEIGHTS:
                parts=[record(d,k,p,c,w) for p in ('fold0','fold1')];choices.append(dict(task=task,metric=d['metrics'][k],config=c,weight=w,parts=parts,objective=min(r['gain'] for r in parts)))
        best=max(choices,key=lambda r:r['objective']);development.extend(choices)
        if best['objective']>v.threshold(d['metrics'][k]):selected.append(best)
        b.save(OUT/'development.json',dict(status='IN_PROGRESS',signature=sig(),records=development));print(json.dumps(dict(stage='development',best=best)),flush=True)
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=development));b.save(OUT/'frozen_selection.json',dict(signature=sig(),selected=selected))
    records=[]
    for chosen in selected:
        k=d['tasks'].index(chosen['task']);parts=[record(d,k,p,chosen['config'],chosen['weight']) for p in ('fold0','fold1','fold2','valid')]
        accepted=bool(min(r['gain'] for r in parts)>0 and np.mean([r['gain'] for r in parts])>v.threshold(d['metrics'][k]))
        records.append(dict(**chosen,confirmation=parts,accepted=accepted));print(json.dumps(dict(stage='confirmation',**records[-1])),flush=True)
        if accepted:predict(d,k,'final',chosen['config'])
        b.save(OUT/'confirmation.json',dict(status='IN_PROGRESS',signature=sig(),records=records))
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']]))
if __name__=='__main__':main()
