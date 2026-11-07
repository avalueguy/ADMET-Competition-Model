"""Exact conditional molecular kernels for the selected low-score tasks."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='3'
import hashlib,json,sys,time
from pathlib import Path
import joblib,numpy as np
from sklearn.svm import SVC
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v125_low_task_baseline as v
b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v128';WEIGHT=.5
TASKS=('cyp3a4_substrate_carbonmangels','herg','cyp2d6_substrate_carbonmangels','pgp_broccatelli')
CONFIGS=[dict(view=view,C=c) for view in ('sum','product') for c in (1.,10.)]
def sig():return hashlib.sha256((b.sha(__file__)+v.sig()).encode()).hexdigest()
def key(cfg):return cfg['view']+'_C'+str(cfg['C'])
def matrices(d,k,part):
    fit,q=v.split(d,k,part);rows=np.r_[fit,q];n=len(fit);raw=d['views']['multiview'];chem=np.zeros((len(rows),n),np.float64)
    for lo in range(0,12288,2048):
        x=raw[d['mi'][rows],lo:lo+2048].astype(float);ref=x[:n];dot=x@ref.T
        tan=np.clip(dot/np.maximum(x.square().sum(1)[:,None]+ref.square().sum(1)[None]-dot,1e-12),0,1) if hasattr(x,'square') else np.clip(dot/np.maximum((x*x).sum(1)[:,None]+(ref*ref).sum(1)[None]-dot,1e-12),0,1)
        chem+=(.5*tan+.5*tan**2)/6
    aux,cols,source,_,_=v.old.assays.assays(d,k,part);assert k not in cols and np.all(d['tid'][source]!=k)
    a=aux[rows].astype(float);tr=a[:n];center=np.array([np.median(x[np.isfinite(x)]) if np.isfinite(x).any() else 0. for x in tr.T])
    scale=np.array([np.std(x[np.isfinite(x)]) if np.isfinite(x).any() else 1. for x in tr.T]).clip(.05)
    z=np.nan_to_num((a-center)/scale).clip(-5,5);active=z[:n].std(0)>.01;z=z[:,active];dim=max(1,z.shape[1])
    dist=np.maximum((z*z).sum(1)[:,None]+(z[:n]*z[:n]).sum(1)[None]-2*z@z[:n].T,0)/dim
    assay=.5*np.exp(-dist/.5)+.5*np.exp(-dist/2)
    results={'sum':.8*chem+.2*assay,'product':chem*(.5+.5*assay)}
    return fit,q,{name:((value[:n]+value[:n].T)*.5,value[n:]) for name,value in results.items()}
def family(d,k,part,configs):
    folder=OUT/'models'/d['tasks'][k]/part;folder.mkdir(parents=True,exist_ok=True);results={};missing=[]
    for cfg in configs:
        p=folder/(key(cfg)+'.npz')
        if p.exists():
            z=np.load(p);assert str(z['signature'])==sig();np.testing.assert_array_equal(z['query'],v.split(d,k,part)[1]);results[key(cfg)]=z['p'].copy()
        else:missing.append(cfg)
    if missing:
        start=time.monotonic();fit,q,mm=matrices(d,k,part)
        for cfg in missing:
            x,qx=mm[cfg['view']];model=SVC(C=cfg['C'],kernel='precomputed',tol=1e-5,cache_size=768)
            model.fit(x,d['y'][fit]);assert model.fit_status_==0;raw=model.decision_function(qx)
            file=folder/(key(cfg)+'.joblib');joblib.dump(model,file,compress=3);replay=joblib.load(file);np.testing.assert_array_equal(raw,replay.decision_function(qx))
            p=expit(raw);np.savez_compressed(folder/(key(cfg)+'.npz'),signature=sig(),fit=fit,query=q,p=p)
            results[key(cfg)]=p
        print(json.dumps(dict(stage='train',task=d['tasks'][k],part=part,models=len(missing),seconds=round(time.monotonic()-start))),flush=True)
    return results
def predict(d,k,part,cfg):return family(d,k,part,[cfg])[key(cfg)]
def record(d,k,part,cfg):
    q=v.split(d,k,part)[1];a=v.baseline(d,k,part);z=predict(d,k,part,cfg);p=v.mix(d['metrics'][k],a,z,WEIGHT)
    return dict(part=part,rows=len(q),baseline=r83.score(d['metrics'][k],d['y'][q],a),candidate=r83.score(d['metrics'][k],d['y'][q],p),gain=r83.gain(d['metrics'][k],d['y'][q],a,p))
def main():
    d=v.load();OUT.mkdir(exist_ok=True);doc=dict(signature=sig(),tasks=TASKS,configs=CONFIGS,weight=WEIGHT,
        method='Exact logcount multiview Tanimoto kernel with target-excluded assay RBF, additive/product; SVC margin.',
        selection='Choose max-min config fold0/1>.0025; freeze before fold2/valid. All4 gains positive and mean>.0025. No fallback.',
        protected_tasks_immutable=True,external_data=False,test_labels_used=False,baseline='v124 carried branches, CYP3A4 exact matched v67 target rows; historical proxy limitations elsewhere')
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(doc))
    else:b.save(p,doc)
    development=[];selected=[]
    for task in TASKS:
        k=d['tasks'].index(task)
        for part in ('fold0','fold1'):family(d,k,part,CONFIGS)
        choices=[]
        for cfg in CONFIGS:
            parts=[record(d,k,part,cfg) for part in ('fold0','fold1')];choices.append(dict(task=task,config=cfg,parts=parts,objective=min(x['gain'] for x in parts)))
        best=max(choices,key=lambda r:r['objective']);development.extend(choices)
        if best['objective']>.0025:selected.append(best)
        print(json.dumps(dict(stage='development',best=best)),flush=True)
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=development))
    b.save(OUT/'frozen_selection.json',dict(signature=sig(),selected=selected))
    records=[]
    for chosen in selected:
        k=d['tasks'].index(chosen['task']);parts=[record(d,k,part,chosen['config']) for part in ('fold0','fold1','fold2','valid')]
        accepted=bool(min(x['gain'] for x in parts)>0 and np.mean([x['gain'] for x in parts])>.0025)
        records.append(dict(**chosen,confirmation=parts,accepted=accepted));print(json.dumps(dict(stage='confirmation',**records[-1])),flush=True)
        if accepted:predict(d,k,'final',chosen['config'])
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']]))
if __name__=='__main__':main()
