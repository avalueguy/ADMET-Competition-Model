"""Low-score hepatocyte and solubility experts, immutable protected tasks."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='3'
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v125_low_task_baseline as v
b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v126';SEEDS=(1264026,1264027);WEIGHT=.5
TASKS=('clearance_hepatocyte_az','solubility_aqsoldb')
def sig():return hashlib.sha256((b.sha(__file__)+v.sig()).encode()).hexdigest()
def predict(d,k,part,view):
    folder=OUT/'models'/d['tasks'][k]/part/view;path=folder/'predictions.npz'
    if path.exists():
        z=np.load(path);assert str(z['signature'])==sig();return z['p'].copy()
    fit,q,x,qx,source=v.features(d,k,part,view);y=d['y'][fit];metric=d['metrics'][k]
    # Two objectives share partitions; query labels do not define targets/scales.
    first=np.log1p(y) if metric=='Spearman' else y
    second=r83.ranknorm(y) if metric=='Spearman' else np.arcsinh((y-np.median(y))/max(np.std(y),.05))
    target=np.c_[first,second].astype(np.float32);center=target.mean(0);scale=target.std(0).clip(.05);target=(target-center)/scale
    folder.mkdir(parents=True,exist_ok=True);members=[];start=time.monotonic()
    for seed in SEEDS:
        net=CatBoostRegressor(loss_function='MultiRMSE',iterations=1000,depth=4,learning_rate=.025,l2_leaf_reg=30,
            random_seed=seed,thread_count=3,verbose=False,allow_writing_files=False,bootstrap_type='Bernoulli',subsample=.85,border_count=64)
        net.fit(x,target);raw=net.predict(qx);file=folder/f'model_{seed}.cbm';net.save_model(str(file))
        reload=CatBoostRegressor();reload.load_model(str(file));np.testing.assert_array_equal(raw,reload.predict(qx))
        if metric=='Spearman':members.append(.5*r83.rankdata(raw[:,0])/len(q)+.5*r83.rankdata(raw[:,1])/len(q))
        else:members.append(raw[:,0]*scale[0]+center[0])
    p=np.mean(members,axis=0);np.savez_compressed(path,signature=sig(),p=p,fit=fit,query=q,source=source,center=center,scale=scale)
    print(json.dumps(dict(stage='train',task=d['tasks'][k],part=part,view=view,seconds=round(time.monotonic()-start))),flush=True);return p
def record(d,k,part,view):
    q=v.split(d,k,part)[1];m=d['metrics'][k];a=v.baseline(d,k,part);p=v.mix(m,a,predict(d,k,part,view),WEIGHT)
    return dict(part=part,rows=len(q),baseline=r83.score(m,d['y'][q],a),candidate=r83.score(m,d['y'][q],p),gain=r83.gain(m,d['y'][q],a,p))
def main():
    d=v.load();OUT.mkdir(exist_ok=True);doc=dict(signature=sig(),tasks=TASKS,seeds=SEEDS,weight=WEIGHT,views=['physchem','full'],
        selection='Choose view max-min gain fold0/1 above .008Spearman/.01relativeMAE; freeze before confirmation.',
        confirmation='All4 gains strictly positive, mean above threshold; no rescue. Protected tasks immutable.',
        baseline='v124 carried branches; solubility historical proxy caveat',test_labels_used=False,external_data=False)
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(doc))
    else:b.save(p,doc)
    selected=[];development=[]
    for task in TASKS:
        k=d['tasks'].index(task);choices=[]
        for view in ('physchem','full'):
            parts=[record(d,k,part,view) for part in ('fold0','fold1')];choices.append(dict(task=task,view=view,parts=parts,objective=min(r['gain'] for r in parts)))
        best=max(choices,key=lambda r:r['objective']);development.extend(choices)
        if best['objective']>v.threshold(d['metrics'][k]):selected.append(best)
        print(json.dumps(dict(stage='development',best=best)),flush=True)
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=development))
    b.save(OUT/'frozen_selection.json',dict(signature=sig(),selected=selected))
    records=[]
    for chosen in selected:
        k=d['tasks'].index(chosen['task']);parts=[record(d,k,part,chosen['view']) for part in ('fold0','fold1','fold2','valid')]
        accepted=bool(min(r['gain'] for r in parts)>0 and np.mean([r['gain'] for r in parts])>v.threshold(d['metrics'][k]))
        records.append(dict(**chosen,confirmation=parts,accepted=accepted));print(json.dumps(dict(stage='confirmation',**records[-1])),flush=True)
        if accepted:predict(d,k,'final',chosen['view'])
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']]))
if __name__=='__main__':main()
