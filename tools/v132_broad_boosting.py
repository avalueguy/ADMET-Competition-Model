"""Broad GPU boosted specialists with shape/pharmacophore and assay context."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import xgboost as xgb
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v131_current_controls as v
b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v132';SEEDS=(1324026,1324027);WEIGHTS=(.35,.75)
CONFIGS=(dict(view='augmented',depth=3,trees=1100),dict(view='full',depth=6,trees=900))
def sig():return hashlib.sha256((b.sha(__file__)+v.sig()).encode()).hexdigest()
def key(c):return c['view']+'_d'+str(c['depth'])
def predict(d,k,part,c):
    folder=OUT/'models'/d['tasks'][k]/part/key(c);file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);assert str(z['signature'])==sig();np.testing.assert_array_equal(z['query'],v.split(d,k,part)[1]);return z['p'].copy()
    fit,q,x,qx,source=v.features(d,k,part,c['view']);y=d['y'][fit];m=d['metrics'][k];cls=m in ('AUROC','AUPRC')
    center=float(np.median(y)) if m=='MAE' else 0.;scale=max(float(np.subtract(*np.quantile(y,[.75,.25]))),.05) if m=='MAE' else 1.
    target=y if cls else (y-center)/scale if m=='MAE' else r83.ranknorm(y)
    objective='binary:logistic' if cls else 'reg:absoluteerror' if m=='MAE' else 'reg:squarederror'
    folder.mkdir(parents=True,exist_ok=True);members=[];start=time.monotonic();dm=xgb.DMatrix(x,label=target);qm=xgb.DMatrix(qx)
    for seed in SEEDS:
        model=xgb.train(dict(objective=objective,max_depth=c['depth'],eta=.025,reg_lambda=20.,reg_alpha=.05,min_child_weight=3.,
            subsample=.85,colsample_bytree=.7,max_bin=64,nthread=4,seed=seed,device='cuda',tree_method='hist',verbosity=0),dm,num_boost_round=c['trees'])
        raw=model.predict(qm);mp=folder/f'model_{seed}.ubj';model.save_model(mp)
        replay=xgb.Booster();replay.load_model(mp);np.testing.assert_array_equal(raw,replay.predict(qm));members.append(raw)
    p=np.mean(members,axis=0).astype(float)*scale+center
    if d['tasks'][k]=='ppbr_az':p=p.clip(0,100)
    np.savez_compressed(file,signature=sig(),fit=fit,query=q,p=p,source=source,center=center,scale=scale)
    print(json.dumps(dict(stage='train',task=d['tasks'][k],part=part,config=c,seconds=round(time.monotonic()-start))),flush=True);return p
def record(d,k,part,c,w):
    q=v.split(d,k,part)[1];m=d['metrics'][k];a=v.baseline(d,k,part);z=predict(d,k,part,c);p=v.mix(m,a,z,w)
    return dict(part=part,rows=len(q),baseline=r83.score(m,d['y'][q],a),expert=r83.score(m,d['y'][q],z),candidate=r83.score(m,d['y'][q],p),gain=r83.gain(m,d['y'][q],a,p))
def main():
    d=v.load();OUT.mkdir(exist_ok=True);doc=dict(signature=sig(),tasks=v.tasks(d),configs=CONFIGS,seeds=SEEDS,weights=WEIGHTS,
        model='GPU XGBoost; augmented descriptors+3Dshape+pharmacophore or full structural counts; target-excluded official assay context.',
        selection='Max minimum gain on fold0/1 across two configurations and fixed weights; freeze before fold2/valid.',
        confirmation='All4 gains>0, mean>metric threshold; rejected task has no fallback. HIA AUROC1 unchanged.',
        baseline='v130 reconstructed/proxy controls documented in v131; previous label reuse; local scores are not official estimates.',test_labels_used=False,external_data=False)
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(doc))
    else:b.save(path,doc)
    development=[];selected=[]
    for task in v.tasks(d):
        k=d['tasks'].index(task);choices=[]
        for c in CONFIGS:
            for w in WEIGHTS:
                parts=[record(d,k,part,c,w) for part in ('fold0','fold1')]
                choices.append(dict(task=task,metric=d['metrics'][k],config=c,weight=w,parts=parts,objective=min(r['gain'] for r in parts)))
        best=max(choices,key=lambda r:r['objective']);development.extend(choices)
        if best['objective']>v.threshold(d['metrics'][k]):selected.append(best)
        b.save(OUT/'development.json',dict(status='IN_PROGRESS',signature=sig(),records=development));print(json.dumps(dict(stage='development',best=best)),flush=True)
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=development))
    b.save(OUT/'frozen_selection.json',dict(signature=sig(),selected=selected))
    records=[]
    for chosen in selected:
        k=d['tasks'].index(chosen['task']);parts=[record(d,k,p,chosen['config'],chosen['weight']) for p in ('fold0','fold1','fold2','valid')]
        accepted=bool(min(r['gain'] for r in parts)>0 and np.mean([r['gain'] for r in parts])>v.threshold(d['metrics'][k]))
        records.append(dict(**chosen,confirmation=parts,accepted=accepted));print(json.dumps(dict(stage='confirmation',**records[-1])),flush=True)
        if accepted:predict(d,k,'final',chosen['config'])
        b.save(OUT/'confirmation.json',dict(status='IN_PROGRESS',signature=sig(),records=records))
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']]))
if __name__=='__main__':main()
