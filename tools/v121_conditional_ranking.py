"""Full structural conditional LambdaRank specialists with fixed blending."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='3'
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import xgboost as xgb
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v117_conditional_structure as v
b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v121';WEIGHT=.4;SEEDS=(1214026,1214027)
TASKS=('cyp2d6_substrate_carbonmangels','cyp3a4_substrate_carbonmangels',
       'clearance_hepatocyte_az','clearance_microsome_az','herg','ames','pgp_broccatelli')

def sig():return hashlib.sha256((b.sha(__file__)+v.sig()).encode()).hexdigest()
def emit(**x):print(json.dumps(x),flush=True)
def predict(d,k,part):
    folder=OUT/'models'/d['tasks'][k]/part;path=folder/'predictions.npz'
    if path.exists():
        z=np.load(path);assert str(z['signature'])==sig();np.testing.assert_array_equal(z['query'],r83.split(d,k,part)[1]);return z['p'].copy()
    fit,q,x,qx,source=v.design(d,k,part,'full');y=d['y'][fit];m=d['metrics'][k]
    label=np.minimum(30,np.floor(31*(r83.rankdata(y)-1)/len(y))) if m=='Spearman' else y
    # A single task-wide query optimizes global discrimination; uniform sampling
    # avoids limiting training to a handful of the current highest predictions.
    dm=xgb.DMatrix(x,label=label);dm.set_group([len(fit)]);qm=xgb.DMatrix(qx)
    objective='rank:map' if m=='AUPRC' else 'rank:pairwise'
    folder.mkdir(parents=True,exist_ok=True);members=[];started=time.monotonic()
    for seed in SEEDS:
        model=xgb.train(dict(objective=objective,max_depth=4,eta=.03,reg_lambda=30,reg_alpha=.1,
            min_child_weight=6,subsample=.9,colsample_bytree=.8,max_bin=64,nthread=3,
            lambdarank_pair_method='mean',lambdarank_num_pair_per_sample=32,
            seed=seed,device='cpu',tree_method='hist',verbosity=0),dm,num_boost_round=650)
        raw=model.predict(qm);mp=folder/f'model_{seed}.ubj';model.save_model(mp)
        replay=xgb.Booster();replay.load_model(mp);np.testing.assert_array_equal(raw,replay.predict(qm))
        members.append(r83.rankdata(raw)/len(raw))
    p=np.mean(members,axis=0)
    np.savez_compressed(path,signature=sig(),fit=fit,query=q,source=source,p=p)
    emit(stage='train',task=d['tasks'][k],part=part,objective=objective,seconds=round(time.monotonic()-started))
    return p
def record(d,k,part):
    q=r83.split(d,k,part)[1];y=d['y'][q];m=d['metrics'][k];a=v.prev.base(d,k,part);z=predict(d,k,part)
    p=(1-WEIGHT)*r83.rankdata(a)/len(a)+WEIGHT*r83.rankdata(z)/len(z)
    return dict(part=part,rows=len(q),baseline_proxy=r83.score(m,y,a),expert=r83.score(m,y,z),
        candidate=r83.score(m,y,p),gain=r83.gain(m,y,a,p))
def run():
    d=v.prev.load();OUT.mkdir(exist_ok=True)
    doc=dict(signature=sig(),tasks=TASKS,weight=WEIGHT,seeds=SEEDS,
        model='CPU XGBoost LambdaRank650treesdepth4:MAP forAP, pairwise forAUROC/Spearman,32 uniform pairs per fit row.',
        features='Full12288 structural fragment counts+435physchem+target-excluded official other-task assays.',
        selection='One fixed model; both fold0/1 gains>.0025 classification/.008Spearman; freeze tasks.',
        confirmation='All four gains positive and mean above practical threshold; no fallback.',
        test_labels_used=False,external_data=False,limitations='Public labels reused. v101 proxy except replayed hERG v85; full v114 OOF unavailable.')
    pp=OUT/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==json.loads(json.dumps(doc))
    else:b.save(pp,doc)
    records=[];selected=[]
    for task in TASKS:
        k=d['tasks'].index(task);m=d['metrics'][k];parts=[record(d,k,p) for p in ('fold0','fold1')]
        r=dict(task=task,metric=m,parts=parts);r['selected']=min(p['gain'] for p in parts)>v.prev.threshold(m)
        records.append(r)
        if r['selected']:selected.append(r)
        emit(stage='development',**r);b.save(OUT/'development.json',dict(status='IN_PROGRESS',records=records))
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=records))
    b.save(OUT/'frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=sig(),selected=selected))
    confirmation=[]
    for r in selected:
        k=d['tasks'].index(r['task']);parts=[record(d,k,p) for p in ('fold0','fold1','fold2','valid')]
        z=dict(**r,confirmation=parts,accepted=bool(min(p['gain'] for p in parts)>0 and np.mean([p['gain'] for p in parts])>v.prev.threshold(r['metric'])))
        confirmation.append(z);emit(stage='confirmation',**z)
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=confirmation,accepted=[r['task'] for r in confirmation if r['accepted']]))

if __name__=='__main__':run()
