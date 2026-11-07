"""Unlabeled query distribution adaptation, with identical unweighted control.

Cross-fitted domain classifiers estimate covariate weights. Query structures,
never query labels, define the target domain. Official data only.
"""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='4'
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier,CatBoostRegressor
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v115_diverse_specialists as prev
b,r83=prev.b,prev.r83
OUT=ROOT/'admet_model_outputs_v116'
TASKS=('solubility_aqsoldb','caco2_wang','ames','bbb_martins','herg',
       'clearance_hepatocyte_az','clearance_microsome_az','ppbr_az')
WEIGHT=.4;SEEDS=(1164026,1164027)

def sig():return hashlib.sha256((b.sha(__file__)+prev.sig()).encode()).hexdigest()
def emit(**x):print(json.dumps(x,ensure_ascii=False),flush=True)
def save(name,x):b.save(OUT/name,x)

def protocol():
    doc=dict(version=116,signature=sig(),tasks=TASKS,seeds=SEEDS,weight=WEIGHT,
        domain='3fold crossfit XGBoost depth3/120trees on435physchem, each query structure weighted nfit/nquery; odds^0.5 clipped [.25,4], fit mean1.',
        target='GPU CatBoost depth6/1400trees/lr.025/l2=25, six fingerprints+435physchem. Logloss/MAE/log1pRMSE. Two seed ensemble.',
        control='Identical fit rows, seeds, model parameters without covariate weights.',
        selection='fold0/1 positive vs historical proxy with mean>1.2%MAE/.008rho/.0025AUC; weighted model must beat same-row unweighted control in both.',
        confirmation='Frozen tasks, fold2 and valid positive vs proxy, weighted-control mean nonnegative. No fallback retuning.',
        transductive_unlabeled_query_structures=True,test_labels_used=False,external_data=False,
        limitations='All labels historically reused; full v114 OOF unavailable. Crossfit domain weighting corrects covariate shift only, not target shift. No official score estimate.')
    OUT.mkdir(exist_ok=True);p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(doc))
    else:b.save(p,doc)

def design(d,k,part):
    fit,q=r83.split(d,k,part);x=d['views']['multiview'][d['mi'][fit]].copy();qx=d['views']['multiview'][d['mi'][q]].copy()
    x[~np.isfinite(x)]=np.nan;qx[~np.isfinite(qx)]=np.nan
    xx=np.r_[d['views']['physchem'][d['mi'][fit]],d['views']['physchem'][d['mi'][q]]].copy()
    xx[~np.isfinite(xx)]=np.nan
    domain=np.r_[np.zeros(len(fit)),np.ones(len(q))]
    weights=np.r_[np.ones(len(fit)),np.full(len(q),len(fit)/len(q))]
    assignment=np.array([int(hashlib.sha256(str(s).encode()).hexdigest()[:10],16)%3 for s in d['canonical'][np.r_[fit,q]]])
    folder=OUT/'models'/d['tasks'][k]/part;folder.mkdir(parents=True,exist_ok=True)
    probability=np.zeros(len(fit))
    for fold in range(3):
        train=assignment!=fold;query=np.flatnonzero(assignment[:len(fit)]==fold)
        model=xgb.train(dict(objective='binary:logistic',max_depth=3,eta=.04,min_child_weight=10,
            subsample=.9,colsample_bytree=.8,reg_lambda=20,nthread=4,seed=116200+fold,tree_method='hist',max_bin=64),
            xgb.DMatrix(xx[train],label=domain[train],weight=weights[train]),num_boost_round=120)
        probability[query]=model.predict(xgb.DMatrix(xx[query]));model.save_model(folder/f'domain_{fold}.ubj')
    odds=np.clip(probability,1e-4,1-1e-4)/(1-np.clip(probability,1e-4,1-1e-4))
    importance=np.clip(np.sqrt(odds),.25,4.);importance/=importance.mean()
    assert np.isfinite(importance).all();return fit,q,x,qx,importance

def predict(d,k,part):
    folder=OUT/'models'/d['tasks'][k]/part;file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);assert str(z['signature'])==sig();np.testing.assert_array_equal(z['query'],r83.split(d,k,part)[1]);return z['weighted'],z['control']
    fit,q,x,qx,weights=design(d,k,part);y=d['y'][fit];m=d['metrics'][k];cls=m in ('AUROC','AUPRC')
    target=np.log1p(y) if m=='Spearman' else y
    result={};started=time.monotonic()
    for kind in ('weighted','control'):
        members=[]
        for seed in SEEDS:
            net=(CatBoostClassifier if cls else CatBoostRegressor)(iterations=1400,depth=6,learning_rate=.025,l2_leaf_reg=25,
                random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False,
                bootstrap_type='Bernoulli',subsample=.85,task_type='GPU',devices='0',border_count=64,
                loss_function='Logloss' if cls else 'MAE' if m=='MAE' else 'RMSE')
            net.fit(x,target,sample_weight=weights if kind=='weighted' else np.ones(len(fit)))
            raw=net.predict(qx,prediction_type='RawFormulaVal');p=expit(raw) if cls else raw
            mp=folder/f'{kind}_{seed}.cbm';net.save_model(str(mp))
            replay=CatBoostClassifier() if cls else CatBoostRegressor();replay.load_model(str(mp))
            rp=replay.predict(qx,prediction_type='RawFormulaVal');np.testing.assert_allclose(raw,rp,rtol=0,atol=1e-10)
            members.append(p)
        result[kind]=np.mean(members,axis=0)
    np.savez_compressed(file,signature=sig(),fit=fit,query=q,importance=weights,**result)
    emit(stage='trained',task=d['tasks'][k],part=part,seconds=round(time.monotonic()-started),effective_fit=float(weights.sum()**2/(weights**2).sum()))
    return result['weighted'],result['control']

def record(d,k,part):
    m=d['metrics'][k];q=r83.split(d,k,part)[1];y=d['y'][q];a=prev.base(d,k,part);z,c=predict(d,k,part)
    p=prev.mix(m,a,z)
    return dict(part=part,rows=len(q),baseline_proxy=r83.score(m,y,a),weighted=r83.score(m,y,z),
        unweighted_control=r83.score(m,y,c),candidate=r83.score(m,y,p),gain=r83.gain(m,y,a,p),weighting_gain=r83.gain(m,y,c,z))

def develop(d):
    protocol();records=[]
    for task in TASKS:
        k=d['tasks'].index(task);m=d['metrics'][k];parts=[record(d,k,p) for p in ('fold0','fold1')]
        selected=bool(all(p['gain']>0 and p['weighting_gain']>0 for p in parts) and np.mean([p['gain'] for p in parts])>prev.threshold(m))
        r=dict(task=task,metric=m,parts=parts,selected=selected);records.append(r);emit(stage='development',**r)
        save('development.json',dict(status='IN_PROGRESS',records=records))
    save('development.json',dict(status='COMPLETE',signature=sig(),records=records))
    save('frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=sig(),selected=[r for r in records if r['selected']]))

def confirm(d):
    frozen=json.loads((OUT/'frozen_selection.json').read_text());assert frozen['signature']==sig();records=[]
    for r in frozen['selected']:
        k=d['tasks'].index(r['task']);parts=[record(d,k,p) for p in ('fold0','fold1','fold2','valid')]
        accepted=bool(all(p['gain']>0 for p in parts) and np.mean([p['gain'] for p in parts])>prev.threshold(r['metric']) and np.mean([p['weighting_gain'] for p in parts])>0)
        rr=dict(**r,confirmation=parts,accepted=accepted);records.append(rr);emit(stage='confirmation',**rr)
        save('confirmation.json',dict(status='IN_PROGRESS',records=records))
    save('confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']]))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['develop','confirm']);args=ap.parse_args()
    dict(develop=develop,confirm=confirm)[args.stage](prev.load())
