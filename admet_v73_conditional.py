"""Conditional endpoint experts using only official OTHER-assay train labels.

Development auxiliaries use train only, including for official-valid queries.
Final auxiliaries use train+valid. The target assay is absent from the design.
This measures prediction conditional on available public assays, not pure
structure-only scaffold generalization. No hidden test labels are parsed.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='4'
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier,CatBoostRegressor
import admet_v67_data as b
b.rdBase.rdkitVersion='2025.03.6'

OUT=b.ROOT/'admet_model_outputs_v73_conditional'
SEED=733026
TASKS=('clearance_hepatocyte_az','clearance_microsome_az','half_life_obach',
       'vdss_lombardo','ppbr_az','bbb_martins','bioavailability_ma',
       'cyp2d6_substrate_carbonmangels','cyp2c9_substrate_carbonmangels',
       'cyp3a4_substrate_carbonmangels','pgp_broccatelli')
PARTS=('fold0','fold1','fold2','valid')

def signature():
    return hashlib.sha256((b.signature()+b.sha(__file__)).encode()).hexdigest()

def rows(d,k,part):
    f=np.load(b.OUT/'folds.npy');task=d['tid']==k
    if part=='final':fit=np.flatnonzero(task&(d['split']<2));q=np.flatnonzero(task&(d['split']==2))
    elif part=='valid':fit=np.flatnonzero(task&(d['split']==0));q=np.flatnonzero(task&(d['split']==1))
    else:
        fold=int(part[-1]);fit=np.flatnonzero(task&(d['split']==0)&(f!=fold));q=np.flatnonzero(task&(d['split']==0)&(f==fold))
    b.assert_excluded(d,fit,q)
    return fit,q

def auxiliary(d,k,part):
    allowed=d['split']<(2 if part=='final' else 1)
    source=np.flatnonzero(allowed&(d['tid']!=k))
    assert not np.any(d['tid'][source]==k) and np.all(d['split'][source]<2)
    cols=[j for j in range(len(d['tasks'])) if j!=k];lookup={j:i for i,j in enumerate(cols)}
    labels={}
    for i in source:
        key=(str(d['canonical'][d['mi'][i]]),int(d['tid'][i]))
        labels.setdefault(key,[]).append(float(d['y'][i]))
    nc=len(cols);aux=np.full((len(d['smiles']),2*nc),np.nan,np.float32)
    aux[:,nc:]=0
    # Values and masks are both absent for the target. Median aggregates
    # replicate measurements; training-only fold predictions never supply Y.
    for i,can in enumerate(d['canonical']):
        for task,col in lookup.items():
            values=labels.get((str(can),task))
            if values:
                aux[i,col]=np.median(values);aux[i,nc+col]=1
    audit=dict(source_rows=len(source),source_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
        allowed_splits=[0,1] if part=='final' else [0],source_tasks=[d['tasks'][j] for j in cols],
        target_task=d['tasks'][k],target_aux_values_and_masks_absent=True,
        test_labels_used=False,external_data=False,pretrained_weights=False,
        conditional_assay_evaluation=True,auxiliary_identity='canonical structure without stereochemistry')
    assert d['tasks'][k] not in audit['source_tasks']
    return aux,audit

def design(d,k,part,fit,q,view):
    aux,audit=auxiliary(d,k,part);mol=d['mi']
    x=d['x'][mol[fit],4096:];qx=d['x'][mol[q],4096:]
    if view=='aux':
        x=np.concatenate([x,aux[mol[fit]]],1);qx=np.concatenate([qx,aux[mol[q]]],1)
    elif view!='control':raise ValueError(view)
    nc=(len(d['tasks'])-1)
    audit.update(query_coverage=float(np.mean(aux[mol[q],nc:].sum(1)>0)),
                 fit_coverage=float(np.mean(aux[mol[fit],nc:].sum(1)>0)),view=view)
    return x,qx,audit

def fit_predict(d,k,part,view='aux',iterations=1400,seed=SEED):
    task=d['tasks'][k];metric=d['metrics'][k];fit,q=rows(d,k,part)
    folder=OUT/task/part/view/str(seed);folder.mkdir(parents=True,exist_ok=True)
    cache=folder/'predictions.npz';sig=signature()
    if cache.exists():
        z=np.load(cache);assert str(z['signature'])==sig
        np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['query'],q)
        meta=json.loads((folder/'metadata.json').read_text());assert b.sha(folder/'model.cbm')==meta['model_sha256']
        assert meta['config']['iterations']==iterations
        return z['p'].copy()
    start=time.monotonic();cls=metric in ('AUROC','AUPRC');tf=b.target(d['y'][fit],metric)
    x,qx,audit=design(d,k,part,fit,q,view)
    cfg=dict(iterations=iterations,depth=5,learning_rate=.035,l2_leaf_reg=15,
        loss_function='Logloss' if cls else 'RMSE',
        eval_metric='AUC' if metric=='AUROC' else 'PRAUC' if metric=='AUPRC' else 'MAE' if metric=='MAE' else 'RMSE',
        task_type='CPU',thread_count=4,random_seed=seed,verbose=False,
        allow_writing_files=False,border_count=128,bootstrap_type='Bernoulli',subsample=.85)
    model=(CatBoostClassifier if cls else CatBoostRegressor)(**cfg)
    _,inv,cnt=np.unique(d['canonical'][d['mi'][fit]],return_inverse=True,return_counts=True)
    weights=1/cnt[inv];weights/=weights.mean()
    if part=='final':model.fit(x,b.encode(d['y'][fit],tf),sample_weight=weights)
    else:model.fit(x,b.encode(d['y'][fit],tf),sample_weight=weights,
        eval_set=(qx,b.encode(d['y'][q],tf)),use_best_model=True,early_stopping_rounds=160)
    p=model.predict_proba(qx)[:,1] if cls else model.predict(qx)
    if metric=='MAE':p=p*tf['scale']+tf['center']
    assert np.isfinite(p).all();model.save_model(str(folder/'model.cbm'))
    np.savez_compressed(cache,p=p,fit=fit,query=q,signature=sig)
    serial={key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in tf.items()}
    b.save(folder/'metadata.json',dict(signature=sig,task=task,part=part,view=view,seed=seed,
        trees=model.tree_count_,config=cfg,target=serial,audit=audit,
        model_sha256=b.sha(folder/'model.cbm'),prediction_sha256=b.sha(cache),
        independent_reload_pending=True))
    b.emit('v73_conditional_fit',task=task,part=part,view=view,trees=model.tree_count_,
        score=b.metric_score(metric,d['y'][q],p) if part!='final' else None,
        coverage=audit['query_coverage'],seconds=round(time.monotonic()-start))
    return p

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tasks',nargs='+',default=TASKS)
    a=parser.parse_args();assert os.name!='nt';d=b.build()
    b.save(OUT/'protocol.json',dict(signature=signature(),tasks=a.tasks,parts=PARTS,
        views=['control','aux'],development_auxiliary_splits=[0],final_auxiliary_splits=[0,1],
        target_task_absent_from_auxiliaries=True,test_labels_used=False,
        conditional_evaluation=True,independent_unbiased_evaluation=False))
    for task in a.tasks:
        k=d['tasks'].index(task)
        for part in PARTS:
            for view in ('control','aux'):fit_predict(d,k,part,view)
    b.save(OUT/'development_completed.json',dict(status='COMPLETE',signature=signature()))

if __name__=='__main__':main()
