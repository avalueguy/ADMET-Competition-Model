"""Conditional assay models use supplied OTHER-task labels, never hidden targets."""
import os,json,time,hashlib
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier,CatBoostRegressor
import admet_v24_models as b
from admet_v25_run import v24_valid
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'admet_model_outputs_v30'
P=json.loads((ROOT/'admet_v30_protocol.json').read_text())

def signature():
    return hashlib.sha256(json.dumps(dict(base=b.signature(),files={n:b.sha(ROOT/n) for n in ['admet_v30_run.py','admet_v30_protocol.json']}),sort_keys=True).encode()).hexdigest()

def auxiliary(d,source_splits,target):
    # Drop the complete target task before touching target values.
    rows=np.flatnonzero(np.isin(d['split'],source_splits)&(d['tid']!=target));assert np.all(d['split'][rows]<2)
    assert not np.any(d['tid'][rows]==target)
    canonical=d['canonical'][d['mi']];identities,iv=np.unique(canonical,return_inverse=True)
    values=np.zeros((len(identities),len(d['tasks'])),np.float32);observed=np.zeros_like(values)
    meta=[]
    for k in range(len(d['tasks'])):
        if k==target:continue
        ix=rows[d['tid'][rows]==k];yy=d['y'][ix];tf=b.Target(yy,d['metrics'][k])
        order=np.argsort(iv[ix],kind='stable');ix=ix[order];group=iv[ix];starts=np.r_[0,np.flatnonzero(np.diff(group))+1,len(ix)]
        for start,end in zip(starts[:-1],starts[1:]):
            cur=ix[start:end];g=group[start]
            y=float(np.mean(d['y'][cur])) if d['metrics'][k] in ['AUROC','AUPRC'] else float(np.median(d['y'][cur]))
            values[g,k]=tf.encode(np.array([y]))[0];observed[g,k]=1.
        meta.append(dict(task=d['tasks'][k],rows=len(ix),observed_identities=int(observed[:,k].sum())))
    assert np.all(values[:,target]==0) and np.all(observed[:,target]==0)
    columns=[k for k in range(len(d['tasks'])) if k!=target]
    # Missing targets cannot be inferred from their own column, even via a mask.
    features=np.concatenate([values[iv][:,columns],observed[iv][:,columns]],1)
    return features,observed[iv][:,columns].sum(1),dict(source_splits=source_splits,target=d['tasks'][target],target_column_removed=True,
             source_rows=len(rows),test_source_rows=0,other_tasks=meta)

def fit(d,k,rows,query,source_splits,dest):
    dest.mkdir(parents=True,exist_ok=True);cache=dest/'predictions.npz';sig=signature()
    if cache.exists():
        z=np.load(cache);assert str(z['signature'])==sig
        np.testing.assert_array_equal(rows,z['fit']);np.testing.assert_array_equal(query,z['query']);return z['p'].copy()
    assert not set(rows)&set(query) and np.all(d['tid'][rows]==k) and np.all(d['split'][rows]<2)
    if np.any(d['split'][query]<2):assert not set(d['rg'][rows])&set(d['rg'][query])
    aux,count,audit=auxiliary(d,source_splits,k)
    dense=np.concatenate([d['desc'],d['emb']],1);pre=b.Preprocessor().fit(dense[np.unique(d['mi'][rows])])
    normalized=pre.transform(dense);desc_dim=d['desc'].shape[1]
    # Scientific labels are features only after removing this task completely.
    rich=np.concatenate([normalized[d['mi']],d['fp2'][d['mi']],d['fcfp'][d['mi']],aux],1)
    narrow=np.concatenate([normalized[d['mi'],:desc_dim],aux],1)
    m=d['metrics'][k];cls=m in ['AUROC','AUPRC'];tf=b.Target(d['y'][rows],m);yy=tf.encode(d['y'][rows])
    klass=CatBoostClassifier if cls else CatBoostRegressor;pred=[];t0=time.time()
    for family,raw in enumerate([rich,narrow]):
        parts=[]
        for seed,depth in zip(P['seeds'],P['depths']):
            params=dict(iterations=P['iterations'],depth=depth,learning_rate=P['learning_rate'],l2_leaf_reg=P['l2_leaf_reg'],
                random_seed=seed,loss_function='Logloss' if cls else ('MAE' if m=='MAE' else 'RMSE'),
                task_type='GPU',devices='0',gpu_ram_part=.14,thread_count=3,border_count=96,
                verbose=False,allow_writing_files=False)
            if cls:
                prevalence=float(yy.mean());params['class_weights']=[1.,min(3.,max(1.,np.sqrt((1-prevalence)/max(prevalence,.01))))]
            model=klass(**params);model.fit(raw[rows],yy);path=dest/f'family{family}_{seed}.cbm';model.save_model(str(path))
            q=model.predict_proba(raw[query])[:,1] if cls else model.predict(raw[query])
            restored=klass();restored.load_model(str(path));again=restored.predict_proba(raw[query])[:,1] if cls else restored.predict(raw[query])
            np.testing.assert_allclose(q,again,rtol=1e-7,atol=1e-7)
            parts.append(q if cls or m=='Spearman' else tf.decode(q))
        pred.append(np.mean(parts,0))
    p=np.stack(pred);assert np.isfinite(p).all()
    np.savez_compressed(cache,p=p,fit=rows,query=query,signature=sig)
    audit.update(signature=sig,fit_rows=len(rows),query_rows=len(query),query_auxiliary_coverage=float(np.mean(count[query]>0)),
                 query_two_auxiliary_coverage=float(np.mean(count[query]>=2)),seconds=time.time()-t0,model_reload_passed=True,
                 conditional_assay_evaluation=True,local_gpu_training=False)
    b.save(dest/'metadata.json',audit);b.emit('v30_fit',task=d['tasks'][k],partition=str(dest.relative_to(OUT)),seconds=round(time.time()-t0),coverage=audit['query_auxiliary_coverage'])
    return p

def evaluate(d):
    folds=np.load(b.OUT/'folds.npy');old=np.load(b.OUT/'cv_predictions.npy')[1];baseline=v24_valid(d);records=[]
    # Start with assay-dense tasks; every task is eventually evaluated.
    priority=['cyp2c9_veith','cyp2d6_veith','cyp3a4_veith','cyp2d6_substrate_carbonmangels','cyp2c9_substrate_carbonmangels','cyp3a4_substrate_carbonmangels',
              'clearance_hepatocyte_az','clearance_microsome_az','bioavailability_ma','half_life_obach','bbb_martins','ppbr_az','dili']
    tasks=priority+[t for t in d['tasks'] if t not in priority]
    for t in tasks:
        k=d['tasks'].index(t);m=d['metrics'][k];tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1))
        vp=fit(d,k,tr,vi,[0],OUT/t/'valid');cp=np.full((2,len(d['y'])),np.nan)
        for fold in range(3):
            query=tr[folds[tr]==fold];rows=tr[folds[tr]!=fold]
            cp[:,query]=fit(d,k,rows,query,[0],OUT/t/f'fold{fold}')
        np.save(OUT/t/'cv_predictions.npy',cp);grid=[]
        for family,name in enumerate(P['families']):
            for weight in P['weights']:
                fg=[]
                for fold in range(3):
                    ix=tr[folds[tr]==fold];fg.append(b.delta(m,d['y'][ix],old[ix],b.blend(m,old[ix],cp[family,ix],weight)))
                vg=b.delta(m,d['y'][vi],baseline[vi],b.blend(m,baseline[vi],vp[family],weight))
                obj=.6*vg+.4*np.mean(fg)-.1*np.std(fg);eligible=vg>1e-4 and np.mean(fg)>-.01 and any(g>=0 for g in fg) and obj>1e-4
                grid.append(dict(family=family,name=name,weight=weight,cv_gains=fg,cv_mean=float(np.mean(fg)),valid_gain=vg,objective=float(obj),eligible=bool(eligible)))
        options=[g for g in grid if g['eligible']];selection=max(options,key=lambda g:g['objective']) if options else None
        records.append(dict(task=t,metric=m,selection=selection,grid=grid,valid_baseline=b.metric(m,d['y'][vi],baseline[vi]),
                            valid_candidates=[b.metric(m,d['y'][vi],q) for q in vp]))
        b.save(OUT/'analysis.json',dict(signature=signature(),status='IN_PROGRESS',tasks=records,limitation=P['limitation']))
        b.emit('v30_selection',task=t,selection=selection)
    b.save(OUT/'analysis.json',dict(signature=signature(),status='EVALUATED',tasks=records,
               selected_tasks=[r['task'] for r in records if r['selection']],limitation=P['limitation']))

def final(d):
    report=json.loads((OUT/'analysis.json').read_text());assert report['signature']==signature();pred={}
    for r in report['tasks']:
        if r['selection'] is None:continue
        k=d['tasks'].index(r['task']);rows=np.flatnonzero((d['tid']==k)&(d['split']<2));query=np.flatnonzero((d['tid']==k)&(d['split']==2))
        pred[r['task']]=fit(d,k,rows,query,[0,1],OUT/r['task']/'final').tolist()
    b.save(OUT/'test_predictions_compact.json',pred)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['evaluate','final','all']);args=parser.parse_args()
    b.remote_only();OUT.mkdir(parents=True,exist_ok=True);b.emit('v30_start',signature=signature());d=b.build_data()
    if args.stage in ['evaluate','all']:evaluate(d)
    if args.stage in ['final','all']:final(d)
