"""Remote fit-only adaptive boosting and kernel model selection.

Every outer fit creates its own inner scaffold holdout. Neither official valid
labels nor outer-heldout labels enter preprocessing, iteration choice or C choice.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import SVC, SVR
from catboost import CatBoostClassifier, CatBoostRegressor
from admet_v6_models import Preprocess, Target
from admet_v8_models import RobustTarget
from admet_v11_models import load_task as load11, references, kernel


def load_task(task):
    d=load11(task)
    d['v12_feature_key']=hashlib.sha256(('v12|'+d['v11_feature_key']).encode()).hexdigest()
    return d


def inner_split(d,ix,y,metric):
    cls=metric in ['AUROC','AUPRC']
    # First feasible split in a fixed list, coverage only; no metric-based split search.
    for seed in [12091,12092,12093,12094,12095]:
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=.22,random_state=seed).split(ix,groups=d['groups'][ix]))
        if len(a)>=20 and len(b)>=15 and (not cls or (len(np.unique(y[a]))==2 and len(np.unique(y[b]))==2)):
            assert not set(d['groups'][ix[a]])&set(d['groups'][ix[b]])
            return a,b,seed
    raise ValueError('No feasible inner scaffold holdout')


def cat_x(d,family):
    if family=='boost_compact':
        return np.concatenate([d['desc'],d['maccs'],d['emb']],axis=1)
    assert family=='boost_fragments'
    return np.concatenate([d['desc'],d['emb'],np.log1p(d['count']),np.log1p(d['ap']),
        np.log1p(d['tt']),np.log1p(d['fcfp']),d['rdkit'],d['maccs']],axis=1)


def select_columns(x,ix,family,d):
    # Retain all descriptors/embeddings, but screen fragments within fit only.
    dense=d['desc'].shape[1]+d['emb'].shape[1]
    if family=='boost_compact':return np.arange(x.shape[1])
    var=np.var(x[ix,dense:],axis=0)
    selected=np.argsort(var,kind='stable')[-min(1536,len(var)):]+dense
    return np.r_[np.arange(dense),selected]


def params(metric,depth,seed,iterations):
    return dict(iterations=int(iterations),depth=depth,learning_rate=.035 if depth==5 else .025,
        l2_leaf_reg=8 if depth==5 else 12,random_seed=seed,task_type='GPU',devices='0',
        gpu_ram_part=.32,thread_count=4,loss_function='Logloss' if metric in ['AUROC','AUPRC'] else ('MAE' if metric=='MAE' else 'RMSE'),
        eval_metric='AUC' if metric=='AUROC' else ('PRAUC' if metric=='AUPRC' else ('MAE' if metric=='MAE' else 'RMSE')),
        bootstrap_type='Bayesian',bagging_temperature=.5,random_strength=.5,border_count=128,
        verbose=False,allow_writing_files=False)


def transformed(y,metric):
    return None if metric in ['AUROC','AUPRC'] else (RobustTarget(y,metric) if metric=='MAE' else Target(y,metric))


def fit_boost(family,d,ix,y,metric,p):
    a,b,seed=inner_split(d,ix,y,metric);x=cat_x(d,family)
    c=select_columns(x,ix[a],family,d);pre=Preprocess().fit(x[ix[a]][:,c])
    train=pre.transform(x[ix[a]][:,c]);hold=pre.transform(x[ix[b]][:,c])
    tf=transformed(y[a],metric);yy=y if tf is None else tf.encode(y)
    cls=metric in ['AUROC','AUPRC'];klass=CatBoostClassifier if cls else CatBoostRegressor
    iterations=[]
    for depth,modelseed in zip(p['boost_depths'],p['seeds']):
        model=klass(**params(metric,depth,modelseed,p['boost_max_iterations']))
        model.fit(train,yy[a],eval_set=(hold,yy[b]),early_stopping_rounds=p['boost_patience'],use_best_model=True)
        iterations.append(max(80,int(model.get_best_iteration())+1))
    # New fit-only preprocessing and vocabulary, refit on all outer fitting rows.
    c=select_columns(x,ix,family,d);pre=Preprocess().fit(x[ix][:,c]);train=pre.transform(x[ix][:,c])
    tf=transformed(y,metric);yy=y if tf is None else tf.encode(y)
    models=[]
    for depth,modelseed,niter in zip(p['boost_depths'],p['seeds'],iterations):
        model=klass(**params(metric,depth,modelseed,niter));model.fit(train,yy);models.append(model)
    return dict(family=family,feature_key=d['v12_feature_key'],classification=cls,metric=metric,
        models=models,columns=c,pre=pre,target=tf,inner_seed=seed,inner_fit_rows=ix[a],inner_hold_rows=ix[b],
        iterations=iterations,reference=ix,train_rows=len(ix))


def kernel_bundle(d,ix):
    pre=Preprocess().fit(d['desc'][ix])
    return dict(pre=pre,reference_desc=pre.transform(d['desc'][ix]),references=references(d,ix),reference=ix)


def fit_kernel(d,ix,y,metric,p):
    from admet_v6_runner import metrics,kind,utility
    a,b,seed=inner_split(d,ix,y,metric);bundle=kernel_bundle(d,ix[a]);ka=kernel(d,ix[a],bundle);kb=kernel(d,ix[b],bundle)
    cls=metric in ['AUROC','AUPRC'];tf=transformed(y[a],metric);yy=y if tf is None else tf.encode(y)
    grid=[]
    for c in p['adaptive_cs']:
        for power in [1,2]:
            model=SVC(C=c,kernel='precomputed',class_weight='balanced',cache_size=1536) if cls else SVR(C=c,kernel='precomputed',epsilon=.04,cache_size=1536)
            model.fit(ka**power,yy[a]);q=expit(model.decision_function(kb**power)) if cls else tf.decode(model.predict(kb**power))
            val=utility(metric,metrics(kind(metric),y[b],q))
            grid.append(dict(c=c,power=power,utility=float(val)))
    best=max(grid,key=lambda r:r['utility'])
    full=kernel_bundle(d,ix);k=kernel(d,ix,full)**best['power'];tf=transformed(y,metric);yy=y if tf is None else tf.encode(y)
    model=SVC(C=best['c'],kernel='precomputed',class_weight='balanced',cache_size=1536) if cls else SVR(C=best['c'],kernel='precomputed',epsilon=.04,cache_size=1536)
    model.fit(k,yy)
    full.update(family='kernel_adaptive',feature_key=d['v12_feature_key'],classification=cls,metric=metric,
        target=tf,models=[model],power=best['power'],c=best['c'],inner_grid=grid,inner_seed=seed,
        inner_fit_rows=ix[a],inner_hold_rows=ix[b],train_rows=len(ix))
    return full


def fit(family,d,ix,y,metric,p):
    from admet_v6_runner import remote_only
    remote_only();ix=np.asarray(ix,dtype=int);y=np.asarray(y,float)
    assert len(ix)==len(y) and family in p['families']+['blend_all']
    if family=='blend_all':
        return dict(family=family,feature_key=d['v12_feature_key'],members=[fit(f,d,ix,y,metric,p) for f in p['families']])
    return fit_kernel(d,ix,y,metric,p) if family=='kernel_adaptive' else fit_boost(family,d,ix,y,metric,p)


def predict(bundle,d,ix):
    assert bundle['feature_key']==d['v12_feature_key'];ix=np.asarray(ix,int)
    if bundle['family']=='blend_all':
        return np.mean([predict(m,d,ix) for m in bundle['members']],axis=0)
    if bundle['family']=='kernel_adaptive':
        k=kernel(d,ix,bundle)**bundle['power'];m=bundle['models'][0]
        return expit(m.decision_function(k)) if bundle['classification'] else bundle['target'].decode(m.predict(k))
    x=bundle['pre'].transform(cat_x(d,bundle['family'])[ix][:,bundle['columns']])
    values=[m.predict_proba(x)[:,1] if bundle['classification'] else bundle['target'].decode(m.predict(x)) for m in bundle['models']]
    return np.mean(values,axis=0)
