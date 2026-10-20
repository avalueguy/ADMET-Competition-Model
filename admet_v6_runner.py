"""Remote-only candidate training and honest, metric-aware release selection."""
import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
from admet_upgrade_v3 import metrics, utility, get_metric
from admet_v6_features import load_task, mixed_kernel, sha
from admet_v6_models import fit, predict

OUT=Path('admet_model_outputs_v6')
DATA=Path('admet_competition_data/data_benchmark')


def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8')


def kind(metric):
    return 'classification' if metric in ['AUROC','AUPRC'] else 'regression'


def emit(**kw):
    print(json.dumps(kw),flush=True)


def partition(task,data):
    folds=pd.read_csv(Path('admet_model_outputs_v5')/task/'folds.csv',keep_default_na=False)
    n=len(data['frames']['train'])
    assert folds.canonical.tolist()==data['canonical'][:n].tolist()
    assert folds.scaffold.tolist()==data['groups'][:n].tolist()
    return folds.fold.to_numpy(int)


def remote_only():
    if os.name=='nt' or Path.cwd()!=Path('/root/autodl-tmp/admet_20260925'):
        raise RuntimeError('Training is restricted to the authorized remote directory')
    if not torch.cuda.is_available():raise RuntimeError('Remote CUDA required')
    torch.set_num_threads(4)


def candidate(task,family,protocol):
    dest=OUT/task;dest.mkdir(parents=True,exist_ok=True)
    final=dest/(family+'_oof.csv')
    if final.exists():return
    data=load_task(task);y=data['frames']['train'].Y.to_numpy(float)
    fold=partition(task,data);metric=get_metric(task);kk=kind(metric)
    if family.startswith('kernel'):data['kernel']=mixed_kernel(data)
    pp=np.full(len(y),np.nan);rows=[];t=time.time()
    for f in range(3):
        it=np.flatnonzero(fold!=f);iv=np.flatnonzero(fold==f)
        assert not set(data['groups'][it])&set(data['groups'][iv])
        assert not set(data['canonical'][it])&set(data['canonical'][iv])
        cache=dest/(family+f'_fold{f}.npz')
        if cache.exists():
            with np.load(cache) as z:
                assert np.array_equal(z['rows'],iv)
                pp[iv]=z['p']
        else:
            model=fit(family,data,it,y[it],metric,protocol)
            pp[iv]=predict(model,data,iv)
            np.savez_compressed(cache,rows=iv,p=pp[iv])
            del model
        m=metrics(kk,y[iv],pp[iv]);rows.append(dict(fold=f,**m))
        emit(stage='candidate',task=task,family=family,fold=f,seconds=round(time.time()-t),**m)
    assert np.isfinite(pp).all()
    pd.DataFrame(dict(row=np.arange(len(y)),fold=fold,Y_true=y,Y_pred=pp)).to_csv(final,index=False)
    save(dest/(family+'_cv.json'),dict(task=task,family=family,folds=rows,
         protocol_sha256=sha('admet_v6_protocol.json'),feature_sha256=data['feature_sha256']))


def legacy_hold(task,data,it,iv,tasks,cache):
    from admet_v5_confirm import train_legacy
    from admet_transfer_upgrade import REL
    from admet_features_v2 import features
    from admet_v5_legacy_mtl import fit_predict
    recipes=json.loads(Path('admet_legacy_recipes.json').read_text())
    forbidden=set(data['groups'][iv]);filtered=copy.copy(tasks)
    for source in REL.get(task,[]):
        if source not in tasks:continue
        df=tasks[source]['train'];_,_,sg=features(df.Drug.tolist(),OUT/'legacy_features')
        mask=np.array([s not in forbidden for s in sg])
        filtered[source]={'train':df.loc[mask].reset_index(drop=True)}
    tr=data['frames']['train'];smiles=tr.Drug.tolist();y=tr.Y.to_numpy(float)
    # For a new split the MTL target training rows must also exclude the holdout.
    npred=None
    if recipes[task]['neural_weight']:
        npred=fit_predict(task,str(cache),forbidden,[smiles[i] for i in iv],tasks,OUT/'legacy_mtl')
    return train_legacy(recipes[task],data['raw'][it],[smiles[i] for i in it],y[it],
                        data['raw'][iv],[smiles[i] for i in iv],task,filtered,OUT/'legacy_features',npred)


def all_tasks():
    return {p.name:{s:pd.read_csv(p/(s+'.csv')) for s in ['train','valid','test']}
            for p in DATA.iterdir() if p.is_dir()}


def baseline(task):
    dest=OUT/task;dest.mkdir(parents=True,exist_ok=True)
    final=dest/'baseline_oof.csv'
    if final.exists():return
    data=load_task(task);y=data['frames']['train'].Y.to_numpy(float);fold=partition(task,data)
    prior=Path('admet_v5_confirmation')/task/'matched_oof.csv'
    if prior.exists():
        p=pd.read_csv(prior);assert np.allclose(p.Y_true,y);base=p.baseline.to_numpy(float)
    else:
        tasks=all_tasks();base=np.full(len(y),np.nan)
        for f in range(3):
            it=np.flatnonzero(fold!=f);iv=np.flatnonzero(fold==f)
            base[iv]=legacy_hold(task,data,it,iv,tasks,f)
            emit(stage='baseline',task=task,fold=f)
    route=json.loads((Path('admet_release_v5')/task/'selection.json').read_text())
    if route['accepted']:
        result=json.loads((Path('admet_model_outputs_v5')/task/'result.json').read_text())
        p=pd.read_csv(Path('admet_model_outputs_v5')/task/(result['chosen']+'_oof.csv'))
        assert np.allclose(p.Y_true,y)
        base=.75*base+.25*p.Y_pred.to_numpy(float)
    pd.DataFrame(dict(row=np.arange(len(y)),fold=fold,Y_true=y,Y_pred=base)).to_csv(final,index=False)


def deltas(metric,old,new):
    gain=utility(metric,new)-utility(metric,old)
    return gain/max(old['mae'],1e-8) if metric=='MAE' else gain


def secondary_ok(metric,old,new):
    if metric=='MAE':return new['spearman']>=old['spearman']-.015
    if metric=='Spearman':return new['mae']<=old['mae']*1.03
    other='auroc' if metric=='AUPRC' else 'auprc'
    return new[other]>=old[other]-.015


def select(task,protocol):
    dest=OUT/task
    if (dest/'selection.json').exists():return json.loads((dest/'selection.json').read_text())
    big={'solubility_aqsoldb','ld50_zhu','ames','cyp2c9_veith','cyp2d6_veith','cyp3a4_veith'}
    expected={'kernel_1','kernel_10','xgb','dense'}
    if task not in big:expected.add('extra')
    if task in {'caco2_wang','lipophilicity_astrazeneca','ppbr_az','solubility_aqsoldb'}:expected.add('graph')
    missing=[f for f in sorted(expected) if not (dest/(f+'_oof.csv')).exists()]
    if missing:raise RuntimeError(f'{task}: candidate pool incomplete: {missing}')
    baseline(task)
    b=pd.read_csv(dest/'baseline_oof.csv');y=b.Y_true.to_numpy();base=b.Y_pred.to_numpy();fold=b.fold.to_numpy()
    metric=get_metric(task);kk=kind(metric);pool={}
    for p in sorted(dest.glob('*_oof.csv')):
        name=p.name[:-8]
        if name=='baseline':continue
        df=pd.read_csv(p);assert np.allclose(df.Y_true,y)
        assert np.array_equal(df.fold.to_numpy(),fold)
        assert df.row.tolist()==list(range(len(y)))
        pool[name]=df.Y_pred.to_numpy()
    for p in sorted((Path('admet_model_outputs_v5')/task).glob('*_oof.csv')):
        df=pd.read_csv(p);assert np.allclose(df.Y_true,y)
        assert np.array_equal(df.fold.to_numpy(),fold)
        assert df.row.tolist()==list(range(len(y)))
        pool['v5_'+p.name[:-8]]=df.Y_pred.to_numpy()
    # A single fixed complementary blend adds diversity without a large simplex search.
    if 'graph' in pool and 'dense' in pool:pool['graph+dense']=.5*pool['graph']+.5*pool['dense']
    rows=[];best=None
    for family,p in pool.items():
        for alpha in protocol['alphas']:
            q=(1-alpha)*base+alpha*p;gains=[];secondary=True;fold_metrics=[]
            for f in range(3):
                ix=fold==f;old=metrics(kk,y[ix],base[ix]);new=metrics(kk,y[ix],q[ix])
                gains.append(deltas(metric,old,new));secondary &= secondary_ok(metric,old,new)
                fold_metrics.append(dict(fold=f,before=old,after=new))
            tol=.01 if metric=='MAE' else .005
            passed=min(gains)>=-tol and np.mean(gains)>0 and secondary
            objective=float(np.mean(gains)-.5*np.std(gains))
            row=dict(task=task,family=family,alpha=alpha,cv_pass=bool(passed),mean_gain=float(np.mean(gains)),
                     worst_gain=float(min(gains)),objective=objective,folds=fold_metrics)
            rows.append(row)
            if passed and objective>0 and (best is None or objective>best['objective']):best=row
    save(dest/'selection_grid.json',rows)
    result=best or dict(task=task,family='baseline',alpha=0.,cv_pass=False)
    result=dict(**result,metric=metric,kind=kk,selection_source='reused train scaffold OOF only',
                protocol_sha256=sha('admet_v6_protocol.json'),pool_names=sorted(pool),pool_complete_at_lock=True)
    save(dest/'selection.json',result);emit(stage='selection',**result)
    return result


def fit_candidate(family,data,it,y,metric,protocol):
    if '+' in family:
        return dict(composite=[fit_candidate(f,data,it,y,metric,protocol) for f in family.split('+')])
    if family.startswith('v5_'):
        from admet_v5_remote_train import fit_family
        v5=json.loads(Path('admet_v5_protocol.json').read_text())
        return dict(legacy=fit_family(family[3:],data['raw'][it],data['emb'][it],y,kind(metric),metric,v5))
    if family.startswith('kernel') and 'kernel' not in data:data['kernel']=mixed_kernel(data)
    return fit(family,data,it,y,metric,protocol)


def predict_candidate(bundle,data,ix):
    if 'composite' in bundle:return np.mean([predict_candidate(b,data,ix) for b in bundle['composite']],0)
    if 'legacy' in bundle:
        from admet_v5_remote_train import predict as p5
        return p5(bundle['legacy'],data['raw'][ix],data['emb'][ix])
    return predict(bundle,data,ix)


def evaluate(task,protocol):
    dest=OUT/task
    if (dest/'decision.json').exists():return
    selected=select(task,protocol)
    if selected['family']=='baseline':
        save(dest/'decision.json',dict(task=task,accepted=False,reason='no robust OOF candidate'));return
    data=load_task(task);tr=data['frames']['train'];va=data['frames']['valid'];n=len(tr);nv=len(va)
    metric=selected['metric'];kk=kind(metric);y=tr.Y.to_numpy(float);yv=va.Y.to_numpy(float)
    model=fit_candidate(selected['family'],data,np.arange(n),y,metric,protocol)
    cand=predict_candidate(model,data,np.arange(n,n+nv));joblib.dump(model,dest/'validation_model.joblib',compress=3)
    base=pd.read_csv(Path('admet_release_v5')/task/'validation_predictions.csv')
    assert np.allclose(base.Y_true,yv)
    q=(1-selected['alpha'])*base.Y_pred.to_numpy()+selected['alpha']*cand
    old=metrics(kk,yv,base.Y_pred);new=metrics(kk,yv,q)
    passed=deltas(metric,old,new)>=0 and secondary_ok(metric,old,new)
    decision=dict(task=task,accepted=False,valid_pass=bool(passed),before=old,after=new,
                  selection=selected,reason='valid failed' if not passed else 'awaiting repeat scaffold confirmation')
    pd.DataFrame(dict(Y_true=yv,baseline=base.Y_pred,candidate=cand,blend=q)).to_csv(dest/'validation_comparison.csv',index=False)
    save(dest/'decision.json',decision);emit(stage='valid',**decision)


def main():
    remote_only();ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['train','baseline','evaluate'])
    ap.add_argument('--families',nargs='*',default=['kernel_1','kernel_10','extra','xgb','dense'])
    ap.add_argument('--tasks',nargs='*');args=ap.parse_args()
    protocol=json.loads(Path('admet_v6_protocol.json').read_text());OUT.mkdir(exist_ok=True)
    tasks=args.tasks or sorted(p.name for p in DATA.iterdir() if p.is_dir())
    with threadpool_limits(limits=6):
        for task in tasks:
            if args.stage=='train':
                for family in args.families:candidate(task,family,protocol)
            elif args.stage=='baseline':baseline(task)
            else:evaluate(task,protocol)


if __name__=='__main__':main()
