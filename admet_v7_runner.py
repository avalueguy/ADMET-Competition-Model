"""Reconstruct v6 comparisons, run a fixed pool and gate each task independently."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
from admet_v6_runner import (remote_only,partition,metrics,kind,deltas,secondary_ok,save,emit,sha,
    all_tasks,legacy_hold,fit_candidate,predict_candidate,get_metric)
from admet_v6_domain_guard import get_oof,nearest,mixed
from admet_v7_models import load_task,fit,predict

OUT=Path('admet_model_outputs_v7')
PFILE=Path('admet_v7_protocol.json')


def protocol():return json.loads(PFILE.read_text())


def route6(task):return json.loads((Path('admet_release_v6')/task/'selection.json').read_text())


def baseline(task,d):
    dest=OUT/task;dest.mkdir(parents=True,exist_ok=True);path=dest/'baseline_oof.csv'
    if path.exists():return pd.read_csv(path)
    b=pd.read_csv(Path('admet_model_outputs_v6')/task/'baseline_oof.csv')
    np.testing.assert_allclose(b.Y_true,d['frames']['train'].Y)
    np.testing.assert_array_equal(b.fold,partition(task,d))
    r=route6(task)
    if r['accepted']:
        s=r['selection'];candidate=get_oof(task,s['family']);alpha=np.full(len(b),s['alpha'])
        if r.get('domain_guard',{}).get('passed'):
            for f in range(3):
                it=np.flatnonzero(b.fold!=f);iv=np.flatnonzero(b.fold==f)
                alpha[iv]=np.where(nearest(d['raw'],it,iv)<.3,0.,s['alpha'])
        b['Y_pred']=(1-alpha)*b.Y_pred+alpha*candidate
    b.to_csv(path,index=False)
    save(dest/'baseline_provenance.json',dict(version=6,route_sha256=sha(Path('admet_release_v6')/task/'selection.json'),
        reconstructed_accepted_candidate=r['accepted'],has_domain_fallback=r.get('domain_guard',{}).get('passed',False)))
    return b


def train(task,family):
    p=protocol();dest=OUT/task;dest.mkdir(parents=True,exist_ok=True);file=dest/(family+'_oof.csv')
    if file.exists():
        assert json.loads((dest/(family+'_cv.json')).read_text())['protocol_sha256']==sha(PFILE)
        return
    d=load_task(task);y=d['frames']['train'].Y.to_numpy(float);fold=partition(task,d)
    prediction=np.full(len(y),np.nan);rows=[];t=time.time()
    for f in range(3):
        it=np.flatnonzero(fold!=f);iv=np.flatnonzero(fold==f)
        assert not set(d['groups'][it])&set(d['groups'][iv])
        cache=dest/(family+f'_fold{f}.npz')
        if cache.exists():
            with np.load(cache) as z:
                assert str(z['protocol_sha256'])==sha(PFILE)
                np.testing.assert_array_equal(z['rows'],iv);prediction[iv]=z['p']
        else:
            model=fit(family,d,it,y[it],get_metric(task),p);prediction[iv]=predict(model,d,iv)
            np.savez_compressed(cache,rows=iv,p=prediction[iv],protocol_sha256=sha(PFILE));del model
        m=metrics(kind(get_metric(task)),y[iv],prediction[iv]);rows.append(dict(fold=f,**m))
        emit(stage='v7_train',task=task,family=family,fold=f,elapsed_s=round(time.time()-t),**m)
    pd.DataFrame(dict(row=np.arange(len(y)),fold=fold,Y_true=y,Y_pred=prediction)).to_csv(file,index=False)
    save(dest/(family+'_cv.json'),dict(folds=rows,protocol_sha256=sha(PFILE),feature_key=d['v7_feature_key']))


def select(task,d):
    p=protocol();dest=OUT/task;locked=dest/'selection.json'
    if locked.exists():return json.loads(locked.read_text())
    expected=p['families_all']+(['chemberta_ft'] if task in p['finetune_tasks'] else [])
    b=baseline(task,d);y=b.Y_true.to_numpy();base=b.Y_pred.to_numpy();fold=b.fold.to_numpy();metric=get_metric(task);kk=kind(metric)
    pool={}
    for name in expected:
        df=pd.read_csv(dest/(name+'_oof.csv'));np.testing.assert_allclose(df.Y_true,y);np.testing.assert_array_equal(df.fold,fold)
        assert df.row.tolist()==list(range(len(y)))
        assert json.loads((dest/(name+'_cv.json')).read_text())['protocol_sha256']==sha(PFILE)
        pool[name]=df.Y_pred.to_numpy()
    grid=[];best=None
    for fam,pred in pool.items():
        for a in p['alphas']:
            q=(1-a)*base+a*pred;checks=[];gains=[]
            for f in range(3):
                mask=fold==f;old=metrics(kk,y[mask],base[mask]);new=metrics(kk,y[mask],q[mask]);gain=deltas(metric,old,new)
                gains.append(gain);checks.append(dict(fold=f,before=old,after=new,passed=bool(gain>=(-.01 if metric=='MAE' else -.005) and secondary_ok(metric,old,new))))
            obj=float(np.mean(gains)-.5*np.std(gains));passed=all(c['passed'] for c in checks) and np.mean(gains)>0 and obj>0
            row=dict(task=task,family=fam,alpha=a,objective=obj,mean_gain=float(np.mean(gains)),worst_gain=float(min(gains)),folds=checks,passed=bool(passed))
            grid.append(row)
            if passed and (best is None or obj>best['objective']):best=row
    save(dest/'selection_grid.json',grid)
    selected=dict(best or dict(task=task,family='baseline',alpha=0,passed=False))
    selected.update(metric=metric,kind=kk,pool=expected,protocol_sha256=sha(PFILE),selection_source='historically reused train scaffold OOF; locked before new valid evaluation')
    save(locked,selected);emit(stage='v7_lock',task=task,family=selected['family'],alpha=selected['alpha'])
    return selected


def subgroup(d,it,iv,y,b,q,metric):
    sim=nearest(d['raw'],it,iv);mask=sim<.3;result=dict(n=int(mask.sum()),threshold=.3,passed=True)
    if mask.sum()<20 or (kind(metric)=='classification' and len(np.unique(y[mask]))<2):
        result['reason']='insufficient subgroup coverage';return result
    old=metrics(kind(metric),y[mask],b[mask]);new=metrics(kind(metric),y[mask],q[mask])
    result.update(before=old,after=new,passed=bool(deltas(metric,old,new)>=(-.03 if metric=='MAE' else -.01)))
    return result


def repeat_baseline(task,d,it,iv):
    dest=OUT/task;cache=dest/'repeat_baseline.csv';y=d['frames']['train'].Y.to_numpy(float)
    if cache.exists():
        df=pd.read_csv(cache);np.testing.assert_array_equal(df.row,iv);return df.Y_pred.to_numpy()
    r=route6(task);prior=Path('admet_model_outputs_v6')/task/'repeat_confirmation.csv'
    if prior.exists():
        df=pd.read_csv(prior);np.testing.assert_array_equal(df.row,iv);np.testing.assert_allclose(df.Y_true,y[iv])
        if r['accepted']:
            if r.get('domain_guard',{}).get('passed'):base=pd.read_csv(prior.parent/'repeat_domain_guard.csv').blend_guarded.to_numpy()
            else:base=df.blend.to_numpy()
        else:base=df.baseline.to_numpy()
    else:
        base=legacy_hold(task,d,it,iv,all_tasks(),'repeat_692509')
        v5=json.loads((Path('admet_release_v5')/task/'selection.json').read_text())
        p6=json.loads(Path('admet_v6_protocol.json').read_text());metric=get_metric(task)
        if v5['accepted']:
            result=json.loads((Path('admet_model_outputs_v5')/task/'result.json').read_text())
            model=fit_candidate('v5_'+result['chosen'],d,it,y[it],metric,p6)
            base=.75*base+.25*predict_candidate(model,d,iv)
        if r['accepted']:
            s=r['selection'];model=fit_candidate(s['family'],d,it,y[it],metric,p6);c=predict_candidate(model,d,iv)
            base=mixed(base,c,s['alpha'],nearest(d['raw'],it,iv)) if r.get('domain_guard',{}).get('passed') else (1-s['alpha'])*base+s['alpha']*c
    pd.DataFrame(dict(row=iv,Y_true=y[iv],Y_pred=base)).to_csv(cache,index=False)
    return base


def evaluate(task):
    p=protocol();dest=OUT/task;path=dest/'decision.json'
    if path.exists():return
    d=load_task(task);s=select(task,d)
    if s['family']=='baseline':save(path,dict(task=task,accepted=False,reason='no robust OOF candidate',selection=s));return
    n=len(d['frames']['train']);nv=len(d['frames']['valid']);y=d['frames']['train'].Y.to_numpy(float)
    metric=s['metric'];kk=s['kind'];alpha=s['alpha'];baseo=baseline(task,d);cand=pd.read_csv(dest/(s['family']+'_oof.csv')).Y_pred.to_numpy()
    domains=[]
    for f in range(3):
        it=np.flatnonzero(baseo.fold!=f);iv=np.flatnonzero(baseo.fold==f);b=baseo.Y_pred.to_numpy()[iv];q=(1-alpha)*b+alpha*cand[iv]
        domains.append(dict(split='fold_'+str(f),**subgroup(d,it,iv,y[iv],b,q,metric)))
    if not all(c['passed'] for c in domains):
        save(path,dict(task=task,accepted=False,selection=s,domains=domains,reason='distant OOF subgroup regression'));return
    it=np.arange(n);iv=np.arange(n,n+nv);yv=d['frames']['valid'].Y.to_numpy(float)
    model=fit(s['family'],d,it,y,metric,p);candidate=predict(model,d,iv)
    joblib.dump(model,dest/'validation_model.joblib',compress=3);del model
    base=pd.read_csv(Path('admet_release_v6')/task/'validation_predictions.csv');np.testing.assert_allclose(base.Y_true,yv);b=base.Y_pred.to_numpy();q=(1-alpha)*b+alpha*candidate
    old=metrics(kk,yv,b);new=metrics(kk,yv,q);domains.append(dict(split='valid',**subgroup(d,it,iv,yv,b,q,metric)))
    pd.DataFrame(dict(Y_true=yv,baseline=b,candidate=candidate,blend=q)).to_csv(dest/'validation_comparison.csv',index=False)
    decision=dict(task=task,accepted=False,before=old,after=new,selection=s,domains=domains)
    valid=deltas(metric,old,new)>=0 and secondary_ok(metric,old,new) and domains[-1]['passed']
    decision.update(valid_pass=bool(valid),reason='valid gate failed')
    emit(stage='v7_valid',task=task,passed=bool(valid),before=old,after=new)
    if not valid:save(path,decision);return
    it,iv=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=692509).split(np.arange(n),y,d['groups'][:n]))
    assert not set(d['groups'][it])&set(d['groups'][iv])
    if kk=='classification' and (len(np.unique(y[it]))<2 or len(np.unique(y[iv]))<2):
        decision['reason']='repeat lacks class coverage';save(path,decision);return
    b=repeat_baseline(task,d,it,iv);model=fit(s['family'],d,it,y[it],metric,p);c=predict(model,d,iv);q=(1-alpha)*b+alpha*c
    old=metrics(kk,y[iv],b);new=metrics(kk,y[iv],q);domains.append(dict(split='repeat',**subgroup(d,it,iv,y[iv],b,q,metric)))
    passed=deltas(metric,old,new)>=(-.01 if metric=='MAE' else -.005) and secondary_ok(metric,old,new) and domains[-1]['passed']
    pd.DataFrame(dict(row=iv,Y_true=y[iv],baseline=b,candidate=c,blend=q)).to_csv(dest/'repeat_comparison.csv',index=False)
    decision.update(accepted=bool(passed),repeat=dict(before=old,after=new,passed=bool(passed),seed=692509),reason='passed all observed gates' if passed else 'repeat gate failed')
    save(path,decision);emit(stage='v7_decision',task=task,accepted=bool(passed),repeat=decision['repeat'])


def main():
    remote_only();ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['train','evaluate']);ap.add_argument('--family');ap.add_argument('--tasks',nargs='*');args=ap.parse_args()
    names=args.tasks or sorted(p.name for p in Path('admet_competition_data/data_benchmark').iterdir() if p.is_dir())
    with threadpool_limits(limits=4):
        for task in names:
            if args.stage=='train':train(task,args.family)
            else:evaluate(task)


if __name__=='__main__':main()
