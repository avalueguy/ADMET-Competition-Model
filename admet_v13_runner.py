"""Remote-only sparse-kernel candidates against the scored v11 baseline."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from admet_v6_runner import remote_only, partition, metrics, kind, deltas, secondary_ok, save, emit, sha, get_metric
from admet_v8_runner import subgroup
from admet_v13_models import load_task, fit, predict

OUT = Path('admet_model_outputs_v13')
PFILE = Path('admet_v13_protocol.json')
PRIOR_RELEASE = Path('admet_release_v11')


def protocol():
    return json.loads(PFILE.read_text())


def domain_check(d, it, iv, y, b, q, metric):
    result=subgroup(d,it,iv,y,b,q,metric)
    if result.get('reason')=='insufficient subgroup coverage':
        result['passed']=False;return result
    result['secondary_passed']=bool(secondary_ok(metric,result['before'],result['after']))
    result['passed']=bool(result['passed'] and result['secondary_passed'])
    return result


def baseline(task,d):
    dest=OUT/task;dest.mkdir(parents=True,exist_ok=True);path=dest/'baseline_oof.csv'
    from admet_v11_runner import baseline as prior_baseline
    b=prior_baseline(task,d).copy()
    r=json.loads((PRIOR_RELEASE/task/'selection.json').read_text())
    if r['accepted']:
        sel=r['selection'];c=pd.read_csv(Path('admet_model_outputs_v11')/task/(sel['family']+'_oof.csv'))
        np.testing.assert_array_equal(c.row,b.row);np.testing.assert_array_equal(c.fold,b.fold);np.testing.assert_allclose(c.Y_true,b.Y_true)
        b['Y_pred']=(1-sel['alpha'])*b.Y_pred+sel['alpha']*c.Y_pred
    np.testing.assert_allclose(b.Y_true,d['frames']['train'].Y);np.testing.assert_array_equal(b.fold,partition(task,d))
    if path.exists():
        old=pd.read_csv(path);np.testing.assert_allclose(old.Y_pred,b.Y_pred,rtol=1e-12,atol=1e-12);return old
    b.to_csv(path,index=False);save(dest/'baseline_provenance.json',dict(release='v11',submission_sha256=protocol()['baseline_sha256']))
    return b


def repeat_baseline(task,d,it,iv):
    r=json.loads((PRIOR_RELEASE/task/'selection.json').read_text())
    if r['accepted']:
        b=pd.read_csv(Path('admet_model_outputs_v11')/task/'repeat_comparison.csv')
        np.testing.assert_array_equal(b.row,iv);np.testing.assert_allclose(b.Y_true,d['frames']['train'].Y.to_numpy()[iv]);return b.blend.to_numpy()
    from admet_v11_runner import repeat_baseline as previous
    return previous(task,d,it,iv)


def train(task, family):
    p = protocol(); assert family in p['families'] and task in p['tasks']
    dest = OUT / task; dest.mkdir(parents=True, exist_ok=True); final = dest / (family + '_oof.csv')
    if final.exists():
        assert json.loads((dest / (family + '_cv.json')).read_text())['protocol_sha256'] == sha(PFILE); return
    d = load_task(task); y = d['frames']['train'].Y.to_numpy(float); fold = partition(task, d)
    metric = get_metric(task); prediction = np.full(len(y), np.nan); rows=[]; start=time.time()
    for f in range(3):
        it=np.flatnonzero(fold != f); iv=np.flatnonzero(fold == f)
        assert not set(d['groups'][it]) & set(d['groups'][iv]); assert not set(d['canonical'][it]) & set(d['canonical'][iv])
        cache=dest/(family + f'_fold{f}.npz')
        if cache.exists():
            with np.load(cache, allow_pickle=False) as z:
                assert str(z['protocol_sha256']) == sha(PFILE); np.testing.assert_array_equal(z['rows'],iv); prediction[iv]=z['p']
        else:
            model=fit(family,d,it,y[it],metric,p); prediction[iv]=predict(model,d,iv)
            save(dest/(family+f'_fold{f}_source_audit.json'),model.get('source_audit',[]))
            np.savez_compressed(cache,rows=iv,p=prediction[iv],protocol_sha256=sha(PFILE)); del model
        m=metrics(kind(metric),y[iv],prediction[iv]); rows.append(dict(fold=f,**m))
        emit(stage='v13_train',task=task,family=family,fold=f,elapsed_s=round(time.time()-start),**m)
    assert np.isfinite(prediction).all()
    pd.DataFrame(dict(row=np.arange(len(y)),fold=fold,Y_true=y,Y_pred=prediction)).to_csv(final,index=False)
    save(dest/(family+'_cv.json'),dict(task=task,family=family,folds=rows,protocol_sha256=sha(PFILE),feature_key=d['v13_feature_key']))


def select(task,d):
    p=protocol();dest=OUT/task;locked=dest/'selection.json'
    if locked.exists():
        result=json.loads(locked.read_text());assert result['protocol_sha256']==sha(PFILE);return result
    b=baseline(task,d);y=b.Y_true.to_numpy();base=b.Y_pred.to_numpy();fold=b.fold.to_numpy();metric=get_metric(task);kk=kind(metric)
    grid=[];best=None;pool={}
    for family in p['families']:
        frame=pd.read_csv(dest/(family+'_oof.csv'));np.testing.assert_allclose(frame.Y_true,y);np.testing.assert_array_equal(frame.fold,fold)
        assert frame.row.tolist()==list(range(len(y)))
        assert json.loads((dest/(family+'_cv.json')).read_text())['protocol_sha256']==sha(PFILE)
        pool[family]=frame.Y_pred.to_numpy()
    for family,cand in pool.items():
        for alpha in p['alphas']:
            q=(1-alpha)*base+alpha*cand;checks=[];gains=[]
            for f in range(3):
                m=fold==f;before=metrics(kk,y[m],base[m]);after=metrics(kk,y[m],q[m]);gain=deltas(metric,before,after);gains.append(gain)
                checks.append(dict(fold=f,before=before,after=after,passed=bool(gain>=(-.01 if metric=='MAE' else -.005) and secondary_ok(metric,before,after))))
            objective=float(np.mean(gains)-.5*np.std(gains));passed=all(c['passed'] for c in checks) and objective>0 and np.mean(gains)>0
            row=dict(task=task,family=family,alpha=alpha,objective=objective,mean_gain=float(np.mean(gains)),worst_gain=float(min(gains)),folds=checks,passed=bool(passed))
            grid.append(row)
            if passed and (best is None or objective>best['objective']):best=row
    result=best or dict(task=task,family='baseline',alpha=0,passed=False)
    result.update(metric=metric,kind=kk,pool=list(pool),protocol_sha256=sha(PFILE),selection_source='reused scaffold OOF; locked before official valid')
    save(dest/'selection_grid.json',grid);save(locked,result);emit(stage='v13_lock',task=task,family=result['family'],alpha=result['alpha']);return result


def evaluate(task):
    p=protocol(); dest=OUT/task; path=dest/'decision.json'
    if path.exists(): return
    d=load_task(task); s=select(task,d)
    if s['family']=='baseline':
        save(path,dict(task=task,accepted=False,reason='no robust OOF candidate',selection=s)); return
    n=len(d['frames']['train']); nv=len(d['frames']['valid']); y=d['frames']['train'].Y.to_numpy(float); metric=s['metric']; kk=s['kind']; alpha=s['alpha']
    baseo=baseline(task,d); cand=pd.read_csv(dest/(s['family']+'_oof.csv')).Y_pred.to_numpy(); domains=[]
    for f in range(3):
        it=np.flatnonzero(baseo.fold!=f); iv=np.flatnonzero(baseo.fold==f); b=baseo.Y_pred.to_numpy()[iv]; q=(1-alpha)*b+alpha*cand[iv]
        domains.append(dict(split='fold_'+str(f),**domain_check(d,it,iv,y[iv],b,q,metric)))
    if not all(x['passed'] for x in domains):
        save(path,dict(task=task,accepted=False,selection=s,domains=domains,reason='distant OOF subgroup regression')); return
    it=np.arange(n); iv=np.arange(n,n+nv); yv=d['frames']['valid'].Y.to_numpy(float)
    model=fit(s['family'],d,it,y,metric,p); candidate=predict(model,d,iv); save(dest/'validation_source_audit.json',model.get('source_audit',[])); del model
    old=pd.read_csv(PRIOR_RELEASE/task/'validation_predictions.csv'); np.testing.assert_allclose(old.Y_true,yv); b=old.Y_pred.to_numpy(); q=(1-alpha)*b+alpha*candidate
    before=metrics(kk,yv,b); after=metrics(kk,yv,q); domains.append(dict(split='valid',**domain_check(d,it,iv,yv,b,q,metric)))
    pd.DataFrame(dict(Y_true=yv,baseline=b,candidate=candidate,blend=q)).to_csv(dest/'validation_comparison.csv',index=False)
    valid=deltas(metric,before,after)>=0 and secondary_ok(metric,before,after) and domains[-1]['passed']
    decision=dict(task=task,accepted=False,before=before,after=after,selection=s,domains=domains,valid_pass=bool(valid),reason='valid gate failed')
    if not valid: save(path,decision); return
    it,iv=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=692509).split(np.arange(n),y,d['groups'][:n])); assert not set(d['groups'][it])&set(d['groups'][iv])
    if kk=='classification' and (len(np.unique(y[it]))<2 or len(np.unique(y[iv]))<2): decision['reason']='repeat lacks class coverage'; save(path,decision); return
    b=repeat_baseline(task,d,it,iv); model=fit(s['family'],d,it,y[it],metric,p); c=predict(model,d,iv); save(dest/'repeat_source_audit.json',model.get('source_audit',[])); del model
    q=(1-alpha)*b+alpha*c; before=metrics(kk,y[iv],b); after=metrics(kk,y[iv],q); domains.append(dict(split='repeat',**domain_check(d,it,iv,y[iv],b,q,metric)))
    passed=deltas(metric,before,after)>=(-.01 if metric=='MAE' else -.005) and secondary_ok(metric,before,after) and domains[-1]['passed']
    pd.DataFrame(dict(row=iv,Y_true=y[iv],baseline=b,candidate=c,blend=q)).to_csv(dest/'repeat_comparison.csv',index=False)
    decision.update(accepted=bool(passed),repeat=dict(before=before,after=after,passed=bool(passed),seed=692509),reason='passed all observed gates' if passed else 'repeat gate failed')
    save(path,decision); emit(stage='v13_decision',task=task,accepted=bool(passed),repeat=decision['repeat'])


def main():
    remote_only(); parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['train','evaluate']); parser.add_argument('--family'); parser.add_argument('--tasks',nargs='*'); args=parser.parse_args()
    p=protocol(); names=args.tasks or p['tasks']
    with threadpool_limits(limits=4):
        for task in names:
            if args.stage=='train':
                for family in p['families'] if args.family=='both' else [args.family]:
                    train(task,family)
            else:
                evaluate(task)


if __name__=='__main__': main()
