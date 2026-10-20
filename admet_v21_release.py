"""Audit and export the v21 selection against the scored v20 submission."""
import argparse
import csv
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from admet_v21_finetune import (ROOT, OUT, PROTOCOL, SOURCE, read_json, save_json, sha,
                                load_task, cache_prefix, materialize, metric_for, utility,
                                gain, anchor_oof, anchor_valid, anchor_test, combine, remote_only)

REL=ROOT/'admet_release_v21'


def bootstrap(y,b,q,groups,metric,repeats=400):
    unique,inverse=np.unique(groups,return_inverse=True)
    members=[np.flatnonzero(inverse==i) for i in range(len(unique))]
    rng=np.random.default_rng(21627)
    values=[]
    for _ in range(repeats):
        ix=np.concatenate([members[i] for i in rng.integers(len(unique),size=len(unique))])
        if metric in ['AUROC','AUPRC'] and len(np.unique(y[ix]))<2:
            continue
        values.append(gain(metric,y[ix],b[ix],q[ix]))
    return dict(repeats=len(values),percentile_95=list(map(float,np.quantile(values,[.025,.975]))),
                positive_fraction=float(np.mean(np.asarray(values)>0)),
                caveat='Conditional scaffold bootstrap after repeated selection; excludes selection uncertainty.')


def repeat(task):
    dest=OUT/task
    route=read_json(dest/'selection.json'); s=route['selection']
    if s['weight']==0:
        return
    d=load_task(task)
    n=len(d['frames']['train']); y=d['frames']['train'].Y.to_numpy(float)
    fit,hold=next(GroupShuffleSplit(n_splits=1,test_size=.25,random_state=692509).split(
        np.arange(n),groups=d['groups'][:n]))
    assert not set(d['groups'][fit])&set(d['groups'][hold])
    basepath=ROOT/'admet_model_outputs_v16'/task/'repeat_comparison.csv'
    if basepath.exists():
        baseline=pd.read_csv(basepath)
        np.testing.assert_array_equal(baseline.row,hold)
        np.testing.assert_array_equal(baseline.Y_true,y[hold])
        b=baseline.blend.to_numpy(float)
    else:
        from admet_v16_runner import repeat_anchor
        b=repeat_anchor(task,d,fit,hold)
    route19=ROOT/'admet_model_outputs_v19'/task/'selection.json'
    if route19.exists() and read_json(route19).get('candidate_mass',0)>0:
        from admet_v19_runner import materialize as historical
        r=read_json(route19); candidates=[b.copy() for _ in r['names']]
        for i,w in enumerate(r['weights'][1:]):
            if w>0:
                candidates[i]=historical(task,d,fit,hold,r['names'][i],'repeat_v21_audit')
        b=combine(b,candidates,r['weights'],r['mode'],r['metric'])
    if task=='cyp2d6_substrate_carbonmangels':
        from admet_v20_rare import combine as simple
        r=read_json(ROOT/'admet_model_outputs_v20_rare'/task/'decision.json')['selection']
        c=pd.read_csv(ROOT/'admet_model_outputs_v20_rare'/task/'rare_repeat.csv')
        np.testing.assert_array_equal(c.row,hold)
        b=simple(b,c.Y_pred.to_numpy(float),r['weight'],r['mode'])
    prefix=cache_prefix(task,d)
    extra=np.concatenate([d['desc'],d['maccs']],axis=1).astype(np.float32)
    c=materialize(task,d,prefix,extra,fit,hold,'repeat')
    q=combine(b,[c],[1-s['weight'],s['weight']],s['mode'],route['metric'])
    result=dict(task=task,selection_fixed=s,rows=len(hold),metric=route['metric'],
                before=utility(route['metric'],y[hold],b),after=utility(route['metric'],y[hold],q),
                gain=gain(route['metric'],y[hold],b,q),used_as_veto=False,
                caveat='Historically reused scaffold split, diagnostic only; not an untouched holdout.',
                uncertainty=bootstrap(y[hold],b,q,d['groups'][hold],route['metric']))
    save_json(dest/'repeat_audit.json',result)
    pd.DataFrame(dict(row=hold,Y_true=y[hold],baseline=b,Y_pred=q)).to_csv(dest/'repeat_comparison.csv',index=False)
    print(json.dumps(result),flush=True)


def audit_task(task):
    dest=OUT/task
    route=read_json(dest/'selection.json')
    s=route['selection']
    d=load_task(task)
    y,fold,b=anchor_oof(task)
    vy,vb=anchor_valid(task)
    n,nv=len(y),len(vy)
    np.testing.assert_array_equal(y,d['frames']['train'].Y)
    np.testing.assert_array_equal(vy,d['frames']['valid'].Y)
    c=pd.read_csv(dest/'finetune_oof.csv')
    cv=pd.read_csv(dest/'finetune_valid.csv').Y_pred.to_numpy(float)
    np.testing.assert_array_equal(c.row,np.arange(n))
    np.testing.assert_array_equal(c.fold,fold)
    np.testing.assert_array_equal(c.Y_true,y)
    metric=route['metric']
    checks=[]
    labels=pd.concat([d['frames']['train'],d['frames']['valid']],ignore_index=True).Y.to_numpy(float)
    # Check every trained stage, not just the final model used for submission.
    import hashlib
    for stage in ['fold0','fold1','fold2','valid']:
        stagepath=dest/stage
        metadata=read_json(stagepath/'metadata.json')
        assert metadata['model_replay_passed']
        assert metadata['protocol_sha256']==sha(PROTOCOL) and metadata['source_sha256']==sha(SOURCE)
        for name,h in metadata['files'].items():
            assert sha(stagepath/name)==h
        if stage.startswith('fold'):
            f=int(stage[-1]); expected_fit=np.flatnonzero(fold!=f); expected_query=np.flatnonzero(fold==f)
            expected_pred=c.Y_pred.to_numpy()[expected_query]
        else:
            expected_fit=np.arange(n); expected_query=np.arange(n,n+nv); expected_pred=cv
        with np.load(stagepath/'predictions.npz') as z:
            np.testing.assert_array_equal(z['fit'],expected_fit)
            np.testing.assert_array_equal(z['query'],expected_query)
            # pandas serializes float32 with its shortest round-tripping decimal.
            # Compare in the cache dtype so parsing into float64 adds no false error.
            np.testing.assert_allclose(z['prediction'],np.asarray(expected_pred,dtype=z['prediction'].dtype),rtol=1e-12,atol=1e-12)
        assert metadata['fit_target_sha256']==hashlib.sha256(np.asarray(labels[expected_fit],dtype='<f8').tobytes()).hexdigest()
    for f in np.unique(fold):
        mask=fold==f
        assert not set(d['groups'][:n][mask])&set(d['groups'][:n][~mask])
        assert not set(d['canonical'][:n][mask])&set(d['canonical'][:n][~mask])
        q=combine(b[mask],[c.Y_pred.to_numpy()[mask]],[1-s['weight'],s['weight']],s['mode'],metric)
        checks.append(dict(fold=int(f),before=utility(metric,y[mask],b[mask]),after=utility(metric,y[mask],q),
                           gain=gain(metric,y[mask],b[mask],q)))
    qv=combine(vb,[cv],[1-s['weight'],s['weight']],s['mode'],metric)
    result=dict(task=task,metric=metric,selection=s,folds=checks,valid_before=utility(metric,vy,vb),
                valid_after=utility(metric,vy,qv),alignment_passed=True,fold_scaffold_overlap=0,
                repeated_selection=True,valid_n=nv,train_n=n)
    result['all_stages_verified']=True
    if (dest/'repeat_audit.json').exists():
        result['repeat']=read_json(dest/'repeat_audit.json')
    if s['weight']>0:
        result['uncertainty']=bootstrap(vy,vb,qv,d['groups'][n:n+nv],metric)
        final=dest/'final'
        meta=read_json(final/'metadata.json')
        assert meta['model_replay_passed'] and meta['protocol_sha256']==sha(PROTOCOL)
        assert meta['source_sha256']==sha(SOURCE)
        for f,h in meta['files'].items():
            assert sha(final/f)==h
        with np.load(final/'predictions.npz') as z:
            np.testing.assert_array_equal(z['fit'],np.arange(n+nv))
            np.testing.assert_array_equal(z['query'],np.arange(n+nv,len(d['smiles'])))
            cand=pd.read_csv(dest/'finetune_test.csv').Y_pred.to_numpy(float)
            np.testing.assert_allclose(z['prediction'],np.asarray(cand,dtype=z['prediction'].dtype),rtol=1e-12,atol=1e-12)
        assert len(cand)==len(anchor_test(task))
        q=combine(anchor_test(task),[cand],[1-s['weight'],s['weight']],s['mode'],metric)
        assert np.isfinite(q).all()
        if metric in ['AUROC','AUPRC']:
            assert ((q>=0)&(q<=1)).all()
        target=REL/task
        target.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(dict(Y=q)).to_csv(target/f'submission_{task}.csv',index=False)
        pd.DataFrame(dict(Y_true=vy,baseline=vb,Y_pred=qv)).to_csv(target/'validation_predictions.csv',index=False)
        result.update(model_replay_passed=True,prediction_sha256=sha(target/f'submission_{task}.csv'))
        save_json(target/'selection.json',route)
    save_json(dest/'audit.json',result)
    return result


def export():
    p=read_json(PROTOCOL)
    assert all((OUT/t/'selection.json').exists() for t in p['tasks'])
    audits=[audit_task(t) for t in p['tasks']]
    changed=[a['task'] for a in audits if a['selection']['weight']>0]
    assert changed, 'No selected improvement; keep v20'
    base=next((ROOT/'admet_submissions/v20').glob('*.csv'))
    assert sha(base)==p['baseline_sha256']
    rows=list(csv.reader(io.StringIO(base.read_bytes().decode('utf-8-sig'),newline='')))
    template=list(csv.reader(io.StringIO((ROOT/'admet_official/submission_template.csv').read_bytes().decode('utf-8-sig'),newline='')))
    predictions={t:pd.read_csv(REL/t/f'submission_{t}.csv').Y.to_numpy(float) for t in changed}
    for row in rows[1:]:
        if row[0] in predictions:
            row[4]=repr(float(predictions[row[0]][int(row[1])]))
    assert len(rows)==len(template)==16380 and rows[0]==template[0]
    seen=set()
    for a,b in zip(rows[1:],template[1:]):
        assert len(a)==6 and a[:4]+a[5:]==b[:4]+b[5:]
        assert np.isfinite(float(a[4]))
        key=(a[0],a[1])
        assert key not in seen
        seen.add(key)
    path=ROOT/'admet_submissions/v21/我从山北来带着兰花草-小分子药物预测技术创新赛-20260927-v21.csv'
    path.parent.mkdir(parents=True,exist_ok=True)
    buf=io.StringIO(newline='')
    csv.writer(buf,lineterminator='\r\n').writerows(rows)
    path.write_bytes(b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8'))
    summary=dict(status='VERIFIED_REMOTE',submission=str(path),sha256=sha(path),bytes=path.stat().st_size,
                 rows=16379,columns=6,tasks=22,encoding='UTF-8 BOM',line_endings='CRLF',
                 baseline='v20',baseline_score=50.019727,baseline_sha256=sha(base),changed=changed,
                 trained_tasks=len(audits),audits=audits,local_gpu_training=False,automatic_submission=False,
                 platform_score=None,source_sha256=sha(SOURCE),protocol_sha256=sha(PROTOCOL))
    save_json(path.parent/'verification.json',summary)
    save_json(ROOT/'admet_diagnostics/v21_summary.json',summary)
    archive=ROOT/'admet_v21_review.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        files=[f for r in [OUT,REL,path.parent] for f in r.rglob('*') if f.is_file() and f.suffix in ['.json','.csv']]
        files+=list(ROOT.glob('admet_v21*.py'))+[PROTOCOL,ROOT/'admet_diagnostics/v21_summary.json']
        for f in files:
            tar.add(f,arcname=str(f.relative_to(ROOT)),recursive=False)
    print(json.dumps(dict(changed=changed,sha256=sha(path),archive_bytes=archive.stat().st_size)),flush=True)


if __name__=='__main__':
    remote_only()
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['export','repeat'],nargs='?',default='export')
    parser.add_argument('--tasks',nargs='*')
    args=parser.parse_args()
    if args.stage=='repeat':
        for task in args.tasks or read_json(PROTOCOL)['tasks']:
            repeat(task)
    else:export()
