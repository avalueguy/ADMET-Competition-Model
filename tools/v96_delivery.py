"""Template-exact normalized-assay release and independent model replay."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse,csv,io,json,sys,tarfile
from collections import Counter
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier,CatBoostRegressor
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v96_normalized_assays as v
from verify_submission import verify
from repack_upload_template import field_spans


def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))


def selected(require=True):
    f=json.loads((v.OUT/'frozen_selection.json').read_text());c=json.loads((v.OUT/'confirmation.json').read_text())
    assert f['signature']==c['signature']==v.sig() and c['status']=='COMPLETE'
    assert c['frozen_sha256']==v.b.sha(v.OUT/'frozen_selection.json')
    rs=[r for r in c['records'] if r['accepted']];assert [r['task'] for r in rs]==c['accepted']
    if require:assert rs,'No eligible expert: no CSV.'
    for r in rs:
        assert all(x['gain']>0 for x in r['confirmation']) and np.mean([x['gain'] for x in r['confirmation']])>v.threshold(r['metric'])
        assert any(all(r[key]==ch[key] for key in ('task','metric','config','weight')) for ch in f['selected'])
    return rs


def export():
    rs=selected();d=v.load();old=read(v.ANCHOR);rows=[r.copy() for r in old];changed={}
    for r in rs:
        k=d['tasks'].index(r['task']);p=v.mix(r['metric'],v.base(d,k,'final'),v.expert(d,k,'final',r['config']),r['weight'])
        pos=[i for i,row in enumerate(old) if i and row[0]==r['task']];assert len(p)==len(pos) and np.isfinite(p).all()
        for i,x in zip(pos,p):rows[i][4]=repr(float(x))
        changed[r['task']]=sum(rows[i][4]!=old[i][4] for i in pos)
    template=ROOT/'admet_official/submission_template.csv';raw=template.read_bytes();ss=field_spans(raw);chunks=[];cursor=0
    for i,row in enumerate(rows[1:],1):
        lo,hi=ss[i][4];chunks.extend((raw[cursor:lo],row[4].encode('ascii')));cursor=hi
    chunks.append(raw[cursor:]);output=b''.join(chunks)
    file=ROOT/'admet_submissions/v96_normalized_assays/我从山北来带着兰花草-小分子药物预测技术创新赛-20261026.csv';file.parent.mkdir(parents=True,exist_ok=True)
    if file.exists():assert file.read_bytes()==output,'Refuse overwrite of different release.'
    else:file.write_bytes(output)
    fmt=verify(file,template)
    v.b.save(v.OUT/'release.json',dict(status='PENDING_INDEPENDENT_REPLAY',version='v96',file=file.relative_to(ROOT).as_posix(),sha256=v.b.sha(file),
        anchor='v94',anchor_sha256=v.ANCHOR_SHA,anchor_official_score_user_reported=50.916,selected=rs,changed_tasks=changed,unchanged_tasks=22-len(changed),
        verification=fmt,official_score=None,test_labels_used=False,external_data_new_branch=False,new_pretrained_weights=False,
        limitations=['Historical proxies not exact v94 OOF; labels reused historically.','Neutral-parent covariates are conditional other official assays.',
                    'Historical baseline includes early pretrained sources.','No estimate or guarantee of official score gain.']))
    v.emit(stage='export',changed=changed,sha256=v.b.sha(file))


def reconstruct(d,k,part,spec,a):
    cols=[j for j in range(len(d['tasks'])) if j!=k]
    source=np.flatnonzero((d['split']<(2 if part=='final' else 1))&(d['tid']!=k))
    np.testing.assert_array_equal(source,a['source']);np.testing.assert_array_equal(cols,a['cols'])
    rows=np.r_[a['fit'],a['query']];nc=len(cols);exact={};neutral={}
    for i in source:
        exact.setdefault((str(d['canonical'][i]),int(d['tid'][i])),[]).append(float(d['y'][i]))
        neutral.setdefault((str(d['neutral_identity'][i]),int(d['tid'][i])),[]).append(float(d['y'][i]))
    ev=np.full((len(rows),nc),np.nan,np.float32);nv=ev.copy();spread=ev.copy()
    for i,row in enumerate(rows):
        for j,task in enumerate(cols):
            ys=exact.get((str(d['canonical'][row]),task))
            if ys:ev[i,j]=np.median(ys)
            ys=neutral.get((str(d['neutral_identity'][row]),task))
            if ys:nv[i,j]=np.median(ys);spread[i,j]=np.max(ys)-np.min(ys)
    ref=np.c_[ev,np.isfinite(ev).astype(np.float32),nv,np.isfinite(nv).astype(np.float32),spread]
    np.testing.assert_array_equal(a['x'][:,:5*nc],ref[:len(a['fit'])])
    np.testing.assert_array_equal(a['qx'][:,:5*nc],ref[len(a['fit']):])
    np.testing.assert_array_equal(a['known_counts'],np.isfinite(nv[len(a['fit']):]).sum(1))
    np.testing.assert_array_equal(a['exact_counts'],np.isfinite(ev[len(a['fit']):]).sum(1))
    if spec['view']=='chemical':
        np.testing.assert_array_equal(a['x'][:,5*nc+1:],d['views']['physchem'][d['mi'][a['fit']]])
        np.testing.assert_array_equal(a['qx'][:,5*nc+1:],d['views']['physchem'][d['mi'][a['query']]])


def audit(d,k,part,spec,mode,seed,steps):
    a=v.prepare(d,k,part,spec);fit,q=a['fit'],a['query'];assert not set(fit)&set(q) and np.all(d['split'][fit]<2)
    if part!='final':assert not set(d['groups'][fit])&set(d['groups'][q])
    reconstruct(d,k,part,spec,a)
    if part!='final':
        saved=d['y'][q].copy();d['y'][q]=1e6;d.pop('_v94_aux',None);d.pop('_v96_assays',None)
        z=v.prepare(d,k,part,spec);d['y'][q]=saved;d.pop('_v94_aux',None);d.pop('_v96_assays',None)
        for key in ('fit','query','x','qx','target','offset','qoffset','cols','source'):np.testing.assert_array_equal(a[key],z[key])
    folder=v.OUT/'models'/d['tasks'][k]/part/spec['name']/mode/str(seed)
    meta=json.loads((folder/'metadata.json').read_text());cache=np.load(folder/'predictions.npz')
    assert str(cache['signature'])==meta['signature']==v.sig() and v.b.sha(folder/'predictions.npz')==meta['predictions_sha256']
    assert v.b.sha(folder/'model.cbm')==meta['model_sha256'] and meta['state']==a['state']
    for key in ('fit','query'):np.testing.assert_array_equal(cache[key],a[key])
    np.testing.assert_array_equal(cache['snapshots'],v.SNAPSHOTS)
    cls=d['metrics'][k] in ('AUROC','AUPRC');net=(CatBoostClassifier if cls else CatBoostRegressor)()
    net.load_model(str(folder/'model.cbm'));records=[]
    for step in steps:
        p=net.predict(a['qx'],prediction_type='RawFormulaVal',ntree_end=step).astype(float)
        if cls and mode=='residual':p=p+a['qoffset']
        expected=cache['values'][v.SNAPSHOTS.index(step)];np.testing.assert_allclose(p,expected,atol=1e-12,rtol=1e-12)
        ids=np.unique([0,len(q)//2,len(q)-1]);sub=net.predict(a['qx'][ids],prediction_type='RawFormulaVal',ntree_end=step).astype(float)
        if cls and mode=='residual':sub=sub+a['qoffset'][ids]
        np.testing.assert_allclose(sub,p[ids],atol=1e-12,rtol=1e-12)
        r=dict(task=d['tasks'][k],part=part,spec=spec['name'],mode=mode,seed=seed,step=step,status='PASS',
            model_sha256=meta['model_sha256'],max_abs_difference=float(np.max(np.abs(p-expected))),
            assay_reconstruction=True,target_query_label_perturbation='PASS' if part!='final' else 'NOT_APPLICABLE')
        records.append(r);v.emit(stage='model_replay',**r)
    return records


def replay():
    d=v.load();rs=selected(False);used={};records=[]
    choices=rs or json.loads((v.OUT/'frozen_selection.json').read_text())['selected']
    if choices:
        for r in choices:
            for part in ('fold0','fold1','fold2','valid','final') if rs else ('fold0','fold1','fold2','valid'):
                for seed in v.SEEDS:used.setdefault((r['task'],part,r['config']['spec'],r['config']['mode'],seed),set()).add(r['config']['step'])
    else:
        for p in (v.OUT/'models').glob('*/*/*/*/*/metadata.json'):
            used[(p.parents[4].name,p.parents[3].name,p.parents[2].name,p.parents[1].name,int(p.parent.name))]={v.SNAPSHOTS[-1]}
    for (task,part,spec,mode,seed),steps in used.items():
        records.extend(audit(d,d['tasks'].index(task),part,next(x for x in v.SPECS if x['name']==spec),mode,seed,sorted(steps)))
    report=dict(status='PASS',models=len({r['model_sha256'] for r in records}),checkpoints=len(records),records=records,no_csv=not bool(rs),test_labels_used=False)
    if rs:
        rel=json.loads((v.OUT/'release.json').read_text());file=ROOT/rel['file'];assert v.b.sha(file)==rel['sha256']
        old,actual=read(v.ANCHOR),read(file);changed=Counter()
        for r in rs:
            k=d['tasks'].index(r['task'])
            for part in ('fold0','fold1','fold2','valid','final'):
                p=v.mix(r['metric'],v.base(d,k,part),v.expert(d,k,part,r['config']),r['weight'])
                if part=='final':np.testing.assert_allclose(p,[float(z[4]) for z in actual[1:] if z[0]==r['task']],atol=1e-12,rtol=1e-12)
                else:
                    q=v.r83.split(d,k,part)[1];record=next(x for x in r['confirmation'] if x['part']==part)
                    np.testing.assert_allclose(v.r83.score(r['metric'],d['y'][q],p),record['candidate'],atol=1e-12,rtol=1e-12)
        for a,z in zip(old[1:],actual[1:]):
            assert a[:4]+a[5:]==z[:4]+z[5:]
            if a[4]!=z[4]:changed[z[0]]+=1
            if z[0] not in rel['changed_tasks']:assert a==z
        assert dict(changed)==rel['changed_tasks'];template=ROOT/'admet_official/submission_template.csv';fmt=verify(file,template)
        raw,ref=file.read_bytes(),template.read_bytes();ss,tt=field_spans(raw),field_spans(ref);chunks=[];cursor=0
        for i in range(1,len(actual)):
            lo,hi=ss[i][4];a,z=tt[i][4];chunks.extend((raw[cursor:lo],ref[a:z]));cursor=hi
        chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
        fmt.update(nonprediction_bytes_exact=True,changed_tasks=dict(changed),checkpoints=len(records),model_reload='PASS',subset_prediction='PASS',target_assay_excluded='PASS')
        v.b.save(file.with_suffix('.verification.json'),fmt);report['submission']=fmt;rel.update(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',verification=fmt)
    v.b.save(v.OUT/'model_verification.json',report)
    if rs:rel['verification_sha256']=v.b.sha(v.OUT/'model_verification.json');v.b.save(v.OUT/'release.json',rel)
    v.emit(stage='verification',status='PASS',checkpoints=len(records),no_csv=not bool(rs))


def package():
    audit=json.loads((v.OUT/'model_verification.json').read_text());assert audit['status']=='PASS';accepted=bool(selected(False));files={}
    paths=[v.OUT,Path(__file__),ROOT/'tools/v96_normalized_assays.py',ROOT/'tools/run_v96_pipeline.py',
        ROOT/'tools/v95_graph_specialists.py',ROOT/'tools/v95_identity_audit.py',v.IDENTITIES,v.IDENTITIES.parent/'coverage.json',
        ROOT/'docs/ITERATION_V96_20261026.md',ROOT/'docs/platform_feedback_v94_20261026.json']
    if accepted:paths.append(ROOT/'admet_submissions/v96_normalized_assays')
    for path in paths:
        for p in ([path] if path.is_file() else path.rglob('*')):
            if p.is_file() and p.suffix!='.pyc' and p.name!='smoke.cbm':files[p.relative_to(ROOT).as_posix()]=p
    prefix='v96_normalized_assays_20261026' if accepted else 'v96_rejected_normalized_assays_20261026'
    folder=ROOT/'releases';mp=folder/(prefix+'.manifest.json')
    deps=[dict(path='releases/'+n,sha256=v.b.sha(folder/n)) for n in ('v86_metric_ranking_20261025.tar.gz',
        'v88_context_kernel_20261025.tar.gz','v92_full_kernel_20261025.tar.gz','v94_isolated_assay_20261025.tar.gz')]
    v.b.save(mp,dict(files=[dict(path=n,sha256=v.b.sha(p),bytes=p.stat().st_size) for n,p in sorted(files.items())],
        dependencies=deps,test_labels_included=False,credentials_included=False,no_csv=not accepted))
    archive=folder/(prefix+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tf:
        for name,p in sorted(files.items()):tf.add(p,arcname=name)
        tf.add(mp,arcname=mp.relative_to(ROOT).as_posix())
    report=dict(path=archive.relative_to(ROOT).as_posix(),sha256=v.b.sha(archive),bytes=archive.stat().st_size,files=len(files),no_csv=not accepted)
    v.b.save(folder/(prefix+'.archive.json'),report);v.emit(stage='package',**report)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['export','verify','package']);args=ap.parse_args()
    dict(export=export,verify=replay,package=package)[args.stage]()
