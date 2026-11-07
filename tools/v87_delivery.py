"""Release v87 with template byte preservation and independent booster replay."""
from __future__ import annotations
import argparse
import csv
import io
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path
import numpy as np
import xgboost as xgb
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import v87_metric_ranking as v
from verify_submission import verify
from repack_upload_template import field_spans


def read(path):return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'),newline='')))
def close(a,b,tol=1e-9):np.testing.assert_allclose(a,b,atol=tol,rtol=tol,equal_nan=True)


def selected():
    frozen=json.loads((v.OUT/'frozen_selection.json').read_text())
    conf=json.loads((v.OUT/'confirmation.json').read_text())
    assert conf['status']=='COMPLETE' and frozen['signature']==conf['signature']==v.signature()
    assert conf['frozen_sha256']==v.b.sha(v.OUT/'frozen_selection.json')
    choices=[r for r in conf['records'] if r['accepted']]
    assert choices,'No eligible model: no CSV export.'
    assert [r['task'] for r in choices]==conf['accepted']
    for r in choices:
        assert all(t['gain']>0 for t in r['confirmation'])
        assert any(all(f[key]==r[key] for key in ('task','metric','config','weight')) for f in frozen['selected'])
    return choices


def export():
    d=v.load();choices=selected();original=read(v.ANCHOR);rows=[r.copy() for r in original];changed={}
    for r in choices:
        k=d['tasks'].index(r['task']);expert=v.fit(d,k,'final',r['config'])
        pred=v.mix(v.baseline(d,k,'final'),expert,r['weight'])
        pos=[i for i,row in enumerate(original) if i and row[0]==r['task']]
        assert len(pos)==len(pred) and np.isfinite(pred).all()
        for i,p in zip(pos,pred):rows[i][4]=repr(float(p))
        changed[r['task']]=sum(rows[i][4]!=original[i][4] for i in pos)
    template=ROOT/'admet_official/submission_template.csv';raw=template.read_bytes();ss=field_spans(raw);cursor=0;chunks=[]
    for i,row in enumerate(rows[1:],1):
        lo,hi=ss[i][4];chunks.extend((raw[cursor:lo],row[4].encode('ascii')));cursor=hi
    chunks.append(raw[cursor:]);output=b''.join(chunks)
    folder=ROOT/'admet_submissions/v87_metric_ranking';folder.mkdir(parents=True,exist_ok=True)
    file=folder/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261025.csv'
    if file.exists():assert file.read_bytes()==output,'Refuse overwriting a different release.'
    else:file.write_bytes(output)
    fmt=verify(file,template)
    v.b.save(v.OUT/'release.json',dict(status='PENDING_INDEPENDENT_REPLAY',version='v87',
        file=file.relative_to(ROOT).as_posix(),sha256=v.b.sha(file),anchor='v86',anchor_sha256=v.ANCHOR_SHA,
        anchor_official_score_user_reported=None,highest_official_score_user_reported=50.889,changed_tasks=changed,unchanged_tasks=22-len(changed),
        selected=choices,verification=fmt,official_score=None,test_labels_used=False,external_data_new_branch=False,
        limitations=['Historical proxies are not full exact v86 OOF; labels were reused in historical iterations.',
            'Expert starts from original v73 endpoint; comparison and rank blend use replayed v86 endpoint.',
            'Other official assay measurements are conditional deployment covariates.',
            'Local improvements do not establish official score improvement or attainment of 52.']))
    v.emit(stage='export',changed_tasks=changed,sha256=v.b.sha(file))


def replay_design(d,k,part,cfg,cache):
    fit,q=v.r83.split(d,k,part)
    np.testing.assert_array_equal(cache['fit'],fit);np.testing.assert_array_equal(cache['query'],q)
    assert not set(fit)&set(q) and np.all(d['split'][fit]<2)
    if part!='final':
        assert not set(d['canonical'][fit])&set(d['canonical'][q])
        assert not set(d['groups'][fit])&set(d['groups'][q])
    source=np.flatnonzero((d['split']<(2 if part=='final' else 1))&(d['tid']!=k))
    cols=[j for j in range(len(d['tasks'])) if j!=k]
    np.testing.assert_array_equal(cache['source'],source);np.testing.assert_array_equal(cache['cols'],cols)
    assert not np.any(d['tid'][source]==k) and not np.any(d['split'][source]==2)
    known={}
    for i in source:known.setdefault((str(d['canonical'][i]),int(d['tid'][i])),[]).append(float(d['y'][i]))
    def aux(rows):
        values=np.full((len(rows),len(cols)),np.nan,dtype=np.float32)
        for i,row in enumerate(rows):
            for j,task in enumerate(cols):
                vals=known.get((str(d['canonical'][row]),task))
                if vals:values[i,j]=np.median(vals)
        return np.c_[values,np.isfinite(values).astype(np.float32)]
    af,aq=aux(fit),aux(q)
    np.testing.assert_array_equal(cache['known_counts'],aq[:,len(cols):].sum(1).astype(int))
    base=v.r83.raw_proxy(d,fit)
    qp=(np.array([float(r['y_pred']) for r in v.b.table(v.r85.r84.ORIGINAL) if r['task']==d['tasks'][k]])
        if part=='final' else v.r83.raw_proxy(d,q))
    z,qz=v.r83.ranklog(base),v.r83.ranklog(qp);beta=v.r83.calibrate(d['y'][fit],z)
    off,qoff=beta[0]+beta[1]*z,beta[0]+beta[1]*qz
    for key,val in [('base_fit',base),('base_query',qp),('z',z),('qz',qz),('beta',beta),('off',off),('qoff',qoff)]:close(cache[key],val)
    x,qx=np.c_[af,z],np.c_[aq,qz]
    if cfg['view'] in ('compact','multiview'):
        key='physchem' if cfg['view']=='compact' else 'multiview'
        x=np.c_[x,d['views'][key][d['mi'][fit]]];qx=np.c_[qx,d['views'][key][d['mi'][q]]]
    x,qx=x.astype(np.float32),qx.astype(np.float32)
    close(cache['x'],x,0);close(cache['qx'],qx,0)
    return q,qx,qoff


def replay():
    choices=selected();d=v.load();release=json.loads((v.OUT/'release.json').read_text());file=ROOT/release['file']
    assert v.b.sha(file)==release['sha256'];actual=read(file);original=read(v.ANCHOR);records=[]
    for r in choices:
        k=d['tasks'].index(r['task']);cfg=r['config']
        for part in ('fold0','fold1','fold2','valid','final'):
            folder=v.OUT/'models'/r['task']/part/cfg['name'];cache=np.load(folder/'predictions.npz');meta=json.loads((folder/'metadata.json').read_text())
            assert str(cache['signature'])==meta['signature']==v.signature()
            assert v.b.sha(folder/'predictions.npz')==meta['predictions_sha256']
            assert meta['config']==cfg and meta['seeds']==list(v.SEEDS)
            np.testing.assert_array_equal(cache['seeds'],v.SEEDS)
            q,qx,qoff=replay_design(d,k,part,cfg,cache);members=[]
            for seed in v.SEEDS:
                mp=folder/f'model_{seed}.ubj';assert v.b.sha(mp)==meta['model_hashes'][mp.name]
                model=xgb.Booster();model.load_model(mp);model.set_param(dict(device='cpu',nthread=4))
                conf=json.loads(model.save_config())['learner']
                assert conf['objective']['name']=={'logloss':'binary:logistic','map':'rank:map','pairwise':'rank:pairwise'}[cfg['method']]
                assert model.num_boosted_rounds()==cfg['trees'] and model.num_features()==qx.shape[1]
                assert any('split' in tree for tree in model.get_dump(dump_format='json')),'Degenerate ranking booster'
                p=model.predict(xgb.DMatrix(qx,base_margin=qoff),output_margin=True).astype(float)
                ids=np.unique([0,len(q)//2,len(q)-1]);batch=model.predict(xgb.DMatrix(qx[ids],base_margin=qoff[ids]),output_margin=True)
                close(batch,p[ids],2e-6)
                bare=model.predict(xgb.DMatrix(qx,base_margin=np.zeros(len(q))),output_margin=True)
                close(p,bare+qoff,3e-5)
                members.append(p)
            members=np.stack(members);close(cache['members'],members,2e-6)
            expert=expit(members.mean(0));close(cache['p'],expert,2e-7)
            pred=v.mix(v.baseline(d,k,part),expert,r['weight'])
            if part!='final':
                rr=next(t for t in r['confirmation'] if t['part']==part)
                close(rr['candidate'],v.r83.score(r['metric'],d['y'][q],pred))
                close(rr['baseline'],v.r83.score(r['metric'],d['y'][q],v.baseline(d,k,part)))
            else:
                pos=[i for i,row in enumerate(actual) if i and row[0]==r['task']]
                close(pred,np.array([float(actual[i][4]) for i in pos]),1e-12)
            records.append(dict(task=r['task'],part=part,status='PASS',models=len(v.SEEDS),
                predictions_sha256=v.b.sha(folder/'predictions.npz')))
    target={r['task'] for r in choices};changed=Counter()
    for a,b in zip(original[1:],actual[1:]):
        assert a[:4]+a[5:]==b[:4]+b[5:]
        if a[4]!=b[4]:changed[b[0]]+=1
        if b[0] not in target:assert a==b
    assert dict(changed)==release['changed_tasks']
    template=ROOT/'admet_official/submission_template.csv';fmt=verify(file,template)
    raw,ref=file.read_bytes(),template.read_bytes();ss,tt=field_spans(raw),field_spans(ref);cursor=0;chunks=[]
    for i in range(1,len(actual)):
        lo,hi=ss[i][4];a,b=tt[i][4];chunks.extend((raw[cursor:lo],ref[a:b]));cursor=hi
    chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
    fmt.update(changed_tasks=dict(changed),unchanged_tasks=22-len(changed),nonprediction_bytes_exact=True,
        models=len(records)*len(v.SEEDS),model_reload='PASS',source_design_reconstruction='PASS')
    v.b.save(v.OUT/'model_verification.json',dict(status='PASS',records=records,submission=fmt,test_labels_used=False))
    v.b.save(file.with_suffix('.verification.json'),fmt)
    release.update(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',verification=fmt,
        verification_sha256=v.b.sha(v.OUT/'model_verification.json'))
    v.b.save(v.OUT/'release.json',release);v.emit(stage='verify',models=fmt['models'],sha256=v.b.sha(file),changed_tasks=dict(changed))


def package():
    release=json.loads((v.OUT/'release.json').read_text());assert release['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM'
    paths=[Path(__file__),ROOT/'tools/v87_metric_ranking.py',ROOT/'tools/v87_diagnostics.py',
        ROOT/'tools/v86_metric_ranking.py',ROOT/'tools/v86_delivery.py',ROOT/'tools/v86_diagnostics.py',
        ROOT/'tools/v85_smooth_residual.py',ROOT/'tools/v83_residual.py',ROOT/'tools/v84_neighbor_transfer.py',
        ROOT/'tools/verify_submission.py',ROOT/'tools/repack_upload_template.py',ROOT/'admet_v77_ranking.py',ROOT/'admet_v67_protocol.json',
        ROOT/'docs/artifact-manifests/runtime-inputs.manifest.json',ROOT/'docs/ITERATION_V87_20261025.md',ROOT/'docs/ITERATION_V86_20261025.md',
        ROOT/'admet_official/submission_template.csv',v.ANCHOR,v.r85.r84.ORIGINAL,
        ROOT/'admet_submissions/v82_targeted',ROOT/'admet_submissions/v83_matched_residual',ROOT/'admet_submissions/v84_neighbor_transfer',
        ROOT/'admet_competition_data/data_benchmark',ROOT/'recovered_history',ROOT/'admet_model_outputs_v67/features',
        ROOT/'admet_model_outputs_v67/folds.npy',ROOT/'admet_model_outputs_v73/features',v.OUT,
        ROOT/'admet_model_outputs_v83/confirmation',ROOT/'admet_model_outputs_v83/baseline',
        ROOT/'admet_model_outputs_v84/profiles',ROOT/'admet_model_outputs_v84/models/clearance_hepatocyte_az',
        ROOT/'admet_model_outputs_v85',ROOT/'admet_model_outputs_v86',ROOT/'admet_submissions/v85_smooth_residual',ROOT/'admet_submissions/v87_metric_ranking']
    files={}
    for path in paths:
        assert path.exists(),path
        for p in ([path] if path.is_file() else path.rglob('*')):
            if p.is_file() and p.suffix!='.pyc' and p.name!='test.csv':files[p.relative_to(ROOT).as_posix()]=p
    prefix='v87_metric_ranking_20261025';folder=ROOT/'releases';folder.mkdir(exist_ok=True)
    mp=folder/(prefix+'.manifest.json');v.b.save(mp,dict(files=[dict(path=n,sha256=v.b.sha(p),bytes=p.stat().st_size) for n,p in sorted(files.items())],test_labels_included=False,credentials_included=False))
    archive=folder/(prefix+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tf:
        for name,p in sorted(files.items()):tf.add(p,arcname=name)
        tf.add(mp,arcname=mp.relative_to(ROOT).as_posix())
    v.b.save(folder/(prefix+'.archive.json'),dict(path=archive.relative_to(ROOT).as_posix(),sha256=v.b.sha(archive),bytes=archive.stat().st_size,files=len(files)))
    v.emit(stage='package',files=len(files),bytes=archive.stat().st_size,sha256=v.b.sha(archive))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['export','verify','package']);args=ap.parse_args()
    dict(export=export,verify=replay,package=package)[args.stage]()
