"""Export only frozen v84 tasks passing confirmation and replay saved models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import admet_v77_ranking as b
import v84_neighbor_transfer as v
from verify_submission import verify
from repack_upload_template import field_spans


def read(path):
    return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'),newline='')))


def selected():
    frozen=json.loads((v.OUT/'frozen_selection.json').read_text())
    conf=json.loads((v.OUT/'confirmation.json').read_text())
    assert frozen['signature']==conf['signature']==v.signature()
    assert conf['status']=='COMPLETE'
    assert conf['frozen_sha256']==b.sha(v.OUT/'frozen_selection.json')
    result=[r for r in conf['records'] if r['accepted']]
    assert result,'No accepted tasks. No submission permitted.'
    assert [r['task'] for r in result]==conf['accepted']
    for r in result:
        assert all(p['gain']>0 for p in r['confirmation'])
        assert any(all(r[k]==f[k] for k in ('task','metric','config','weight')) for f in frozen['selected'])
    return result


def export():
    choices=selected();d=v.load();profile=v.profiles(d,True)
    original=read(v.ANCHOR);rows=[r.copy() for r in original];changed={}
    for r in choices:
        k=d['tasks'].index(r['task']);q=v.old.split(d,k,'final')[1]
        pos=[i for i,row in enumerate(original) if i and row[0]==r['task']]
        assert len(pos)==len(q)
        base=np.array([float(original[i][4]) for i in pos])
        if r['task'] not in v.old.BASE_SPECS and r['task']!='ppbr_az':
            historical=read(v.ORIGINAL)
            assert [row for row in original[1:] if row[0]==r['task']]==[row for row in historical[1:] if row[0]==r['task']]
        pred=v.fit(d,k,'final',r['config'],profile)
        mixed=v.mix(r['metric'],base,pred,r['weight'])
        for i,p in zip(pos,mixed):rows[i][4]=repr(float(p))
        changed[r['task']]=sum(rows[i][4]!=original[i][4] for i in pos)
    template=ROOT/'admet_official/submission_template.csv'
    raw=template.read_bytes();spans=field_spans(raw);chunks=[];cursor=0
    for i,row in enumerate(rows[1:],1):
        lo,hi=spans[i][4];chunks.extend((raw[cursor:lo],row[4].encode('ascii')));cursor=hi
    chunks.append(raw[cursor:])
    dest=ROOT/'admet_submissions/v84_neighbor_transfer';dest.mkdir(parents=True,exist_ok=True)
    file=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261025.csv'
    output=b''.join(chunks)
    if file.exists():assert file.read_bytes()==output,'Refuse replacing different submission.'
    else:file.write_bytes(output)
    fmt=verify(file,template)
    b.save(v.OUT/'release.json',dict(status='PENDING_INDEPENDENT_REPLAY',version='v84',
        file=file.relative_to(ROOT).as_posix(),sha256=b.sha(file),anchor='v83',anchor_sha256=v.ANCHOR_SHA,
        anchor_official_score_user_reported=50.870,changed_tasks=changed,unchanged_tasks=22-len(changed),
        selected=choices,verification=fmt,official_score=None,test_labels_used=False,external_data_new_branch=False,
        limitations=['Development labels reused historically; confirmation is not untouched independent data.',
            'Historical full ensemble OOF unavailable; comparisons use historical proxies and replayed later residual layers.',
            'Source assays on same or similar molecules are conditional deployment information.',
            'Final candidate residualizes original v73 endpoints then blends with v83. This mirrors development candidate base vs replayed v83 layer, not a full v83 residual model.',
            'Anchor contains historical pretrained sources; no new external data or pretrained weights in v84.']))
    v.emit(stage='export',changed_tasks=changed,sha256=b.sha(file))


def close(a,b1,tol=1e-10):
    np.testing.assert_allclose(a,b1,atol=tol,rtol=tol,equal_nan=True)


def replay_profiles(d,final):
    """Independently reconstruct sample summaries on CPU float64."""
    stage='train_valid' if final else 'train';folder=v.OUT/'profiles'/stage
    z=np.load(folder/'profiles.npz');assert str(z['signature'])==v.signature();profiles=z['x']
    with np.load(ROOT/'admet_model_outputs_v67/features/official_chemistry.npz') as chem:
        x=chem['x'][:,:4096].astype(np.float64);canonical=chem['canonical']
    norms=np.square(x).sum(1)
    checks=[]
    for k,task in enumerate(d['tasks']):
        state=np.load(folder/f'{task}.npz');source=state['source'];refs=state['refs'];values=state['values'];labels=state['labels']
        expected=np.flatnonzero((d['tid']==k)&(d['split']<(2 if final else 1)))
        np.testing.assert_array_equal(source,expected)
        assert not np.any(d['split'][source]==2)
        un,inv=np.unique(d['canonical'][source],return_inverse=True)
        rr=np.array([d['mi'][source[np.flatnonzero(inv==j)[0]]] for j in range(len(un))])
        yy=np.array([np.median(d['y'][source[inv==j]]) for j in range(len(un))])
        np.testing.assert_array_equal(refs,rr);close(values,yy);np.testing.assert_array_equal(canonical[refs],un)
        metric=d['metrics'][k]
        if metric=='MAE':
            center=float(np.median(values));scale=max(float(np.quantile(values,.75)-np.quantile(values,.25)),1e-3)
            close(state['center'],center);close(state['scale'],scale);close(labels,(values-center)/scale)
        elif metric=='Spearman':close(labels,v.old.ranknorm(values))
        else:close(labels,values)
        queries=state['audit_queries'];chosen_ix=state['audit_ix'];chosen_top=state['audit_top']
        dot=x[queries]@x[refs].T
        sim=np.clip(dot/np.maximum(norms[queries,None]+norms[refs][None]-dot,1e-8),0,1)
        take=min(v.NEIGHBOR_COUNT,len(refs));ix=np.argsort(-sim,axis=1,kind='stable')[:,:take]
        tops=np.take_along_axis(sim,ix,axis=1);y=labels[ix]
        w4=np.maximum(tops**4,1e-12);w16=np.maximum(tops**16,1e-12)
        mean4=(w4*y).sum(1)/w4.sum(1);mean16=(w16*y).sum(1)/w16.sum(1)
        sd=np.sqrt((w4*(y-mean4[:,None])**2).sum(1)/w4.sum(1))
        actual=np.c_[mean4,mean16,sd,tops[:,0],tops.mean(1)]
        close(actual[:,3:],profiles[queries,k,3:],2e-5)
        # Independently verify chosen neighbor identities and their exact
        # label-weighted aggregates, allowing arbitrary choices only at ties.
        assert np.all((chosen_ix>=0)&(chosen_ix<len(refs)))
        for ixrow in chosen_ix:assert len(np.unique(ixrow))==take
        exact_top=np.take_along_axis(sim,chosen_ix,axis=1)
        close(exact_top,chosen_top,2e-5)
        assert np.all(exact_top[:,-1]>=tops[:,-1]-2e-5)
        y=labels[chosen_ix];top=chosen_top.astype(np.float64)
        w4=np.maximum(top**4,1e-12);w16=np.maximum(top**16,1e-12)
        mean4=(w4*y).sum(1)/w4.sum(1);mean16=(w16*y).sum(1)/w16.sum(1)
        sd=np.sqrt((w4*(y-mean4[:,None])**2).sum(1)/w4.sum(1))
        reconstructed=np.c_[mean4,mean16,sd,top[:,0],top.mean(1)]
        close(reconstructed,profiles[queries,k],2e-5)
        checks.append(dict(task=task,source_rows=len(source),source_molecules=len(refs),
            source_state='PASS',similarity_replay='PASS',
            max_aggregate_difference=float(np.max(np.abs(actual[:,:3]-profiles[queries,k,:3])))))
    return checks


def replay_models():
    d=v.load();choices=selected();profile=v.profiles(d,True)
    release=json.loads((v.OUT/'release.json').read_text());file=ROOT/release['file']
    assert b.sha(file)==release['sha256'];original,actual=read(v.ANCHOR),read(file)
    model_records=[]
    for r in choices:
        k=d['tasks'].index(r['task']);metric=r['metric'];iscls=metric in ('AUROC','AUPRC');cfg=r['config']
        for part in ('fold0','fold1','fold2','valid','final'):
            fit,q=v.old.split(d,k,part);folder=v.OUT/'models'/r['task']/part/cfg['name']
            cache=np.load(folder/'predictions.npz');meta=json.loads((folder/'metadata.json').read_text())
            assert meta['signature']==str(cache['signature'])==v.signature()
            np.testing.assert_array_equal(cache['fit'],fit);np.testing.assert_array_equal(cache['query'],q)
            assert not set(fit)&set(q) and np.all(d['split'][fit]<2)
            if part!='final':
                assert not set(d['groups'][fit])&set(d['groups'][q])
                assert not set(d['canonical'][fit])&set(d['canonical'][q])
            aux,cols,source,counts=v.old.aux(d,k,part=='final')
            np.testing.assert_array_equal(source,cache['source']);np.testing.assert_array_equal(counts[q],cache['known_counts'])
            base=v.old.raw_proxy(d,fit)
            qp=(np.array([float(z['y_pred']) for z in b.table(v.ORIGINAL) if z['task']==r['task']])
                if part=='final' else v.old.raw_proxy(d,q))
            y=d['y'][fit];state=meta['state']
            if iscls:
                z,qz=v.old.ranklog(base),v.old.ranklog(qp);beta=v.old.calibrate(y,z)
                off,qoff=beta[0]+beta[1]*z,beta[0]+beta[1]*qz;scale=1.;close(state['beta'],beta)
            elif metric=='MAE':
                center=float(np.median(y));scale=max(float(np.quantile(y,.75)-np.quantile(y,.25)),1e-3)
                z,qz=(base-center)/scale,(qp-center)/scale;off,qoff=np.zeros(len(base)),qp
                close([state['center'],state['scale']],[center,scale])
            else:
                z,qz=v.old.ranknorm(base),v.old.ranknorm(qp);xx=np.c_[np.ones(len(z)),z]
                beta=np.linalg.solve(xx.T@xx+np.diag([1e-8,1.]),xx.T@v.old.ranknorm(y));beta[1]=max(beta[1],.01)
                off,qoff=np.zeros(len(base)),beta[0]+beta[1]*qz;scale=1.;close(state['beta'],beta)
            pr=profile if part=='final' else v.profiles(d,False)
            x=v.design(d,k,fit,z,cfg,pr,aux,cols);qx=v.design(d,k,q,qz,cfg,pr,aux,cols)
            for key,value in [('x',x),('qx',qx),('off',off),('qoff',qoff),('base_fit',base),('base_query',qp)]:close(cache[key],value)
            # Perturb every target-assay profile field: design must be unchanged.
            changed=pr.copy();changed[:,k]=np.nan
            close(qx,v.design(d,k,q,qz,cfg,changed,aux,cols))
            members=[]
            for seed in v.SEEDS:
                mp=folder/f'model_{seed}.cbm';assert b.sha(mp)==meta['model_hashes'][mp.name]
                model=(CatBoostClassifier if iscls else CatBoostRegressor)();model.load_model(str(mp))
                params=model.get_all_params();assert params['random_seed']==seed and params['depth']==cfg['depth'] and model.tree_count_==cfg['trees']
                raw=model.predict(qx,prediction_type='RawFormulaVal');ids=np.unique([0,len(qx)//2,len(qx)-1])
                close(raw[ids],model.predict(qx[ids],prediction_type='RawFormulaVal'))
                members.append(qoff+raw*scale)
                model_records.append(dict(path=mp.relative_to(ROOT).as_posix(),sha256=b.sha(mp),status='PASS'))
            members=np.stack(members);pred=expit(members.mean(0)) if iscls else members.mean(0)
            if r['task']=='ppbr_az':pred=pred.clip(0,100)
            close(members,cache['members']);close(pred,cache['p'])
            mixed=v.mix(metric,v.baseline(d,k,part),pred,r['weight'])
            if part!='final':
                saved=next(t for t in r['confirmation'] if t['part']==part)
                close(v.old.score(metric,d['y'][q],mixed),saved['candidate'])
            else:
                pos=[i for i,row in enumerate(original) if i and row[0]==r['task']]
                close(mixed,np.array([float(actual[i][4]) for i in pos]),1e-14)
    changed=Counter()
    tasks={r['task'] for r in choices}
    for a,b1 in zip(original[1:],actual[1:]):
        assert a[:4]+a[5:]==b1[:4]+b1[5:]
        if a[4]!=b1[4]:changed[b1[0]]+=1
        if b1[0] not in tasks:assert a==b1
    assert dict(changed)==release['changed_tasks']
    template=ROOT/'admet_official/submission_template.csv';fmt=verify(file,template)
    raw,ref=file.read_bytes(),template.read_bytes();ss,tt=field_spans(raw),field_spans(ref);cursor=0;chunks=[]
    for i in range(1,len(actual)):
        lo,hi=ss[i][4];a,b1=tt[i][4];chunks.extend((raw[cursor:lo],ref[a:b1]));cursor=hi
    chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
    profile_audits={stage:replay_profiles(d,stage=='final') for stage in ('development','final')}
    fmt.update(nonprediction_bytes_exact=True,changed_tasks=dict(changed),unchanged_tasks=22-len(changed),model_reload='PASS',models=len(model_records))
    result=dict(status='PASS',records=model_records,profile_source_replay=profile_audits,
        target_profile_perturbation='PASS',submission=fmt,test_labels_used=False)
    b.save(v.OUT/'model_verification.json',result);b.save(file.with_suffix('.verification.json'),fmt)
    release.update(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',verification=fmt,
        verification_sha256=b.sha(v.OUT/'model_verification.json'))
    b.save(v.OUT/'release.json',release)
    v.emit(stage='verify',models=len(model_records),changed_tasks=dict(changed),sha256=b.sha(file))


def package():
    release=json.loads((v.OUT/'release.json').read_text());assert release['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM'
    paths=[Path(__file__),ROOT/'tools/v84_neighbor_transfer.py',ROOT/'tools/v84_uncertainty.py',ROOT/'tools/v83_residual.py',ROOT/'tools/v83_delivery.py',
        ROOT/'tools/verify_submission.py',ROOT/'tools/repack_upload_template.py',ROOT/'admet_v77_ranking.py',
        ROOT/'admet_v67_protocol.json',ROOT/'docs/artifact-manifests/runtime-inputs.manifest.json',
        ROOT/'docs/ITERATION_V84_20261025.md',ROOT/'admet_official/submission_template.csv',v.ANCHOR,v.ORIGINAL,
        ROOT/'admet_competition_data/data_benchmark',ROOT/'recovered_history',
        ROOT/'admet_model_outputs_v67/features',ROOT/'admet_model_outputs_v73/features',
        ROOT/'admet_model_outputs_v67/folds.npy',v.OUT,
        ROOT/'admet_model_outputs_v83/confirmation',ROOT/'admet_model_outputs_v83/baseline',
        ROOT/'admet_submissions/v82_targeted',ROOT/'admet_submissions/v83_matched_residual',
        ROOT/'admet_submissions/v84_neighbor_transfer']
    files={}
    for path in paths:
        assert path.exists(),path
        for p in ([path] if path.is_file() else path.rglob('*')):
            if p.is_file() and p.suffix!='.pyc' and p.name!='test.csv':files[p.relative_to(ROOT).as_posix()]=p
    manifest=dict(files=[dict(path=name,sha256=b.sha(p),bytes=p.stat().st_size) for name,p in sorted(files.items())],
        test_labels_included=False,credentials_included=False)
    folder=ROOT/'releases';folder.mkdir(exist_ok=True);prefix='v84_neighbor_transfer_20261025'
    mp=folder/(prefix+'.manifest.json');b.save(mp,manifest)
    archive=folder/(prefix+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tf:
        for name,p in sorted(files.items()):tf.add(p,arcname=name)
        tf.add(mp,arcname=mp.relative_to(ROOT).as_posix())
    b.save(folder/(prefix+'.archive.json'),dict(path=archive.relative_to(ROOT).as_posix(),
        sha256=b.sha(archive),bytes=archive.stat().st_size,files=len(files)))
    v.emit(stage='package',sha256=b.sha(archive),bytes=archive.stat().st_size,files=len(files))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['export','verify','package']);args=ap.parse_args()
    dict(export=export,verify=replay_models,package=package)[args.stage]()
