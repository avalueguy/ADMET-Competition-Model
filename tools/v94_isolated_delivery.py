"""Exact-template release and independent CPU replay for v94 isolated experts."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
import argparse,csv,io,json,sys,tarfile
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v94_isolated_assay as v
from verify_submission import verify
from repack_upload_template import field_spans
def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def selected(require=True):
    f=json.loads((v.OUT/'frozen_selection.json').read_text());c=json.loads((v.OUT/'confirmation.json').read_text());assert f['signature']==c['signature']==v.sig() and c['status']=='COMPLETE'
    assert c['frozen_sha256']==v.b.sha(v.OUT/'frozen_selection.json');rs=[r for r in c['records'] if r['accepted']];assert [r['task'] for r in rs]==c['accepted']
    if require:assert rs,'No eligible expert: no CSV.'
    for r in rs:
        assert all(x['gain']>0 for x in r['confirmation']) and np.mean([x['gain'] for x in r['confirmation']])>v.threshold(r['metric'])
        assert any(all(r[key]==ch[key] for key in ('task','metric','config','weight')) for ch in f['selected'])
    return rs

def export():
    rs=selected();d=v.load();old=read(v.ANCHOR);rows=[r.copy() for r in old];changed={}
    for r in rs:
        k=d['tasks'].index(r['task']);p=v.mix(r['metric'],v.base(d,k,'final'),v.expert(d,k,'final',r['config']),r['weight']);pos=[i for i,row in enumerate(old) if i and row[0]==r['task']];assert len(p)==len(pos) and np.isfinite(p).all()
        for i,x in zip(pos,p):rows[i][4]=repr(float(x))
        changed[r['task']]=sum(rows[i][4]!=old[i][4] for i in pos)
    template=ROOT/'admet_official/submission_template.csv';raw=template.read_bytes();ss=field_spans(raw);chunks=[];cursor=0
    for i,row in enumerate(rows[1:],1):
        lo,hi=ss[i][4];chunks.extend((raw[cursor:lo],row[4].encode('ascii')));cursor=hi
    chunks.append(raw[cursor:]);output=b''.join(chunks);file=ROOT/'admet_submissions/v94_isolated_assay/我从山北来带着兰花草-小分子药物预测技术创新赛-20261025.csv';file.parent.mkdir(parents=True,exist_ok=True)
    if file.exists():assert file.read_bytes()==output,'Refuse overwrite of different release.'
    else:file.write_bytes(output)
    fmt=verify(file,template);v.b.save(v.OUT/'release.json',dict(status='PENDING_INDEPENDENT_REPLAY',version='v94',file=file.relative_to(ROOT).as_posix(),sha256=v.b.sha(file),anchor='v92',anchor_sha256=v.ANCHOR_SHA,anchor_official_score_user_reported=50.915,selected=rs,changed_tasks=changed,unchanged_tasks=22-len(changed),verification=fmt,official_score=None,test_labels_used=False,external_data_new_branch=False,new_pretrained_weights=False,limitations=['Historical proxies not exact v92 OOF; labels reused historically.','Other official assays are conditional inputs; entire target assay excluded.','Historical baseline includes early pretrained sources.','No estimate or guarantee of official score gain.']))
    v.emit(stage='export',changed=changed,sha256=v.b.sha(file))

def audit(d,k,part,seed,steps):
    a=v.prepare(d,k,part);fit,q=a['fit'],a['query'];assert not set(fit)&set(q) and np.all(d['split'][fit]<2)
    if part!='final':assert not set(d['groups'][fit])&set(d['groups'][q]) and not set(d['canonical'][fit])&set(d['canonical'][q])
    assert k not in a['cols'] and np.all(d['tid'][a['source']]!=k) and np.all(d['split'][a['source']]<(2 if part=='final' else 1))
    # Independent reconstruction of all measured covariates.
    cols=[j for j in range(len(d['tasks'])) if j!=k];source=np.flatnonzero((d['split']<(2 if part=='final' else 1))&(d['tid']!=k));np.testing.assert_array_equal(a['source'],source);lookup={}
    for i in source:lookup.setdefault((str(d['canonical'][i]),int(d['tid'][i])),[]).append(float(d['y'][i]))
    vals=np.full((len(fit)+len(q),len(cols)),np.nan)
    for ii,row in enumerate(np.r_[fit,q]):
        for j,t in enumerate(cols):
            ys=lookup.get((str(d['canonical'][row]),t))
            if ys:vals[ii,j]=np.median(ys)
    vals=vals.astype(np.float32);ref=vals[:len(fit)];center=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in ref.T]);scale=np.array([np.std(c[np.isfinite(c)]) if np.isfinite(c).any() else 1. for c in ref.T]).clip(.05);mask=np.isfinite(vals).astype(np.float32);assay=np.c_[np.nan_to_num((vals-center)/scale).clip(-5,5)*mask,mask].astype(np.float32)
    np.testing.assert_array_equal(a['cols'],cols);np.testing.assert_array_equal(a['assay_center'],center);np.testing.assert_array_equal(a['assay_scale'],scale);np.testing.assert_array_equal(a['assay'],assay)
    # End-to-end input isolation, beyond a single example's target masking.
    if part!='final':
        saved=d['y'][q].copy();d['y'][q]=1e6;d.pop('_v94_aux',None);perturbed=v.prepare(d,k,part);d['y'][q]=saved;d.pop('_v94_aux',None)
        for key in ('fp','desc','assay','target','offset','qoffset','fit','query','cols','source'):np.testing.assert_array_equal(a[key],perturbed[key])
    folder=v.OUT/'models'/d['tasks'][k]/part/str(seed);meta=json.loads((folder/'metadata.json').read_text());cache=np.load(folder/'predictions.npz');assert str(cache['signature'])==meta['signature']==v.sig() and v.b.sha(folder/'predictions.npz')==meta['predictions_sha256']
    for key in ('fit','query','qoffset'):np.testing.assert_array_equal(cache[key],a[key])
    np.testing.assert_array_equal(cache['snapshots'],v.SNAPSHOTS);ts=v.tensors(a,'cpu');records=[]
    for step in steps:
        path=folder/f'checkpoint_{step}.pt';assert v.b.sha(path)==meta['checkpoint_hashes'][path.name];ck=torch.load(path,map_location='cpu',weights_only=False);assert ck['signature']==v.sig() and ck['seed']==seed and ck['step']==step and ck['state']==a['state'];np.testing.assert_array_equal(ck['fit'],fit);np.testing.assert_array_equal(ck['query'],q)
        net=v.Net(ck['desc_dim'],ck['assay_dim']).eval();net.load_state_dict(ck['weights']);pred=v.infer(net,a,ts,113);expected=cache['values'][v.SNAPSHOTS.index(step)];np.testing.assert_allclose(pred,expected,atol=4e-5,rtol=3e-5)
        ids=np.unique([0,len(q)//2,len(q)-1]);ix=ids+len(fit)
        with torch.inference_mode():sub=net(*(x[ix] for x in ts)).numpy()
        np.testing.assert_allclose(sub,pred[ids],atol=2e-5,rtol=3e-5);r=dict(task=d['tasks'][k],part=part,seed=seed,step=step,status='PASS',sha256=v.b.sha(path),cpu_max_abs_difference=float(np.max(np.abs(pred-expected))),target_query_label_perturbation='PASS' if part!='final' else 'NOT_APPLICABLE',assay_reconstruction=True);records.append(r);v.emit(stage='checkpoint_replay',**r)
    return records

def replay():
    d=v.load();rs=selected(False);used={};records=[]
    if rs:
        for r in rs:
            for part in ('fold0','fold1','fold2','valid','final'):
                for seed in v.SEEDS:used.setdefault((r['task'],part,seed),set()).add(r['config']['step'])
    else:
        for p in (v.OUT/'models').glob('*/*/*/metadata.json'):used[(p.parent.parent.parent.name,p.parent.parent.name,int(p.parent.name))]=set(v.SNAPSHOTS)
    for (task,part,seed),steps in used.items():records.extend(audit(d,d['tasks'].index(task),part,seed,sorted(steps)))
    report=dict(status='PASS',checkpoints=len(records),records=records,no_csv=not bool(rs),test_labels_used=False)
    if rs:
        rel=json.loads((v.OUT/'release.json').read_text());file=ROOT/rel['file'];assert v.b.sha(file)==rel['sha256'];old,actual=read(v.ANCHOR),read(file);changed=Counter()
        for r in rs:
            k=d['tasks'].index(r['task'])
            for part in ('fold0','fold1','fold2','valid','final'):
                pred=v.mix(r['metric'],v.base(d,k,part),v.expert(d,k,part,r['config']),r['weight'])
                if part=='final':np.testing.assert_allclose(pred,[float(z[4]) for z in actual[1:] if z[0]==r['task']],atol=1e-12,rtol=1e-12)
                else:
                    q=v.r83.split(d,k,part)[1];record=next(x for x in r['confirmation'] if x['part']==part);np.testing.assert_allclose(v.r83.score(r['metric'],d['y'][q],pred),record['candidate'],atol=1e-12,rtol=1e-12)
        for a,z in zip(old[1:],actual[1:]):
            assert a[:4]+a[5:]==z[:4]+z[5:]
            if a[4]!=z[4]:changed[z[0]]+=1
            if z[0] not in rel['changed_tasks']:assert a==z
        assert dict(changed)==rel['changed_tasks'];template=ROOT/'admet_official/submission_template.csv';fmt=verify(file,template);raw,ref=file.read_bytes(),template.read_bytes();ss,tt=field_spans(raw),field_spans(ref);chunks=[];cursor=0
        for i in range(1,len(actual)):
            lo,hi=ss[i][4];a,z=tt[i][4];chunks.extend((raw[cursor:lo],ref[a:z]));cursor=hi
        chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
        fmt.update(nonprediction_bytes_exact=True,changed_tasks=dict(changed),checkpoints=len(records),cpu_model_reload='PASS',subset_prediction='PASS',target_assay_excluded='PASS');v.b.save(file.with_suffix('.verification.json'),fmt);report['submission']=fmt;rel.update(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',verification=fmt)
    v.b.save(v.OUT/'model_verification.json',report)
    if rs:rel['verification_sha256']=v.b.sha(v.OUT/'model_verification.json');v.b.save(v.OUT/'release.json',rel)
    v.emit(stage='verification',status='PASS',checkpoints=len(records),no_csv=not bool(rs))

def package():
    audit=json.loads((v.OUT/'model_verification.json').read_text());assert audit['status']=='PASS';accepted=bool(selected(False));files={}
    paths=[v.OUT,Path(__file__),ROOT/'tools/v94_isolated_assay.py',ROOT/'tools/run_v94_confirm_serialization.py',ROOT/'tools/v93_balanced_multitask.py',ROOT/'docs/ITERATION_V94_20261025.md',ROOT/'docs/ITERATION_V94_ATTEMPT1_INVALID_20261025.md',ROOT/'docs/platform_feedback_v92_20261025.json']
    if accepted:paths.append(ROOT/'admet_submissions/v94_isolated_assay')
    for path in paths:
        for p in ([path] if path.is_file() else path.rglob('*')):
            if p.is_file() and p.suffix!='.pyc':files[p.relative_to(ROOT).as_posix()]=p
    prefix='v94_isolated_assay_20261025' if accepted else 'v94_rejected_isolated_assay_20261025';folder=ROOT/'releases';mp=folder/(prefix+'.manifest.json');deps=[dict(path='releases/'+n,sha256=v.b.sha(folder/n)) for n in ('v86_metric_ranking_20261025.tar.gz','v88_context_kernel_20261025.tar.gz','v92_full_kernel_20261025.tar.gz')]
    v.b.save(mp,dict(files=[dict(path=n,sha256=v.b.sha(p),bytes=p.stat().st_size) for n,p in sorted(files.items())],dependencies=deps,test_labels_included=False,credentials_included=False,no_csv=not accepted));archive=folder/(prefix+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tf:
        for name,p in sorted(files.items()):tf.add(p,arcname=name)
        tf.add(mp,arcname=mp.relative_to(ROOT).as_posix())
    report=dict(path=archive.relative_to(ROOT).as_posix(),sha256=v.b.sha(archive),bytes=archive.stat().st_size,files=len(files),no_csv=not accepted);v.b.save(folder/(prefix+'.archive.json'),report);v.emit(stage='package',**report)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['export','verify','package']);a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');dict(export=export,verify=replay,package=package)[a.stage]()
