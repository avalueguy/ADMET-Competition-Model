"""Template-exact v95 release with independent CPU checkpoint replay."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
import argparse,csv,io,json,sys,tarfile
from collections import Counter
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v95_graph_specialists as v
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
    # Raw field-span patching is required by the template's identifier/byte contract.
    template=ROOT/'admet_official/submission_template.csv';raw=template.read_bytes();ss=field_spans(raw);chunks=[];cursor=0
    for i,row in enumerate(rows[1:],1):
        lo,hi=ss[i][4];chunks.extend((raw[cursor:lo],row[4].encode('ascii')));cursor=hi
    chunks.append(raw[cursor:]);output=b''.join(chunks)
    file=ROOT/'admet_submissions/v95_graph_specialists/我从山北来带着兰花草-小分子药物预测技术创新赛-20261026.csv';file.parent.mkdir(parents=True,exist_ok=True)
    if file.exists():assert file.read_bytes()==output,'Refuse overwrite of different release.'
    else:file.write_bytes(output)
    fmt=verify(file,template)
    v.b.save(v.OUT/'release.json',dict(status='PENDING_INDEPENDENT_REPLAY',version='v95',file=file.relative_to(ROOT).as_posix(),sha256=v.b.sha(file),
        anchor='v94',anchor_sha256=v.ANCHOR_SHA,anchor_official_score_user_reported=50.916,selected=rs,changed_tasks=changed,unchanged_tasks=22-len(changed),
        verification=fmt,official_score=None,test_labels_used=False,external_data_new_branch=False,new_pretrained_weights=False,
        limitations=['Historical proxies not exact v94 OOF; labels reused historically.','Target task completely excluded from auxiliary covariates.',
                    'Historical baseline includes early pretrained sources.','No estimate or guarantee of official score gain.']))
    v.emit(stage='export',changed=changed,sha256=v.b.sha(file))


def reconstruct_assays(d,k,a,part):
    cols=[j for j in range(len(d['tasks'])) if j!=k]
    source=np.flatnonzero((d['split']<(2 if part=='final' else 1))&(d['tid']!=k))
    np.testing.assert_array_equal(a['source'],source);lookup={}
    for i in source:lookup.setdefault((str(d['canonical'][i]),int(d['tid'][i])),[]).append(float(d['y'][i]))
    vals=np.full((len(a['fit'])+len(a['query']),len(cols)),np.nan)
    for ii,row in enumerate(np.r_[a['fit'],a['query']]):
        for j,t in enumerate(cols):
            ys=lookup.get((str(d['canonical'][row]),t))
            if ys:vals[ii,j]=np.median(ys)
    vals=vals.astype(np.float32);ref=vals[:len(a['fit'])]
    center=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in ref.T])
    scale=np.array([np.std(c[np.isfinite(c)]) if np.isfinite(c).any() else 1. for c in ref.T]).clip(.05)
    mask=np.isfinite(vals).astype(np.float32);assay=np.c_[np.nan_to_num((vals-center)/scale).clip(-5,5)*mask,mask].astype(np.float32)
    np.testing.assert_array_equal(a['cols'],cols);np.testing.assert_array_equal(a['assay_center'],center)
    np.testing.assert_array_equal(a['assay_scale'],scale);np.testing.assert_array_equal(a['assay'],assay)


def audit(d,k,part,seed,steps):
    a=v.prepare(d,k,part);fit,q=a['fit'],a['query'];assert not set(fit)&set(q) and np.all(d['split'][fit]<2)
    if part!='final':assert not set(d['groups'][fit])&set(d['groups'][q]) and not set(d['canonical'][fit])&set(d['canonical'][q])
    reconstruct_assays(d,k,a,part)
    if part!='final':
        saved=d['y'][q].copy();d['y'][q]=1e6;d.pop('_v94_aux',None)
        perturbed=v.prepare(d,k,part);d['y'][q]=saved;d.pop('_v94_aux',None)
        for key in ('fp','desc','assay','target','offset','qoffset','fit','query','cols','source'):np.testing.assert_array_equal(a[key],perturbed[key])
    folder=v.OUT/'models'/d['tasks'][k]/part/str(seed)
    meta=json.loads((folder/'metadata.json').read_text());cache=np.load(folder/'predictions.npz')
    assert str(cache['signature'])==meta['signature']==v.sig() and v.b.sha(folder/'predictions.npz')==meta['predictions_sha256']
    for key in ('fit','query','qoffset'):np.testing.assert_array_equal(cache[key],a[key])
    np.testing.assert_array_equal(cache['snapshots'],v.SNAPSHOTS);ts=v.tensors(a,'cpu');records=[]
    for step in steps:
        path=folder/f'checkpoint_{step}.pt';assert v.b.sha(path)==meta['checkpoint_hashes'][path.name]
        ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['signature']==v.sig() and ck['seed']==seed and ck['step']==step and ck['state']==a['state']
        np.testing.assert_array_equal(ck['fit'],fit);np.testing.assert_array_equal(ck['query'],q)
        net=v.Net(ck['desc_dim'],ck['assay_dim']).eval();net.load_state_dict(ck['weights'])
        pred=v.infer(net,d,a,ts,113);expected=cache['values'][v.SNAPSHOTS.index(step)]
        np.testing.assert_allclose(pred,expected,atol=6e-5,rtol=4e-5)
        ids=np.unique([0,len(q)//2,len(q)-1]);ix=ids+len(fit)
        with torch.inference_mode():sub=net(v.pack(v.graphs(d),d['mi'][q[ids]],'cpu'),*(x[ix] for x in ts)).numpy()
        np.testing.assert_allclose(sub,pred[ids],atol=4e-5,rtol=4e-5)
        r=dict(task=d['tasks'][k],part=part,seed=seed,step=step,status='PASS',sha256=v.b.sha(path),
            cpu_max_abs_difference=float(np.max(np.abs(pred-expected))),target_query_label_perturbation='PASS' if part!='final' else 'NOT_APPLICABLE',assay_reconstruction=True)
        records.append(r);v.emit(stage='checkpoint_replay',**r)
    return records


def replay():
    d=v.load();rs=selected(False);used={};records=[]
    if rs:
        for r in rs:
            for part in ('fold0','fold1','fold2','valid','final'):
                for seed in v.SEEDS:used.setdefault((r['task'],part,seed),set()).add(r['config']['step'])
    else:
        # Replay every frozen candidate, or each task's final checkpoint if none was frozen.
        frozen=json.loads((v.OUT/'frozen_selection.json').read_text())['selected']
        if frozen:
            for r in frozen:
                for part in ('fold0','fold1','fold2','valid'):
                    for seed in v.SEEDS:used.setdefault((r['task'],part,seed),set()).add(r['config']['step'])
        else:
            for p in (v.OUT/'models').glob('*/*/*/metadata.json'):
                used[(p.parent.parent.parent.name,p.parent.parent.name,int(p.parent.name))]={v.SNAPSHOTS[-1]}
    for (task,part,seed),steps in used.items():records.extend(audit(d,d['tasks'].index(task),part,seed,sorted(steps)))
    protocol=json.loads((v.OUT/'protocol.json').read_text())
    data_audit=json.loads((v.OUT/'data_audit.json').read_text())
    graph_sha=v.b.sha(v.old_graph.OUT/'features/graphs.pt')
    assert data_audit['graph_cache_sha256']==graph_sha
    report=dict(status='PASS',checkpoints=len(records),records=records,no_csv=not bool(rs),test_labels_used=False,graph_cache_sha256=graph_sha)
    if rs:
        rel=json.loads((v.OUT/'release.json').read_text());file=ROOT/rel['file'];assert v.b.sha(file)==rel['sha256']
        old,actual=read(v.ANCHOR),read(file);changed=Counter()
        for r in rs:
            k=d['tasks'].index(r['task'])
            for part in ('fold0','fold1','fold2','valid','final'):
                pred=v.mix(r['metric'],v.base(d,k,part),v.expert(d,k,part,r['config']),r['weight'])
                if part=='final':np.testing.assert_allclose(pred,[float(z[4]) for z in actual[1:] if z[0]==r['task']],atol=1e-12,rtol=1e-12)
                else:
                    q=v.r83.split(d,k,part)[1];record=next(x for x in r['confirmation'] if x['part']==part)
                    np.testing.assert_allclose(v.r83.score(r['metric'],d['y'][q],pred),record['candidate'],atol=1e-12,rtol=1e-12)
        for a,z in zip(old[1:],actual[1:]):
            assert a[:4]+a[5:]==z[:4]+z[5:]
            if a[4]!=z[4]:changed[z[0]]+=1
            if z[0] not in rel['changed_tasks']:assert a==z
        assert dict(changed)==rel['changed_tasks'];template=ROOT/'admet_official/submission_template.csv';fmt=verify(file,template)
        raw,ref=file.read_bytes(),template.read_bytes();ss,tt=field_spans(raw),field_spans(ref);chunks=[];cursor=0
        for i in range(1,len(actual)):
            lo,hi=ss[i][4];a,z=tt[i][4];chunks.extend((raw[cursor:lo],ref[a:z]));cursor=hi
        chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
        fmt.update(nonprediction_bytes_exact=True,changed_tasks=dict(changed),checkpoints=len(records),cpu_model_reload='PASS',subset_prediction='PASS',target_assay_excluded='PASS')
        v.b.save(file.with_suffix('.verification.json'),fmt);report['submission']=fmt
        rel.update(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',verification=fmt)
    v.b.save(v.OUT/'model_verification.json',report)
    if rs:rel['verification_sha256']=v.b.sha(v.OUT/'model_verification.json');v.b.save(v.OUT/'release.json',rel)
    v.emit(stage='verification',status='PASS',checkpoints=len(records),no_csv=not bool(rs))


def package():
    audit=json.loads((v.OUT/'model_verification.json').read_text());assert audit['status']=='PASS'
    accepted=bool(selected(False));files={}
    paths=[v.OUT,Path(__file__),ROOT/'tools/v95_graph_specialists.py',ROOT/'tools/run_v95_pipeline.py',
        ROOT/'admet_v79_joint_graph.py',v.old_graph.OUT/'features/graphs.pt',
        ROOT/'docs/ITERATION_V95_20261026.md',ROOT/'docs/platform_feedback_v94_20261026.json']
    if accepted:paths.append(ROOT/'admet_submissions/v95_graph_specialists')
    for path in paths:
        for p in ([path] if path.is_file() else path.rglob('*')):
            if p.is_file() and p.suffix!='.pyc' and p.name!='smoke.pt':files[p.relative_to(ROOT).as_posix()]=p
    prefix='v95_graph_specialists_20261026' if accepted else 'v95_rejected_graph_specialists_20261026'
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
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['export','verify','package']);a=ap.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest');dict(export=export,verify=replay,package=package)[a.stage]()
