"""Template-preserving v94 export, independent checkpoint replay and archive."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
import argparse,csv,io,json,sys,tarfile
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v94_masked_assay as v
from verify_submission import verify
from repack_upload_template import field_spans

def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def selected(require=True):
    f=json.loads((v.OUT/'frozen_selection.json').read_text());c=json.loads((v.OUT/'confirmation.json').read_text());assert f['signature']==c['signature']==v.sig() and c['status']=='COMPLETE'
    assert c['frozen_sha256']==v.b.sha(v.OUT/'frozen_selection.json');rs=[r for r in c['records'] if r['accepted']];assert [r['task'] for r in rs]==c['accepted']
    if require:assert rs,'No eligible model, no CSV.'
    for r in rs:
        assert all(x['gain']>0 for x in r['confirmation']) and np.mean([x['gain'] for x in r['confirmation']])>v.threshold(r['metric'])
        assert any(all(r[k]==ch[k] for k in ('task','metric','config','weight')) for ch in f['selected'])
    return rs

def export():
    rs=selected();d=v.load();old=read(v.ANCHOR);rows=[r.copy() for r in old];changed={}
    for r in rs:
        k=d['tasks'].index(r['task']);p=v.mix(r['metric'],v.base(d,k,'final'),v.expert(d,k,'final',r['config']),r['weight'])
        pos=[i for i,row in enumerate(old) if i and row[0]==r['task']];assert len(p)==len(pos) and np.isfinite(p).all()
        for i,x in zip(pos,p):rows[i][4]=repr(float(x))
        changed[r['task']]=sum(rows[i][4]!=old[i][4] for i in pos)
    raw=(ROOT/'admet_official/submission_template.csv').read_bytes();ss=field_spans(raw);chunks=[];cursor=0
    for i,row in enumerate(rows[1:],1):
        lo,hi=ss[i][4];chunks.extend((raw[cursor:lo],row[4].encode('ascii')));cursor=hi
    chunks.append(raw[cursor:]);output=b''.join(chunks);file=ROOT/'admet_submissions/v94_masked_assay/我从山北来带着兰花草-小分子药物预测技术创新赛-20261025.csv';file.parent.mkdir(parents=True,exist_ok=True)
    if file.exists():assert file.read_bytes()==output,'Refuse overwriting different release.'
    else:file.write_bytes(output)
    fmt=verify(file,ROOT/'admet_official/submission_template.csv')
    v.b.save(v.OUT/'release.json',dict(status='PENDING_INDEPENDENT_REPLAY',version='v94',file=file.relative_to(ROOT).as_posix(),sha256=v.b.sha(file),anchor='v92',anchor_sha256=v.ANCHOR_SHA,anchor_official_score_user_reported=50.915,selected=rs,changed_tasks=changed,unchanged_tasks=22-len(changed),verification=fmt,official_score=None,test_labels_used=False,external_data_new_branch=False,new_pretrained_weights=False,limitations=['Historical proxies not exact v92 OOF; labels reused historically.', 'Other official assays are conditional inputs; target values and masks removed.','Baseline includes historical pretrained sources.','No platform points estimate or guaranteed gain.']))
    v.emit(stage='export',changed=changed,sha256=v.b.sha(file))

def replay():
    d=v.load();rs=selected(False);used={}
    if rs:
        for part in ('fold0','fold1','fold2','valid','final'):used[part]=sorted({r['config']['step'] for r in rs})
    else:
        for path in (v.OUT/'models').glob('*/*/metadata.json'):used[path.parent.parent.name]=list(v.SNAPSHOTS)
    records=[]
    for part,steps in used.items():
        a=v.prepare(d,part);q=a['query'];fit=a['fit'];np.testing.assert_array_equal(fit,v.split(d,part)[0])
        assert not set(fit)&set(q) and np.all(d['split'][fit]<2)
        if part.startswith('fold'):assert not set(d['groups'][fit])&set(d['groups'][q]) and not set(d['canonical'][fit])&set(d['canonical'][q])
        elif part=='valid':np.testing.assert_array_equal(fit,np.flatnonzero(d['split']==0))
        else:np.testing.assert_array_equal(fit,np.flatnonzero(d['split']<2))
        # Fit-only normalization is independently reconstructed from raw descriptors.
        raw=d['views']['physchem'].astype(float);mm=np.unique(d['mi'][fit]);median=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in raw[mm].T]);full=np.where(np.isfinite(raw),raw,median);mean=full[mm].mean(0);std=full[mm].std(0).clip(.05)
        np.testing.assert_array_equal(a['median'],median);np.testing.assert_array_equal(a['mean'],mean);np.testing.assert_array_equal(a['std'],std)
        fp=torch.from_numpy(np.sqrt(d['views']['multiview'][:,:12288]).astype(np.float32));desc=torch.from_numpy(((full-mean)/std).clip(-5,5).astype(np.float32))
        # Reconstruct the conditional lookup independently from allowed official rows.
        source=np.flatnonzero(d['split']<(2 if part=='final' else 1));np.testing.assert_array_equal(a['source'],source)
        cans,inv=np.unique(d['canonical'],return_inverse=True);values=np.zeros((len(cans),len(d['tasks'])),np.float32);masks=np.zeros_like(values);lookup={}
        for i in source:lookup.setdefault((int(inv[i]),int(d['tid'][i])),[]).append(float(d['y'][i]))
        for (i,k),ys in lookup.items():values[i,k]=np.clip((np.median(ys)-a['assay_center'][k])/a['assay_scale'][k],-5,5);masks[i,k]=1.
        np.testing.assert_array_equal(a['assay'],values);np.testing.assert_array_equal(a['mask'],masks);np.testing.assert_array_equal(a['assay_row'],inv)
        assay=torch.from_numpy(values);mask=torch.from_numpy(masks)
        for seed in v.SEEDS:
            folder=v.OUT/'models'/part/str(seed);meta=json.loads((folder/'metadata.json').read_text());cache=np.load(folder/'predictions.npz');assert str(cache['signature'])==meta['signature']==v.sig() and v.b.sha(folder/'predictions.npz')==meta['predictions_sha256']
            np.testing.assert_array_equal(cache['query'],q);np.testing.assert_array_equal(cache['fit'],fit);np.testing.assert_array_equal(cache['qoffset'],a['qoffset']);np.testing.assert_array_equal(cache['snapshots'],v.SNAPSHOTS)
            for step in steps:
                path=folder/f'checkpoint_{step}.pt';assert v.b.sha(path)==meta['checkpoint_hashes'][path.name];ck=torch.load(path,map_location='cpu',weights_only=False)
                assert ck['signature']==v.sig() and ck['seed']==seed and ck['step']==step
                for key in ('fit','query','median','mean','std','qoffset','source','assay_center','assay_scale'):np.testing.assert_array_equal(ck[key],a[key])
                assert ck['states']==a['states'];net=v.Net(ck['desc_dim'],ck['tasks']).eval();net.load_state_dict(ck['state']);out=[]
                with torch.inference_mode():
                    for start in range(0,len(q),347):
                        ix=q[start:start+347];tid=torch.from_numpy(d['tid'][ix]);out.append(net(fp[d['mi'][ix]],desc[d['mi'][ix]],assay[inv[ix]],mask[inv[ix]],tid).numpy())
                pred=np.concatenate(out).astype(float);expected=cache['values'][v.SNAPSHOTS.index(step)];np.testing.assert_allclose(pred,expected,atol=5e-5,rtol=3e-5)
                ids=np.unique([0,len(q)//2,len(q)-1]);ix=q[ids]
                with torch.inference_mode():
                    tid=torch.from_numpy(d['tid'][ix]);vv=assay[inv[ix]].clone();ms=mask[inv[ix]].clone()
                    sub=net(fp[d['mi'][ix]],desc[d['mi'][ix]],vv,ms,tid).numpy()
                    vv[torch.arange(len(ix)),tid]=1e6;ms[torch.arange(len(ix)),tid]=1.
                    changed_target=net(fp[d['mi'][ix]],desc[d['mi'][ix]],vv,ms,tid).numpy()
                    np.testing.assert_array_equal(sub,changed_target)
                np.testing.assert_allclose(sub,pred[ids],atol=2e-5,rtol=3e-5)
                record=dict(part=part,seed=seed,step=step,status='PASS',sha256=v.b.sha(path),cpu_max_abs_difference=float(np.max(np.abs(pred-expected))),subset_prediction=True);records.append(record);v.emit(stage='checkpoint_replay',**record)
                del net
    report=dict(status='PASS',records=records,checkpoints=len(records),source_rows_and_normalization=True,test_labels_used=False,no_csv=not bool(rs))
    if rs:
        rel=json.loads((v.OUT/'release.json').read_text());file=ROOT/rel['file'];assert v.b.sha(file)==rel['sha256'];old,actual=read(v.ANCHOR),read(file);changed=Counter()
        for r in rs:
            k=d['tasks'].index(r['task'])
            for part in ('fold0','fold1','fold2','valid','final'):
                p=v.mix(r['metric'],v.base(d,k,part),v.expert(d,k,part,r['config']),r['weight'])
                if part=='final':np.testing.assert_allclose(p,[float(z[4]) for z in actual[1:] if z[0]==r['task']],atol=1e-12,rtol=1e-12)
                else:
                    ix=v.r83.split(d,k,part)[1];rec=next(x for x in r['confirmation'] if x['part']==part);np.testing.assert_allclose(v.r83.score(r['metric'],d['y'][ix],p),rec['candidate'],atol=1e-12,rtol=1e-12)
        for a,z in zip(old[1:],actual[1:]):
            assert a[:4]+a[5:]==z[:4]+z[5:]
            if a[4]!=z[4]:changed[z[0]]+=1
            if z[0] not in rel['changed_tasks']:assert a==z
        assert dict(changed)==rel['changed_tasks'];template=ROOT/'admet_official/submission_template.csv';fmt=verify(file,template);raw,ref=file.read_bytes(),template.read_bytes();ss,tt=field_spans(raw),field_spans(ref);chunks=[];cursor=0
        for i in range(1,len(actual)):
            lo,hi=ss[i][4];a,z=tt[i][4];chunks.extend((raw[cursor:lo],ref[a:z]));cursor=hi
        chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
        fmt.update(nonprediction_bytes_exact=True,changed_tasks=dict(changed),checkpoints=len(records),cpu_model_reload='PASS',subset_prediction='PASS');v.b.save(file.with_suffix('.verification.json'),fmt);report['submission']=fmt
        rel.update(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',verification=fmt)
    v.b.save(v.OUT/'model_verification.json',report)
    if rs:rel['verification_sha256']=v.b.sha(v.OUT/'model_verification.json');v.b.save(v.OUT/'release.json',rel)
    v.emit(stage='verification',checkpoints=len(records),no_csv=not bool(rs),status='PASS')

def package():
    audit=json.loads((v.OUT/'model_verification.json').read_text());assert audit['status']=='PASS';accepted=bool(selected(False));files={}
    paths=[v.OUT,Path(__file__),ROOT/'tools/v94_masked_assay.py',ROOT/'docs/ITERATION_V94_20261025.md',ROOT/'docs/platform_feedback_v92_20261025.json',ROOT/'tools/v93_balanced_multitask.py']
    if accepted:paths.append(ROOT/'admet_submissions/v94_masked_assay')
    for path in paths:
        for p in ([path] if path.is_file() else path.rglob('*')):
            if p.is_file() and p.suffix!='.pyc':files[p.relative_to(ROOT).as_posix()]=p
    prefix='v94_masked_assay_20261025' if accepted else 'v94_rejected_masked_assay_20261025';folder=ROOT/'releases';mp=folder/(prefix+'.manifest.json')
    deps=[dict(path='releases/'+n,sha256=v.b.sha(folder/n)) for n in ('v86_metric_ranking_20261025.tar.gz','v88_context_kernel_20261025.tar.gz','v92_full_kernel_20261025.tar.gz')]
    v.b.save(mp,dict(files=[dict(path=n,sha256=v.b.sha(p),bytes=p.stat().st_size) for n,p in sorted(files.items())],dependencies=deps,test_labels_included=False,credentials_included=False,no_csv=not accepted))
    archive=folder/(prefix+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tf:
        for name,p in sorted(files.items()):tf.add(p,arcname=name)
        tf.add(mp,arcname=mp.relative_to(ROOT).as_posix())
    report=dict(path=archive.relative_to(ROOT).as_posix(),sha256=v.b.sha(archive),bytes=archive.stat().st_size,files=len(files),no_csv=not accepted);v.b.save(folder/(prefix+'.archive.json'),report);v.emit(stage='package',**report)

if __name__=='__main__':
    raise SystemExit('Attempt1 invalid. No export permitted. Use v94_isolated_delivery.py.')
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['export','verify','package']);args=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');dict(export=export,verify=replay,package=package)[args.stage]()
