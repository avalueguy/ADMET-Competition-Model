"""Byte-preserving v88 export and independent CPU feature/kernel/model replay."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
import argparse
import csv
import io
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v88_context_kernel as v
from verify_submission import verify
from repack_upload_template import field_spans


def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def close(a,b,tol=1e-9):np.testing.assert_allclose(a,b,atol=tol,rtol=tol,equal_nan=True)


def selected():
    f=json.loads((v.OUT/'frozen_selection.json').read_text());c=json.loads((v.OUT/'confirmation.json').read_text())
    assert c['status']=='COMPLETE' and f['signature']==c['signature']==v.signature()
    assert c['frozen_sha256']==v.b.sha(v.OUT/'frozen_selection.json')
    choices=[r for r in c['records'] if r['accepted']];assert choices,'No accepted models; no CSV.'
    assert [r['task'] for r in choices]==c['accepted']
    for r in choices:
        assert all(p['gain']>0 for p in r['confirmation'])
        assert any(all(r[k]==rr[k] for k in ('task','metric','config','weight')) for rr in f['selected'])
    return choices


def export():
    d=v.load();choices=selected();old=read(v.ANCHOR);rows=[r.copy() for r in old];changed={}
    for r in choices:
        k=d['tasks'].index(r['task']);p=v.mix(r['metric'],v.baseline(d,k,'final'),v.expert(d,k,'final',r['config']),r['weight'])
        pos=[i for i,row in enumerate(old) if i and row[0]==r['task']];assert len(p)==len(pos)
        for i,number in zip(pos,p):rows[i][4]=repr(float(number))
        changed[r['task']]=sum(rows[i][4]!=old[i][4] for i in pos)
    t=ROOT/'admet_official/submission_template.csv';raw=t.read_bytes();ss=field_spans(raw);chunks=[];cursor=0
    for i,row in enumerate(rows[1:],1):
        lo,hi=ss[i][4];chunks.extend((raw[cursor:lo],row[4].encode('ascii')));cursor=hi
    chunks.append(raw[cursor:]);output=b''.join(chunks)
    folder=ROOT/'admet_submissions/v88_context_kernel';folder.mkdir(exist_ok=True,parents=True)
    file=folder/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261025.csv'
    if file.exists():assert file.read_bytes()==output,'Refuse overwriting different release.'
    else:file.write_bytes(output)
    fmt=verify(file,t)
    v.b.save(v.OUT/'release.json',dict(status='PENDING_INDEPENDENT_REPLAY',version='v88',file=file.relative_to(ROOT).as_posix(),
        sha256=v.b.sha(file),anchor='v86',anchor_sha256=v.ANCHOR_SHA,anchor_official_score_user_reported=50.900,
        changed_tasks=changed,unchanged_tasks=22-len(changed),selected=choices,verification=fmt,official_score=None,
        test_labels_used=False,external_data_new_branch=False,limitations=[
            'Historical proxy is not full exact v86 OOF; development and confirmation labels historically reused.',
            'Other official assay measurements and chemical-neighbor assay summaries are conditional information.',
            'Expert residualizes original v73 endpoint, then blends with current v86. No new external data or pretrained weights.',
            'Historical anchor includes early pretrained sources; local gains cannot establish official score or 52 attainment.']))
    v.emit(stage='export',changed=changed,sha256=v.b.sha(file))


def audit_design(d,k,part,cfg,cache):
    """Rebuild raw auxiliary sources and kernel geometry without v88 builders."""
    fit,q=v.r83.split(d,k,part);rows=np.r_[fit,q];n=len(fit)
    np.testing.assert_array_equal(cache['fit'],fit);np.testing.assert_array_equal(cache['query'],q)
    assert not set(fit)&set(q) and np.all(d['split'][fit]<2)
    if part!='final':
        assert not set(d['canonical'][fit])&set(d['canonical'][q]);assert not set(d['groups'][fit])&set(d['groups'][q])
    cols=[j for j in range(len(d['tasks'])) if j!=k]
    source=np.flatnonzero((d['split']<(2 if part=='final' else 1))&(d['tid']!=k))
    np.testing.assert_array_equal(cache['cols'],cols);np.testing.assert_array_equal(cache['source'],source)
    assert k not in cols and not np.any(d['split'][source]==2)
    known={}
    for i in source:known.setdefault((str(d['canonical'][i]),int(d['tid'][i])),[]).append(float(d['y'][i]))
    values=np.full((len(rows),len(cols)),np.nan,dtype=np.float32)
    for i,row in enumerate(rows):
        for j,col in enumerate(cols):
            val=known.get((str(d['canonical'][row]),col))
            if val:values[i,j]=np.median(val)
    mask=np.isfinite(values);np.testing.assert_array_equal(cache['known_counts'],mask[n:].sum(1))
    center=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in values[:n].T])
    scale=np.array([np.std(c[np.isfinite(c)]) if np.isfinite(c).any() else 1. for c in values[:n].T]).clip(.05)
    close(cache['measured_center'],center);close(cache['measured_scale'],scale)
    measured=np.c_[np.nan_to_num((values-center)/scale).clip(-5,5)*mask,mask].astype(np.float32).astype(float)
    stage='train_valid' if part=='final' else 'train';pf=v.r84.OUT/'profiles'/stage
    profile=np.load(pf/'profiles.npz');assert str(profile['signature'])==v.r84.signature()
    # Source identities and labels for every retained neighboring assay.
    for col in cols:
        saved=np.load(pf/(d['tasks'][col]+'.npz'))
        si=np.flatnonzero((d['tid']==col)&(d['split']<(2 if part=='final' else 1)))
        np.testing.assert_array_equal(saved['source'],si)
        un,inv=np.unique(d['canonical'][si],return_inverse=True)
        refs=np.array([d['mi'][si[np.flatnonzero(inv==j)[0]]] for j in range(len(un))])
        vals=np.array([np.median(d['y'][si[inv==j]]) for j in range(len(un))])
        np.testing.assert_array_equal(saved['refs'],refs);close(saved['values'],vals)
    neighbor=profile['x'][d['mi'][rows]][:,cols,:].reshape(len(rows),-1).astype(float)
    nc=np.median(neighbor[:n],axis=0);ns=neighbor[:n].std(0).clip(.05)
    close(cache['neighbor_center'],nc);close(cache['neighbor_scale'],ns)
    neighbor=((neighbor-nc)/ns).clip(-5,5).astype(np.float32).astype(float)
    physical=d['views']['physchem'][d['mi'][rows]].astype(float)
    pm=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in physical[:n].T])
    full=np.where(np.isfinite(physical),physical,pm);pc=full[:n].mean(0);ps=full[:n].std(0).clip(.05)
    for key,val in [('physical_median',pm),('physical_mean',pc),('physical_scale',ps)]:close(cache[key],val)
    physical=((full-pc)/ps).clip(-5,5).astype(np.float32).astype(float)
    un,first=np.unique(d['canonical'][fit],return_index=True)
    chosen=np.random.default_rng(v.SEED+k).permutation(len(un))[:min(v.LANDMARKS,len(un))];li=first[chosen]
    np.testing.assert_array_equal(cache['landmarks'],fit[li]);np.testing.assert_array_equal(cache['landmark_positions'],li)
    def rbf(x):
        ref=x[li];dist=np.maximum(0,(x*x).sum(1)[:,None]+(ref*ref).sum(1)[None]-2*x@ref.T)/x.shape[1]
        return .5*np.exp(-dist/.5)+.5*np.exp(-dist/2.)
    mk=rbf(measured);pheno=.5*mk+.5*rbf(neighbor)
    if cfg['view']=='measured':kernel=mk
    elif cfg['view']=='phenotype':kernel=pheno
    else:
        fp=d['fingerprint'][d['mi'][rows]].astype(float);ref=fp[li];dot=fp@ref.T
        tan=np.clip(dot/np.maximum((fp*fp).sum(1)[:,None]+(ref*ref).sum(1)[None]-dot,1e-8),0,1)
        chemistry=.8*(.6*tan+.4*tan*tan)+.2*rbf(physical)
        kernel=.25*pheno+.75*pheno*chemistry
    base=v.r83.raw_proxy(d,fit)
    qp=(np.array([float(r['y_pred']) for r in v.b.table(v.r84.ORIGINAL) if r['task']==d['tasks'][k]]) if part=='final' else v.r83.raw_proxy(d,q))
    y=d['y'][fit];metric=d['metrics'][k]
    if metric in ('AUROC','AUPRC'):
        z,qz=v.r83.ranklog(base),v.r83.ranklog(qp);beta=v.r83.calibrate(y,z)
        off,qoff=beta[0]+beta[1]*z,beta[0]+beta[1]*qz;target=y;state=dict(beta=beta.tolist(),scale=1.)
    elif metric=='MAE':
        center=float(np.median(y));scale=max(float(np.quantile(y,.75)-np.quantile(y,.25)),1e-3)
        z,qz=(base-center)/scale,(qp-center)/scale;off=np.zeros(n);qoff=qp;target=(y-base)/scale;state=dict(center=center,scale=scale)
    else:
        z,qz=v.r83.ranknorm(base),v.r83.ranknorm(qp);xx=np.c_[np.ones(n),z]
        beta=np.linalg.solve(xx.T@xx+np.diag([1e-8,1.]),xx.T@v.r83.ranknorm(y));beta[1]=max(beta[1],.01)
        off=xx@beta;qoff=beta[0]+beta[1]*qz;target=v.r83.ranknorm(y)-off;state=dict(beta=beta.tolist(),scale=1.)
    for key,val in [('base_fit',base),('base_query',qp),('off',off),('qoff',qoff),('target',target)]:close(cache[key],val)
    rebuilt=np.c_[np.ones(len(rows)),kernel@cache['whitener'],np.r_[z,qz]]
    close(np.r_[cache['x'],cache['qx']],rebuilt,2e-3)
    saved=json.loads((v.OUT/'design'/d['tasks'][k]/part/cfg['view']/'state.json').read_text())
    for key,val in state.items():close(saved[key],val)
    return state


def solve(x,target,off,metric,alpha):
    p=x.shape[1];reg=np.full(p,float(alpha));reg[0]*=.01;coef=np.zeros(p)
    for _ in range(15 if metric in ('AUROC','AUPRC') else 12 if metric=='MAE' else 1):
        if metric in ('AUROC','AUPRC'):
            probability=expit(off+x@coef);weight=(probability*(1-probability)).clip(.03);work=x@coef+(target-probability)/weight
        elif metric=='MAE':weight=(1/np.sqrt((target-x@coef)**2+.05**2)).clip(.1,20);work=target
        else:weight=np.ones(len(x));work=target
        lhs=x.T@(weight[:,None]*x);lhs.flat[::p+1]+=reg;new=np.linalg.solve(lhs,x.T@(weight*work))
        if np.max(np.abs(new-coef))<1e-7:coef=new;break
        coef=new
    return coef


def replay():
    d=v.load();choices=selected();release=json.loads((v.OUT/'release.json').read_text());file=ROOT/release['file']
    assert v.b.sha(file)==release['sha256'];old,actual=read(v.ANCHOR),read(file);records=[]
    # The reused profile artifact has an independent CPU sample audit as well.
    import v84_delivery
    pa={p:v84_delivery.replay_profiles(d,p=='final') for p in ('development','final')}
    for r in choices:
        k=d['tasks'].index(r['task']);cfg=r['config'];metric=r['metric']
        for part in ('fold0','fold1','fold2','valid','final'):
            dest=v.OUT/'models'/r['task']/part/cfg['name'];cache=np.load(dest/'predictions.npz');meta=json.loads((dest/'metadata.json').read_text())
            assert str(cache['signature'])==meta['signature']==v.signature();assert meta['config']==cfg
            assert v.b.sha(dest/'predictions.npz')==meta['predictions_sha256']
            path=v.OUT/'design'/r['task']/part/cfg['view']/'design.npz';assert v.b.sha(path)==meta['design_sha256']
            design=np.load(path);assert str(design['signature'])==v.signature();state=audit_design(d,k,part,cfg,design)
            coef=solve(design['x'],design['target'],design['off'],metric,cfg['alpha']);close(cache['coef'],coef)
            delta=design['qx']@coef;close(cache['delta'],delta);ids=np.unique([0,len(delta)//2,len(delta)-1]);close(delta[ids],design['qx'][ids]@coef)
            raw=design['qoff']+delta*state['scale'];expert=expit(raw) if metric in ('AUROC','AUPRC') else raw
            if r['task']=='ppbr_az':expert=expert.clip(0,100)
            pred=v.mix(metric,v.baseline(d,k,part),expert,r['weight']);q=design['query']
            if part!='final':
                rr=next(p for p in r['confirmation'] if p['part']==part);close(rr['candidate'],v.r83.score(metric,d['y'][q],pred))
                close(rr['baseline'],v.r83.score(metric,d['y'][q],v.baseline(d,k,part)))
            else:close(pred,np.array([float(row[4]) for row in actual[1:] if row[0]==r['task']]),1e-12)
            records.append(dict(task=r['task'],part=part,status='PASS',model_sha256=v.b.sha(dest/'predictions.npz')))
    changed=Counter();tasks={r['task'] for r in choices}
    for a,z in zip(old[1:],actual[1:]):
        assert a[:4]+a[5:]==z[:4]+z[5:]
        if a[4]!=z[4]:changed[z[0]]+=1
        if z[0] not in tasks:assert a==z
    assert dict(changed)==release['changed_tasks']
    t=ROOT/'admet_official/submission_template.csv';fmt=verify(file,t);raw,ref=file.read_bytes(),t.read_bytes();ss,tt=field_spans(raw),field_spans(ref);cursor=0;chunks=[]
    for i in range(1,len(actual)):
        lo,hi=ss[i][4];a,z=tt[i][4];chunks.extend((raw[cursor:lo],ref[a:z]));cursor=hi
    chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
    fmt.update(nonprediction_bytes_exact=True,changed_tasks=dict(changed),unchanged_tasks=22-len(changed),models=len(records),model_reload='PASS',coefficient_refit='PASS',source_kernel_reconstruction='PASS')
    v.b.save(v.OUT/'model_verification.json',dict(status='PASS',records=records,profile_source_audit=pa,submission=fmt,test_labels_used=False))
    v.b.save(file.with_suffix('.verification.json'),fmt)
    release.update(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',verification=fmt,verification_sha256=v.b.sha(v.OUT/'model_verification.json'))
    v.b.save(v.OUT/'release.json',release);v.emit(stage='verify',models=len(records),changed=dict(changed),sha256=v.b.sha(file))


def package():
    release=json.loads((v.OUT/'release.json').read_text());assert release['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM'
    # Incremental archive explicitly depends on the already verified v86 base.
    dependency=ROOT/'releases/v86_metric_ranking_20261025.tar.gz'
    assert v.b.sha(dependency)=='1e017a0924314982bdf2f4eeab28f0c4546c12ea6b28bc2db02ef7349bdfbf8b'
    paths=[v.OUT,Path(__file__),ROOT/'tools/v88_context_kernel.py',ROOT/'tools/v88_diagnostics.py',ROOT/'tools/v84_delivery.py',
        ROOT/'tools/v87_metric_ranking.py',ROOT/'docs/ITERATION_V88_20261025.md',ROOT/'docs/platform_feedback_v86_20261025.json',ROOT/'admet_submissions/v88_context_kernel']
    files={}
    for path in paths:
        assert path.exists(),path
        for p in ([path] if path.is_file() else path.rglob('*')):
            if p.is_file() and p.suffix!='.pyc' and p.name!='test.csv':files[p.relative_to(ROOT).as_posix()]=p
    folder=ROOT/'releases';prefix='v88_context_kernel_20261025';mp=folder/(prefix+'.manifest.json')
    v.b.save(mp,dict(files=[dict(path=n,sha256=v.b.sha(p),bytes=p.stat().st_size) for n,p in sorted(files.items())],
        dependency_archive=dependency.relative_to(ROOT).as_posix(),dependency_sha256=v.b.sha(dependency),test_labels_included=False,credentials_included=False))
    archive=folder/(prefix+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tf:
        for name,p in sorted(files.items()):tf.add(p,arcname=name)
        tf.add(mp,arcname=mp.relative_to(ROOT).as_posix())
    report=dict(path=archive.relative_to(ROOT).as_posix(),sha256=v.b.sha(archive),bytes=archive.stat().st_size,files=len(files),dependency_archive=dependency.name)
    v.b.save(folder/(prefix+'.archive.json'),report);v.emit(stage='package',**report)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['export','verify','package']);args=ap.parse_args()
    dict(export=export,verify=replay,package=package)[args.stage]()
