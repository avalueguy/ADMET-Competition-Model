"""v92 template-preserving delivery and independent CPU kernel/model replay."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
import argparse,csv,io,json,sys,tarfile
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v92_full_kernel as v
from verify_submission import verify
from repack_upload_template import field_spans

def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def close(a,b,tol=1e-10):np.testing.assert_allclose(a,b,atol=tol,rtol=tol,equal_nan=True)
def selected(require=True):
    f=json.loads((v.OUT/'frozen_selection.json').read_text());c=json.loads((v.OUT/'confirmation.json').read_text())
    assert f['signature']==c['signature']==v.sig() and c['status']=='COMPLETE'
    assert c['frozen_sha256']==v.b.sha(v.OUT/'frozen_selection.json')
    rs=[r for r in c['records'] if r['accepted']];assert [r['task'] for r in rs]==c['accepted']
    if require:assert rs,'No eligible model: no CSV.'
    for r in rs:
        assert all(x['gain']>0 for x in r['confirmation'])
        assert any(all(r[key]==ch[key] for key in ('task','metric','config','weight')) for ch in f['selected'])
    return rs

def export():
    rs=selected();d=v.load();old=read(v.ANCHOR);rows=[r.copy() for r in old];changed={}
    for r in rs:
        k=d['tasks'].index(r['task']);p=v.mix(r['metric'],v.base(d,k,'final'),v.expert(d,k,'final',r['config']),r['weight'])
        pos=[i for i,row in enumerate(old) if i and row[0]==r['task']];assert len(p)==len(pos) and np.isfinite(p).all()
        for i,x in zip(pos,p):rows[i][4]=repr(float(x))
        changed[r['task']]=sum(rows[i][4]!=old[i][4] for i in pos)
    template=ROOT/'admet_official/submission_template.csv';raw=template.read_bytes();ss=field_spans(raw);cursor=0;chunks=[]
    for i,row in enumerate(rows[1:],1):
        lo,hi=ss[i][4];chunks.extend((raw[cursor:lo],row[4].encode('ascii')));cursor=hi
    chunks.append(raw[cursor:]);output=b''.join(chunks)
    folder=ROOT/'admet_submissions/v92_full_kernel';folder.mkdir(exist_ok=True,parents=True)
    file=folder/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261025.csv'
    if file.exists():assert file.read_bytes()==output,'Refuse overwriting different release.'
    else:file.write_bytes(output)
    fmt=verify(file,template)
    v.b.save(v.OUT/'release.json',dict(status='PENDING_INDEPENDENT_REPLAY',version='v92',file=file.relative_to(ROOT).as_posix(),sha256=v.b.sha(file),
        anchor='v88',anchor_sha256=v.ANCHOR_SHA,anchor_official_score_user_reported=50.910,selected=rs,changed_tasks=changed,unchanged_tasks=22-len(changed),
        verification=fmt,official_score=None,test_labels_used=False,external_data_new_branch=False,new_pretrained_weights=False,
        limitations=['Historical proxy not complete v88 OOF; labels reused historically.','No auxiliary assay inputs in new branch.',
            'Historical anchor includes early pretrained sources.','Local gain cannot be translated to official score or assurance of reaching52.']))
    v.emit(stage='export',changed=changed,sha256=v.b.sha(file))

def cpu_kernel(d,rows,fit,view):
    raw=d['views']['multiview'][:,:12288]
    slices=[(0,2048),(2048,4096)] if view=='ecfp' else [(j,j+2048) for j in range(0,12288,2048)]
    output=np.zeros((len(rows),len(fit)),np.float64)
    for lo,hi in slices:
        ref=(raw[d['mi'][fit],lo:hi]>0).astype(np.float64);norm=ref.sum(1)
        for start in range(0,len(rows),256):
            x=(raw[d['mi'][rows[start:start+256]],lo:hi]>0).astype(np.float64);dot=x@ref.T
            tan=np.clip(dot/np.maximum(x.sum(1)[:,None]+norm[None]-dot,1e-8),0,1)
            output[start:start+len(x)]+=(.5*tan+.5*tan**2)/len(slices)
    return output

def state(d,k,part):
    fit,q=v.r83.split(d,k,part);base=v.r83.raw_proxy(d,fit);metric=d['metrics'][k];y=d['y'][fit]
    qp=(np.array([float(r['y_pred']) for r in v.b.table(v.prev.prev.r88.r84.ORIGINAL) if r['task']==d['tasks'][k]]) if part=='final' else v.r83.raw_proxy(d,q))
    if metric in ('AUROC','AUPRC'):
        z,qz=v.r83.ranklog(base),v.r83.ranklog(qp);beta=v.r83.calibrate(y,z);prob=expit(beta[0]+beta[1]*z)
        center=float(y.mean());scale=1.;direct=y-center;residual=y-prob;qoff=beta[0]+beta[1]*qz
    elif metric=='MAE':
        center=float(np.median(y));scale=max(float(np.quantile(y,.75)-np.quantile(y,.25)),1e-3)
        direct=(y-center)/scale;residual=(y-base)/scale;qoff=qp
    else:
        z,qz=v.r83.ranknorm(base),v.r83.ranknorm(qp);x=np.c_[np.ones(len(z)),z];beta=np.linalg.solve(x.T@x+np.diag([1e-8,1.]),x.T@v.r83.ranknorm(y));beta[1]=max(beta[1],.01)
        center=0.;scale=1.;direct=v.r83.ranknorm(y);residual=direct-x@beta;qoff=beta[0]+beta[1]*qz
    return dict(fit=fit,query=q,target=np.c_[direct,residual],center=center,scale=scale,qoff=qoff)

def infer(a,metric,cfg):
    j=list(a['alphas']).index(cfg['alpha']);col=0 if cfg['mode']=='direct' else 1;x=a['values'][j,:,col]
    if cfg['mode']=='direct':p=x*float(a['scale'])+float(a['center'])
    elif metric in ('AUROC','AUPRC'):p=expit(a['qoff'])+x
    else:p=a['qoff']+x*float(a['scale'])
    return expit(4*(p-.5)) if metric in ('AUROC','AUPRC') else p

def audit(d,k,part,view):
    path=v.OUT/'models'/d['tasks'][k]/part/(view+'.npz');a=dict(np.load(path));meta=json.loads(path.with_suffix('.json').read_text())
    assert str(a['signature'])==meta['signature']==v.sig() and v.b.sha(path)==meta['sha256']
    expected=state(d,k,part)
    for key,x in expected.items():close(a[key],x,0)
    np.testing.assert_array_equal(a['alphas'],v.ALPHAS)
    fit,q=a['fit'],a['query'];assert not set(fit)&set(q) and np.all(d['split'][fit]<2)
    if part!='final':assert not set(d['groups'][fit])&set(d['groups'][q]) and not set(d['canonical'][fit])&set(d['canonical'][q])
    qe=cpu_kernel(d,q,fit,view);values=[];errors=[]
    ids=np.unique(np.linspace(0,len(fit)-1,min(17,len(fit)),dtype=int));tr=cpu_kernel(d,fit[ids],fit,view)
    for j,alpha in enumerate(v.ALPHAS):
        coef=a['coefficients'][j].astype(float);p=qe@coef;close(a['values'][j],p,3e-4);values.append(p)
        residual=tr@coef+alpha*coef[ids]-a['target'][ids];error=float(np.max(np.abs(residual)));assert error<.004,(view,part,error);errors.append(error)
    return a,dict(task=d['tasks'][k],part=part,view=view,status='PASS',sha256=v.b.sha(path),query_rows=len(q),cpu_linear_system_rows=len(ids),
        cpu_prediction_max_difference=float(np.max(np.abs(np.stack(values)-a['values']))),cpu_linear_system_max_residual=max(errors))

def replay():
    rs=selected();d=v.load();rel=json.loads((v.OUT/'release.json').read_text());file=ROOT/rel['file'];assert v.b.sha(file)==rel['sha256']
    old,actual=read(v.ANCHOR),read(file);records=[]
    for r in rs:
        k=d['tasks'].index(r['task']);metric=r['metric'];cfg=r['config']
        for part in ('fold0','fold1','fold2','valid','final'):
            a,rec=audit(d,k,part,cfg['view']);p=infer(a,metric,cfg)
            if r['task']=='ppbr_az':p=p.clip(0,100)
            close(p,v.expert(d,k,part,cfg),0);ids=np.unique([0,len(p)//2,len(p)-1]);sub=dict(a);sub['values']=a['values'][:,ids];sub['qoff']=a['qoff'][ids]
            sp=infer(sub,metric,cfg)
            if r['task']=='ppbr_az':sp=sp.clip(0,100)
            close(sp,p[ids],0);base=v.base(d,k,part);pred=v.mix(metric,base,p,r['weight'])
            if part=='final':close(pred,np.array([float(row[4]) for row in actual[1:] if row[0]==r['task']]),1e-12)
            else:
                rr=next(x for x in r['confirmation'] if x['part']==part);close(rr['candidate'],v.r83.score(metric,d['y'][a['query']],pred));close(rr['baseline'],v.r83.score(metric,d['y'][a['query']],base))
            records.append(rec)
            v.emit(stage='cpu_replay_part',**rec)
    accepted={r['task'] for r in rs};changed=Counter()
    for a,z in zip(old[1:],actual[1:]):
        assert a[:4]+a[5:]==z[:4]+z[5:]
        if a[4]!=z[4]:changed[z[0]]+=1
        if z[0] not in accepted:assert a==z
    assert dict(changed)==rel['changed_tasks'];template=ROOT/'admet_official/submission_template.csv';fmt=verify(file,template)
    raw,ref=file.read_bytes(),template.read_bytes();ss,tt=field_spans(raw),field_spans(ref);chunks=[];cursor=0
    for i in range(1,len(actual)):
        lo,hi=ss[i][4];aa,zz=tt[i][4];chunks.extend((raw[cursor:lo],ref[aa:zz]));cursor=hi
    chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
    fmt.update(nonprediction_bytes_exact=True,changed_tasks=dict(changed),unchanged_tasks=22-len(changed),kernel_models=len(records),model_reload='PASS',cpu_prediction_replay='PASS',cpu_linear_system_sample='PASS',source_reconstruction='PASS',subset_prediction='PASS')
    v.b.save(v.OUT/'model_verification.json',dict(status='PASS',records=records,submission=fmt,test_labels_used=False))
    v.b.save(file.with_suffix('.verification.json'),fmt);rel.update(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',verification=fmt,verification_sha256=v.b.sha(v.OUT/'model_verification.json'));v.b.save(v.OUT/'release.json',rel)
    v.emit(stage='verify',models=len(records),changed=dict(changed),sha256=v.b.sha(file))

def rejected():
    assert not selected(False);d=v.load();records=[]
    for path in sorted((v.OUT/'models').rglob('*.npz')):
        view=path.stem;part=path.parent.name;task=path.parent.parent.name;k=d['tasks'].index(task);_,rec=audit(d,k,part,view);records.append(rec)
    v.b.save(v.OUT/'model_verification.json',dict(status='PASS',records=records,no_csv=True));v.emit(stage='rejected_replay',models=len(records),status='PASS')

def diagnostics():
    d=v.load();rs=selected();records=[]
    for r in rs:
        k=d['tasks'].index(r['task']);metric=r['metric']
        for part in ('fold2','valid'):
            q=v.r83.split(d,k,part)[1];old=v.base(d,k,part);p=v.mix(metric,old,v.expert(d,k,part,r['config']),r['weight']);y=d['y'][q]
            groups,inv=np.unique(d['groups'][q],return_inverse=True)
            if len(groups)<20:
                rec=dict(task=r['task'],part=part,metric=metric,rows=len(q),scaffolds=len(groups),point_gain=v.r83.gain(metric,y,old,p),
                    replicates=0,percentile_95=None,positive_fraction=None,status='NOT_ESTIMABLE_TOO_FEW_GROUPS',
                    reason='Cached exclusion grouping has fewer than20 groups; cluster bootstrap would be degenerate. Do not interpret it as evidence of certainty.')
                records.append(rec);v.emit(stage='bootstrap_report_only',**rec);continue
            blocks=[np.flatnonzero(inv==j) for j in range(len(groups))];rng=np.random.default_rng(920923+k+(100 if part=='valid' else 0));gains=[]
            for _ in range(1500):
                ix=np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))])
                if metric in ('AUROC','AUPRC') and len(np.unique(y[ix]))<2:continue
                gain=v.r83.gain(metric,y[ix],old[ix],p[ix])
                if np.isfinite(gain):gains.append(gain)
            rec=dict(task=r['task'],part=part,metric=metric,rows=len(q),scaffolds=len(groups),point_gain=v.r83.gain(metric,y,old,p),replicates=len(gains),percentile_95=np.quantile(gains,[.025,.975]).tolist(),positive_fraction=float(np.mean(np.array(gains)>0)));records.append(rec);v.emit(stage='bootstrap_report_only',**rec)
    v.b.save(v.OUT/'diagnostics.json',dict(status='COMPLETE',selection_effect='NONE_REPORT_ONLY',bootstrap=records,limitations='Historically reused labels and proxy baseline; intervals do not remove selection bias or estimate official points.'))

def package():
    audit=json.loads((v.OUT/'model_verification.json').read_text());assert audit['status']=='PASS';accepted=bool(selected(False));files={}
    paths=[v.OUT,Path(__file__),ROOT/'tools/v92_full_kernel.py',ROOT/'tools/v91_local_residual.py',ROOT/'tools/v90_broad_residual.py',ROOT/'docs/ITERATION_V92_20261025.md',ROOT/'docs/platform_feedback_v88_20261025.json']
    if accepted:paths.append(ROOT/'admet_submissions/v92_full_kernel')
    for path in paths:
        for p in ([path] if path.is_file() else path.rglob('*')):
            if p.is_file() and p.suffix!='.pyc':files[p.relative_to(ROOT).as_posix()]=p
    prefix='v92_full_kernel_20261025' if accepted else 'v92_rejected_full_kernel_20261025';folder=ROOT/'releases';mp=folder/(prefix+'.manifest.json')
    deps=[dict(path='releases/'+name,sha256=v.b.sha(folder/name)) for name in ('v86_metric_ranking_20261025.tar.gz','v88_context_kernel_20261025.tar.gz')]
    v.b.save(mp,dict(files=[dict(path=n,sha256=v.b.sha(p),bytes=p.stat().st_size) for n,p in sorted(files.items())],dependencies=deps,test_labels_included=False,credentials_included=False,no_csv=not accepted))
    archive=folder/(prefix+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tf:
        for name,p in sorted(files.items()):tf.add(p,arcname=name)
        tf.add(mp,arcname=mp.relative_to(ROOT).as_posix())
    report=dict(path=archive.relative_to(ROOT).as_posix(),sha256=v.b.sha(archive),bytes=archive.stat().st_size,files=len(files),no_csv=not accepted);v.b.save(folder/(prefix+'.archive.json'),report);v.emit(stage='package',**report)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['export','verify','rejected','diagnostics','package']);args=ap.parse_args();dict(export=export,verify=replay,rejected=rejected,diagnostics=diagnostics,package=package)[args.stage]()
