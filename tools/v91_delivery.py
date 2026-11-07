"""v91 byte-preserving export and independent CPU nearest-neighbor replay."""
from __future__ import annotations
import argparse,csv,io,json,sys,tarfile
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v91_local_residual as v
from verify_submission import verify
from repack_upload_template import field_spans

def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def close(a,b,tol=1e-10):np.testing.assert_allclose(a,b,atol=tol,rtol=tol,equal_nan=True)
def selected(require=True):
    f=json.loads((v.OUT/'frozen_selection.json').read_text());c=json.loads((v.OUT/'confirmation.json').read_text())
    assert f['signature']==c['signature']==v.sig() and c['status']=='COMPLETE'
    assert c['frozen_sha256']==v.b.sha(v.OUT/'frozen_selection.json')
    rs=[r for r in c['records'] if r['accepted']];assert [r['task'] for r in rs]==c['accepted']
    if require:assert rs,'No eligible candidate; no CSV.'
    for r in rs:
        assert all(p['gain']>0 for p in r['confirmation'])
        assert any(all(ch[key]==r[key] for key in ('task','metric','config','weight')) for ch in f['selected'])
    return rs

def export():
    rs=selected();d=v.load();old=read(v.ANCHOR);rows=[r.copy() for r in old];changed={}
    for r in rs:
        k=d['tasks'].index(r['task']);p=v.mix(r['metric'],v.base(d,k,'final'),v.expert(d,k,'final',r['config']),r['weight'])
        pos=[i for i,row in enumerate(old) if i and row[0]==r['task']];assert len(pos)==len(p) and np.isfinite(p).all()
        for i,x in zip(pos,p):rows[i][4]=repr(float(x))
        changed[r['task']]=sum(rows[i][4]!=old[i][4] for i in pos)
    template=ROOT/'admet_official/submission_template.csv';raw=template.read_bytes();ss=field_spans(raw);cursor=0;chunks=[]
    for i,row in enumerate(rows[1:],1):
        lo,hi=ss[i][4];chunks.extend((raw[cursor:lo],row[4].encode('ascii')));cursor=hi
    chunks.append(raw[cursor:]);output=b''.join(chunks)
    folder=ROOT/'admet_submissions/v91_local_residual';folder.mkdir(parents=True,exist_ok=True)
    file=folder/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261025.csv'
    if file.exists():assert file.read_bytes()==output,'Refuse overwriting different release.'
    else:file.write_bytes(output)
    fmt=verify(file,template)
    v.b.save(v.OUT/'release.json',dict(status='PENDING_INDEPENDENT_REPLAY',version='v91',file=file.relative_to(ROOT).as_posix(),sha256=v.b.sha(file),
        anchor='v88',anchor_sha256=v.ANCHOR_SHA,anchor_official_score_user_reported=50.910,changed_tasks=changed,unchanged_tasks=22-len(changed),
        selected=rs,verification=fmt,official_score=None,test_labels_used=False,external_data_new_branch=False,new_pretrained_weights=False,
        limitations=['Historical proxy not full exact v88 OOF; labels historically reused.','New branch uses own-task fit labels only; exact molecule matches excluded.',
            'Historical anchor includes earlier pretrained sources.','Local gain does not estimate official points or reaching52.']))
    v.emit(stage='export',changed=changed,sha256=v.b.sha(file))

def reconstruct(d,k,part):
    fit,q=v.r83.split(d,k,part);base=v.r83.raw_proxy(d,fit)
    qp=(np.array([float(r['y_pred']) for r in v.b.table(v.prev.r88.r84.ORIGINAL) if r['task']==d['tasks'][k]]) if part=='final' else v.r83.raw_proxy(d,q))
    y=d['y'][fit];metric=d['metrics'][k]
    if metric in ('AUROC','AUPRC'):
        z,qz=v.r83.ranklog(base),v.r83.ranklog(qp);beta=v.r83.calibrate(y,z);pr=expit(beta[0]+beta[1]*z)
        residual=y-pr;curvature=(pr*(1-pr)).clip(.03);qoff=beta[0]+beta[1]*qz;scale=1.
    elif metric=='MAE':
        scale=max(float(np.quantile(y,.75)-np.quantile(y,.25)),1e-3);residual=(y-base)/scale;curvature=np.ones(len(fit));qoff=qp
    else:
        z,qz=v.r83.ranknorm(base),v.r83.ranknorm(qp);x=np.c_[np.ones(len(z)),z]
        beta=np.linalg.solve(x.T@x+np.diag([1e-8,1.]),x.T@v.r83.ranknorm(y));beta[1]=max(beta[1],.01)
        residual=v.r83.ranknorm(y)-x@beta;curvature=np.ones(len(fit));qoff=beta[0]+beta[1]*qz;scale=1.
    unique,inverse=np.unique(d['canonical'][fit],return_inverse=True)
    first=np.array([np.flatnonzero(inverse==j)[0] for j in range(len(unique))]);refs=fit[first]
    rr=np.array([np.median(residual[inverse==j]) for j in range(len(unique))]);hh=np.array([np.median(curvature[inverse==j]) for j in range(len(unique))])
    return dict(fit=fit,query=q,base_fit=base,base_query=qp,residual=residual,curvature=curvature,qoff=qoff,scale=scale,
        refrows=refs,refs=unique,reference_residual=rr,reference_curvature=hh)

def audit_neighbors(d,k,part,view):
    path=v.OUT/'neighbors'/d['tasks'][k]/part/(view+'.npz');a=dict(np.load(path));meta=json.loads(path.with_suffix('.json').read_text())
    assert meta['signature']==str(a['signature'])==v.sig() and v.b.sha(path)==meta['sha256']
    state=reconstruct(d,k,part)
    for key,x in state.items():
        if key=='refs':np.testing.assert_array_equal(a[key],x)
        else:close(a[key],x,0)
    fit,q=a['fit'],a['query'];assert not set(fit)&set(q)
    if part!='final':
        assert not set(d['groups'][fit])&set(d['groups'][q]) and not set(d['canonical'][fit])&set(d['canonical'][q])
    assert np.all(d['tid'][a['refrows']]==k) and np.all(d['split'][a['refrows']]<2)
    fp=d['fingerprint'] if view=='morgan' else d['views']['multiview'][:,:-435]
    # Full neighbor list arithmetic; CPU independent float32 GEMM may reorder exact ties.
    ref=fp[d['mi'][a['refrows']]].astype(float);rn=np.square(ref).sum(1)
    ids=np.unique(np.linspace(0,len(q)-1,min(17,len(q)),dtype=int));x=fp[d['mi'][q[ids]]].astype(float)
    dot=x@ref.T;sim=np.clip(dot/np.maximum(np.square(x).sum(1)[:,None]+rn[None]-dot,1e-8),0,1)
    sim[d['canonical'][q[ids],None]==a['refs'][None,:]]=0
    order=np.sort(sim,axis=1)[:,::-1][:,:a['similarity'].shape[1]]
    close(order,a['similarity'][ids],2e-6)
    actual=np.take_along_axis(sim,a['neighbor_positions'][ids],axis=1);close(actual,a['similarity'][ids],2e-6)
    assert np.all(a['neighbor_positions']>=0) and np.all(a['neighbor_positions']<len(a['refrows']))
    return a,dict(task=d['tasks'][k],part=part,view=view,status='PASS',query_rows=len(q),cpu_neighbor_samples=len(ids),sha256=v.b.sha(path))

def independent_infer(a,metric,cfg):
    take=min(cfg['neighbors'],a['similarity'].shape[1]);weights=a['similarity'][:,:take].astype(float)**cfg['power']
    ix=a['neighbor_positions'][:,:take];residual=a['reference_residual'][ix];curv=a['reference_curvature'][ix]
    if metric=='MAE':
        correction=[]
        for w,r in zip(weights,residual):
            order=np.argsort(r);total=w.sum();position=np.searchsorted(np.cumsum(w[order]),total*.5,side='left');position=min(int(position),take-1)
            correction.append(r[order[position]]*total/(total+cfg['alpha']))
        correction=np.array(correction)
    else:correction=np.sum(weights*residual,axis=1)/(np.sum(weights*curv,axis=1)+cfg['alpha'])
    raw=a['qoff']+float(a['scale'])*correction
    return expit(raw) if metric in ('AUROC','AUPRC') else raw

def replay():
    rs=selected();d=v.load();rel=json.loads((v.OUT/'release.json').read_text());file=ROOT/rel['file'];assert v.b.sha(file)==rel['sha256']
    old,actual=read(v.ANCHOR),read(file);records=[]
    for r in rs:
        k=d['tasks'].index(r['task']);metric=r['metric'];cfg=r['config']
        for part in ('fold0','fold1','fold2','valid','final'):
            a,record=audit_neighbors(d,k,part,cfg['view']);p=independent_infer(a,metric,cfg)
            if r['task']=='ppbr_az':p=p.clip(0,100)
            close(p,v.expert(d,k,part,cfg));ids=np.unique([0,len(p)//2,len(p)-1]);sub=dict(a)
            for key in ('qoff','similarity','neighbor_positions'):sub[key]=a[key][ids]
            sp=independent_infer(sub,metric,cfg)
            if r['task']=='ppbr_az':sp=sp.clip(0,100)
            close(sp,p[ids]);base=v.base(d,k,part);pred=v.mix(metric,base,p,r['weight'])
            if part=='final':close(pred,np.array([float(row[4]) for row in actual[1:] if row[0]==r['task']]),1e-12)
            else:
                rr=next(x for x in r['confirmation'] if x['part']==part);close(rr['candidate'],v.r83.score(metric,d['y'][a['query']],pred));close(rr['baseline'],v.r83.score(metric,d['y'][a['query']],base))
            records.append(record)
    accepted={r['task'] for r in rs};changed=Counter()
    for a,z in zip(old[1:],actual[1:]):
        assert a[:4]+a[5:]==z[:4]+z[5:]
        if a[4]!=z[4]:changed[z[0]]+=1
        if z[0] not in accepted:assert a==z
    assert dict(changed)==rel['changed_tasks'];template=ROOT/'admet_official/submission_template.csv';fmt=verify(file,template)
    raw,ref=file.read_bytes(),template.read_bytes();ss,tt=field_spans(raw),field_spans(ref);chunks=[];cursor=0
    for i in range(1,len(actual)):
        lo,hi=ss[i][4];a,z=tt[i][4];chunks.extend((raw[cursor:lo],ref[a:z]));cursor=hi
    chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
    fmt.update(nonprediction_bytes_exact=True,changed_tasks=dict(changed),unchanged_tasks=22-len(changed),neighbor_artifacts=len(records),model_reload='PASS',cpu_neighbor_audit='PASS',source_reconstruction='PASS',subset_prediction='PASS')
    v.b.save(v.OUT/'model_verification.json',dict(status='PASS',records=records,submission=fmt,test_labels_used=False))
    v.b.save(file.with_suffix('.verification.json'),fmt);rel.update(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',verification=fmt,verification_sha256=v.b.sha(v.OUT/'model_verification.json'));v.b.save(v.OUT/'release.json',rel)
    v.emit(stage='verify',neighbors=len(records),changed=dict(changed),sha256=v.b.sha(file))

def rejected():
    assert not selected(False);d=v.load();records=[]
    for path in sorted((v.OUT/'neighbors').rglob('*.npz')):
        view=path.stem;part=path.parent.name;task=path.parent.parent.name;k=d['tasks'].index(task)
        _,rec=audit_neighbors(d,k,part,view);records.append(rec)
    v.b.save(v.OUT/'model_verification.json',dict(status='PASS',records=records,no_csv=True));v.emit(stage='rejected_replay',artifacts=len(records),status='PASS')

def package():
    audit=json.loads((v.OUT/'model_verification.json').read_text());assert audit['status']=='PASS';accepted=bool(selected(False));files={}
    paths=[v.OUT,Path(__file__),ROOT/'tools/v91_local_residual.py',ROOT/'tools/v90_broad_residual.py',ROOT/'docs/ITERATION_V91_20261025.md',ROOT/'docs/platform_feedback_v88_20261025.json']
    if accepted:paths.append(ROOT/'admet_submissions/v91_local_residual')
    for path in paths:
        for p in ([path] if path.is_file() else path.rglob('*')):
            if p.is_file() and p.suffix!='.pyc':files[p.relative_to(ROOT).as_posix()]=p
    prefix='v91_local_residual_20261025' if accepted else 'v91_rejected_local_residual_20261025';folder=ROOT/'releases';mp=folder/(prefix+'.manifest.json')
    deps=[dict(path='releases/'+name,sha256=v.b.sha(folder/name)) for name in ('v86_metric_ranking_20261025.tar.gz','v88_context_kernel_20261025.tar.gz')]
    v.b.save(mp,dict(files=[dict(path=n,sha256=v.b.sha(p),bytes=p.stat().st_size) for n,p in sorted(files.items())],dependencies=deps,test_labels_included=False,credentials_included=False,no_csv=not accepted))
    archive=folder/(prefix+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tf:
        for name,p in sorted(files.items()):tf.add(p,arcname=name)
        tf.add(mp,arcname=mp.relative_to(ROOT).as_posix())
    report=dict(path=archive.relative_to(ROOT).as_posix(),sha256=v.b.sha(archive),bytes=archive.stat().st_size,files=len(files),no_csv=not accepted)
    v.b.save(folder/(prefix+'.archive.json'),report);v.emit(stage='package',**report)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['export','verify','rejected','package']);args=ap.parse_args();dict(export=export,verify=replay,rejected=rejected,package=package)[args.stage]()
