"""Byte-preserving v90 release with independent source and CatBoost replay."""
from __future__ import annotations
import argparse, csv, io, json, sys, tarfile
from collections import Counter
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v90_broad_residual as v
from verify_submission import verify
from repack_upload_template import field_spans

def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def close(a,b,tol=1e-10):np.testing.assert_allclose(a,b,atol=tol,rtol=tol,equal_nan=True)
def choices(require=True):
    f=json.loads((v.OUT/'frozen_selection.json').read_text()); c=json.loads((v.OUT/'confirmation.json').read_text())
    assert f['signature']==c['signature']==v.sig() and c['status']=='COMPLETE'
    assert c['frozen_sha256']==v.b.sha(v.OUT/'frozen_selection.json')
    rs=[r for r in c['records'] if r['accepted']]; assert [r['task'] for r in rs]==c['accepted']
    if require:assert rs,'No accepted model: no CSV.'
    for r in rs:
        assert all(p['gain']>0 for p in r['confirmation'])
        assert any(all(r[key]==ch[key] for key in ('task','metric','config','weight')) for ch in f['selected'])
    return rs

def export():
    rs=choices(); d=v.load(); old=read(v.ANCHOR); rows=[r.copy() for r in old]; changed={}
    for r in rs:
        k=d['tasks'].index(r['task']); p=v.mix(v.base(d,k,'final'),v.expert(d,k,'final',r['config']),r['weight'])
        pos=[i for i,row in enumerate(old) if i and row[0]==r['task']]; assert len(pos)==len(p) and np.isfinite(p).all()
        for i,x in zip(pos,p):rows[i][4]=repr(float(x))
        changed[r['task']]=sum(rows[i][4]!=old[i][4] for i in pos)
    template=ROOT/'admet_official/submission_template.csv'; raw=template.read_bytes(); ss=field_spans(raw); cursor=0; chunks=[]
    for i,row in enumerate(rows[1:],1):
        lo,hi=ss[i][4]; chunks.extend((raw[cursor:lo],row[4].encode('ascii'))); cursor=hi
    chunks.append(raw[cursor:]); output=b''.join(chunks)
    folder=ROOT/'admet_submissions/v90_broad_residual'; folder.mkdir(exist_ok=True,parents=True)
    file=folder/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261025.csv'
    if file.exists():assert file.read_bytes()==output,'Refuse overwriting different release.'
    else:file.write_bytes(output)
    fmt=verify(file,template)
    v.b.save(v.OUT/'release.json',dict(status='PENDING_INDEPENDENT_REPLAY',version='v90',file=file.relative_to(ROOT).as_posix(),
        sha256=v.b.sha(file),anchor='v88',anchor_sha256=v.ANCHOR_SHA,anchor_official_score_user_reported=50.910,
        changed_tasks=changed,unchanged_tasks=22-len(changed),selected=rs,verification=fmt,official_score=None,
        test_labels_used=False,external_data_new_branch=False,new_pretrained_weights=False,
        limitations=['Historical proxies are not full v88 OOF; labels have been reused.',
            'Original v73 endpoint trains residual; current v88 is blended once.',
            'Other official assay measurements are conditional features.',
            'Local gain is not an estimate of platform score or evidence of reaching 52.']))
    v.emit(stage='export',changed=changed,sha256=v.b.sha(file))

def audit_design(d,k,part):
    fit,q=v.r83.split(d,k,part)
    assert not set(fit)&set(q) and np.all(d['split'][fit]<2)
    if part!='final':
        assert not set(d['canonical'][fit])&set(d['canonical'][q])
        assert not set(d['groups'][fit])&set(d['groups'][q])
    cols=[j for j in range(len(d['tasks'])) if j!=k]
    source=np.flatnonzero((d['split']<(2 if part=='final' else 1))&(d['tid']!=k))
    assert np.all(d['tid'][source]!=k) and np.all(d['split'][source]<2)
    known={}
    for i in source:known.setdefault((str(d['canonical'][i]),int(d['tid'][i])),[]).append(float(d['y'][i]))
    def aux(rows):
        vals=np.full((len(rows),len(cols)),np.nan,np.float32)
        for i,row in enumerate(rows):
            for j,t in enumerate(cols):
                values=known.get((str(d['canonical'][row]),t))
                if values:vals[i,j]=np.median(values)
        return np.c_[vals,np.isfinite(vals).astype(np.float32)]
    af,aq=aux(fit),aux(q)
    base=v.r83.raw_proxy(d,fit)
    qp=(np.array([float(r['y_pred']) for r in v.b.table(v.r88.r84.ORIGINAL) if r['task']==d['tasks'][k]]) if part=='final' else v.r83.raw_proxy(d,q))
    z,qz=v.r83.ranklog(base),v.r83.ranklog(qp); beta=v.r83.calibrate(d['y'][fit],z)
    off,qoff=beta[0]+beta[1]*z,beta[0]+beta[1]*qz
    return dict(fit=fit,query=q,source=source,cols=np.array(cols),known_counts=aq[:,len(cols):].sum(1).astype(int),
        base_fit=base,base_query=qp,beta=beta,off=off,qoff=qoff,x=np.c_[af,z].astype(np.float32),qx=np.c_[aq,qz].astype(np.float32))

def replay_models(d,targets):
    records=[]; designs={}; raws={}
    for task,part,cfg,seed in targets:
        k=d['tasks'].index(task); key=(task,part)
        if key not in designs:designs[key]=audit_design(d,k,part)
        a=designs[key]; folder=v.OUT/'models'/task/part/cfg['name']/str(seed)
        meta=json.loads((folder/'meta.json').read_text()); path=folder/'prediction.npz'
        assert meta['signature']==v.sig() and meta['config']==cfg and meta['seed']==seed
        assert v.b.sha(path)==meta['prediction_sha256'] and v.b.sha(folder/'model.cbm')==meta['model_sha256']
        with np.load(path) as cache:
            assert str(cache['signature'])==v.sig()
            for name,expected in a.items():close(cache[name],expected,0)
            model=CatBoostClassifier().load_model(str(folder/'model.cbm'))
            assert model.tree_count_==cfg['trees'] and model.get_param('depth')==cfg['depth']
            raw=a['qoff']+model.predict(a['qx'],prediction_type='RawFormulaVal',thread_count=4)
            close(raw,cache['raw']); ids=np.unique([0,len(raw)//2,len(raw)-1])
            sub=a['qoff'][ids]+model.predict(a['qx'][ids],prediction_type='RawFormulaVal',thread_count=4)
            close(sub,raw[ids]); raws[(task,part,cfg['name'],seed)]=raw
        records.append(dict(task=task,part=part,config=cfg['name'],seed=seed,status='PASS',model_sha256=v.b.sha(folder/'model.cbm')))
    return records,raws

def replay():
    rs=choices(); d=v.load(); rel=json.loads((v.OUT/'release.json').read_text()); file=ROOT/rel['file']
    assert v.b.sha(file)==rel['sha256']; old,actual=read(v.ANCHOR),read(file)
    targets=[(r['task'],part,r['config'],seed) for r in rs for part in ('fold0','fold1','fold2','valid','final') for seed in v.SEEDS]
    records,raws=replay_models(d,targets)
    for r in rs:
        k=d['tasks'].index(r['task'])
        for part in ('fold0','fold1','fold2','valid','final'):
            expert=expit(np.mean([raws[(r['task'],part,r['config']['name'],seed)] for seed in v.SEEDS],axis=0))
            base=v.base(d,k,part); pred=v.mix(base,expert,r['weight']); q=v.r83.split(d,k,part)[1]
            if part=='final':close(pred,np.array([float(row[4]) for row in actual[1:] if row[0]==r['task']]),1e-12)
            else:
                rr=next(p for p in r['confirmation'] if p['part']==part)
                close(rr['candidate'],v.r83.score(r['metric'],d['y'][q],pred))
                close(rr['baseline'],v.r83.score(r['metric'],d['y'][q],base))
    accepted={r['task'] for r in rs}; changed=Counter()
    for a,z in zip(old[1:],actual[1:]):
        assert a[:4]+a[5:]==z[:4]+z[5:]
        if a[4]!=z[4]:changed[z[0]]+=1
        if z[0] not in accepted:assert a==z
    assert dict(changed)==rel['changed_tasks']
    template=ROOT/'admet_official/submission_template.csv'; fmt=verify(file,template)
    raw,ref=file.read_bytes(),template.read_bytes(); ss,tt=field_spans(raw),field_spans(ref); chunks=[]; cursor=0
    for i in range(1,len(actual)):
        lo,hi=ss[i][4]; a,z=tt[i][4]; chunks.extend((raw[cursor:lo],ref[a:z])); cursor=hi
    chunks.append(raw[cursor:]); assert b''.join(chunks)==ref
    fmt.update(nonprediction_bytes_exact=True,changed_tasks=dict(changed),unchanged_tasks=22-len(changed),models=len(records),model_reload='PASS',source_reconstruction='PASS',subset_prediction='PASS')
    v.b.save(v.OUT/'model_verification.json',dict(status='PASS',records=records,submission=fmt,test_labels_used=False))
    v.b.save(file.with_suffix('.verification.json'),fmt)
    rel.update(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',verification=fmt,verification_sha256=v.b.sha(v.OUT/'model_verification.json'))
    v.b.save(v.OUT/'release.json',rel); v.emit(stage='verify',models=len(records),changed=dict(changed),sha256=v.b.sha(file))

def rejected():
    assert not choices(False); d=v.load(); targets=[]
    for path in sorted((v.OUT/'models').rglob('meta.json')):
        meta=json.loads(path.read_text()); targets.append((meta['task'],meta['part'],meta['config'],meta['seed']))
    records,_=replay_models(d,targets)
    v.b.save(v.OUT/'model_verification.json',dict(status='PASS',models=len(records),records=records,no_csv=True))
    v.emit(stage='rejected_replay',models=len(records),status='PASS',no_csv=True)

def package():
    audit=json.loads((v.OUT/'model_verification.json').read_text()); assert audit['status']=='PASS'
    accepted=bool(choices(False)); files={}
    paths=[v.OUT,Path(__file__),ROOT/'tools/v90_broad_residual.py',ROOT/'docs/ITERATION_V90_20261025.md',ROOT/'docs/platform_feedback_v88_20261025.json']
    if accepted:paths.append(ROOT/'admet_submissions/v90_broad_residual')
    for path in paths:
        for p in ([path] if path.is_file() else path.rglob('*')):
            if p.is_file() and p.suffix!='.pyc':files[p.relative_to(ROOT).as_posix()]=p
    prefix='v90_broad_residual_20261025' if accepted else 'v90_rejected_broad_residual_20261025'
    folder=ROOT/'releases'; mp=folder/(prefix+'.manifest.json')
    deps=[dict(path='releases/'+name,sha256=v.b.sha(folder/name)) for name in ('v86_metric_ranking_20261025.tar.gz','v88_context_kernel_20261025.tar.gz')]
    v.b.save(mp,dict(files=[dict(path=n,sha256=v.b.sha(p),bytes=p.stat().st_size) for n,p in sorted(files.items())],dependencies=deps,test_labels_included=False,credentials_included=False,no_csv=not accepted))
    archive=folder/(prefix+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tf:
        for name,p in sorted(files.items()):tf.add(p,arcname=name)
        tf.add(mp,arcname=mp.relative_to(ROOT).as_posix())
    report=dict(path=archive.relative_to(ROOT).as_posix(),sha256=v.b.sha(archive),bytes=archive.stat().st_size,files=len(files),no_csv=not accepted)
    v.b.save(folder/(prefix+'.archive.json'),report); v.emit(stage='package',**report)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['export','verify','rejected','package']); args=ap.parse_args()
    dict(export=export,verify=replay,rejected=rejected,package=package)[args.stage]()
