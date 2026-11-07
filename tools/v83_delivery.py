"""Export, independently replay and package an eligible frozen v83 candidate."""
from __future__ import annotations
import argparse, csv, io, json, sys, tarfile
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
from catboost import CatBoostClassifier,CatBoostRegressor

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import admet_v77_ranking as b
import v83_residual as v
from verify_submission import verify
from repack_upload_template import field_spans

def read(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def data():
    d=b.load_data();d['legacy_folds']=np.load(ROOT/'recovered_history/legacy_v24_folds.npy');v.TASK_NAMES=d['tasks'];return d
def selected():
    conf=json.loads((v.OUT/'confirmation.json').read_text());assert conf['status']=='COMPLETE'
    assert b.sha(v.OUT/'frozen_selection.json')==conf['frozen_sha256']
    frozen=json.loads((v.OUT/'frozen_selection.json').read_text())
    assert frozen['signature']==b.sha(ROOT/'tools/v83_residual.py')
    choices=[r for r in conf['records'] if r['accepted']]
    assert choices,'No eligible candidate; no CSV export allowed.'
    assert [r['task'] for r in choices]==conf['accepted']
    for r in choices:
        assert all(p['gain']>0 for p in r['confirmation'])
        assert any(all(r[key]==f[key] for key in ('task','metric','config','weight')) for f in frozen['selected'])
    return choices
def mix(metric,old,new,w):
    if metric=='MAE':return (1-w)*old+w*new
    return (1-w)*rankdata(old)/len(old)+w*rankdata(new)/len(new)
def same(a,b,tol=1e-11):np.testing.assert_allclose(a,b,atol=tol,rtol=tol,equal_nan=True)

def export():
    d=data();choices=selected();assert b.sha(v.ANCHOR)==v.ANCHOR_SHA
    original=read(v.ANCHOR);rows=[r.copy() for r in original];changed={}
    for r in choices:
        task=r['task'];k=d['tasks'].index(task);q=v.split(d,k,'final')[1]
        pos=[i for i,row in enumerate(original) if i and row[0]==task]
        assert len(pos)==len(q)
        old=np.array([float(original[i][4]) for i in pos])
        # Accepted v83 PPBR uses the unchanged pre-residual endpoint. Refuse
        # exporting a modified endpoint without a matching final-base audit.
        assert task not in v.BASE_SPECS,'Final base for an already modified endpoint needs a separate audit.'
        historical=read(ROOT/'admet_submissions/v73_v72/我从山北来带着兰花草-小分子药物预测技术创新赛-20261021.csv')
        assert [row for row in original[1:] if row[0]==task]==[row for row in historical[1:] if row[0]==task]
        new=v.fit(d,k,'final',r['config'],v.SEEDS,'final')
        pred=mix(r['metric'],old,new,r['weight'])
        for i,p in zip(pos,pred):rows[i][4]=repr(float(p))
        changed[task]=sum(rows[i][4]!=original[i][4] for i in pos)
    template=ROOT/'admet_official/submission_template.csv';raw=template.read_bytes();spans=field_spans(raw);chunks=[];cursor=0
    for i,row in enumerate(rows[1:],1):
        lo,hi=spans[i][4];chunks.extend((raw[cursor:lo],row[4].encode('ascii')));cursor=hi
    chunks.append(raw[cursor:]);dest=ROOT/'admet_submissions/v83_matched_residual';dest.mkdir(parents=True,exist_ok=True)
    file=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261025.csv';file.write_bytes(b''.join(chunks))
    audit=verify(file,template)
    b.save(v.OUT/'release.json',dict(status='PENDING_INDEPENDENT_REPLAY',version='v83',file=file.relative_to(ROOT).as_posix(),
        sha256=b.sha(file),anchor='v82',anchor_sha256=v.ANCHOR_SHA,anchor_official_score_user_reported=50.850,
        changed_tasks=changed,unchanged_tasks=22-len(changed),selected=choices,verification=audit,
        official_score=None,test_labels_used=False,external_data_new_branch=False,pretrained_weights_new_branch=False,
        limitations=['All development labels reused historically; confirmation is not new independent test data.',
          'Historical full ensemble OOF unavailable; residual baselines replay only the corresponding later residual layer over historical proxies.',
          'Anchor includes earlier pretrained sources; no new external data or weights.',
          'Other official assay values may be available on the same molecule; evaluation is conditional.',
          'Ranks use the fixed full fit and query cohorts.']))
    print(json.dumps(dict(stage='export',changed_tasks=changed,sha256=b.sha(file)),ensure_ascii=False),flush=True)

def verify_models():
    d=data();choices=selected();release=json.loads((v.OUT/'release.json').read_text());file=ROOT/release['file']
    assert b.sha(v.ANCHOR)==release['anchor_sha256'] and b.sha(file)==release['sha256']
    original,actual=read(v.ANCHOR),read(file);records=[];changed=Counter()
    for r in choices:
        task=r['task'];k=d['tasks'].index(task);metric=r['metric'];cfg=r['config'];iscls=metric in ('AUROC','AUPRC')
        fit,q=v.split(d,k,'final');folder=v.OUT/'final'/task/'final'/cfg['name'];zcache=np.load(folder/'predictions.npz');meta=json.loads((folder/'metadata.json').read_text())
        assert meta['signature']==b.sha(ROOT/'tools/v83_residual.py')
        np.testing.assert_array_equal(zcache['fit'],fit);np.testing.assert_array_equal(zcache['query'],q)
        aux,cols,source,counts=v.aux(d,k,True);assert k not in cols and not np.any(d['tid'][source]==k)
        np.testing.assert_array_equal(source,zcache['source']);np.testing.assert_array_equal(counts[q],zcache['known_counts'])
        base=v.raw_proxy(d,fit);hist=ROOT/'admet_submissions/v73_v72/我从山北来带着兰花草-小分子药物预测技术创新赛-20261021.csv'
        qp=np.array([float(row['y_pred']) for row in b.table(hist) if row['task']==task]);y=d['y'][fit]
        if iscls:
            z,qz=v.ranklog(base),v.ranklog(qp);beta=v.calibrate(y,z);off=beta[0]+beta[1]*z;qoff=beta[0]+beta[1]*qz;scale=1.;same(meta['state']['beta'],beta)
        elif metric=='MAE':
            scale=max(float(np.quantile(y,.75)-np.quantile(y,.25)),1e-3);center=float(np.median(y));z=(base-center)/scale;qz=(qp-center)/scale;off=np.zeros(len(fit));qoff=qp
            same([meta['state']['center'],meta['state']['scale']],[center,scale])
        else:
            z,qz=v.ranknorm(base),v.ranknorm(qp);target=v.ranknorm(y);xx=np.c_[np.ones(len(z)),z]
            beta=np.linalg.solve(xx.T@xx+np.diag([1e-8,1.]),xx.T@target);beta[1]=max(beta[1],.01);off=np.zeros(len(fit));qoff=beta[0]+beta[1]*qz;scale=1.;same(meta['state']['beta'],beta)
        x=v.features(aux[fit],cols,task,cfg['view'],z);qx=v.features(aux[q],cols,task,cfg['view'],qz)
        same(zcache['x'],x);same(zcache['qx'],qx);same(zcache['off'],off);same(zcache['qoff'],qoff);same(zcache['base_fit'],base);same(zcache['base_query'],qp)
        members=[]
        for seed in v.SEEDS:
            mp=folder/f'model_{seed}.cbm';assert b.sha(mp)==meta['model_hashes'][mp.name]
            m=(CatBoostClassifier if iscls else CatBoostRegressor)();m.load_model(str(mp));params=m.get_all_params();assert m.tree_count_==cfg['trees'] and params['depth']==cfg['depth'] and params['random_seed']==seed
            raw=m.predict(qx,prediction_type='RawFormulaVal');ids=np.unique([0,len(qx)//2,len(qx)-1]);same(raw[ids],m.predict(qx[ids],prediction_type='RawFormulaVal'))
            members.append(qoff+raw*scale);records.append(dict(path=mp.relative_to(ROOT).as_posix(),sha256=b.sha(mp),status='PASS'))
        members=np.stack(members);pred=expit(members.mean(0)) if iscls else members.mean(0)
        if task=='ppbr_az':pred=pred.clip(0,100)
        same(members,zcache['members']);same(pred,zcache['p'])
        pos=[i for i,row in enumerate(original) if i and row[0]==task];old=np.array([float(original[i][4]) for i in pos]);pred=mix(metric,old,pred,r['weight'])
        same(pred,np.array([float(actual[i][4]) for i in pos]),1e-14)
    target_tasks={r['task'] for r in choices}
    for old,new in zip(original[1:],actual[1:]):
        assert old[:4]+old[5:]==new[:4]+new[5:]
        if old[4]!=new[4]:changed[old[0]]+=1
        if old[0] not in target_tasks:assert old==new
    assert dict(changed)==release['changed_tasks']
    template=ROOT/'admet_official/submission_template.csv';fmt=verify(file,template);raw=file.read_bytes();ref=template.read_bytes();ss,tt=field_spans(raw),field_spans(ref);cursor=0;chunks=[]
    for i in range(1,len(actual)):
        lo,hi=ss[i][4];a,b1=tt[i][4];chunks.extend((raw[cursor:lo],ref[a:b1]));cursor=hi
    chunks.append(raw[cursor:]);assert b''.join(chunks)==ref
    fmt.update(changed_tasks=dict(changed),unchanged_tasks=22-len(changed),nonprediction_bytes_exact=True,model_reload='PASS',models=len(records))
    result=dict(status='PASS',records=records,submission=fmt,test_labels_used=False,external_data_new_branch=False)
    b.save(v.OUT/'model_verification.json',result);b.save(file.with_suffix('.verification.json'),fmt)
    release.update(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',verification=fmt,verification_sha256=b.sha(v.OUT/'model_verification.json'));b.save(v.OUT/'release.json',release)
    print(json.dumps(dict(stage='verify',models=len(records),changed_tasks=dict(changed),sha256=b.sha(file)),ensure_ascii=False),flush=True)

def package():
    release=json.loads((v.OUT/'release.json').read_text());assert release['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM'
    paths=[ROOT/'tools/v83_residual.py',Path(__file__),ROOT/'admet_v77_ranking.py',ROOT/'admet_v67_protocol.json',
      ROOT/'tools/verify_submission.py',ROOT/'tools/repack_upload_template.py',
      ROOT/'docs/artifact-manifests/runtime-inputs.manifest.json',ROOT/'docs/ITERATION_V83_20261025.md',
      ROOT/'admet_official/submission_template.csv',v.ANCHOR,ROOT/'admet_submissions/v73_v72',
      ROOT/'recovered_history',ROOT/'admet_competition_data/data_benchmark',v.OUT,ROOT/'admet_submissions/v83_matched_residual',
      ROOT/'admet_model_outputs_v67/features',ROOT/'admet_model_outputs_v73/features',ROOT/'admet_model_outputs_v67/folds.npy']
    files={}
    for p in paths:
        for q in ([p] if p.is_file() else p.rglob('*')):
            if q.is_file() and q.suffix not in ('.pyc',) and q.name!='test.csv':files[q.relative_to(ROOT).as_posix()]=q
    manifest=dict(files=[dict(path=name,sha256=b.sha(p),bytes=p.stat().st_size) for name,p in sorted(files.items())],credentials_included=False,test_labels_included=False)
    dest=ROOT/'releases';dest.mkdir(exist_ok=True);prefix='v83_matched_residual_20261025';mp=dest/(prefix+'.manifest.json');b.save(mp,manifest)
    archive=dest/(prefix+'.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for name,p in sorted(files.items()):tar.add(p,arcname=name)
        tar.add(mp,arcname=mp.relative_to(ROOT).as_posix())
    b.save(dest/(prefix+'.archive.json'),dict(path=archive.relative_to(ROOT).as_posix(),sha256=b.sha(archive),bytes=archive.stat().st_size,files=len(files)))
    print(json.dumps(dict(stage='package',sha256=b.sha(archive),bytes=archive.stat().st_size,files=len(files))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['export','verify','package']);a=parser.parse_args()
    {'export':export,'verify':verify_models,'package':package}[a.stage]()
