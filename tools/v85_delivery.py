"""Byte-preserving release and independent source/design/model replay for v85."""
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
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import v85_smooth_residual as v
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
        assert any(all(f[key]==r[key] for key in ('task','metric','config','strength')) for f in frozen['selected'])
    return choices


def export():
    d=v.load();choices=selected();original=read(v.ANCHOR);rows=[r.copy() for r in original];changed={}
    for r in choices:
        k=d['tasks'].index(r['task']);pred=v.predict(d,k,'final',r['config'],r['strength'])
        pos=[i for i,row in enumerate(original) if i and row[0]==r['task']]
        assert len(pred)==len(pos)
        for i,p in zip(pos,pred):rows[i][4]=repr(float(p))
        changed[r['task']]=sum(rows[i][4]!=original[i][4] for i in pos)
    template=ROOT/'admet_official/submission_template.csv';raw=template.read_bytes();ss=field_spans(raw);cursor=0;chunks=[]
    for i,row in enumerate(rows[1:],1):
        lo,hi=ss[i][4];chunks.extend((raw[cursor:lo],row[4].encode('ascii')));cursor=hi
    chunks.append(raw[cursor:]);output=b''.join(chunks)
    folder=ROOT/'admet_submissions/v85_smooth_residual';folder.mkdir(parents=True,exist_ok=True)
    file=folder/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261025.csv'
    if file.exists():assert file.read_bytes()==output,'Refuse overwriting a different release.'
    else:file.write_bytes(output)
    fmt=verify(file,template)
    v.b.save(v.OUT/'release.json',dict(status='PENDING_INDEPENDENT_REPLAY',version='v85',
        file=file.relative_to(ROOT).as_posix(),sha256=v.b.sha(file),anchor='v84',anchor_sha256=v.ANCHOR_SHA,
        anchor_official_score_user_reported=50.880,changed_tasks=changed,unchanged_tasks=22-len(changed),
        selected=choices,verification=fmt,official_score=None,test_labels_used=False,external_data_new_branch=False,
        limitations=['Historical proxies are not full exact v84 OOF; labels were reused in historical iterations.',
            'Candidate starts from original v73 endpoint, compared with replayed v84 endpoint in development.',
            'Other official assay measurements are conditional deployment covariates.',
            'Local improvements do not establish official score improvement or attainment of 52.']))
    v.emit(stage='export',changed_tasks=changed,sha256=v.b.sha(file))


def replay_design(d,k,part,cfg):
    folder=v.OUT/'design'/d['tasks'][k]/part/cfg['view'];cache=np.load(folder/'design.npz');state=json.loads((folder/'state.json').read_text())
    assert str(cache['signature'])==v.signature()
    fit,q,base,qp,z,qz,off,qoff,target,expected=v.base_state(d,k,part)
    np.testing.assert_array_equal(cache['fit'],fit);np.testing.assert_array_equal(cache['query'],q)
    assert not set(fit)&set(q) and np.all(d['split'][fit]<2)
    if part!='final':
        assert not set(d['canonical'][fit])&set(d['canonical'][q])
        assert not set(d['groups'][fit])&set(d['groups'][q])
    for key,val in [('base_fit',base),('base_query',qp),('off',off),('qoff',qoff),('target',target)]:close(cache[key],val)
    for key,val in expected.items():close(state[key],val)
    if cfg['view']=='assay':
        aux,cols,source,counts=v.r83.aux(d,k,part=='final');assert k not in cols
        np.testing.assert_array_equal(cache['aux_source'],source);np.testing.assert_array_equal(cache['aux_cols'],cols)
        assert not np.any(d['tid'][source]==k) and not np.any(d['split'][source]==2)
        vals=aux[fit,:len(cols)]
        center=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in vals.T])
        std=np.array([np.std(c[np.isfinite(c)]) if np.isfinite(c).any() else 1. for c in vals.T]).clip(.05)
        close(center,cache['aux_center']);close(std,cache['aux_std'])
        def trans(rows,score):
            a=aux[rows];mask=a[:,len(cols):];val=np.nan_to_num((a[:,:len(cols)]-center)/std).clip(-5,5)*mask
            return np.c_[np.ones(len(rows)),val,mask,score]
        x,qx=trans(fit,z),trans(q,qz)
        close(cache['x'],x);close(cache['qx'],qx)
    else:
        # CPU float64 reconstruction of the GPU-produced kernels and the
        # saved fit-only whitening transform. No target labels enter kernels.
        landmarks=cache['landmarks'];assert set(landmarks).issubset(set(fit))
        unique,first=np.unique(d['canonical'][fit],return_index=True)
        chosen=np.random.default_rng(v.SEED+k).permutation(len(unique))[:min(v.LANDMARKS,len(unique))]
        np.testing.assert_array_equal(landmarks,fit[first[chosen]])
        rows=np.r_[fit,q];mi=d['mi'][rows];lm=d['mi'][landmarks]
        if cfg['view']=='chem':
            f=d['fingerprint'].astype(float);a,b=f[mi],f[lm]
            dot=a@b.T;sim=np.clip(dot/np.maximum(np.square(a).sum(1)[:,None]+np.square(b).sum(1)[None]-dot,1e-8),0,1)
            kernel=.6*sim+.4*sim**2
        else:
            full=d['views']['physchem'];a=full[d['mi'][fit]].astype(float)
            med=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in a.T])
            a=np.where(np.isfinite(a),a,med);mean=a.mean(0);std=a.std(0).clip(.05)
            close(cache['physical_median'],med);close(cache['physical_mean'],mean);close(cache['physical_std'],std)
            f=((np.where(np.isfinite(full),full,med)-mean)/std).clip(-5,5).astype(np.float32).astype(float)
            a,b=f[mi],f[lm];dist=np.maximum(0,np.square(a).sum(1)[:,None]+np.square(b).sum(1)[None]-2*a@b.T)/f.shape[1]
            kernel=.5*np.exp(-dist/.5)+.5*np.exp(-dist/2.)
        transform=cache['whitener'];features=kernel@transform
        rebuilt=np.c_[np.ones(len(rows)),features,np.r_[z,qz]]
        # Kernel matrix multiply is float32 on GPU, spectral reconstruction
        # can amplify rounding in small eigenvalues; retain a strict bound.
        close(np.r_[cache['x'],cache['qx']],rebuilt,2e-3)
    return cache,state


def replay():
    choices=selected();d=v.load();release=json.loads((v.OUT/'release.json').read_text());file=ROOT/release['file']
    assert v.b.sha(file)==release['sha256'];actual=read(file);original=read(v.ANCHOR);records=[]
    for r in choices:
        k=d['tasks'].index(r['task']);metric=r['metric'];cfg=r['config']
        for part in ('fold0','fold1','fold2','valid','final'):
            design,state=replay_design(d,k,part,cfg)
            folder=v.OUT/'models'/r['task']/part/cfg['name'];cache=np.load(folder/'predictions.npz');meta=json.loads((folder/'metadata.json').read_text())
            assert str(cache['signature'])==meta['signature']==v.signature()
            assert v.b.sha(folder/'predictions.npz')==meta['prediction_sha256']
            np.testing.assert_array_equal(cache['query'],design['query'])
            coef=v.solve_model(design['x'],design['target'],design['off'],metric,cfg['alpha'])
            close(cache['coef'],coef);delta=design['qx']@coef;close(cache['delta'],delta)
            ids=np.unique([0,len(delta)//2,len(delta)-1]);close(delta[ids],design['qx'][ids]@cache['coef'])
            raw=design['qoff']+r['strength']*delta*state['scale'];pred=expit(raw) if metric in ('AUROC','AUPRC') else raw
            if r['task']=='ppbr_az':pred=pred.clip(0,100)
            if part!='final':
                q=design['query'];rr=next(t for t in r['confirmation'] if t['part']==part)
                close(rr['candidate'],v.r83.score(metric,d['y'][q],pred))
                close(rr['baseline'],v.r83.score(metric,d['y'][q],v.baseline(d,k,part)))
            else:
                pos=[i for i,row in enumerate(actual) if i and row[0]==r['task']]
                close(pred,np.array([float(actual[i][4]) for i in pos]),1e-12)
            records.append(dict(task=r['task'],part=part,model=folder.relative_to(ROOT).as_posix(),
                status='PASS',model_sha256=v.b.sha(folder/'predictions.npz')))
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
        models=len(records),model_reload='PASS',coefficient_refit='PASS')
    v.b.save(v.OUT/'model_verification.json',dict(status='PASS',records=records,submission=fmt,test_labels_used=False))
    v.b.save(file.with_suffix('.verification.json'),fmt)
    release.update(status='VERIFIED_CANDIDATE_AWAITING_PLATFORM',verification=fmt,
        verification_sha256=v.b.sha(v.OUT/'model_verification.json'))
    v.b.save(v.OUT/'release.json',release);v.emit(stage='verify',models=len(records),sha256=v.b.sha(file),changed_tasks=dict(changed))


def package():
    release=json.loads((v.OUT/'release.json').read_text());assert release['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM'
    paths=[Path(__file__),ROOT/'tools/v85_smooth_residual.py',ROOT/'tools/v85_diagnostics.py',ROOT/'tools/v83_residual.py',ROOT/'tools/v84_neighbor_transfer.py',
        ROOT/'tools/verify_submission.py',ROOT/'tools/repack_upload_template.py',ROOT/'admet_v77_ranking.py',ROOT/'admet_v67_protocol.json',
        ROOT/'docs/artifact-manifests/runtime-inputs.manifest.json',ROOT/'docs/ITERATION_V85_20261025.md',
        ROOT/'admet_official/submission_template.csv',v.ANCHOR,v.r84.ORIGINAL,ROOT/'admet_submissions/v82_targeted',
        ROOT/'admet_competition_data/data_benchmark',ROOT/'recovered_history',ROOT/'admet_model_outputs_v67/features',
        ROOT/'admet_model_outputs_v67/folds.npy',ROOT/'admet_model_outputs_v73/features',v.OUT,
        ROOT/'admet_model_outputs_v83/confirmation',ROOT/'admet_model_outputs_v83/baseline',
        ROOT/'admet_model_outputs_v84/profiles',ROOT/'admet_model_outputs_v84/models/clearance_hepatocyte_az',
        ROOT/'admet_submissions/v83_matched_residual',ROOT/'admet_submissions/v85_smooth_residual']
    files={}
    for path in paths:
        assert path.exists(),path
        for p in ([path] if path.is_file() else path.rglob('*')):
            if p.is_file() and p.suffix!='.pyc' and p.name!='test.csv':files[p.relative_to(ROOT).as_posix()]=p
    prefix='v85_smooth_residual_20261025';folder=ROOT/'releases';folder.mkdir(exist_ok=True)
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
