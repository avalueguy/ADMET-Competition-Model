"""Task-specific SMILES sequence kernels and censored ordinal specialists.

Official training labels only; train-only preprocessing; query labels are used
only after predictions are saved. Selection and confirmation are separate.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
import argparse, hashlib, json, sys, time
from pathlib import Path
import joblib
import numpy as np
import torch
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, SVR
from catboost import CatBoostClassifier, CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
import v102_count_margin as old
import v86_metric_ranking as herg_old
from repack_upload_template import field_spans
from verify_submission import verify
b, r83 = old.b, old.r83
OUT = ROOT/'admet_model_outputs_v115'
TASKS = ('herg', 'ames', 'bbb_martins', 'caco2_wang',
         'clearance_hepatocyte_az', 'clearance_microsome_az',
         'cyp3a4_substrate_carbonmangels', 'solubility_aqsoldb',
         'half_life_obach', 'vdss_lombardo')
WEIGHT = .4
CS = (.3, 3., 30.)
SEEDS = (1154026, 1154027)

def sig():
    return hashlib.sha256((b.sha(__file__)+old.sig()).encode()).hexdigest()

def emit(**x):
    print(json.dumps(x, ensure_ascii=False), flush=True)

def save(name, x):
    b.save(OUT/name, x)

def load():
    d = old.load()
    assert not np.isfinite(d['y'][d['split']==2]).any()
    return d

def base(d, k, part):
    # hERG uses the v85 branch carried into v114. Remaining histories do not
    # all have reconstructible OOF; their v101 proxy is explicitly approximate.
    if d['tasks'][k] == 'herg':
        return herg_old.baseline(d, k, part)
    return old.base(d, k, part)

def mix(m, a, z):
    if m == 'MAE':
        return (1-WEIGHT)*a + WEIGHT*z
    return (1-WEIGHT)*r83.rankdata(a)/len(a) + WEIGHT*r83.rankdata(z)/len(z)

def threshold(m):
    return .012 if m == 'MAE' else .008 if m == 'Spearman' else .0025

def protocol():
    doc = dict(version=115, signature=sig(), tasks=TASKS, weight=WEIGHT,
        C=CS, seeds=SEEDS, selection=['fold0','fold1'], confirmation=['fold2','valid'],
        methods=['Canonical SMILES character TF-IDF 2-5gram SVM/SVR',
                 'Sequence plus circular count Tanimoto and 2D RBF',
                 'Censored ordinal MultiLogloss and log-property regression'],
        rules='One configuration per task frozen on fold0/1. Confirm both remaining parts positive; all four mean exceeds threshold. No fallback after rejection.',
        baseline='v114 output; v101 reconstructed proxy on legacy splits except hERG v85.',
        official_baseline_score=51.091, test_labels_used=False, external_data=False,
        limitations='Historical public validation labels are repeatedly reused. Proxy controls are not complete v114 OOF; local metrics cannot predict official points.')
    OUT.mkdir(exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists():
        assert json.loads(path.read_text())==json.loads(json.dumps(doc))
    else:
        b.save(path,doc)

def matrices(d,k,part):
    fit,q=r83.split(d,k,part); fi=d['mi'][fit];qi=d['mi'][q]
    vec=TfidfVectorizer(analyzer='char',ngram_range=(2,5),lowercase=False,
                        min_df=1,sublinear_tf=True,norm='l2',dtype=np.float64)
    x=vec.fit_transform(d['canonical'][fit]);qx=vec.transform(d['canonical'][q])
    tr=(x@x.T).toarray();qe=(qx@x.T).toarray()
    tr=.5*tr+.5*tr**2;qe=.5*qe+.5*qe**2
    tc=np.zeros_like(tr);qc=np.zeros_like(qe)
    for view in range(3):
        ref=torch.as_tensor(old.count_raw(d,fi,view),device='cuda');rn=ref.square().sum(1)
        for rows,dest in ((fi,tc),(qi,qc)):
            for lo in range(0,len(rows),256):
                raw=torch.as_tensor(old.count_raw(d,rows[lo:lo+256],view),device='cuda')
                dot=raw@ref.T
                t=(dot/(raw.square().sum(1)[:,None]+rn[None]-dot).clamp_min(1e-8)).clamp(0,1)
                dest[lo:lo+len(raw)]+=(.5*t+.5*t.square()).cpu().numpy()/3
    xx,qq,center,scale,active=old.kernel.normalized(d['views']['physchem'],fi,qi)
    def rbf(qq):
        out=np.empty((len(qq),len(xx)))
        rn=(xx.astype(float)**2).sum(1)
        for lo in range(0,len(qq),256):
            z=qq[lo:lo+256].astype(float)
            dist=np.maximum((z*z).sum(1)[:,None]+rn[None]-2*z@xx.astype(float).T,0)/max(1,len(active))
            out[lo:lo+len(z)]=.5*np.exp(-.5*dist)+.5*np.exp(-2*dist)
        return out
    result={'sequence':((tr+tr.T)/2,qe),
            'joint':(.45*tr+.45*tc+.1*rbf(xx),.45*qe+.45*qc+.1*rbf(qq))}
    torch.cuda.empty_cache()
    return result,vec

def kernel_family(d,k,part,cfgs=None):
    m=d['metrics'][k];cls=m in ('AUROC','AUPRC')
    cfgs=cfgs or [dict(kind='kernel',view=v,C=c) for v in ('sequence','joint') for c in CS]
    folder=OUT/'models'/d['tasks'][k]/part
    missing=[c for c in cfgs if not (folder/(key(c)+'.npz')).exists()]
    result={}
    if missing:
        mm,vec=matrices(d,k,part);fit,q=r83.split(d,k,part);y=d['y'][fit]
        center=float(np.median(y)) if m=='MAE' else 0.
        scale=max(float(np.subtract(*np.quantile(y,[.75,.25]))),.001) if m=='MAE' else 1.
        yt=y if cls else (y-center)/scale if m=='MAE' else r83.ranknorm(y)
        folder.mkdir(parents=True,exist_ok=True)
        joblib.dump(vec,folder/'sequence_vectorizer.joblib',compress=3)
        for c in missing:
            started=time.monotonic();tr,qe=mm[c['view']]
            net=(SVC(C=c['C'],kernel='precomputed',tol=1e-4,cache_size=1024) if cls else
                 SVR(C=c['C'],epsilon=.05,kernel='precomputed',tol=1e-4,cache_size=1024))
            net.fit(tr,yt)
            raw=net.decision_function(qe) if cls else net.predict(qe)
            p=expit(raw) if cls else raw*scale+center
            file=folder/(key(c)+'.joblib');joblib.dump(net,file,compress=3)
            reload=joblib.load(file);rp=reload.decision_function(qe) if cls else reload.predict(qe)
            np.testing.assert_allclose(raw,rp,rtol=0,atol=1e-12)
            np.savez_compressed(folder/(key(c)+'.npz'),p=p,raw=raw,fit=fit,query=q,
                signature=sig(),center=center,scale=scale)
            emit(stage='train',task=d['tasks'][k],part=part,config=c,seconds=round(time.monotonic()-started))
    for c in cfgs:
        z=np.load(folder/(key(c)+'.npz'));assert str(z['signature'])==sig()
        np.testing.assert_array_equal(z['query'],r83.split(d,k,part)[1]);result[key(c)]=z['p'].copy()
    return result

def key(c):
    return '_'.join(str(c[k]) for k in sorted(c))

def ordinal(d,k,part,c):
    folder=OUT/'models'/d['tasks'][k]/part;file=folder/(key(c)+'.npz')
    if file.exists():
        z=np.load(file);assert str(z['signature'])==sig();return z['p'].copy()
    fit,q=r83.split(d,k,part);y=d['y'][fit];assert d['metrics'][k]=='Spearman'
    raw=d['views']['physchem'] if c['view']=='physchem' else d['views']['multiview']
    x=raw[d['mi'][fit]].copy();qx=raw[d['mi'][q]].copy()
    x[~np.isfinite(x)]=np.nan;qx[~np.isfinite(qx)]=np.nan
    thresholds=(np.array([3.1,5.,10.,20.,50.,100.,149.]) if 'clearance_' in d['tasks'][k]
                else np.unique(np.quantile(y,[.1,.2,.35,.5,.65,.8,.9])))
    targets=(y[:,None]>thresholds).astype(float);members=[];started=time.monotonic()
    folder.mkdir(parents=True,exist_ok=True)
    for seed in SEEDS:
        params=dict(iterations=700,depth=5,learning_rate=.035,l2_leaf_reg=20,
            random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False,
            bootstrap_type='Bernoulli',subsample=.85,border_count=64)
        if c['kind']=='ordinal':
            net=CatBoostClassifier(loss_function='MultiLogloss',**params).fit(x,targets)
            pred=net.predict_proba(qx).sum(1)
        else:
            net=CatBoostRegressor(loss_function='RMSE',**params).fit(x,np.log1p(y))
            pred=net.predict(qx)
        mp=folder/(key(c)+f'_{seed}.cbm');net.save_model(str(mp))
        replay=CatBoostClassifier() if c['kind']=='ordinal' else CatBoostRegressor();replay.load_model(str(mp))
        pp=replay.predict_proba(qx).sum(1) if c['kind']=='ordinal' else replay.predict(qx)
        np.testing.assert_array_equal(pred,pp);members.append(pred)
    p=np.mean(members,axis=0)
    np.savez_compressed(file,p=p,fit=fit,query=q,thresholds=thresholds,signature=sig())
    emit(stage='train',task=d['tasks'][k],part=part,config=c,seconds=round(time.monotonic()-started))
    return p

def configs(m):
    a=[dict(kind='kernel',view=v,C=c) for v in ('sequence','joint') for c in CS]
    if m=='Spearman':
        a += [dict(kind=kind,view=view) for kind in ('ordinal','logreg') for view in ('physchem','multiview')]
    return a

def predict(d,k,part,c):
    return kernel_family(d,k,part,[c])[key(c)] if c['kind']=='kernel' else ordinal(d,k,part,c)

def develop(d):
    protocol();records=[];selected=[]
    for task in TASKS:
        k=d['tasks'].index(task);m=d['metrics'][k]
        bases={p:base(d,k,p) for p in ('fold0','fold1')}
        family={p:kernel_family(d,k,p) for p in bases}
        choices=[]
        for c in configs(m):
            parts=[]
            for part,a in bases.items():
                q=r83.split(d,k,part)[1];y=d['y'][q]
                z=family[part][key(c)] if c['kind']=='kernel' else ordinal(d,k,part,c)
                p=mix(m,a,z)
                parts.append(dict(part=part,rows=len(q),baseline_proxy=r83.score(m,y,a),
                    expert=r83.score(m,y,z),candidate=r83.score(m,y,p),gain=r83.gain(m,y,a,p)))
            choices.append(dict(task=task,metric=m,config=c,parts=parts,objective=min(p['gain'] for p in parts)))
        best=max(choices,key=lambda c:c['objective']);records.extend(choices)
        if best['objective']>threshold(m):selected.append(best)
        emit(stage='development',**best)
        save('development.json',dict(status='IN_PROGRESS',records=records))
    save('development.json',dict(status='COMPLETE',signature=sig(),records=records))
    save('frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=sig(),selected=selected))

def confirm(d):
    frozen=json.loads((OUT/'frozen_selection.json').read_text());assert frozen['signature']==sig()
    records=[]
    for r in frozen['selected']:
        k=d['tasks'].index(r['task']);m=r['metric'];parts=[]
        for part in ('fold0','fold1','fold2','valid'):
            q=r83.split(d,k,part)[1];y=d['y'][q];a=base(d,k,part);z=predict(d,k,part,r['config']);p=mix(m,a,z)
            parts.append(dict(part=part,rows=len(q),baseline_proxy=r83.score(m,y,a),expert=r83.score(m,y,z),
                candidate=r83.score(m,y,p),gain=r83.gain(m,y,a,p)))
        rr=dict(**r,confirmation=parts,accepted=bool(min(p['gain'] for p in parts)>0 and np.mean([p['gain'] for p in parts])>threshold(m)))
        records.append(rr);emit(stage='confirmation',**rr)
        save('confirmation.json',dict(status='IN_PROGRESS',records=records))
    save('confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']]))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['develop','confirm','protocol']);a=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    if a.stage=='protocol':protocol()
    else:dict(develop=develop,confirm=confirm)[a.stage](load())
