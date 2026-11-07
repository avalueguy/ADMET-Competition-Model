"""Target-excluded neutral-parent assay transfer against frozen v94.

Exact molecular identities and neutral-largest-parent identities are separate
input columns. No target assay, mask or measurement enters any input source.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier,CatBoostRegressor,Pool
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v95_graph_specialists as prev
b,r83=prev.b,prev.r83
OUT=ROOT/'admet_model_outputs_v96';ANCHOR,ANCHOR_SHA=prev.ANCHOR,prev.ANCHOR_SHA
SEEDS=(964026,964027)
SPECS=(dict(name='assay_d3',view='assay',depth=3),dict(name='chemical_d5',view='chemical',depth=5))
STEPS=1000
SNAPSHOTS=(100,300,1000)
MODES=('direct','residual')
WEIGHTS=(.25,.5,1.)
IDENTITIES=ROOT/'admet_model_outputs_v95/identity_audit/identities.npz'
IDENTITIES_SHA='5d4d1868d48c770ea43d64929dc07efe98cd5a494a07438955ef40a1087dd30a'


def sig():return hashlib.sha256((b.sha(__file__)+prev.sig()+IDENTITIES_SHA).encode()).hexdigest()
def emit(**x):print(json.dumps(x,ensure_ascii=False),flush=True)
def load():
    d=prev.load();assert b.sha(IDENTITIES)==IDENTITIES_SHA
    d['neutral_identity']=np.load(IDENTITIES)['keys'][d['mi'],1]
    return d
def base(d,k,part):return prev.base(d,k,part)
def threshold(metric):return prev.threshold(metric)
def mix(metric,old,new,w):return prev.mix(metric,old,new,w)


def assays(d,k,part):
    final=part=='final';key=(k,final);memo=d.setdefault('_v96_assays',{})
    if key in memo:return memo[key]
    exact,cols,source,counts=r83.aux(d,k,final)
    assert k not in cols and np.all(d['tid'][source]!=k)
    lookup={}
    for i in source:lookup.setdefault((str(d['neutral_identity'][i]),int(d['tid'][i])),[]).append(float(d['y'][i]))
    neutral=np.full_like(exact,np.nan);spread=np.full((len(d['y']),len(cols)),np.nan,np.float32)
    for i in np.flatnonzero(d['tid']==k):
        for j,t in enumerate(cols):
            values=lookup.get((str(d['neutral_identity'][i]),t))
            if values:
                neutral[i,j]=np.median(values);spread[i,j]=np.max(values)-np.min(values)
    neutral[:,len(cols):]=np.isfinite(neutral[:,:len(cols)])
    # Separate columns preserve disagreements caused by different charge/salt forms.
    result=(np.c_[exact,neutral,spread].astype(np.float32),np.array(cols),source,
            neutral[:,len(cols):].sum(1).astype(int),counts)
    memo[key]=result
    return result


def prepare(d,k,part,spec):
    fit,q=r83.split(d,k,part);metric=d['metrics'][k]
    auxiliary,cols,source,counts,exact_counts=assays(d,k,part)
    oldfit=r83.raw_proxy(d,fit)
    original=ROOT/'admet_submissions/v73_v72/我从山北来带着兰花草-小分子药物预测技术创新赛-20261021.csv'
    qp=np.array([float(r['y_pred']) for r in b.table(original) if r['task']==d['tasks'][k]]) if part=='final' else r83.raw_proxy(d,q)
    if metric in ('AUROC','AUPRC'):
        z,qz=r83.ranklog(oldfit),r83.ranklog(qp)
        beta=r83.calibrate(d['y'][fit],z);off=beta[0]+beta[1]*z;qoff=beta[0]+beta[1]*qz
        state=dict(center=0.,scale=1.,beta=beta.tolist());targets=np.c_[d['y'][fit],d['y'][fit]]
    elif metric=='MAE':
        yy=d['y'][fit];center=float(np.median(yy));scale=max(float(np.quantile(yy,.75)-np.quantile(yy,.25)),.001)
        state=dict(center=center,scale=scale);z=(oldfit-center)/scale;qz=(qp-center)/scale
        targets=np.c_[(yy-center)/scale,(yy-oldfit)/scale];off=np.zeros(len(fit));qoff=qp
    else:
        z,qz=r83.ranknorm(oldfit),r83.ranknorm(qp);design=np.c_[np.ones(len(z)),z];yy=r83.ranknorm(d['y'][fit])
        beta=np.linalg.solve(design.T@design+np.diag([1e-8,1.]),design.T@yy);beta[1]=max(beta[1],.01)
        state=dict(center=0.,scale=1.,beta=beta.tolist());targets=np.c_[yy,yy-design@beta]
        off=np.zeros(len(fit));qoff=beta[0]+beta[1]*qz
    x=np.c_[auxiliary[fit],z];qx=np.c_[auxiliary[q],qz]
    if spec['view']=='chemical':
        x=np.c_[x,d['views']['physchem'][d['mi'][fit]]]
        qx=np.c_[qx,d['views']['physchem'][d['mi'][q]]]
    x=np.nan_to_num(x,nan=np.nan,posinf=np.nan,neginf=np.nan).astype(np.float32)
    qx=np.nan_to_num(qx,nan=np.nan,posinf=np.nan,neginf=np.nan).astype(np.float32)
    return dict(fit=fit,query=q,x=x,qx=qx,target=targets.astype(np.float32),offset=off,qoffset=qoff,
        state=state,cols=cols,source=source,known_counts=counts[q],exact_counts=exact_counts[q])


def train(d,k,part,spec,mode,seed):
    folder=OUT/'models'/d['tasks'][k]/part/spec['name']/mode/str(seed);file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);meta=json.loads((folder/'metadata.json').read_text())
        assert str(z['signature'])==meta['signature']==sig() and b.sha(file)==meta['predictions_sha256']
        assert b.sha(folder/'model.cbm')==meta['model_sha256']
        np.testing.assert_array_equal(z['query'],r83.split(d,k,part)[1]);return z['values'].copy()
    folder.mkdir(parents=True,exist_ok=True);started=time.monotonic();a=prepare(d,k,part,spec)
    m=d['metrics'][k];cls=m in ('AUROC','AUPRC');j=MODES.index(mode)
    params=dict(iterations=STEPS,depth=spec['depth'],learning_rate=.025,l2_leaf_reg=25,
        random_seed=seed,thread_count=2,verbose=False,allow_writing_files=False,
        bootstrap_type='Bernoulli',subsample=.85,task_type='CPU',border_count=64,
        loss_function='Logloss' if cls else 'MAE' if m=='MAE' else 'RMSE')
    model=(CatBoostClassifier if cls else CatBoostRegressor)(**params)
    offset=a['offset'] if cls and j==1 else np.zeros(len(a['fit']))
    model.fit(Pool(a['x'],a['target'][:,j],baseline=offset))
    values=[]
    for step in SNAPSHOTS:
        p=model.predict(a['qx'],prediction_type='RawFormulaVal',ntree_end=step).astype(float)
        if cls and j==1:p=p+a['qoffset']
        values.append(p)
    values=np.stack(values);model.save_model(str(folder/'model.cbm'))
    np.savez_compressed(file,signature=sig(),fit=a['fit'],query=a['query'],values=values,snapshots=SNAPSHOTS)
    b.save(folder/'metadata.json',dict(signature=sig(),task=d['tasks'][k],part=part,spec=spec,mode=mode,seed=seed,
        state=a['state'],model_sha256=b.sha(folder/'model.cbm'),predictions_sha256=b.sha(file),seconds=time.monotonic()-started,
        neutral_identity_sha256=IDENTITIES_SHA,target_assay_excluded=True,loss=params['loss_function']))
    emit(stage='train',task=d['tasks'][k],part=part,spec=spec['name'],mode=mode,seed=seed,seconds=round(time.monotonic()-started))
    return values


def expert(d,k,part,cfg):
    memo=d.setdefault('_v96_experts',{});key=(k,part,cfg['spec'],cfg['mode'])
    if key not in memo:
        spec=next(s for s in SPECS if s['name']==cfg['spec'])
        values=np.mean([train(d,k,part,spec,cfg['mode'],s) for s in SEEDS],axis=0)
        a=prepare(d,k,part,spec);memo[key]=(values,a['state'],a['qoffset'])
    values,state,qoff=memo[key];raw=values[SNAPSHOTS.index(cfg['step'])];m=d['metrics'][k]
    if m in ('AUROC','AUPRC'):return expit(raw)
    p=raw*state['scale']+(state['center'] if cfg['mode']=='direct' else qoff)
    return p.clip(0,100) if d['tasks'][k]=='ppbr_az' else p


def smoke(d):
    OUT.mkdir(exist_ok=True);k=d['tasks'].index('vdss_lombardo');a=prepare(d,k,'fold0',SPECS[0]);q=a['query']
    saved=d['y'][q].copy();d['y'][q]=1e6;d.pop('_v94_aux',None);d.pop('_v96_assays',None)
    z=prepare(d,k,'fold0',SPECS[0]);d['y'][q]=saved;d.pop('_v94_aux',None);d.pop('_v96_assays',None)
    for key in ('fit','query','x','qx','target','offset','qoffset','cols','source'):np.testing.assert_array_equal(a[key],z[key])
    assert np.all(d['tid'][a['source']]!=k)
    # Lightweight smoke fit does not populate production caches.
    model=CatBoostRegressor(iterations=5,depth=2,loss_function='RMSE',verbose=False,allow_writing_files=False,thread_count=2)
    model.fit(a['x'],a['target'][:,0]);p=model.predict(a['qx']);file=OUT/'smoke.cbm';model.save_model(str(file))
    other=CatBoostRegressor();other.load_model(str(file));np.testing.assert_array_equal(p,other.predict(a['qx']))
    b.save(OUT/'smoke.json',dict(status='PASS',signature=sig(),target_query_label_perturbation_invariance=True,
        target_assay_source_absent=True,model_reload=True,neutral_identity_sha256=IDENTITIES_SHA))
    emit(stage='smoke',status='PASS')


def develop(d):
    OUT.mkdir(exist_ok=True)
    b.save(OUT/'protocol.json',dict(version=96,signature=sig(),baseline='v94',baseline_official_score_user_reported=50.916,
        anchor_sha256=ANCHOR_SHA,seeds=SEEDS,specs=SPECS,modes=MODES,steps=STEPS,snapshots=SNAPSHOTS,weights=WEIGHTS,
        selection_parts=['fold0','fold1'],confirmation_parts=['fold2','valid'],
        selection='One config maximizing minimum gain; >1% relative MAE, .005 Spearman, .002 AUC/AP in both folds.',
        confirmation='All four gains positive, mean exceeds threshold, no fallback.',
        inputs='Separate exact and neutral largest-parent other-task values and masks, neutral measurement range, baseline rank, optional 435 descriptors.',
        identity_choice='Neutral largest parent fixed from structure-only coverage audit; tautomer merging not used.',
        neutral_identity_sha256=IDENTITIES_SHA,test_labels_used=False,external_data=False,new_pretrained_weights=False,
        limitations='Chemical forms may have different assay values; separate columns and spreads retain this distinction. Conditional official assays and historically reused proxy validation. No official gain estimate.'))
    b.save(OUT/'data_audit.json',dict(input_hashes=d['sources'],test_labels_finite=0,neutral_identity_sha256=IDENTITIES_SHA))
    records=[];selected=[]
    for k,task in enumerate(d['tasks']):
        m=d['metrics'][k];bases={p:base(d,k,p) for p in ('fold0','fold1')}
        for spec in SPECS:
            for mode in MODES:
                for step in SNAPSHOTS:
                    cfg=dict(spec=spec['name'],mode=mode,step=step);ps={p:expert(d,k,p,cfg) for p in bases}
                    for w in WEIGHTS:
                        parts=[]
                        for part,old in bases.items():
                            q=r83.split(d,k,part)[1];p=mix(m,old,ps[part],w)
                            parts.append(dict(part=part,rows=len(q),baseline=r83.score(m,d['y'][q],old),candidate=r83.score(m,d['y'][q],p),gain=r83.gain(m,d['y'][q],old,p)))
                        records.append(dict(task=task,metric=m,config=cfg,weight=w,parts=parts,objective=min(x['gain'] for x in parts)))
        best=max((r for r in records if r['task']==task),key=lambda r:r['objective']);emit(stage='development',**best)
        if best['objective']>threshold(m):selected.append(best)
        b.save(OUT/'development.json',dict(status='IN_PROGRESS',signature=sig(),records=records))
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=records))
    b.save(OUT/'frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=sig(),selected=selected,confirmation_inspected=False))
    emit(stage='frozen',selected=selected)


def confirm(d):
    frozen=json.loads((OUT/'frozen_selection.json').read_text());assert frozen['signature']==sig();records=[]
    for ch in frozen['selected']:
        k=d['tasks'].index(ch['task']);m=ch['metric'];parts=[]
        for part in ('fold0','fold1','fold2','valid'):
            q=r83.split(d,k,part)[1];old=base(d,k,part);p=mix(m,old,expert(d,k,part,ch['config']),ch['weight'])
            parts.append(dict(part=part,rows=len(q),baseline=r83.score(m,d['y'][q],old),candidate=r83.score(m,d['y'][q],p),gain=r83.gain(m,d['y'][q],old,p)))
        r=dict(**ch,confirmation=parts,accepted=bool(all(x['gain']>0 for x in parts) and np.mean([x['gain'] for x in parts])>threshold(m)))
        records.append(r);emit(stage='confirmation',**r)
        b.save(OUT/'confirmation.json',dict(status='IN_PROGRESS',signature=sig(),records=records))
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']],frozen_sha256=b.sha(OUT/'frozen_selection.json')))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','develop','confirm']);args=ap.parse_args()
    dict(smoke=smoke,develop=develop,confirm=confirm)[args.stage](load())
