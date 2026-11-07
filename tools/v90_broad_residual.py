"""Official-assay residual search over the user-reported v88 anchor."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='4'
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier, Pool
from scipy.special import expit
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import admet_v77_ranking as b
import v83_residual as r83
import v88_context_kernel as r88
OUT=ROOT/'admet_model_outputs_v90'
ANCHOR=ROOT/'admet_submissions/v88_context_kernel/我从山北来带着兰花草-小分子药物预测技术创新赛-20261025.csv'
ANCHOR_SHA='b8b339db361ca11499300f998fa9418f3e947db751c1c3cd960936b59defd20d'
SEEDS=(904026,904027,904028)
CONFIGS=[dict(name=f'd{d}_t{t}',depth=d,trees=t) for d,t in ((2,50),(2,100),(3,100),(3,200),(4,200))]
WEIGHTS=(.25,.5,.75,1.)
def sig():
    paths=(__file__,b.__file__,r83.__file__,r88.__file__,r88.r84.__file__,r88.r85.__file__,r88.r86.__file__,r88.r87.__file__)
    return hashlib.sha256(''.join(b.sha(p) for p in paths).encode()).hexdigest()
def emit(**x): print(json.dumps(x,ensure_ascii=False),flush=True)
def load():
    d=r88.load(); assert b.sha(ANCHOR)==ANCHOR_SHA; return d

def base(d,k,part):
    if part=='final':return np.array([float(r['y_pred']) for r in b.table(ANCHOR) if r['task']==d['tasks'][k]])
    p=r88.baseline(d,k,part)
    if d['tasks'][k]=='cyp3a4_substrate_carbonmangels':
        conf=json.loads((r88.OUT/'confirmation.json').read_text())
        ch=next(r for r in conf['records'] if r['task']==d['tasks'][k] and r['accepted'])
        p=r88.mix(ch['metric'],p,r88.expert(d,k,part,ch['config']),ch['weight'])
    return p

def design(d,k,part):
    fit,q=r83.split(d,k,part); a,cols,source,counts=r83.aux(d,k,part=='final'); assert k not in cols
    oldfit=r83.raw_proxy(d,fit)
    # Residualize the original endpoint only, then blend once with current anchor.
    oldquery=(np.array([float(r['y_pred']) for r in b.table(r88.r84.ORIGINAL) if r['task']==d['tasks'][k]]) if part=='final' else r83.raw_proxy(d,q))
    z,qz=r83.ranklog(oldfit),r83.ranklog(oldquery); beta=r83.calibrate(d['y'][fit],z)
    off,qoff=beta[0]+beta[1]*z,beta[0]+beta[1]*qz
    return dict(fit=fit,query=q,source=source,cols=np.array(cols),known_counts=counts[q],base_fit=oldfit,base_query=oldquery,
        beta=beta,off=off,qoff=qoff,x=np.c_[a[fit],z].astype(np.float32),qx=np.c_[a[q],qz].astype(np.float32))

def fit_one(d,k,part,cfg,seed):
    task=d['tasks'][k]; folder=OUT/'models'/task/part/cfg['name']/str(seed); path=folder/'prediction.npz'
    if path.exists():
        meta=json.loads((folder/'meta.json').read_text())
        assert meta['signature']==sig() and meta['config']==cfg and meta['seed']==seed
        assert b.sha(path)==meta['prediction_sha256'] and b.sha(folder/'model.cbm')==meta['model_sha256']
        with np.load(path) as z:
            assert str(z['signature'])==sig(); np.testing.assert_array_equal(z['query'],r83.split(d,k,part)[1]); return z['raw'].copy()
    started=time.monotonic(); a=design(d,k,part); folder.mkdir(parents=True,exist_ok=True)
    m=CatBoostClassifier(iterations=cfg['trees'],depth=cfg['depth'],learning_rate=.025,l2_leaf_reg=50,loss_function='Logloss',
        random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False,bootstrap_type='Bernoulli',subsample=.85,task_type='GPU',devices='0')
    m.fit(Pool(a['x'],d['y'][a['fit']],baseline=a['off'])); raw=a['qoff']+m.predict(a['qx'],prediction_type='RawFormulaVal')
    assert np.isfinite(raw).all(); m.save_model(str(folder/'model.cbm')); np.savez_compressed(path,signature=sig(),raw=raw,**a)
    b.save(folder/'meta.json',dict(signature=sig(),task=task,part=part,config=cfg,seed=seed,fit_rows=len(a['fit']),query_rows=len(a['query']),
        source_rows=len(a['source']),target_excluded=True,model_sha256=b.sha(folder/'model.cbm'),prediction_sha256=b.sha(path),seconds=time.monotonic()-started))
    return raw

def expert(d,k,part,cfg):return expit(np.mean([fit_one(d,k,part,cfg,s) for s in SEEDS],axis=0))
def mix(old,p,w):return (1-w)*rankdata(old)/len(old)+w*rankdata(p)/len(p)

def develop(d):
    OUT.mkdir(exist_ok=True)
    b.save(OUT/'protocol.json',dict(version=90,signature=sig(),baseline='v88',anchor_sha256=ANCHOR_SHA,
        baseline_official_score_user_reported=50.910,seeds=SEEDS,configs=CONFIGS,weights=WEIGHTS,
        selection_parts=['fold0','fold1'],confirmation_parts=['fold2','valid'],
        selection='Max minimum declared-metric gain, >0.0005 both folds; freeze one task/config/weight.',
        confirmation='All four gains strictly positive; no fallback tuning.',
        ensemble='sigmoid(mean(raw logits including fixed calibrated historical base)) at every stage',
        test_labels_used=False,external_data=False,new_pretrained_weights=False,target_assay_measurements_and_masks_excluded=True,
        limitations='Historical raw proxies are not full v88 OOF. Historical labels reused. Other official assays are conditional information. Local gains do not estimate platform points.'))
    b.save(OUT/'data_audit.json',dict(input_hashes=d['sources'],legacy_folds_sha256=b.sha(ROOT/'recovered_history/legacy_v24_folds.npy'),
        test_labels_finite=int(np.isfinite(d['y'][d['split']==2]).sum())))
    tasks=[k for k,m in enumerate(d['metrics']) if m in ('AUROC','AUPRC')]; records=[]
    for k in tasks:
        task,metric=d['tasks'][k],d['metrics'][k]; bases={p:base(d,k,p) for p in ('fold0','fold1')}
        for cfg in CONFIGS:
            pp={p:expert(d,k,p,cfg) for p in bases}
            for w in WEIGHTS:
                parts=[]
                for part,old in bases.items():
                    q=r83.split(d,k,part)[1]; p=mix(old,pp[part],w)
                    s0,s1=r83.score(metric,d['y'][q],old),r83.score(metric,d['y'][q],p)
                    parts.append(dict(part=part,rows=len(q),baseline=s0,candidate=s1,gain=s1-s0))
                records.append(dict(task=task,metric=metric,config=cfg,weight=w,parts=parts,objective=min(x['gain'] for x in parts)))
        best=max((r for r in records if r['task']==task),key=lambda r:r['objective']); emit(stage='development',**best)
        b.save(OUT/'development.json',dict(status='IN_PROGRESS',signature=sig(),records=records))
    selected=[]
    for k in tasks:
        best=max((r for r in records if r['task']==d['tasks'][k]),key=lambda r:r['objective'])
        if best['objective']>.0005:selected.append(best)
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=records))
    b.save(OUT/'frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=sig(),selected=selected,confirmation_inspected=False))
    emit(stage='frozen',selected=selected)

def confirm(d):
    fr=json.loads((OUT/'frozen_selection.json').read_text()); assert fr['signature']==sig(); records=[]
    for ch in fr['selected']:
        k=d['tasks'].index(ch['task']); parts=[]
        for part in ('fold0','fold1','fold2','valid'):
            p=expert(d,k,part,ch['config']); q=r83.split(d,k,part)[1]; old=base(d,k,part); new=mix(old,p,ch['weight'])
            s0,s1=r83.score(ch['metric'],d['y'][q],old),r83.score(ch['metric'],d['y'][q],new)
            parts.append(dict(part=part,baseline=s0,candidate=s1,gain=s1-s0,rows=len(q)))
        rec=dict(**ch,confirmation=parts,accepted=all(x['gain']>0 for x in parts)); records.append(rec); emit(stage='confirmation',**rec)
        b.save(OUT/'confirmation.json',dict(status='IN_PROGRESS',signature=sig(),records=records,frozen_sha256=b.sha(OUT/'frozen_selection.json')))
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']],frozen_sha256=b.sha(OUT/'frozen_selection.json')))

def smoke(d):
    k=next(k for k,m in enumerate(d['metrics']) if m=='AUPRC'); a=design(d,k,'fold0')
    assert not set(a['fit'])&set(a['query']) and not set(d['groups'][a['fit']])&set(d['groups'][a['query']])
    assert k not in a['cols'] and np.all(d['tid'][a['source']]!=k) and np.all(d['split'][a['source']]==0)
    folder=OUT/'smoke'; folder.mkdir(parents=True,exist_ok=True)
    m=CatBoostClassifier(iterations=3,depth=2,verbose=False,task_type='GPU',devices='0',allow_writing_files=False,thread_count=4,random_seed=904000)
    m.fit(Pool(a['x'],d['y'][a['fit']],baseline=a['off'])); raw=m.predict(a['qx'],prediction_type='RawFormulaVal'); m.save_model(str(folder/'model.cbm'))
    replay=CatBoostClassifier().load_model(str(folder/'model.cbm')); np.testing.assert_allclose(raw,replay.predict(a['qx'],prediction_type='RawFormulaVal'),atol=1e-12)
    idx=np.unique([0,len(raw)//2,len(raw)-1]); np.testing.assert_allclose(raw[idx],replay.predict(a['qx'][idx],prediction_type='RawFormulaVal'),atol=1e-12)
    from sklearn.metrics import roc_auc_score, average_precision_score
    y,p=np.array([0,1,0,1]),np.array([.2,.4,.8,.9])
    assert r83.score('AUROC',y,p)==roc_auc_score(y,p) and r83.score('AUPRC',y,p)==average_precision_score(y,p)
    assert r83.score('AUPRC',y,p)!=r83.score('AUROC',y,p)
    b.save(folder/'result.json',dict(status='PASS',signature=sig(),task=d['tasks'][k],declared_metric=d['metrics'][k],source_rows=len(a['source']),model_reload='PASS',subset_prediction='PASS',target_assay_excluded=True))
    emit(stage='smoke',status='PASS',task=d['tasks'][k])

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['smoke','develop','confirm']); args=ap.parse_args()
    dict(smoke=smoke,develop=develop,confirm=confirm)[args.stage](load())
