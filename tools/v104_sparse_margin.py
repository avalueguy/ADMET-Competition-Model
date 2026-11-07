"""Unfolded-count kernels with local exponential bandwidth and margin loss."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse,hashlib,json,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np,joblib
from scipy import sparse
from scipy.special import expit
from sklearn.svm import SVC,SVR
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v102_count_margin as prev
import v104_sparse_features as features
b,r83=prev.b,prev.r83
OUT=ROOT/'admet_model_outputs_v104';ANCHOR,ANCHOR_SHA=prev.ANCHOR,prev.ANCHOR_SHA
CS=(1.,10.);VIEWS=('count_t2','count_rbf4','binary_rbf4');WEIGHTS=(.25,.5,1.)

def sig():return hashlib.sha256((b.sha(__file__)+prev.sig()+features.sig()).encode()).hexdigest()
def emit(**x):print(json.dumps(x,ensure_ascii=False),flush=True)
def load():
    d=prev.load();meta=json.loads((features.OUT/'audit.json').read_text());assert meta['schema']['signature']==features.sig()
    xx=[]
    for j in range(2):
        p=features.OUT/f'view{j}.npz';assert b.sha(p)==meta['views'][j]['sha256'];xx.append(sparse.load_npz(p))
    d['unfolded_fingerprints']=xx;return d
def base(d,k,part):return prev.base(d,k,part)
def mix(m,a,z,w):return prev.mix(m,a,z,w)
def threshold(m):return prev.threshold(m)
def configs(m):return [dict(view=v,mode=z,C=c) for v in VIEWS for z in (('direct',) if m in ('AUROC','AUPRC') else ('direct','residual')) for c in CS]

def similarity(d,rows,fit,binary=False):
    out=np.zeros((len(rows),len(fit)),np.float64);fi,qi=d['mi'][fit],d['mi'][rows]
    for raw in d['unfolded_fingerprints']:
        ref=raw[fi].astype(float);query=raw[qi].astype(float)
        if binary:ref.data.fill(1.);query.data.fill(1.)
        rn=np.asarray(ref.multiply(ref).sum(1)).ravel();qn=np.asarray(query.multiply(query).sum(1)).ravel()
        for lo in range(0,len(rows),256):
            dot=(query[lo:lo+256]@ref.T).toarray();t=dot/np.maximum(qn[lo:lo+len(dot),None]+rn[None]-dot,1e-10)
            out[lo:lo+len(dot)]+=.5*np.clip(t,0,1)
    return out

def matrices(d,k,part):
    fit,q=r83.split(d,k,part);rows=np.r_[fit,q];n=len(fit)
    count=similarity(d,rows,fit);binary=similarity(d,rows,fit,True)
    views={'count_t2':.5*count+.5*count**2,'count_rbf4':np.exp(4*(count-1)),'binary_rbf4':np.exp(4*(binary-1))}
    return {v:((x[:n]+x[:n].T)*.5,x[n:]) for v,x in views.items()}

def train_one(d,k,part,cfg,a,mm):
    folder=OUT/'models'/d['tasks'][k]/part/f"{cfg['view']}_{cfg['mode']}_c{cfg['C']}";file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);meta=json.loads((folder/'metadata.json').read_text());assert str(z['signature'])==meta['signature']==sig()
        assert b.sha(file)==meta['prediction_sha256'] and b.sha(folder/'model.joblib')==meta['model_sha256']
        np.testing.assert_array_equal(z['query'],a['query']);return z['p']
    started=time.monotonic();metric=d['metrics'][k];cls=metric in ('AUROC','AUPRC');tr,q=mm[cfg['view']]
    model=SVC(C=cfg['C'],kernel='precomputed',tol=1e-4,cache_size=768) if cls else SVR(C=cfg['C'],epsilon=.03,kernel='precomputed',tol=1e-4,cache_size=768)
    yt=d['y'][a['fit']] if cls else a['target'][:,('direct','residual').index(cfg['mode'])];model.fit(tr,yt);assert model.fit_status_==0
    raw=model.decision_function(q) if cls else model.predict(q)
    p=expit(raw) if cls else raw*float(a['scale'])+(float(a['center']) if cfg['mode']=='direct' else a['qoff'])
    if d['tasks'][k]=='ppbr_az':p=p.clip(0,100)
    folder.mkdir(parents=True,exist_ok=True);joblib.dump(model,folder/'model.joblib',compress=3);np.savez_compressed(file,signature=sig(),**a,raw=raw,p=p)
    b.save(folder/'metadata.json',dict(signature=sig(),task=d['tasks'][k],part=part,config=cfg,prediction_sha256=b.sha(file),model_sha256=b.sha(folder/'model.joblib'),seconds=time.monotonic()-started,query_labels_used=False))
    emit(stage='train',task=d['tasks'][k],part=part,config=cfg,seconds=round(time.monotonic()-started));return p

def family(d,k,part,cs=None):
    cs=configs(d['metrics'][k]) if cs is None else cs;memo=d.setdefault('_v104',{});missing=[c for c in cs if (k,part,json.dumps(c,sort_keys=True)) not in memo]
    if missing:
        a=prev.state(d,k,part);mm=matrices(d,k,part)
        with ThreadPoolExecutor(max_workers=2) as pool:ps=list(pool.map(lambda c:train_one(d,k,part,c,a,mm),missing))
        for c,p in zip(missing,ps):memo[(k,part,json.dumps(c,sort_keys=True))]=p
    return {json.dumps(c,sort_keys=True):memo[(k,part,json.dumps(c,sort_keys=True))] for c in cs}
def expert(d,k,part,cfg):return family(d,k,part,[cfg])[json.dumps(cfg,sort_keys=True)]

def protocol():
    OUT.mkdir(exist_ok=True);doc=dict(version=104,signature=sig(),baseline='v101',anchor_sha256=ANCHOR_SHA,baseline_official_score_user_reported=50.932,
        C=list(CS),views=list(VIEWS),weights=list(WEIGHTS),
        features='Unfolded 32-bit circular radius2 and feature radius3 count fingerprints; equal-view generalized Tanimoto. Binary counterpart separate.',
        kernels='Count .5T+.5T^2 or exp(4*(T-1)); binary exp(4*(T-1)). SVC hinge, SVR epsilon .03; fit-only label transforms.',
        selection='Max minimum fold0/1 gain >1% MAE/.005 Spearman/.002 AUC/AP; freeze one configuration per task.',
        confirmation='All four gains positive and average above threshold; no failed fallback.',test_labels_used=False,new_external_data=False,new_pretrained_weights=False,
        limitations='Historically reused labels; current baseline validation is approximate, not full exact v101 OOF; no official-score estimate.')
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==doc
    else:b.save(p,doc)

def develop(d):
    protocol();records=[];selected=[];order=sorted(range(len(d['tasks'])),key=lambda k:int(((d['tid']==k)&(d['split']==0)).sum()))
    for k in order:
        m=d['metrics'][k];bases={p:base(d,k,p) for p in ('fold0','fold1')};ps={p:family(d,k,p) for p in bases}
        for cfg in configs(m):
            for w in WEIGHTS:
                parts=[]
                for part,old in bases.items():
                    q=r83.split(d,k,part)[1];p=mix(m,old,ps[part][json.dumps(cfg,sort_keys=True)],w)
                    parts.append(dict(part=part,rows=len(q),baseline=r83.score(m,d['y'][q],old),candidate=r83.score(m,d['y'][q],p),gain=r83.gain(m,d['y'][q],old,p)))
                records.append(dict(task=d['tasks'][k],metric=m,config=cfg,weight=w,parts=parts,objective=min(x['gain'] for x in parts)))
        best=max((r for r in records if r['task']==d['tasks'][k]),key=lambda r:r['objective']);emit(stage='development',**best)
        if best['objective']>threshold(m):selected.append(best)
        b.save(OUT/'development.json',dict(status='IN_PROGRESS',signature=sig(),records=records))
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=records));b.save(OUT/'frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=sig(),selected=selected,confirmation_inspected=False));emit(stage='frozen',selected=selected)

def confirm(d):
    f=json.loads((OUT/'frozen_selection.json').read_text());assert f['signature']==sig();records=[]
    for ch in f['selected']:
        k=d['tasks'].index(ch['task']);m=ch['metric'];parts=[]
        for part in ('fold0','fold1','fold2','valid'):
            q=r83.split(d,k,part)[1];old=base(d,k,part);p=mix(m,old,expert(d,k,part,ch['config']),ch['weight'])
            parts.append(dict(part=part,rows=len(q),baseline=r83.score(m,d['y'][q],old),candidate=r83.score(m,d['y'][q],p),gain=r83.gain(m,d['y'][q],old,p)))
        r=dict(**ch,confirmation=parts,accepted=bool(all(x['gain']>0 for x in parts) and np.mean([x['gain'] for x in parts])>threshold(m)))
        records.append(r);emit(stage='confirmation',**r);b.save(OUT/'confirmation.json',dict(status='IN_PROGRESS',signature=sig(),records=records))
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,accepted=[r['task'] for r in records if r['accepted']],frozen_sha256=b.sha(OUT/'frozen_selection.json')))

def smoke(d):
    protocol();k=d['tasks'].index('cyp2c9_substrate_carbonmangels');fit,q=r83.split(d,k,'fold0');fit=fit[:70]
    for binary in (False,True):
        x=similarity(d,fit,fit,binary);np.testing.assert_allclose(x,x.T,atol=1e-12);assert np.linalg.eigvalsh(x).min()>-1e-8
        z=similarity(d,q[:7],fit,binary);assert np.isfinite(z).all()
    b.save(OUT/'smoke.json',dict(status='PASS',signature=sig(),sparse_kernels_psd=True,official_structures_only=True));emit(stage='smoke',status='PASS')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['protocol','smoke','develop','confirm']);args=ap.parse_args()
    if args.stage=='protocol':protocol()
    else:dict(smoke=smoke,develop=develop,confirm=confirm)[args.stage](load())
