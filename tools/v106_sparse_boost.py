"""Task-specific sparse substructure boosting, fit-only prevalence pruning."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np,xgboost as xgb,torch
from scipy import sparse
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v104_sparse_margin as prev
b,r83=prev.b,prev.r83
OUT=ROOT/'admet_model_outputs_v106';ANCHOR,ANCHOR_SHA=prev.ANCHOR,prev.ANCHOR_SHA
SEEDS=(1064026,1064027);DEPTHS=(3,6);STEPS=1800;SNAPSHOTS=(400,1800);WEIGHTS=(.25,.5,1.)

def sig():return hashlib.sha256((b.sha(__file__)+prev.sig()).encode()).hexdigest()
def emit(**x):print(json.dumps(x,ensure_ascii=False),flush=True)
def load():return prev.load()
def base(d,k,part):return prev.base(d,k,part)
def mix(m,a,z,w):return prev.mix(m,a,z,w)
def threshold(m):return prev.threshold(m)
def configs(m):return [dict(depth=dep,step=t) for dep in DEPTHS for t in SNAPSHOTS]

def prepare(d,k,part):
    a=prev.prev.state(d,k,part);fit,q=a['fit'],a['query'];pieces=[];qpieces=[];columns=[]
    for raw in d['unfolded_fingerprints']:
        xx=raw[d['mi'][fit]];cols=np.flatnonzero(np.asarray((xx>0).sum(0)).ravel()>=3);columns.append(cols)
        x=xx[:,cols].copy();qx=raw[d['mi'][q]][:,cols].copy();x.data=np.log1p(x.data);qx.data=np.log1p(qx.data)
        pieces.append(x);qpieces.append(qx)
    # Explicit NaNs are retained as missing; sparse substructure absence is
    # handled consistently by XGBoost's learned missing branch.
    pieces.append(sparse.csr_matrix(d['views']['physchem'][d['mi'][fit]]));qpieces.append(sparse.csr_matrix(d['views']['physchem'][d['mi'][q]]))
    return dict(**a,x=sparse.hstack(pieces,format='csr'),qx=sparse.hstack(qpieces,format='csr'),columns=columns)

def train(d,k,part,depth,seed):
    folder=OUT/'models'/d['tasks'][k]/part/f'depth{depth}'/str(seed);file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);meta=json.loads((folder/'metadata.json').read_text());assert str(z['signature'])==meta['signature']==sig()
        assert b.sha(file)==meta['prediction_sha256'] and b.sha(folder/'model.ubj')==meta['model_sha256'];np.testing.assert_array_equal(z['query'],r83.split(d,k,part)[1]);return z['values']
    started=time.monotonic();a=prepare(d,k,part);m=d['metrics'][k];cls=m in ('AUROC','AUPRC');y=d['y'][a['fit']] if cls else a['target'][:,0]
    dm=xgb.DMatrix(a['x'],label=y,nthread=4);qm=xgb.DMatrix(a['qx'],nthread=4)
    model=xgb.train(dict(objective='binary:logistic' if cls else 'reg:absoluteerror' if m=='MAE' else 'reg:squarederror',tree_method='hist',device='cuda',
        max_depth=depth,max_bin=64,eta=.025,min_child_weight=5,subsample=.85,colsample_bytree=.7,reg_lambda=20,reg_alpha=.05,seed=seed,nthread=4),dm,num_boost_round=STEPS)
    values=np.stack([model.predict(qm,iteration_range=(0,t),output_margin=cls) for t in SNAPSHOTS]).astype(float)
    if not cls:values=values*float(a['scale'])+float(a['center'])
    if d['tasks'][k]=='ppbr_az':values=values.clip(0,100)
    assert np.isfinite(values).all();folder.mkdir(parents=True,exist_ok=True);model.save_model(folder/'model.ubj')
    np.savez_compressed(file,signature=sig(),fit=a['fit'],query=a['query'],values=values,column0=a['columns'][0],column1=a['columns'][1],snapshots=SNAPSHOTS)
    b.save(folder/'metadata.json',dict(signature=sig(),task=d['tasks'][k],part=part,depth=depth,seed=seed,model_sha256=b.sha(folder/'model.ubj'),prediction_sha256=b.sha(file),
        fit_columns=[len(c) for c in a['columns']],query_labels_used=False,seconds=time.monotonic()-started));emit(stage='train',task=d['tasks'][k],part=part,depth=depth,seed=seed,seconds=round(time.monotonic()-started));return values

def expert(d,k,part,cfg):
    memo=d.setdefault('_v106',{});key=(k,part,cfg['depth'])
    if key not in memo:memo[key]=np.mean([train(d,k,part,cfg['depth'],s) for s in SEEDS],axis=0)
    z=memo[key][SNAPSHOTS.index(cfg['step'])];return expit(z) if d['metrics'][k] in ('AUROC','AUPRC') else z

def protocol():
    OUT.mkdir(exist_ok=True);doc=dict(version=106,signature=sig(),baseline='v101',baseline_official_score_user_reported=50.932,anchor_sha256=ANCHOR_SHA,
        seeds=list(SEEDS),depths=list(DEPTHS),steps=STEPS,snapshots=list(SNAPSHOTS),weights=list(WEIGHTS),
        feature_selection='Keep unfolded count columns present in >=3 fit observations, log1p counts; append435 official-derived2D descriptors. Query does not affect columns.',
        training='Task-specific GPU XGBoost direct labels, logistic/absoluteerror/squarederror by metric. No residuals or auxiliary assays.',
        selection='Max minimum fold0/1 gain >1%MAE/.005Spearman/.002AUC/AP; one frozen configuration per task.',
        confirmation='All four gains positive and average above threshold; no failed fallback.',test_labels_used=False,new_external_data=False,new_pretrained_weights=False,
        limitations='Historical validation reused; base control is proxy not exactv101OOF; no official-score estimate.')
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==doc
    else:b.save(p,doc)

def develop(d):
    protocol();records=[];selected=[];order=sorted(range(len(d['tasks'])),key=lambda k:int(((d['tid']==k)&(d['split']==0)).sum()))
    for k in order:
        m=d['metrics'][k];bases={p:base(d,k,p) for p in ('fold0','fold1')}
        for cfg in configs(m):
            pp={p:expert(d,k,p,cfg) for p in bases}
            for w in WEIGHTS:
                parts=[]
                for part,old in bases.items():
                    q=r83.split(d,k,part)[1];p=mix(m,old,pp[part],w)
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
    protocol();k=d['tasks'].index('herg');a=prepare(d,k,'fold0');q=a['query'];saved=d['y'][q].copy();d['y'][q]=1e6;z=prepare(d,k,'fold0');d['y'][q]=saved
    for key in ('fit','query','target','qoff'):np.testing.assert_array_equal(a[key],z[key])
    for x,y in zip(a['columns'],z['columns']):np.testing.assert_array_equal(x,y)
    for key in ('x','qx'):
        for attr in ('data','indices','indptr'):np.testing.assert_array_equal(getattr(a[key],attr),getattr(z[key],attr))
    b.save(OUT/'smoke.json',dict(status='PASS',signature=sig(),fit_only_feature_pruning=True,query_label_perturbation=True));emit(stage='smoke',status='PASS')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['protocol','smoke','develop','confirm']);args=ap.parse_args();torch.set_num_threads(2)
    if args.stage=='protocol':protocol()
    else:dict(smoke=smoke,develop=develop,confirm=confirm)[args.stage](load())
