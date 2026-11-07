"""Conformer shape specialists with target-excluded official-assay context."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier,CatBoostRegressor,Pool
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v96_normalized_assays as prev
import v98_conformer_features as features
b,r83=prev.b,prev.r83
OUT=ROOT/'admet_model_outputs_v98';ANCHOR,ANCHOR_SHA=prev.ANCHOR,prev.ANCHOR_SHA
SEEDS=(984026,984027)
SPECS=(dict(name='shape_d5',view='shape',depth=5),dict(name='joint_d5',view='joint',depth=5))
STEPS=1200;SNAPSHOTS=(150,450,1200);MODES=('direct','residual');WEIGHTS=(.25,.5,1.)

def sig():return hashlib.sha256((b.sha(__file__)+prev.sig()+features.sig()).encode()).hexdigest()
def emit(**x):print(json.dumps(x,ensure_ascii=False),flush=True)
def load():
    d=prev.load();file=features.OUT/'conformers.npz';meta=json.loads((features.OUT/'audit.json').read_text())
    assert meta['signature']==features.sig() and b.sha(file)==meta['sha256']
    z=np.load(file);assert str(z['signature'])==features.sig();d['conformer_features']=z['x']
    return d
def base(d,k,part):return prev.base(d,k,part)
def threshold(m):return prev.threshold(m)
def mix(m,old,p,w):return prev.mix(m,old,p,w)

def prepare(d,k,part,spec):
    other=next(s for s in prev.SPECS if s['view']=='chemical');a=prev.prepare(d,k,part,other)
    nf=5*(len(d['tasks'])-1)+1
    scalar=d['views']['physchem'];shape=d['conformer_features']
    pieces=[];qpieces=[]
    if spec['view']=='joint':pieces.append(a['x'][:,:nf]);qpieces.append(a['qx'][:,:nf])
    pieces.extend((scalar[d['mi'][a['fit']]],shape[d['mi'][a['fit']]]))
    qpieces.extend((scalar[d['mi'][a['query']]],shape[d['mi'][a['query']]]))
    a['x']=np.nan_to_num(np.concatenate(pieces,axis=1),nan=np.nan,posinf=np.nan,neginf=np.nan).astype(np.float32)
    a['qx']=np.nan_to_num(np.concatenate(qpieces,axis=1),nan=np.nan,posinf=np.nan,neginf=np.nan).astype(np.float32)
    return a

def train(d,k,part,spec,mode,seed):
    folder=OUT/'models'/d['tasks'][k]/part/spec['name']/mode/str(seed);file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);meta=json.loads((folder/'metadata.json').read_text())
        assert str(z['signature'])==meta['signature']==sig() and b.sha(file)==meta['predictions_sha256'] and b.sha(folder/'model.cbm')==meta['model_sha256']
        np.testing.assert_array_equal(z['query'],r83.split(d,k,part)[1]);return z['values'].copy()
    folder.mkdir(parents=True,exist_ok=True);started=time.monotonic();a=prepare(d,k,part,spec);m=d['metrics'][k];cls=m in ('AUROC','AUPRC');j=MODES.index(mode)
    params=dict(iterations=STEPS,depth=spec['depth'],learning_rate=.025,l2_leaf_reg=20,
        random_seed=seed,thread_count=2,verbose=False,allow_writing_files=False,bootstrap_type='Bernoulli',subsample=.85,
        task_type='CPU',border_count=64,loss_function='Logloss' if cls else 'MAE' if m=='MAE' else 'RMSE')
    net=(CatBoostClassifier if cls else CatBoostRegressor)(**params)
    offset=a['offset'] if cls and j==1 else np.zeros(len(a['fit']))
    net.fit(Pool(a['x'],a['target'][:,j],baseline=offset));values=[]
    for step in SNAPSHOTS:
        p=net.predict(a['qx'],prediction_type='RawFormulaVal',ntree_end=step).astype(float)
        if cls and j==1:p=p+a['qoffset']
        values.append(p)
    values=np.stack(values);net.save_model(str(folder/'model.cbm'))
    np.savez_compressed(file,signature=sig(),fit=a['fit'],query=a['query'],values=values,snapshots=SNAPSHOTS)
    b.save(folder/'metadata.json',dict(signature=sig(),task=d['tasks'][k],part=part,spec=spec,mode=mode,seed=seed,state=a['state'],
        model_sha256=b.sha(folder/'model.cbm'),predictions_sha256=b.sha(file),seconds=time.monotonic()-started,loss=params['loss_function']))
    emit(stage='train',task=d['tasks'][k],part=part,spec=spec['name'],mode=mode,seed=seed,seconds=round(time.monotonic()-started));return values

def expert(d,k,part,cfg):
    memo=d.setdefault('_v98_experts',{});key=(k,part,cfg['spec'],cfg['mode'])
    if key not in memo:
        spec=next(s for s in SPECS if s['name']==cfg['spec']);values=np.mean([train(d,k,part,spec,cfg['mode'],s) for s in SEEDS],axis=0)
        a=prepare(d,k,part,spec);memo[key]=(values,a['state'],a['qoffset'])
    values,state,qoff=memo[key];raw=values[SNAPSHOTS.index(cfg['step'])];m=d['metrics'][k]
    if m in ('AUROC','AUPRC'):return expit(raw)
    p=raw*state['scale']+(state['center'] if cfg['mode']=='direct' else qoff)
    return p.clip(0,100) if d['tasks'][k]=='ppbr_az' else p

def smoke(d):
    k=d['tasks'].index('ppbr_az');a=prepare(d,k,'fold0',SPECS[1]);q=a['query'];saved=d['y'][q].copy()
    d['y'][q]=1e6;d.pop('_v94_aux',None);d.pop('_v96_assays',None);z=prepare(d,k,'fold0',SPECS[1]);d['y'][q]=saved
    d.pop('_v94_aux',None);d.pop('_v96_assays',None)
    for key in ('fit','query','x','qx','target','offset','qoffset','cols','source'):np.testing.assert_array_equal(a[key],z[key])
    net=CatBoostRegressor(iterations=5,depth=2,loss_function='MAE',verbose=False,allow_writing_files=False,thread_count=2)
    net.fit(a['x'],a['target'][:,0]);file=OUT/'smoke.cbm';net.save_model(str(file));other=CatBoostRegressor();other.load_model(str(file))
    np.testing.assert_array_equal(net.predict(a['qx']),other.predict(a['qx']))
    b.save(OUT/'smoke.json',dict(status='PASS',signature=sig(),target_query_label_perturbation_invariance=True,model_reload=True));emit(stage='smoke',status='PASS')

def develop(d):
    OUT.mkdir(exist_ok=True);b.save(OUT/'protocol.json',dict(version=98,signature=sig(),baseline='v94',baseline_official_score_user_reported=50.916,
        anchor_sha256=ANCHOR_SHA,seeds=SEEDS,specs=SPECS,modes=MODES,steps=STEPS,snapshots=SNAPSHOTS,weights=WEIGHTS,
        selection_parts=['fold0','fold1'],confirmation_parts=['fold2','valid'],selection='Max minimum development gain; 1% relative MAE/.005 Spearman/.002 AUC/AP.',
        confirmation='All four gains positive and mean above threshold; no fallback.',features='1283 deterministic conformer descriptors and 435 2D descriptors, optional target-excluded neutral/exact assays.',
        test_labels_used=False,external_data=False,new_pretrained_weights=False,limitations='Historical proxy/label reuse. Approximate conformer ensemble from official structures. Local gain does not estimate platform score.'))
    b.save(OUT/'data_audit.json',dict(input_hashes=d['sources'],test_labels_finite=0,conformer_features_sha256=b.sha(features.OUT/'conformers.npz')))
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

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['smoke','develop','confirm']);args=ap.parse_args();dict(smoke=smoke,develop=develop,confirm=confirm)[args.stage](load())
