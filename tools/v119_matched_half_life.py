"""Matched v114 half-life refit: exact carried model and fitting rows.

New models use official target-excluded assays with a fixed log/rank two-target
tree ensemble. Protocol is saved before fitting. Historical labels are reused.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='3'
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v118_exact_half_life as exact
import v117_conditional_structure as conditional
v=conditional.prev;b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v119';WEIGHT=.5;SEEDS=(1194026,1194027)

def sig():return hashlib.sha256((b.sha(__file__)+conditional.sig()+b.sha(exact.__file__)).encode()).hexdigest()
def emit(**x):print(json.dumps(x),flush=True)
def split(d,k,part):
    folder='valid_report' if part=='valid' else part
    z=exact.archive(f'admet_model_outputs_v68/half_life_obach/{folder}/predictions.npz')
    fit,q=z['fit'],z['query']
    assert np.all(d['tid'][fit]==k) and not set(fit)&set(q)
    assert not np.any(d['split'][fit]==2)
    return fit,q
def base(d,k,part):
    if part in ('valid','final'):return exact.baseline(d,k,part)
    fit,q=split(d,k,part);ps=[]
    for seed in (672031,672032):
        z=exact.archive(f'admet_model_outputs_v67/cv/{part}/shared/{seed}/predictions.npz')
        pos={int(i):j for j,i in enumerate(z['query'])};ps.append(z['p'][[pos[int(i)] for i in q]])
        np.testing.assert_array_equal(fit,z['fit'][d['tid'][z['fit']]==k])
    z=exact.archive(f'admet_model_outputs_v68/half_life_obach/{part}/predictions.npz')
    return .65*r83.rankdata(np.mean(ps,axis=0))/len(q)+.35*r83.rankdata(z['p'][2])/len(q)
def design(d,k,part,view):
    fit,q=split(d,k,part)
    aux,cols,source,_,_=conditional.assays.assays(d,k,part)
    assert k not in cols and not np.any(d['tid'][source]==k)
    raw=d['views']['multiview' if view=='full' else 'physchem']
    x=np.c_[aux[fit],raw[d['mi'][fit]]].astype(np.float32)
    qx=np.c_[aux[q],raw[d['mi'][q]]].astype(np.float32)
    x[~np.isfinite(x)]=np.nan;qx[~np.isfinite(qx)]=np.nan
    return fit,q,x,qx,source
def predict(d,k,part,view):
    folder=OUT/'models'/part/view;file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);assert str(z['signature'])==sig();return z['p'].copy()
    fit,q,x,qx,source=design(d,k,part,view);y=d['y'][fit]
    target=np.c_[np.log1p(y),r83.ranknorm(y)].astype(np.float32)
    # Balance the two losses using only fit rows.
    center=target.mean(0);scale=target.std(0).clip(.05);target=(target-center)/scale
    folder.mkdir(parents=True,exist_ok=True);members=[];started=time.monotonic()
    for seed in SEEDS:
        net=CatBoostRegressor(loss_function='MultiRMSE',iterations=900,depth=4,learning_rate=.025,
            l2_leaf_reg=25,random_seed=seed,thread_count=3,verbose=False,allow_writing_files=False,
            bootstrap_type='Bernoulli',subsample=.85,border_count=64,random_strength=.5)
        net.fit(x,target);raw=net.predict(qx);path=folder/f'model_{seed}.cbm';net.save_model(str(path))
        replay=CatBoostRegressor();replay.load_model(str(path));np.testing.assert_array_equal(raw,replay.predict(qx))
        members.append(.5*r83.rankdata(raw[:,0])/len(q)+.5*r83.rankdata(raw[:,1])/len(q))
    p=np.mean(members,axis=0)
    np.savez_compressed(file,p=p,signature=sig(),fit=fit,query=q,target_center=center,target_scale=scale,source=source)
    emit(stage='train',part=part,view=view,fit=len(fit),query=len(q),seconds=round(time.monotonic()-started))
    return p
def record(d,k,part,view):
    fit,q=split(d,k,part);a=base(d,k,part);z=predict(d,k,part,view)
    p=.5*r83.rankdata(a)/len(a)+.5*r83.rankdata(z)/len(z)
    y=d['y'][q];return dict(part=part,rows=len(q),baseline=r83.score('Spearman',y,a),expert=r83.score('Spearman',y,z),
        candidate=r83.score('Spearman',y,p),gain=r83.gain('Spearman',y,a,p))
def run():
    d=v.load();k=d['tasks'].index('half_life_obach');OUT.mkdir(exist_ok=True)
    doc=dict(signature=sig(),baseline='Exact v114 half-life branch from v68',task='half_life_obach',
        model='Fixed two-target CatBoost MultiRMSE on fit-standardized log1p and Gaussian rank;2seeds900treesdepth4.',
        views=['physchem','full'],weight=WEIGHT,seeds=SEEDS,
        selection='fold0/1 choose view maxminimum gain>.008; frozen before fold2 and valid.',
        confirmation='Each of four gains positive, mean>.008. No fallback after failure.',
        baseline_fit='Archived v67/v68 exact fit/query arrays; official valid removes globally overlapping valid structure groups.',
        test_labels_used=False,external_data=False,limitations='Previously reused public labels. Matched refit, not independent new validation.')
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==doc
    else:b.save(path,doc)
    # Every feature/target source must be invariant to the task query labels.
    for part in ('fold0','fold1','fold2','valid','final'):
        a=design(d,k,part,'full');saved=d['y'][a[1]].copy();d['y'][a[1]]=987654.;d.pop('_v96_assays',None)
        z=design(d,k,part,'full');d['y'][a[1]]=saved;d.pop('_v96_assays',None)
        for x,y in zip(a,z):np.testing.assert_array_equal(x,y)
    b.save(OUT/'input_audit.json',dict(status='PASS',signature=sig(),target_query_label_perturbation=True))
    choices=[dict(view=view,parts=[record(d,k,p,view) for p in ('fold0','fold1')]) for view in ('physchem','full')]
    for r in choices:r['objective']=min(p['gain'] for p in r['parts'])
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=choices))
    best=max(choices,key=lambda r:r['objective']);selected=best['objective']>.008
    b.save(OUT/'frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=sig(),selected=selected,record=best))
    emit(stage='selection',selected=selected,**best)
    parts=[record(d,k,p,best['view']) for p in ('fold0','fold1','fold2','valid')] if selected else []
    accepted=bool(parts and min(p['gain'] for p in parts)>0 and np.mean([p['gain'] for p in parts])>.008)
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),accepted=accepted,record=best,parts=parts))
    emit(stage='confirmation',accepted=accepted,parts=parts)
    if accepted:predict(d,k,'final',best['view'])

if __name__=='__main__':run()
