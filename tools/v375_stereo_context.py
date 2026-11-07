"""Stereochemistry-preserving conditional MAE experts with replicate integration."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(k,'4')
import hashlib,json,time
import numpy as np,joblib
from scipy.stats import qmc
from catboost import CatBoostRegressor
import v371_mae_experts as shared
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
TASKS=shared.TASKS
CONFIGS=tuple(dict(name=f'{identity}_{mode}_d{depth}',family='context',identity=identity,mode=mode,depth=depth)
    for identity,mode,depth in (('stereo','median',4),('stereo','median',6),('stereo','integrated',4),('stereo','integrated',6)))
WEIGHTS=(.1,.2,.45,.75,1.)
SEEDS=(3751106,3751107)
sha=shared.sha
save=shared.save
emit=shared.emit
clip=shared.clip
target=shared.target
def setup():shared.setup()
def signature():return hashlib.sha256((sha(__file__)+sha(shared.__file__)+sha(ROOT/'data_manifest.json')+sha(ROOT/'stereo_keys.npz')+json.dumps([CONFIGS,WEIGHTS,SEEDS],sort_keys=True)).encode()).hexdigest()
def load():
    shared.ROOT=ROOT;d=shared.load();d.update(dict(np.load(ROOT/'stereo_keys.npz')));return d
def ctrl(task,part):return shared.ctrl(task,part)
def design(d,task,part):return shared.design(d,task,part)

def context(d,task,part):
    key=(task,part=='final');memo=d.setdefault('_stereo_context',{})
    if key in memo:return memo[key]
    target_id=d['tasks'].index(task);cols=[i for i in range(22) if i!=target_id];source=np.flatnonzero((d['tid']!=target_id)&(d['split']<(2 if part=='final' else 1)))
    assert np.all(d['tid'][source]!=target_id) and np.isfinite(d['y'][source]).all()
    pools=[{},{}]
    for row in source:
        for j,name in enumerate(('isomeric','neutral_stereo')):
            pools[j].setdefault((str(d[name][d['mi'][row]]),int(d['tid'][row])),[]).append(float(d['y'][row]))
    for pool in pools:
        for k,v in pool.items():pool[k]=np.sort(np.array(v,float))
    rows=np.flatnonzero(d['tid']==target_id);n=len(cols);mat=np.full((len(d['y']),5*n),np.nan,np.float32)
    for row in rows:
        for j,t in enumerate(cols):
            for block,name in enumerate(('isomeric','neutral_stereo')):
                values=pools[block].get((str(d[name][d['mi'][row]]),t))
                if values is not None:
                    mat[row,2*block*n+j]=np.median(values)
                    if block==1:mat[row,4*n+j]=np.ptp(values)
    mat[:,n:2*n]=np.isfinite(mat[:,:n]);mat[:,3*n:4*n]=np.isfinite(mat[:,2*n:3*n])
    value=dict(median=mat,pools=pools,cols=cols,source=source);memo[key]=value;return value

def samples(d,rows,task,part,mode):
    ctx=context(d,task,part);median=ctx['median'][rows];cols=ctx['cols'];n=len(cols)
    if mode=='median':return [median]
    # Fixed deterministic Sobol empirical marginalization; target assay never present.
    uniforms=qmc.Sobol(d=2*n,scramble=False).random_base2(2)
    mats=[median.copy() for _ in range(4)]
    for i,row in enumerate(rows):
        for block,name in enumerate(('isomeric','neutral_stereo')):
            for j,t in enumerate(cols):
                values=ctx['pools'][block].get((str(d[name][d['mi'][row]]),t))
                if values is None or len(values)==1:continue
                ids=np.minimum((uniforms[:,block*n+j]*len(values)).astype(int),len(values)-1)
                for index in range(4):mats[index][i,2*block*n+j]=values[ids[index]]
    return mats

def features(d,a,task,part,c):
    fit,q=a['fit'],a['query'];raw=d['fingerprints'];variance=raw[d['mi'][fit]].astype(float).var(0);columns=np.sort(np.argsort(-variance,kind='stable')[:1024])
    def build(rows,original):
        #105-column context layout is frozen from v96; retain other structural channels.
        rest=np.c_[original[:,105:],raw[d['mi'][rows]][:,columns]]
        return [np.where(np.isfinite(x),np.clip(x,-1e18,1e18),np.nan).astype(np.float32) for m in samples(d,rows,task,part,c['mode']) for x in [np.c_[m,rest]]]
    xs,qs=build(fit,a['x']),build(q,a['qx']);return xs,qs,columns

def predict(d,task,part,c):
    setup();folder=OUT/'tasks'/task/part/c['name'];path=folder/'predictions.npz';sig=signature()
    if path.exists():
        z=np.load(path);assert str(z['signature'])==sig;return z['p']
    start=time.monotonic();a=design(d,task,part);xs,qs,columns=features(d,a,task,part,c);x=np.concatenate(xs)
    y,center,scale=target(d['y'][a['fit']]);ys=np.tile(y,len(xs));models=[];predictions=[]
    rounds={'caco2_wang':800,'ld50_zhu':1100,'lipophilicity_astrazeneca':1100,'ppbr_az':1000,'solubility_aqsoldb':1400}[task]
    for seed in SEEDS:
        net=CatBoostRegressor(loss_function='MAE',iterations=rounds,depth=c['depth'],learning_rate=.025,l2_leaf_reg=30,
            random_seed=seed,task_type='GPU',devices='0',thread_count=4,verbose=False,allow_writing_files=False,border_count=64,bootstrap_type='Bernoulli',subsample=.85)
        net.fit(x,ys);predictions.append(np.mean([net.predict(qx) for qx in qs],0)*scale+center);models.append(net)
    p=clip(task,np.mean(predictions,0));assert np.isfinite(p).all();folder.mkdir(parents=True,exist_ok=True)
    state=dict(signature=sig,config=c,fit=a['fit'],query=a['query'],models=models,columns=columns,center=center,scale=scale)
    joblib.dump(state,folder/'model.joblib',compress=3);np.savez_compressed(path,signature=sig,fit=a['fit'],query=a['query'],p=p)
    old=a['x'][:,:105];new=context(d,task,part)['median'][a['fit']];changed=~np.all((old==new)|(np.isnan(old)&np.isnan(new)),axis=1)
    save(folder/'metadata.json',dict(signature=sig,task=task,part=part,config=c,seconds=time.monotonic()-start,model_sha256=sha(folder/'model.joblib'),predictions_sha256=sha(path),source_target_excluded=True,changed_fit_context_rows=int(changed.sum()),fit_context_rows=len(old),training_replicates=len(xs)))
    emit('fit_done',task=task,part=part,config=c['name'],seconds=time.monotonic()-start,changed_fit_context=int(changed.sum()))
    return p

def replay(d,task,part,c):
    a=design(d,task,part);state=joblib.load(OUT/'tasks'/task/part/c['name']/'model.joblib');assert state['signature']==signature()
    for key in ('fit','query'):np.testing.assert_array_equal(a[key],state[key])
    xs,qs,columns=features(d,a,task,part,c);np.testing.assert_array_equal(columns,state['columns'])
    predictions=[np.mean([net.predict(qx) for qx in qs],0)*state['scale']+state['center'] for net in state['models']]
    return clip(task,np.mean(predictions,0))

def freeze():
    v=dict(signature=signature(),configs=CONFIGS,weights=WEIGHTS,seeds=SEEDS,tasks=TASKS,baseline='v371 local exact CSV; human official51.270',all_fitting_remote=True,external_data=False,test_labels_used=False,
        hypothesis='Fit-only diagnosis shows stereo mixtures in nonisomeric duplicates. Rebuild other-assay exact/neutral keys retaining stereo; compare medians and deterministic4-point empirical integrations, two fixeddepths.',
        identity='Largest fragment and uncharge with stereo preserved for ordinary sanitized molecules. Original input string fallback for two invalid-valence molecules; no silent graph changes.',
        selection='Unique maxmin fold0/1 config/weight frozen before fold2/valid, no fallback after failure.',
        honesty='Current target and mask excluded in every context. Train-only other labels for all validation; final additionally other valid. No query target labels, no early stopping on query.',
        limitations='Historical baseline partial proxy, public labels repeatedly used. Integrated mean over finite contexts need not be optimal marginal conditional median. Stereo repair is an empirical hypothesis, not guaranteed gain.',
        strict='All4positive, developmentalminimum1.2percent, mean1.2percent and both500groupbootstrap90lowerpositive.',experimental='All4positive, explicitly experimental if strict fails.')
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(v))
    else:save(path,v)
