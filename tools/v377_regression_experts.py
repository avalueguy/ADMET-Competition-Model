"""Nine task experts with complete, fixed learning curves; remote CUDA only."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(key,'4')
import hashlib
import json
import time
import joblib
import numpy as np
import torch
from xgboost import XGBRegressor
import v371_mae_experts as shared

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
TASKS=('caco2_wang','ld50_zhu','lipophilicity_astrazeneca','ppbr_az','solubility_aqsoldb',
       'clearance_hepatocyte_az','clearance_microsome_az','half_life_obach','vdss_lombardo')
SEEDS=(3771105,3771106)
HORIZONS=(640,1920,3840)
BASES=(dict(name='native_d3',depth=3,warp=False,reg=6.),dict(name='native_d6',depth=6,warp=False,reg=18.),
       dict(name='asinh_d3',depth=3,warp=True,reg=6.))
CONFIGS=tuple(dict(name=b['name']+'_n'+str(n),base=b['name'],rounds=n) for b in BASES for n in HORIZONS)
WEIGHTS=(.1,.25,.5,.75,1.)
sha=shared.sha
save=shared.save
emit=shared.emit


def signature():return hashlib.sha256((sha(__file__)+sha(shared.__file__)+sha(ROOT/'data_manifest.json')+json.dumps([BASES,HORIZONS,SEEDS,WEIGHTS],sort_keys=True)).encode()).hexdigest()
def setup():shared.setup()
def load():
 shared.ROOT=ROOT;return shared.load()
def ctrl(t,p):return shared.ctrl(t,p)
def design(d,t,p):return shared.design(d,t,p)


def arrays(d,a):
 raw=d['fingerprints'];values=raw[d['mi'][a['fit']]].astype(np.float64);variance=values.var(0)
 columns=np.sort(np.argsort(-variance,kind='stable')[:2048])
 def build(rows,context):
  z=np.c_[context,raw[d['mi'][rows]][:,columns]]
  return np.where(np.isfinite(z),np.clip(z,-1e18,1e18),np.nan).astype(np.float32)
 return build(a['fit'],a['x']),build(a['query'],a['qx']),columns


def transform(y,warp):
 center=float(np.median(y));scale=max(float(np.subtract(*np.quantile(y,[.75,.25]))),.05)
 z=(y-center)/scale
 if warp:
  weights=np.sqrt(1+z*z);weights=np.clip(weights/np.median(weights),.25,20.)
  return np.arcsinh(z),center,scale,weights
 return z,center,scale,np.ones(len(y))


def inverse(z,center,scale,warp,task):
 p=(np.sinh(np.clip(z,-20,20)) if warp else z)*scale+center
 if task=='ppbr_az':p=np.clip(p,0,100)
 if task in TASKS[5:]:p=np.maximum(p,0)
 return p


def train_curve(d,task,part,b):
 setup();folder=OUT/'curves'/task/part/b['name'];folder.mkdir(parents=True,exist_ok=True)
 path=folder/'state.joblib';sig=signature();a=design(d,task,part)
 if path.exists():
  state=joblib.load(path);assert state['signature']==sig
  return state
 start=time.monotonic();x,qx,columns=arrays(d,a);y,center,scale,weights=transform(d['y'][a['fit']],b['warp']);models=[]
 for seed in SEEDS:
  net=XGBRegressor(objective='reg:quantileerror',quantile_alpha=.5,n_estimators=max(HORIZONS),max_depth=b['depth'],
   learning_rate=.018,min_child_weight=3,reg_lambda=b['reg'],reg_alpha=.02,subsample=.85,colsample_bytree=.8,
   max_bin=128,tree_method='hist',device='cuda',n_jobs=4,random_state=seed)
  net.fit(x,y,sample_weight=weights,verbose=False);models.append(net)
  emit('seed_curve_done',task=task,part=part,base=b['name'],seed=seed,rounds=max(HORIZONS))
 state=dict(signature=sig,config=b,fit=a['fit'],query=a['query'],columns=columns,center=center,scale=scale,
            weights=weights,models=models,seconds=time.monotonic()-start)
 joblib.dump(state,path,compress=3)
 # All horizons are materialized before any outer development evaluation.
 for n in HORIZONS:
  name=b['name']+'_n'+str(n);dest=OUT/'tasks'/task/part/name;dest.mkdir(parents=True,exist_ok=True)
  preds=[inverse(net.predict(qx,iteration_range=(0,n)).astype(float),center,scale,b['warp'],task) for net in models]
  p=np.mean(preds,0);assert np.isfinite(p).all()
  np.savez_compressed(dest/'predictions.npz',signature=sig,fit=a['fit'],query=a['query'],p=p)
  save(dest/'metadata.json',dict(signature=sig,base=b['name'],rounds=n,model_path=path.relative_to(ROOT).as_posix(),model_sha256=sha(path),prediction_sha256=sha(dest/'predictions.npz')))
 emit('curve_done',task=task,part=part,base=b['name'],seconds=time.monotonic()-start)
 return state


def predict(d,t,p,c):
 path=OUT/'tasks'/t/p/c['name']/'predictions.npz'
 if not path.exists():train_curve(d,t,p,next(b for b in BASES if b['name']==c['base']))
 z=np.load(path);assert str(z['signature'])==signature();return z['p']


def replay(d,t,p,c):
 a=design(d,t,p);s=joblib.load(OUT/'curves'/t/p/c['base']/'state.joblib');assert s['signature']==signature()
 for k in ('fit','query'):np.testing.assert_array_equal(a[k],s[k])
 x,qx,cols=arrays(d,a);np.testing.assert_array_equal(cols,s['columns'])
 pred=[inverse(net.predict(qx,iteration_range=(0,c['rounds'])).astype(float),s['center'],s['scale'],s['config']['warp'],t) for net in s['models']]
 return np.mean(pred,0)


def freeze():
 save(OUT/'protocol.json',dict(signature=signature(),tasks=TASKS,bases=BASES,horizons=HORIZONS,seeds=SEEDS,weights=WEIGHTS,
  goal='Mean of nine per-task relative native-unit MAE reductions >=5%, report five scored-MAE endpoints separately.',
  selection='Entire fixed learning curves complete on fold0/1 before selection. Unique max-min choice. No confirmation fallback.',
  ranking='Four Spearman tasks use calibrated historical score reference for MAE; require positive official rank gains as well to update submission.',
  transform='Asinh weighted quantile loss approximates native absolute error locally; no claim of exact equivalence. Native MAE used for all evaluations.',
  no_early_abandonment='Train all3840trees and both seeds for each base; evaluate640/1920/3840, even if first horizon weak.',
  all_fitting_remote=True,external_data=False,test_labels_used=False,
  limitations='Historical ensemble control partly proxy. Rank calibration is an explicit reference, not recovery of historical physical units. Public labels reused.'))
