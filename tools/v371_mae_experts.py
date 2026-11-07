"""Five independent MAE specialists: median kernels, quantile trees, honest local corrections."""
from pathlib import Path
import os
for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ.setdefault(n,'4')
import hashlib,json,time,platform
import joblib,numpy as np,torch
from scipy.linalg import cho_factor,cho_solve
from sklearn.svm import SVR
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
TASKS=('caco2_wang','ld50_zhu','lipophilicity_astrazeneca','ppbr_az','solubility_aqsoldb')
SEEDS=(3711106,3711107)
CONFIGS=(
 dict(name='median_kernel_C3',family='kernel',C=3.,epsilon=.03),
 dict(name='median_kernel_C30',family='kernel',C=30.,epsilon=.03),
 dict(name='quantile_d3',family='tree',depth=3,prior=False),
 dict(name='quantile_d6',family='tree',depth=6,prior=False),
 dict(name='physical_quantile_d3',family='tree',depth=3,prior=True),
 dict(name='physical_quantile_d6',family='tree',depth=6,prior=True),
 dict(name='crossfit_local_median',family='local',ridge=.3,neighbors=32,shrink=.5))
# More weight is admissible for a strong new expert, but only development chooses it.
WEIGHTS=(.2,.45,.75,1.)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(p)
def emit(stage,**x):print(json.dumps(dict(stage=stage,**x)),flush=True)
def signature():return hashlib.sha256((sha(__file__)+sha(ROOT/'data_manifest.json')+json.dumps([CONFIGS,SEEDS,WEIGHTS],sort_keys=True)).encode()).hexdigest()
def setup():
 assert platform.system()=='Linux' and torch.cuda.is_available(),'Fitting on authorized remote CUDA host only'
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
def load():
 d=dict(np.load(ROOT/'official_rows.npz'));d.update(json.loads((ROOT/'data_manifest.json').read_text()))
 f=dict(np.load(ROOT/'expert_features.npz'));d.update(f);d['rg']=d['groups'][d['mi']];return d
def ctrl(task,part):return dict(np.load(ROOT/'controls'/task/(part+'.npz')))
def design(d,task,part):
 a=dict(np.load(ROOT/'design'/task/(part+'.npz')));c=ctrl(task,part)
 for k in ('fit','query'):np.testing.assert_array_equal(a[k],c[k])
 assert np.all(d['tid'][a['source']]!=d['tasks'].index(task)) and np.all(d['split'][a['source']]<(2 if part=='final' else 1))
 if part!='final':
  assert not np.intersect1d(d['rg'][a['fit']],d['rg'][a['query']]).size
  assert not np.intersect1d(d['canonical'][d['mi'][a['fit']]],d['canonical'][d['mi'][a['query']]]).size
 return a
def robust(fit,query):
 fit,query=fit.astype(float),query.astype(float)
 med=np.array([np.median(x[np.isfinite(x)]) if np.isfinite(x).any() else 0. for x in fit.T]);clean=np.where(np.isfinite(fit),fit,med)
 scale=np.subtract(*np.quantile(clean,[.75,.25],axis=0)).clip(.1)
 keep=np.flatnonzero((np.std(clean,axis=0)>.0001)|(np.isfinite(fit).std(0)>0))
 def apply(x):return np.c_[np.clip((np.where(np.isfinite(x),x,med)-med)/scale,-8,8)[:,keep],(~np.isfinite(x[:,keep])).astype(float)]
 return apply(fit),apply(query),dict(median=med,scale=scale,keep=keep)
def rbf(x,q):
 a,b,state=robust(x,q);width=max(1,a.shape[1]);aa=torch.as_tensor(a,dtype=torch.float32,device='cuda');bb=torch.as_tensor(b,dtype=torch.float32,device='cuda')
 def block(z):
  norm=aa.square().sum(1);outputs=[]
  for lo in range(0,len(z),256):
   zz=z[lo:lo+256];dist=(zz.square().sum(1)[:,None]+norm[None]-2*zz@aa.T).clamp_min(0)/width;outputs.append((.5*torch.exp(-dist/.5)+.5*torch.exp(-dist/2.)).cpu().numpy())
  return np.concatenate(outputs)
 return block(aa),block(bb),state
def kernels(d,a):
 fit,q=a['fit'],a['query'];n=len(fit);train=np.zeros((n,n),np.float32);query=np.zeros((len(q),n),np.float32)
 for j in range(6):
  f=d['fingerprints'][:,j*2048:(j+1)*2048]
  xx=torch.as_tensor(f[d['mi'][fit]],device='cuda');qq=torch.as_tensor(f[d['mi'][q]],device='cuda');norm=xx.square().sum(1)
  for z,out in ((xx,train),(qq,query)):
   for lo in range(0,len(z),256):
    zz=z[lo:lo+256];dot=zz@xx.T;tan=(dot/(zz.square().sum(1)[:,None]+norm[None]-dot).clamp_min(1e-8)).clamp(0,1);out[lo:lo+len(zz)]+=(.85/6*(.5*tan+.5*tan.square())).cpu().numpy()
  del xx,qq
 width=d['descriptor'].shape[1]+d['physical'].shape[1]
 # Augmented block contains context followed by shape/physical channels; transform jointly,
 # and add a separately named 10-dimensional physicochemical kernel.
 r,rq,state=rbf(a['x'],a['qx']);train+=.1*r;query+=.1*rq
 p,pq,physical=rbf(d['physical'][d['mi'][fit]],d['physical'][d['mi'][q]])
 train+=.05*p;query+=.05*pq;train=(train+train.T)*.5
 assert np.isfinite(train).all() and np.isfinite(query).all()
 return train,query,dict(augmented=state,physical=physical)
def cached_kernels(d,a,task,part):
 path=OUT/'kernels'/task/(part+'.npz')
 if path.exists():
  z=np.load(path);assert str(z['signature'])==signature();return z['train'],z['query']
 k,q,s=kernels(d,a);path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,signature=signature(),train=k,query=q)
 return k,q
def target(y):
 center=float(np.median(y));scale=max(float(np.subtract(*np.quantile(y,[.75,.25]))),.05)
 return (y-center)/scale,center,scale
def physical_prior(d,fit,q):
 x,qx,state=robust(d['physical'][d['mi'][fit]],d['physical'][d['mi'][q]])
 y,center,scale=target(d['y'][fit]);model=QuantileRegressor(quantile=.5,alpha=.02,solver='highs').fit(x,y)
 return model.predict(x)*scale+center,model.predict(qx)*scale+center,dict(model=model,state=state,center=center,scale=scale)
def tree_design(d,a):
 fit,q=a['fit'],a['query'];raw=d['fingerprints'];var=raw[d['mi'][fit]].astype(float).var(0);columns=np.sort(np.argsort(-var,kind='stable')[:1024])
 x=np.c_[a['x'],raw[d['mi'][fit]][:,columns]];qx=np.c_[a['qx'],raw[d['mi'][q]][:,columns]]
 def apply(z):return np.where(np.isfinite(z),np.clip(z,-1e20,1e20),np.nan).astype(np.float32)
 return apply(x),apply(qx),columns
def weighted_median(values,weights):
 order=np.argsort(values,axis=1,kind='stable');v=np.take_along_axis(values,order,1);w=np.take_along_axis(weights,order,1);ix=(np.cumsum(w,1)<w.sum(1,keepdims=True)*.5).sum(1);return v[np.arange(len(v)),np.minimum(ix,v.shape[1]-1)]
def clip(task,p):return np.clip(p,0,100) if task=='ppbr_az' else p
def predict(d,task,part,c):
 setup();folder=OUT/'tasks'/task/part/c['name'];file=folder/'predictions.npz';sig=signature()
 if file.exists():
  z=np.load(file);assert str(z['signature'])==sig;return z['p']
 start=time.monotonic();a=design(d,task,part);fit,q=a['fit'],a['query'];y=d['y'][fit];yy,center,scale=target(y);folder.mkdir(parents=True,exist_ok=True)
 emit('fit_start',task=task,part=part,config=c['name'],fit=len(fit),query=len(q))
 state=dict(signature=sig,config=c,fit=fit,query=q,center=center,scale=scale)
 if c['family']=='kernel':
  k,kq=cached_kernels(d,a,task,part);net=SVR(kernel='precomputed',C=c['C'],epsilon=c['epsilon'],tol=1e-4,cache_size=1000,max_iter=200000).fit(k,yy)
  assert net.fit_status_==0,'SVR optimizer failed to converge';p=net.predict(kq)*scale+center;state.update(model=net)
 elif c['family']=='tree':
  x,qx,cols=tree_design(d,a);off=np.zeros(len(fit));qoff=np.zeros(len(q));prior=None
  if c['prior']:off,qoff,prior=physical_prior(d,fit,q)
  z,c0,s0=target(y-off);predictions=[];models=[]
  rounds={'caco2_wang':650,'ld50_zhu':1100,'lipophilicity_astrazeneca':1100,'ppbr_az':900,'solubility_aqsoldb':1400}[task]
  for seed in SEEDS:
   net=XGBRegressor(objective='reg:quantileerror',quantile_alpha=.5,n_estimators=rounds,max_depth=c['depth'],learning_rate=.025,min_child_weight=10,reg_lambda=25.,reg_alpha=.2,subsample=.85,colsample_bytree=.8,max_bin=128,tree_method='hist',device='cuda',n_jobs=4,random_state=seed)
   net.fit(x,z,verbose=False);predictions.append(net.predict(qx).astype(float)*s0+c0+qoff);models.append(net)
  p=np.mean(predictions,0);state.update(models=models,columns=cols,prior=prior,target_center=c0,target_scale=s0)
 else:
  k,kq=cached_kernels(d,a,task,part);groups=d['rg'][fit];oof=np.full(len(fit),np.nan);splits=[]
  for tr,hold in GroupKFold(3).split(fit,groups=groups):
   ytr,c0,s0=target(y[tr]);mat=k[np.ix_(tr,tr)].astype(float)+np.eye(len(tr))*c['ridge'];coef=cho_solve(cho_factor(mat),ytr);oof[hold]=k[np.ix_(hold,tr)]@coef*s0+c0;splits.append(dict(train=fit[tr],held=fit[hold]))
  assert np.isfinite(oof).all();residual=y-oof;coef=cho_solve(cho_factor(k.astype(float)+np.eye(len(fit))*c['ridge']),yy);base=kq@coef*scale+center
  count=min(c['neighbors'],len(fit));ix=np.argsort(-kq,axis=1,kind='stable')[:,:count];sim=np.take_along_axis(kq,ix,1);weights=np.maximum(sim,1e-6)**4
  correction=weighted_median(residual[ix],weights);p=base+c['shrink']*correction;state.update(coef=coef,residual=residual,oof=oof,inner_splits=splits)
 p=clip(task,p);assert np.isfinite(p).all();path=folder/'model.joblib';joblib.dump(state,path,compress=3)
 reloaded=joblib.load(path);assert reloaded['signature']==sig
 np.savez_compressed(file,signature=sig,fit=fit,query=q,p=p)
 save(folder/'metadata.json',dict(status='COMPLETE',signature=sig,task=task,part=part,config=c,seconds=time.monotonic()-start,model_sha256=sha(path),predictions_sha256=sha(file),fit=len(fit),query=len(q),source_rows=len(a['source']),source_target_excluded=True))
 emit('fit_done',task=task,part=part,config=c['name'],seconds=time.monotonic()-start)
 return p
def replay(d,task,part,c):
 a=design(d,task,part);folder=OUT/'tasks'/task/part/c['name'];s=joblib.load(folder/'model.joblib');assert s['signature']==signature()
 for key in ('fit','query'):np.testing.assert_array_equal(s[key],a[key])
 if c['family']=='kernel':
  _,kq=cached_kernels(d,a,task,part);p=s['model'].predict(kq)*s['scale']+s['center']
 elif c['family']=='tree':
  _,qx,columns=tree_design(d,a);np.testing.assert_array_equal(columns,s['columns']);off=np.zeros(len(a['query']))
  if s['prior'] is not None:
   st=s['prior'];raw=d['physical'][d['mi'][a['query']]].astype(float);v=st['state'];xx=np.c_[np.clip((np.where(np.isfinite(raw),raw,v['median'])-v['median'])/v['scale'],-8,8)[:,v['keep']],(~np.isfinite(raw[:,v['keep']])).astype(float)];off=st['model'].predict(xx)*st['scale']+st['center']
  p=np.mean([net.predict(qx).astype(float)*s['target_scale']+s['target_center']+off for net in s['models']],0)
 else:
  _,kq=cached_kernels(d,a,task,part);base=kq@s['coef']*s['scale']+s['center'];count=min(c['neighbors'],len(a['fit']));ix=np.argsort(-kq,axis=1,kind='stable')[:,:count];weights=np.maximum(np.take_along_axis(kq,ix,1),1e-6)**4;p=base+c['shrink']*weighted_median(s['residual'][ix],weights)
 return clip(task,p)
def freeze():
 value=dict(signature=signature(),configs=CONFIGS,seeds=SEEDS,weights=WEIGHTS,tasks=TASKS,baseline='v364:51.269',goal='Each of all five MAE tasks reduced by at least10%; goal not achieved until measured.',
  training_host='connect.nmb2.seetacloud.com:25870',external_data=False,external_weights=False,test_labels_used=False,
  selection='Max-min relative MAE gain on fold0/1, unique config/weight frozen before fold2/valid; no fallback.',
  strict='all4positive, developmental minimum>1.2%, four-part mean>1.2%, both500paired-group-bootstrap90 lower bounds>0.',
  experimental='All4positive even if strict CI/practical threshold fails, retain explicit experimental label.',
  architecture='Two exact epsilon-insensitive median kernel penalties; raw/physical-prior two-depth GPU quantile tree experts; crossfit local median innovation with 3 inner group folds.',
  kernel='Six count fingerprint Tanimoto(.5t+.5t²), total.85; augmented descriptor/contextRBF .10; physicalRBF .05.',
  tree='1024 fit-variance selected fingerprint columns plus augmented context; two seeds, original-scale median objective, fixed task budgets.',
  honesty='No historical predictions are fitting inputs. Local correction residuals generated by inner scaffold OOF ref; outerfit-only feature scaling. Query groups excluded from target fit. Other official task observations allowed, target and mask excluded.',
  limitations='Historical baseline OOF partly proxy; public labels long reused. Local improvements do not establish hidden-test/official improvement.')
 p=OUT/'protocol.json'
 if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(value))
 else:save(p,value)
