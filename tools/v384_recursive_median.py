"""Remote-only diagonal recursive feature metrics with absolute-loss kernels.

Not a reproduction of a published RFM benchmark: this bounded variant uses
epsilon-zero SVR, damped diagonal gradient moments and scaffold-held selection.
"""
from pathlib import Path
import hashlib,json,time
import joblib,numpy as np,torch
from sklearn.svm import SVR
import v377_regression_experts as base
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'
TASKS=base.TASKS
BASES=tuple(dict(name=f'laplace_C{int(c)}',C=c,bandwidth=1.) for c in (3.,30.))
HORIZONS=(0,2,5,8)
CONFIGS=tuple(dict(name=b['name']+'_r'+str(r),base=b['name'],round=r) for b in BASES for r in HORIZONS)
WEIGHTS=(.1,.25,.5,.75,1.)
SETTINGS=dict(fingerprint_columns_per_view=128,context_columns=256,epsilon=0.,tol=1e-5,
 max_iter=1000000,metric_damping=.5,uniform_shrink=.2,metric_min=.05,metric_max=20.,gradient_rows=512)
sha,save,emit=base.sha,base.save,base.emit
def signature():
 return hashlib.sha256((sha(__file__)+sha(base.__file__)+sha(base.shared.__file__)+sha(ROOT/'data_manifest.json')+json.dumps([BASES,HORIZONS,WEIGHTS,SETTINGS],sort_keys=True)).encode()).hexdigest()
def setup():base.setup()
def load():base.ROOT=ROOT;base.shared.ROOT=ROOT;return base.load()
def ctrl(t,p):return base.ctrl(t,p)
def design(d,t,p):return base.design(d,t,p)
def configs(t):return CONFIGS
def hist(raw):
 x=np.maximum(np.expm1(raw.astype(np.float64)),0);return np.sqrt(x/x.sum(1,keepdims=True).clip(1.))
def make_arrays(d,a,state=None):
 fi,qi=d['mi'][a['fit']],d['mi'][a['query']]
 parts=[];qparts=[];blocks=[]
 for j in range(6):
  raw=d['fingerprints'][:,j*2048:(j+1)*2048];f,q=hist(raw[fi]),hist(raw[qi])
  if state is None:
   cols=np.sort(np.argsort(-f.var(0),kind='stable')[:SETTINGS['fingerprint_columns_per_view']]);center=f[:,cols].mean(0)
   scale=max(float(np.sqrt(np.square(f[:,cols]-center).sum(1).mean())),1e-6)
   s=dict(columns=cols,center=center,scale=scale)
  else:s=state['blocks'][j];cols=s['columns']
  parts.append((f[:,cols]-s['center'])/s['scale']*np.sqrt(.8/6))
  qparts.append((q[:,cols]-s['center'])/s['scale']*np.sqrt(.8/6));blocks.append(s)
 if state is None:
  f,q,robust=base.shared.robust(a['x'],a['qx'])
  cols=np.sort(np.argsort(-f.var(0),kind='stable')[:SETTINGS['context_columns']]);center=f[:,cols].mean(0)
  scale=max(float(np.sqrt(np.square(f[:,cols]-center).sum(1).mean())),1e-6)
  numeric=dict(robust=robust,columns=cols,center=center,scale=scale)
 else:
  numeric=state['numeric'];r=numeric['robust']
  def apply(x):
   return np.c_[np.clip((np.where(np.isfinite(x),x,r['median'])-r['median'])/r['scale'],-8,8)[:,r['keep']],(~np.isfinite(x[:,r['keep']])).astype(float)]
  f,q=apply(a['x']),apply(a['qx']);cols=numeric['columns']
 parts.append((f[:,cols]-numeric['center'])/numeric['scale']*np.sqrt(.2))
 qparts.append((q[:,cols]-numeric['center'])/numeric['scale']*np.sqrt(.2))
 x,qx=np.column_stack(parts).astype(np.float32),np.column_stack(qparts).astype(np.float32)
 assert np.isfinite(x).all() and np.isfinite(qx).all()
 return x,qx,dict(blocks=blocks,numeric=numeric)
def kernel(x,ref,metric,bandwidth):
 ref=torch.as_tensor(ref,device='cuda');metric=torch.as_tensor(metric,dtype=torch.float32,device='cuda')
 weighted=ref*metric;norm=(weighted*ref).sum(1);out=[]
 for lo in range(0,len(x),256):
  z=torch.as_tensor(x[lo:lo+256],device='cuda');dist=((z*z*metric).sum(1)[:,None]+norm[None]-2*z@weighted.T).clamp_min(0).sqrt()
  out.append(torch.exp(-dist/bandwidth).cpu().numpy())
 return np.concatenate(out)
def gradient_moment(x,model,metric,bw):
 # Deterministic fit-only subset. Gradient is with respect to the original
 # standardized inputs, before multiplication by the diagonal metric.
 ix=np.linspace(0,len(x)-1,min(len(x),SETTINGS['gradient_rows']),dtype=int)
 support=x[model.support_];alpha=torch.as_tensor(model.dual_coef_[0],dtype=torch.float32,device='cuda')
 ref=torch.as_tensor(support,device='cuda');m=torch.as_tensor(metric,dtype=torch.float32,device='cuda')
 weighted=ref*m;norm=(ref*weighted).sum(1);acc=torch.zeros(x.shape[1],device='cuda')
 for lo in range(0,len(ix),128):
  z=torch.as_tensor(x[ix[lo:lo+128]],device='cuda');dist=((z*z*m).sum(1)[:,None]+norm[None]-2*z@weighted.T).clamp_min(0).sqrt()
  ratio=torch.where(dist>1e-5,torch.exp(-dist/bw)/dist.clamp_min(1e-5),torch.zeros_like(dist))*alpha[None]
  grad=-(z*ratio.sum(1)[:,None]-ratio@ref)*m/bw
  acc+=grad.square().sum(0)
 g=(acc/len(ix)).cpu().numpy().astype(float);assert np.isfinite(g).all()
 g=g/max(g.mean(),1e-12)
 new=(1-SETTINGS['uniform_shrink'])*g+SETTINGS['uniform_shrink']
 new=SETTINGS['metric_damping']*metric+(1-SETTINGS['metric_damping'])*new
 new=np.clip(new,SETTINGS['metric_min'],SETTINGS['metric_max']);return new/new.mean()
def native(z,s,t):
 p=z*s['scale']+s['center']
 return np.clip(p,0,100) if t=='ppbr_az' else np.maximum(p,0) if t in TASKS[5:] else p
def train_curve(d,t,p,b):
 setup();folder=OUT/'curves'/t/p/b['name'];path=folder/'state.joblib';a=design(d,t,p)
 if path.exists():
  s=joblib.load(path);assert s['signature']==signature();return s
 started=time.monotonic();x,qx,transform=make_arrays(d,a)
 y,center,scale=base.shared.target(d['y'][a['fit']]);metric=np.ones(x.shape[1]);snapshots=[];history=[]
 for r in range(max(HORIZONS)+1):
  k=kernel(x,x,metric,b['bandwidth']);k=(k+k.T)*.5;np.fill_diagonal(k,1.)
  net=SVR(kernel='precomputed',C=b['C'],epsilon=0.,tol=SETTINGS['tol'],max_iter=SETTINGS['max_iter'],cache_size=1024).fit(k,y)
  assert net.fit_status_==0,'Every SVR must converge; no early partial candidate'
  history.append(dict(round=r,iterations=int(net.n_iter_),metric_min=float(metric.min()),metric_max=float(metric.max()),training_MAE=float(np.abs(y-net.predict(k)).mean())))
  if r in HORIZONS:snapshots.append(dict(round=r,model=net,metric=metric.copy()))
  emit('round_done',task=t,part=p,base=b['name'],round=r,iterations=int(net.n_iter_))
  if r<max(HORIZONS):metric=gradient_moment(x,net,metric,b['bandwidth'])
 s=dict(signature=signature(),base=b,fit=a['fit'],query=a['query'],transform=transform,center=center,scale=scale,snapshots=snapshots,history=history)
 folder.mkdir(parents=True,exist_ok=True);joblib.dump(s,path,compress=3)
 for snapshot in snapshots:
  dest=OUT/'tasks'/t/p/(b['name']+'_r'+str(snapshot['round']));dest.mkdir(parents=True,exist_ok=True)
  q=kernel(qx,x,snapshot['metric'],b['bandwidth']);prediction=native(snapshot['model'].predict(q),s,t)
  np.savez_compressed(dest/'predictions.npz',signature=signature(),fit=a['fit'],query=a['query'],p=prediction)
  save(dest/'metadata.json',dict(model_path=path.relative_to(ROOT).as_posix(),model_sha256=sha(path),prediction_sha256=sha(dest/'predictions.npz')))
 save(folder/'metadata.json',dict(signature=signature(),model_sha256=sha(path),history=history,seconds=time.monotonic()-started))
 emit('curve_done',task=t,part=p,base=b['name'],seconds=time.monotonic()-started);return s
def predict(d,t,p,c):
 path=OUT/'tasks'/t/p/c['name']/'predictions.npz'
 if not path.exists():train_curve(d,t,p,next(b for b in BASES if b['name']==c['base']))
 z=np.load(path);assert str(z['signature'])==signature();return z['p']
def replay(d,t,p,c):
 a=design(d,t,p);s=joblib.load(OUT/'curves'/t/p/c['base']/'state.joblib');assert s['signature']==signature()
 for k in ('fit','query'):np.testing.assert_array_equal(a[k],s[k])
 x,qx,_=make_arrays(d,a,s['transform']);z=next(z for z in s['snapshots'] if z['round']==c['round'])
 return native(z['model'].predict(kernel(qx,x,z['metric'],s['base']['bandwidth'])),s,t)
def freeze():
 save(OUT/'protocol.json',dict(signature=signature(),baseline='complete v382, official user score51.299',tasks=TASKS,bases=BASES,horizons=HORIZONS,weights=WEIGHTS,settings=SETTINGS,
  all_fitting_remote=True,no_external_data=True,no_external_weights=True,
  method='Diagonal recursive kernel metric: damped average squared SVR input gradients, absolute loss on native labels; fit-only selected histogram/context inputs.',
  ablation='Round0 fixed isotropic metric and rounds2/5/8 learned metric. Two C values, all nine fitted rounds, no early abandonment.',
  budget='9tasks x2 development partitions x2C x9 converged solves =324 development fits. Four saved horizons per curve. Confirmation/final only frozen developmental winners.',
  selection='Unique maxmin fold0/1 config and blend weight, frozen before fold2/valid. No confirmation fallback. Rankings must improve Spearman and MAE.',
  limitations='Not an exact reproduction of published RFM. Diagonal metric can overfit. Reused public partitions and legacy proxy controls; rank MAE references calibrated. No hidden-test gain guarantee.'))
