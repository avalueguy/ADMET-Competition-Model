"""Remote-only monotone target warps for native-metric median experts.

Native and empirical-rank controls overlap v380/v381, now evaluated relative
to complete v382. Asinh SVRs and bounded PPBR logit are explicit additions.
"""
from pathlib import Path
import hashlib,json,time
import joblib,numpy as np
from scipy.special import ndtr,ndtri,expit
from scipy.stats import rankdata
from sklearn.svm import SVR
import v380_histogram_median as geometry
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs';TASKS=geometry.TASKS
WEIGHTS=(.1,.25,.5,.75,1.)
SETTINGS=dict(epsilon=0.,tol=1e-5,max_iter=1000000,ppbr_pseudocount=.5,derivative_weight_cap=20.)
CONFIGS=tuple(dict(name=f'{view}_{mode}_C{int(c)}',view=view,mode=mode,C=c)
 for view in ('sqrt','tanimoto') for mode in ('native','asinh','weighted_asinh','gaussian_rank','bounded_logit')for c in(3.,30.))
sha,save,emit=geometry.sha,geometry.save,geometry.emit
def signature():return hashlib.sha256((sha(__file__)+sha(geometry.__file__)+sha(geometry.base.__file__)+sha(geometry.base.shared.__file__)+sha(ROOT/'data_manifest.json')+json.dumps([CONFIGS,WEIGHTS,SETTINGS],sort_keys=True)).encode()).hexdigest()
def setup():geometry.setup()
def load():geometry.ROOT=ROOT;geometry.base.ROOT=ROOT;geometry.base.shared.ROOT=ROOT;return geometry.load()
def ctrl(t,p):return geometry.ctrl(t,p)
def design(d,t,p):return geometry.design(d,t,p)
def configs(t):return tuple(c for c in CONFIGS if c['mode']!='bounded_logit'or t=='ppbr_az')
def cached(d,t,p):
 path=OUT/'kernels'/t/(p+'.npz');a=design(d,t,p)
 if path.exists():
  z=np.load(path);assert str(z['signature'])==signature()
  for k in('fit','query'):np.testing.assert_array_equal(a[k],z[k])
  return {v:(z[v+'_train'],z[v+'_query'])for v in('sqrt','tanimoto')}
 k,q=geometry.matrices(d,a);tan,tq,_=geometry.base.shared.kernels(d,a)
 values=dict(sqrt=(k['sqrt_polynomial'],q['sqrt_polynomial']),tanimoto=(tan,tq))
 path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,signature=signature(),fit=a['fit'],query=a['query'],**{v+'_'+s:z for v,pair in values.items()for s,z in zip(('train','query'),pair)})
 return values
def target(y,c):
 yy,center,scale=geometry.base.shared.target(y);state=dict(center=center,scale=scale,mode=c['mode']);weights=np.ones(len(y))
 if c['mode']in('asinh','weighted_asinh'):
  transformed=np.arcsinh(yy)
  if c['mode']=='weighted_asinh':weights=np.clip(np.sqrt(1+yy*yy),1/SETTINGS['derivative_weight_cap'],SETTINGS['derivative_weight_cap']);weights/=np.median(weights)
 elif c['mode']=='gaussian_rank':transformed=ndtri((rankdata(y,method='average')-.5)/len(y));state['training_y']=y.copy()
 elif c['mode']=='bounded_logit':
  assert y.min()>=0 and y.max()<=100
  transformed=np.log((y+SETTINGS['ppbr_pseudocount'])/(100-y+SETTINGS['ppbr_pseudocount']))
  state['warp_center']=float(np.median(transformed));state['warp_scale']=max(float(np.subtract(*np.quantile(transformed,[.75,.25]))),.05)
  transformed=(transformed-state['warp_center'])/state['warp_scale']
 else:transformed=yy
 return transformed,weights,state
def native(z,s,t):
 mode=s['mode']
 if mode in('asinh','weighted_asinh'):p=np.sinh(np.clip(z,-20,20))*s['scale']+s['center']
 elif mode=='gaussian_rank':p=np.quantile(s['training_y'],ndtr(z).clip(0,1),method='linear')
 elif mode=='bounded_logit':p=(100+2*SETTINGS['ppbr_pseudocount'])*expit(z*s['warp_scale']+s['warp_center'])-SETTINGS['ppbr_pseudocount']
 else:p=z*s['scale']+s['center']
 return np.clip(p,0,100)if t=='ppbr_az'else np.maximum(p,0)if t in TASKS[5:]else p
def predict(d,t,p,c):
 setup();folder=OUT/'tasks'/t/p/c['name'];path=folder/'predictions.npz'
 if path.exists():
  z=np.load(path);assert str(z['signature'])==signature();return z['p']
 a=design(d,t,p);k,q=cached(d,t,p)[c['view']];yy,weights,s=target(d['y'][a['fit']],c);started=time.monotonic()
 net=SVR(kernel='precomputed',C=c['C'],epsilon=0.,tol=SETTINGS['tol'],max_iter=SETTINGS['max_iter'],cache_size=1024).fit(k,yy,sample_weight=weights)
 assert net.fit_status_==0,'Full convex solve required, no partial-result screening'
 s.update(signature=signature(),config=c,fit=a['fit'],query=a['query'],model=net,weights=weights,iterations=int(net.n_iter_))
 pred=native(net.predict(q),s,t);assert np.isfinite(pred).all();folder.mkdir(parents=True,exist_ok=True)
 joblib.dump(s,folder/'state.joblib',compress=3);np.savez_compressed(path,signature=signature(),fit=a['fit'],query=a['query'],p=pred)
 save(folder/'metadata.json',dict(signature=signature(),iterations=int(net.n_iter_),seconds=time.monotonic()-started,converged=True,model_sha256=sha(folder/'state.joblib'),prediction_sha256=sha(path)))
 emit('expert_done',task=t,part=p,config=c['name'],iterations=int(net.n_iter_));return pred
def replay(d,t,p,c):
 a=design(d,t,p);s=joblib.load(OUT/'tasks'/t/p/c['name']/'state.joblib');assert s['signature']==signature()
 for k in('fit','query'):np.testing.assert_array_equal(a[k],s[k])
 _,q=cached(d,t,p)[c['view']];return native(s['model'].predict(q),s,t)
def freeze():
 save(OUT/'protocol.json',dict(signature=signature(),baseline='complete v382:51.299',tasks=TASKS,configs=CONFIGS,weights=WEIGHTS,settings=SETTINGS,
  budget='Nine endpoints, two partitions, two kernels, two C, four transforms; PPBR has an additional bounded-logit transform:296 converged development fits.',
  development_fits=296,all_fitting_remote=True,no_external_data=True,no_external_weights=True,
  novelty='Asinh kernels and pseudocount bounded PPBR logit extend existing native/rank controls; not claiming target warping itself is new.',
  theory='A strictly increasing target transform preserves the unconstrained population conditional median, but finite regularized SVR predictions differ. Weighted asinh is only a local derivative compensation approximation and changes the finite objective.',
  selection='Complete all296 fits before maxmin fold0/1 unique config/weight freeze. Confirm only frozen choice on fold2/valid, no fallback. Spearman tasks require both MAE and rank gains.',
  limitations='Reused public data, legacy proxy controls; four rank MAE references calibrated. No official score estimate.'))
