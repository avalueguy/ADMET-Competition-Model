import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='4'
import sys,json,pathlib,numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
from scipy.stats import rankdata
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier
sys.path.insert(0,'/root/autodl-tmp/v76');import admet_v77_ranking as b
import admet_v67_data as bb
D=b.load_data(); legacy=np.load(b.ROOT/'recovered_history/legacy_v24_folds.npy')
OUT=b.ROOT/'admet_model_outputs_v80_offset';OUT.mkdir(exist_ok=True)
# Fixed target scopes with exact unchanged v56/v57 baseline endpoint support.
TASKS=['cyp2c9_veith','cyp3a4_veith','pgp_broccatelli']
known={}
for i in np.flatnonzero(D['split']==0):
 known.setdefault((str(D['canonical'][i]),int(D['tid'][i])),[]).append(float(D['y'][i]))
records=[]
for task in TASKS:
 k=D['tasks'].index(task);metric=D['metrics'][k]
 tr=np.flatnonzero((D['tid']==k)&(D['split']==0));va=np.flatnonzero((D['tid']==k)&(D['split']==1));allr=np.r_[tr,va]
 p=D['proxy_cv'][tr];vp=D['proxy_valid'][va];y=D['y'][tr];vy=D['y'][va];ff=legacy[tr]
 cols=[j for j in range(len(D['tasks'])) if j!=k]
 aux=np.full((len(allr),2*len(cols)),np.nan)
 for j,t in enumerate(cols):
  for z,i in enumerate(allr):
   a=known.get((str(D['canonical'][i]),t))
   if a:aux[z,j]=np.median(a)
 aux[:,len(cols):]=np.isfinite(aux[:,:len(cols)])
 # rank to stable log-odds; monotonic base calibration fitted on meta-training rows only
 pp=logit(np.clip((rankdata(p)-.5)/len(p),.002,.998));vpp=logit(np.clip((rankdata(vp)-.5)/len(vp),.002,.998))
 A=aux[:len(tr)];VA=aux[len(tr):]
 def design(a):
  med=np.nanmedian(A[a,:len(cols)],axis=0);med=np.nan_to_num(med)
  mean=np.nanmean(A[a,:len(cols)],axis=0);mean=np.nan_to_num(mean)
  sd=np.nanstd(A[a,:len(cols)],axis=0);sd=np.nan_to_num(sd,nan=1.).clip(.05)
  def trans(x):
   z=(np.where(np.isfinite(x[:,:len(cols)]),x[:,:len(cols)],med)-mean)/sd
   z*=x[:,len(cols):]
   return np.c_[z.clip(-5,5),x[:,len(cols):]]
  return trans(A),trans(VA)
 def fit_glm(a,query,alpha,withdesc=False):
  x,vx=design(a);n=len(a)
  # fit monotone calibrated base on train only
  z=np.c_[np.ones(len(tr)),pp]
  def fun0(theta):
   eta=z[a]@theta;r=expit(eta)-y[a]
   return (np.logaddexp(0,eta)-y[a]*eta).mean()+.001*np.square(theta[1]), z[a].T@r/len(r)+[0,.002*theta[1]]
  fit=minimize(fun0,np.array([0.,1.]),jac=True,bounds=[(None,None),(.01,10.)],method='L-BFGS-B')
  offset=z@fit.x;voffset=np.c_[np.ones(len(va)),vpp]@fit.x
  # additive covariates modify calibrated base logits; baseline intercept fixed
  # include intercept small regularized residual to absorb class-prior shifts
  x=np.c_[np.ones(len(tr)),x];vx=np.c_[np.ones(len(va)),vx]
  def fun(theta):
   eta=offset[a]+x[a]@theta;r=expit(eta)-y[a]
   return (np.logaddexp(0,eta)-y[a]*eta).mean()+.5*alpha*np.square(theta).sum(), x[a].T@r/len(r)+alpha*theta
  fit2=minimize(fun,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500})
  return offset+x@fit2.x,voffset+vx@fit2.x
 for alpha in [.01,.1,1.]:
  out=np.empty(len(tr))
  for fold in range(3):
   a=np.flatnonzero(ff!=fold);q=np.flatnonzero(ff==fold);r,v=fit_glm(a,q,alpha);out[q]=r[q]
  r,v=fit_glm(np.arange(len(tr)),None,alpha)
  rec=dict(task=task,method='glm',alpha=alpha,parts={})
  for part,q,old,new,yy in [('fold0',ff==0,p[ff==0],out[ff==0],y[ff==0]),('fold1',ff==1,p[ff==1],out[ff==1],y[ff==1]),('fold2',ff==2,p[ff==2],out[ff==2],y[ff==2]),('valid',None,vp,v,vy)]:
   rec['parts'][part]=dict(base=bb.metric_score(metric,yy,old),expert=bb.metric_score(metric,yy,new),gains={str(w):bb.gain(metric,yy,old,bb.blend(metric,old,new,w)) for w in [.1,.25,.5,.75,1.]})
  print(json.dumps(rec),flush=True);records.append(rec)
b.save(OUT/'probe.json',dict(records=records))
