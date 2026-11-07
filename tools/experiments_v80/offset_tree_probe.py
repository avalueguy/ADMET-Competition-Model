exec(open('/tmp/v80_offset_probe.py').read().split('records=[]')[0])
from catboost import Pool
from scipy.optimize import minimize
records=[]
for task in TASKS:
 k=D['tasks'].index(task);metric=D['metrics'][k]
 tr=np.flatnonzero((D['tid']==k)&(D['split']==0));va=np.flatnonzero((D['tid']==k)&(D['split']==1));allr=np.r_[tr,va]
 p=D['proxy_cv'][tr];vp=D['proxy_valid'][va];y=D['y'][tr];vy=D['y'][va];ff=legacy[tr]
 cols=[j for j in range(len(D['tasks'])) if j!=k];aux=np.full((len(allr),2*len(cols)),np.nan)
 for j,t in enumerate(cols):
  for z,i in enumerate(allr):
   a=known.get((str(D['canonical'][i]),t))
   if a:aux[z,j]=np.median(a)
 aux[:,len(cols):]=np.isfinite(aux[:,:len(cols)])
 pp=logit(np.clip((rankdata(p)-.5)/len(p),.002,.998));vpp=logit(np.clip((rankdata(vp)-.5)/len(vp),.002,.998))
 for config in [(2,100),(2,300),(3,100),(3,300),(4,100),(4,300)]:
  out=np.empty(len(tr))
  def fitpred(a,query):
   z=np.c_[np.ones(len(tr)),pp]
   def fun0(theta):
    eta=z[a]@theta;r=expit(eta)-y[a]
    return (np.logaddexp(0,eta)-y[a]*eta).mean()+.001*np.square(theta[1]),z[a].T@r/len(r)+[0,.002*theta[1]]
   base=minimize(fun0,[0.,1.],jac=True,bounds=[(None,None),(.01,10.)],method='L-BFGS-B').x
   offset=z@base;voffset=np.c_[np.ones(len(va)),vpp]@base
   # target-assay features wholly absent; no structure means residual learned only from other measured assays
   xx=np.c_[aux[:len(tr)],pp];vx=np.c_[aux[len(tr):],vpp]
   model=CatBoostClassifier(iterations=config[1],depth=config[0],learning_rate=.025,l2_leaf_reg=50,loss_function='Logloss',random_seed=802026,thread_count=4,verbose=False,allow_writing_files=False,bootstrap_type='Bernoulli',subsample=.85)
   model.fit(Pool(xx[a],y[a],baseline=offset[a]))
   return offset+model.predict(xx,prediction_type='RawFormulaVal'),voffset+model.predict(vx,prediction_type='RawFormulaVal')
  for fold in [0,1,2]:
   a=np.flatnonzero(ff!=fold);q=np.flatnonzero(ff==fold);r,v=fitpred(a,q);out[q]=r[q]
  r,v=fitpred(np.arange(len(tr)),None)
  rec=dict(task=task,method='offset_tree',config=config,parts={})
  for part,old,new,yy in [('fold0',p[ff==0],out[ff==0],y[ff==0]),('fold1',p[ff==1],out[ff==1],y[ff==1]),('fold2',p[ff==2],out[ff==2],y[ff==2]),('valid',vp,v,vy)]:
   rec['parts'][part]=dict(base=bb.metric_score(metric,yy,old),expert=bb.metric_score(metric,yy,new),gains={str(w):bb.gain(metric,yy,old,bb.blend(metric,old,new,w)) for w in [.1,.25,.5,.75,1.]})
  print(json.dumps(rec),flush=True);records.append(rec)
b.save(OUT/'tree_probe.json',dict(records=records))
