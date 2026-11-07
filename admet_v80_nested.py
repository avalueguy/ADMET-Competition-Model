import os
for v in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[v]='4'
import sys,json,time,numpy as np,pathlib,joblib
from scipy.special import logit,expit
from scipy.optimize import minimize
from scipy.stats import rankdata
from catboost import CatBoostClassifier,Pool
from sklearn.metrics import average_precision_score as ap
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent));import admet_v77_ranking as b
D=b.load_data();F=np.load(b.ROOT/'recovered_history/legacy_v24_folds.npy');OUT=b.ROOT/'admet_model_outputs_v80_nested';OUT.mkdir(exist_ok=True)
SPEC=json.loads((b.ROOT/'admet_v80_protocol.json').read_text());k=D['tasks'].index(SPEC['task'])
known={}
for i in np.flatnonzero((D['split']==0)&(D['tid']!=k)):
 known.setdefault((str(D['canonical'][i]),int(D['tid'][i])),[]).append(float(D['y'][i]))
cols=[j for j in range(len(D['tasks'])) if j!=k]
aux=np.full((len(D['y']),2*len(cols)),np.nan,np.float32)
for i in np.flatnonzero(D['tid']==k):
 for j,t in enumerate(cols):
  a=known.get((str(D['canonical'][i]),t))
  if a:aux[i,j]=np.median(a)
aux[:,len(cols):]=np.isfinite(aux[:,:len(cols)])

def base_model(tr,q,folder):
 folder.mkdir(parents=True,exist_ok=True);f=folder/'predictions.npz'
 if f.exists():
  z=np.load(f);np.testing.assert_array_equal(z['fit'],tr);np.testing.assert_array_equal(z['query'],q);return z['p'].copy()
 assert not set(D['groups'][tr])&set(D['groups'][q])
 xx=D['views']['multiview'][D['mi'][tr]];qx=D['views']['multiview'][D['mi'][q]]
 net=CatBoostClassifier(iterations=1200,depth=6,learning_rate=.035,l2_leaf_reg=20,loss_function='Logloss',task_type='GPU',devices='0',random_seed=801001,thread_count=4,verbose=False,allow_writing_files=False,border_count=128,bootstrap_type='Bernoulli',subsample=.85,nan_mode='Min')
 start=time.time();net.fit(xx,D['y'][tr]);p=net.predict_proba(qx)[:,1];net.save_model(str(folder/'model.cbm'));np.savez_compressed(f,fit=tr,query=q,p=p)
 print(json.dumps(dict(stage='base',folder=str(folder),nfit=len(tr),nquery=len(q),score=ap(D['y'][q],p),seconds=round(time.time()-start))),flush=True)
 return p

def logitrank(x):return logit(np.clip((rankdata(x)-.5)/len(x),.002,.998))
def residual(tr,q,p,qp,folder):
 folder.mkdir(parents=True,exist_ok=True)
 z=logitrank(p);qz=logitrank(qp);y=D['y'][tr]
 xx=np.c_[np.ones(len(tr)),z]
 def obj(theta):
  eta=xx@theta;r=expit(eta)-y
  return (np.logaddexp(0,eta)-y*eta).mean()+.001*theta[1]**2,xx.T@r/len(r)+[0,.002*theta[1]]
 beta=minimize(obj,[0.,1.],jac=True,bounds=[(None,None),(.01,10.)],method='L-BFGS-B').x
 offset=xx@beta;qoff=np.c_[np.ones(len(q)),qz]@beta
 x=np.c_[aux[tr],z].astype(np.float32);qx=np.c_[aux[q],qz].astype(np.float32);members=[]
 for seed in SPEC['seeds']:
  net=CatBoostClassifier(iterations=SPEC['trees'],depth=SPEC['depth'],learning_rate=SPEC['learning_rate'],l2_leaf_reg=SPEC['l2_leaf_reg'],loss_function='Logloss',random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False,bootstrap_type='Bernoulli',subsample=.85)
  net.fit(Pool(x,y,baseline=offset));pp=qoff+net.predict(qx,prediction_type='RawFormulaVal');net.save_model(str(folder/f'model_{seed}.cbm'));members.append(pp)
 np.savez_compressed(folder/'predictions.npz',fit=tr,query=q,base_fit=p,base_query=qp,p=np.mean(members,axis=0),members=np.stack(members),base_beta=beta,qx=qx,qoff=qoff)
 return np.mean(members,axis=0)
records=[]
for part in ['legacy_fold2','official_valid']:
 active=D['tid']==k
 if part=='legacy_fold2':tr=np.flatnonzero(active&(D['split']==0)&(F!=2));q=np.flatnonzero(active&(D['split']==0)&(F==2))
 else:tr=np.flatnonzero(active&(D['split']==0));q=np.flatnonzero(active&(D['split']==1))
 folder=OUT/part;outer=base_model(tr,q,folder/'outer_base')
 # 3 scaffold folds wholly inside outer fitting scope
 groups=np.unique(D['groups'][tr]);order=np.random.default_rng(801020).permutation(groups);assign={str(g):j%3 for j,g in enumerate(order)};inner=np.array([assign[str(g)] for g in D['groups'][tr]])
 p=np.empty(len(tr))
 for f in [0,1,2]:
  a=inner!=f;z=inner==f;p[z]=base_model(tr[a],tr[z],folder/f'inner_base{f}')
 new=residual(tr,q,p,outer,folder/'residual');gain=ap(D['y'][q],new)-ap(D['y'][q],outer)
 rec=dict(part=part,nfit=len(tr),nquery=len(q),baseline_ap=ap(D['y'][q],outer),candidate_ap=ap(D['y'][q],new),gain=gain,accepted=gain>0,nested=True,target_source_excluded=True,outer_labels_never_used=True)
 records.append(rec);b.save(OUT/'confirmation.json',dict(status='IN_PROGRESS',records=records,frozen_spec=SPEC));print(json.dumps(rec),flush=True)
b.save(OUT/'confirmation.json',dict(status='COMPLETE',accepted=all(r['accepted'] for r in records),records=records,frozen_spec=SPEC))
