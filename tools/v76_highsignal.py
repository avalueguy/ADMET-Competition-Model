from __future__ import annotations
import os,sys,json,csv
from pathlib import Path
import numpy as np
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.metrics import roc_auc_score,average_precision_score,mean_absolute_error
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));import admet_v67_data as b
b.rdBase.rdkitVersion='2025.03.6'

def score(m,y,p):
 if m=='AUROC': return roc_auc_score(y,p)
 if m=='AUPRC': return average_precision_score(y,p)
 if m=='MAE': return -mean_absolute_error(y,p)
 return spearmanr(y,p).statistic

def fit(m,X,y,Xq,seed,it):
 cls=m in ('AUROC','AUPRC'); kw=dict(iterations=it,depth=5,learning_rate=.03,l2_leaf_reg=18,random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False,border_count=128,bootstrap_type='Bernoulli',subsample=.9,task_type='GPU',devices='0')
 if cls: kw.update(loss_function='Logloss',eval_metric='AUC' if m=='AUROC' else 'PRAUC'); net=CatBoostClassifier(**kw)
 else: kw.update(loss_function='MAE' if m=='MAE' else 'RMSE',eval_metric='MAE' if m=='MAE' else 'RMSE'); net=CatBoostRegressor(**kw)
 net.fit(X,y,verbose=False); return net.predict_proba(Xq)[:,1] if cls else net.predict(Xq)

def main():
 d=b.build(); out=ROOT/'admet_model_outputs_v76';out.mkdir(exist_ok=True); baseline=np.load(ROOT/'recovered_history/baseline_valid.npy')
 with np.load(ROOT/'admet_model_outputs_v73/features/views.npz') as z:v={k:z[k] for k in z.files}
 # Use the v73 multiview feature recipe, but calibrate on each task's rank/metric
 # through repeated seeds and median/mean aggregation. This targets the high-signal
 # classification endpoints where v73's single seed is unstable.
 X=np.concatenate([d['x'][:,:4096],v['extra'],d['x'][:,4096:],v['parent'],v['priors']],1).astype(np.float32)
 choices=['ames','dili','cyp2d6_veith','cyp2c9_veith','cyp3a4_veith','pgp_broccatelli','bbb_martins','bioavailability_ma']
 results=[]
 for task in choices:
  k=d['tasks'].index(task);m=d['metrics'][k];tr=np.flatnonzero((d['tid']==k)&(d['split']==0));va=np.flatnonzero((d['tid']==k)&(d['split']==1)); ps=[]
  for s in range(4): ps.append(fit(m,X[d['mi'][tr]],d['y'][tr],X[d['mi'][va]],76000+k*19+s,1800))
  p=np.mean(ps,axis=0); results.append({'task':task,'metric':m,'score':float(score(m,d['y'][va],p)),'baseline':float(score(m,d['y'][va],baseline[va])),'gain':float(score(m,d['y'][va],p)-score(m,d['y'][va],baseline[va]))}); print(json.dumps(results[-1]),flush=True)
  trf=np.flatnonzero((d['tid']==k)&(d['split']<2));te=np.flatnonzero((d['tid']==k)&(d['split']==2)); fs=[]
  for s in range(4): fs.append(fit(m,X[d['mi'][trf]],d['y'][trf],X[d['mi'][te]],77000+k*19+s,1800))
  np.savez_compressed(out/f'{task}_ensemble.npz',p=np.mean(fs,axis=0),query=te,metric=m,valid_gain=results[-1]['gain'])
 (out/'highsignal_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')

if __name__=='__main__':main()
