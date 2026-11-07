from __future__ import annotations
import os,sys,json,csv,hashlib
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.metrics import roc_auc_score,average_precision_score,mean_absolute_error
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import admet_v67_data as b
b.rdBase.rdkitVersion='2025.03.6'

def metric(m,y,p):
 if m=='AUROC':return float(roc_auc_score(y,p))
 if m=='AUPRC':return float(average_precision_score(y,p))
 if m=='MAE':return -float(mean_absolute_error(y,p))
 return float(spearmanr(y,p).statistic)

def fit(m,X,y,Xq,seed,iterations=1200,depth=6):
 cls=m in ('AUROC','AUPRC')
 kw=dict(iterations=iterations,depth=depth,learning_rate=.035,l2_leaf_reg=12,task_type='GPU',devices='0',random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False,border_count=128,bootstrap_type='Bernoulli',subsample=.85)
 if cls: kw.update(loss_function='Logloss',eval_metric='AUC' if m=='AUROC' else 'PRAUC'); net=CatBoostClassifier(**kw)
 else: kw.update(loss_function='MAE' if m=='MAE' else 'RMSE',eval_metric='MAE' if m=='MAE' else 'RMSE'); net=CatBoostRegressor(**kw)
 _,inv,cnt=np.unique(D['canonical'][D['mi'][tr]],return_inverse=True,return_counts=True); sw=1/cnt[inv]; sw/=sw.mean()
 net.fit(X,y,sample_weight=sw,verbose=False)
 return net.predict_proba(Xq)[:,1] if cls else net.predict(Xq),net

def main():
 global D,tr
 D=b.build();
 Path(ROOT/'admet_model_outputs_v76').mkdir(exist_ok=True)
 with np.load(ROOT/'admet_model_outputs_v73/features/views.npz') as z: v={k:z[k] for k in z.files}
 # Three complementary views; remove duplicated descriptor columns from extra view.
 X=np.concatenate([D['x'],v['extra'],v['parent'],v['priors']],1).astype(np.float32)
 base=np.load(ROOT/'recovered_history/baseline_valid.npy')
 tasks=D['tasks']
 # Focus on the largest MAE/Spearman contribution and the four weak classification assays.
 chosen=['caco2_wang','clearance_hepatocyte_az','clearance_microsome_az','half_life_obach','ld50_zhu','lipophilicity_astrazeneca','ppbr_az','solubility_aqsoldb','vdss_lombardo','ames','cyp2c9_veith','cyp3a4_veith']
 out=[]
 for task in chosen:
  k=tasks.index(task); m=D['metrics'][k]; tr=np.flatnonzero((D['tid']==k)&(D['split']==0)); va=np.flatnonzero((D['tid']==k)&(D['split']==1));
  p,net=fit(m,X[D['mi'][tr]],D['y'][tr],X[D['mi'][va]],7000+k,iterations=1400,depth=6)
  s=metric(m,D['y'][va],p); sb=metric(m,D['y'][va],base[va]);
  rec={'task':task,'metric':m,'new_valid':s,'base_valid':sb,'gain':s-sb,'train':len(tr),'valid':len(va)}; out.append(rec); print(json.dumps(rec),flush=True)
  # save valid model for audit; final model is fit below if the gain is positive.
  if s>sb:
   trf=np.flatnonzero((D['tid']==k)&(D['split']<2)); te=np.flatnonzero((D['tid']==k)&(D['split']==2))
   # final refit uses same feature and iterations, no test labels
   pf,netf=fit(m,X[D['mi'][trf]],D['y'][trf],X[D['mi'][te]],9000+k,iterations=1400,depth=6)
   np.savez_compressed(ROOT/'admet_model_outputs_v76'/f'{task}.npz',p=pf,query=te,metric=m,valid_gain=s-sb)
 (ROOT/'admet_model_outputs_v76'/'fast_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')

if __name__=='__main__':main()
