"""Conservative cross-task transfer challenger.

The target task never contributes its valid/test labels to transfer features.
Only a candidate that improves both official valid metrics over the release is
published. The release directory is immutable.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import joblib, numpy as np, pandas as pd
from rdkit import Chem
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from scipy.stats import spearmanr
from xgboost import XGBClassifier, XGBRegressor
from admet_features_v2 import features
from admet_competition_train import predict_model
from admet_upgrade_v3 import tanimoto_neighbour

REL = {
 'cyp2c9_substrate_carbonmangels':['cyp2c9_veith','cyp2d6_veith','cyp3a4_veith'],
 'cyp2d6_substrate_carbonmangels':['cyp2d6_veith','cyp2c9_veith','cyp3a4_veith'],
 'cyp3a4_substrate_carbonmangels':['cyp3a4_veith','cyp2d6_veith','cyp2c9_veith'],
 'bbb_martins':['lipophilicity_astrazeneca','solubility_aqsoldb','pgp_broccatelli'],
 'caco2_wang':['lipophilicity_astrazeneca','solubility_aqsoldb','ppbr_az'],
 'bioavailability_ma':['bbb_martins','pgp_broccatelli','hia_hou'],
 'dili':['ames','herg','lipophilicity_astrazeneca'],
 'herg':['dili','lipophilicity_astrazeneca'],
 'clearance_hepatocyte_az':['clearance_microsome_az','lipophilicity_astrazeneca'],
 'clearance_microsome_az':['clearance_hepatocyte_az','lipophilicity_astrazeneca'],
 'half_life_obach':['clearance_hepatocyte_az','clearance_microsome_az','vdss_lombardo'],
 'vdss_lombardo':['ppbr_az','lipophilicity_astrazeneca'],
 'ppbr_az':['lipophilicity_astrazeneca','vdss_lombardo'],
 'ld50_zhu':['ames','dili','lipophilicity_astrazeneca'],
 'solubility_aqsoldb':['lipophilicity_astrazeneca','caco2_wang','ppbr_az'],
 'lipophilicity_astrazeneca':['solubility_aqsoldb','caco2_wang','ppbr_az'],
}

def kind(df):
 y=df.Y.to_numpy(float); return 'classification' if set(np.unique(y)).issubset({0,1}) else 'regression'
def mm(k,y,p):
 if k=='classification': return {'auroc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p))}
 r=float(spearmanr(y,p).statistic); return {'mae':float(mean_absolute_error(y,p)),'spearman':r if np.isfinite(r) else 0.}
def score(metric,m): return -m['mae'] if metric=='MAE' else m[metric.lower()]
def dominates(k,a,b):
 return b['auroc']>=a['auroc'] and b['auprc']>=a['auprc'] if k=='classification' else b['mae']<=a['mae'] and b['spearman']>=a['spearman']

def transfer(task, smiles, tasks, cache):
 x,_,_=features(smiles,cache); cols=[]
 for src in REL.get(task,[]):
  if src not in tasks: continue
  df=tasks[src]['train']; sx,_,_=features(df.Drug.tolist(),cache); sy=df.Y.to_numpy(float)
  knn=tanimoto_neighbour(sx,sy,x); cols.append(knn)
  # exact canonical-label column is especially useful for duplicate assays
  lut={}
  for s,v in zip(df.Drug,sy):
   m=Chem.MolFromSmiles(str(s)); c=Chem.MolToSmiles(m,canonical=True) if m else ''
   lut.setdefault(c,[]).append(float(v))
  ex=[]
  for s in smiles:
   m=Chem.MolFromSmiles(str(s)); c=Chem.MolToSmiles(m,canonical=True) if m else ''
   ex.append(np.mean(lut.get(c,[np.mean(sy)])))
  cols.append(np.asarray(ex))
 return x,np.column_stack(cols).astype(np.float32) if cols else np.empty((len(smiles),0),np.float32)

def heads(k):
 if k=='classification':
  return {'xgb':XGBClassifier(n_estimators=500,max_depth=4,learning_rate=.035,min_child_weight=2,subsample=.9,colsample_bytree=.7,reg_lambda=4,tree_method='hist',n_jobs=6,random_state=4201),'extra':ExtraTreesClassifier(n_estimators=400,max_features=.3,class_weight='balanced',n_jobs=6,random_state=4201),'linear':LogisticRegression(C=.2,class_weight='balanced',solver='liblinear',max_iter=2000)}
 return {'xgb':XGBRegressor(n_estimators=550,max_depth=4,learning_rate=.03,min_child_weight=2,subsample=.9,colsample_bytree=.7,reg_lambda=5,tree_method='hist',n_jobs=6,random_state=4201),'extra':ExtraTreesRegressor(n_estimators=400,max_features=.3,n_jobs=6,random_state=4201),'linear':Ridge(alpha=40)}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--tasks',nargs='*'); ap.add_argument('--output-dir',type=Path,default=Path('admet_model_outputs_v4')); a=ap.parse_args()
 root=Path('admet_competition_data/data_benchmark'); release=Path('admet_release'); out=a.output_dir; out.mkdir(exist_ok=True); cache=out/'features'
 tasks={p.name:{s:pd.read_csv(p/f'{s}.csv') for s in ['train','valid','test']} for p in root.iterdir() if (p/'train.csv').exists()}; names=a.tasks or sorted(tasks); rows=[]
 for i,t in enumerate(names,1):
  print(f'[{i}/{len(names)}] {t}',flush=True); f=tasks[t]; k=kind(f['train']); metric=pd.read_csv('admet_official/submission_template.csv').query('task==@t').metric.iloc[0]
  tr,va,te=f['train'],f['valid'],f['test']; y=tr.Y.to_numpy(float); yv=va.Y.to_numpy(float)
  xtr,ttr=transfer(t,tr.Drug.tolist(),tasks,cache); xva,tva=transfer(t,va.Drug.tolist(),tasks,cache); xte,tte=transfer(t,te.Drug.tolist(),tasks,cache)
  xtr=np.nan_to_num(np.hstack([xtr,ttr]), nan=0., posinf=0., neginf=0.); xva=np.nan_to_num(np.hstack([xva,tva]), nan=0., posinf=0., neginf=0.); xte=np.nan_to_num(np.hstack([xte, tte]), nan=0., posinf=0., neginf=0.); pred={}
  for n,m in heads(k).items():
   m.fit(xtr[:,4096:] if n=='linear' else xtr,y); pred[n]=predict_model(m,xva[:,4096:] if n=='linear' else xva,k)
  pred['equal']=np.mean(list(pred.values()),axis=0); base=pd.read_csv(release/t/'validation_predictions.csv').Y_pred.to_numpy(float); bm=mm(k,yv,base); best=('baseline',0.,base,bm)
  for n,p in pred.items():
   for z in [.1,.2,.3,.5,.75,1.]:
    q=(1-z)*base+z*p; qmm=mm(k,yv,q)
    if dominates(k,bm,qmm) and score(metric,qmm)>score(metric,best[3])+1e-8: best=(n,z,q,qmm)
  n,z,q,qmm=best; accept=n!='baseline'; folder=out/t; folder.mkdir(parents=True,exist_ok=True)
  pd.DataFrame({'Y_true':yv,'baseline':base,'selected':q,**pred}).to_csv(folder/'candidate_predictions.csv',index=False)
  if accept:
   allx=np.vstack([xtr,xva]); ally=np.r_[y,yv]; testpred={}
   for h,m in heads(k).items():
    m.fit(allx[:,4096:] if h=='linear' else allx,ally); testpred[h]=predict_model(m,xte[:,4096:] if h=='linear' else xte,k)
   p=testpred[n] if n!='equal' else np.mean(list(testpred.values()),axis=0); old=pd.read_csv(release/t/f'submission_{t}.csv').Y.to_numpy(float); p=(1-z)*old+z*p
   pd.DataFrame({'Drug_ID':te.Drug_ID,'Y':p}).to_csv(folder/f'submission_{t}.csv',index=False); pd.DataFrame({'Drug_ID':va.Drug_ID,'Y_true':yv,'Y_pred':q}).to_csv(folder/'validation_predictions.csv',index=False)
  else:
   for fn in [f'submission_{t}.csv','validation_predictions.csv']: shutil.copy2(release/t/fn,folder/fn)
  row={'task':t,'metric':metric,'accepted':accept,'head':n,'alpha':z}; row.update({'base_'+j:v for j,v in bm.items()}); row.update({'candidate_'+j:v for j,v in qmm.items()}); rows.append(row); print(json.dumps(row),flush=True)
 pd.DataFrame(rows).to_csv(out/'transfer_metrics.csv',index=False)

if __name__=='__main__': main()
