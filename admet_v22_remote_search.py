import json, pathlib, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from scipy.stats import spearmanr, rankdata
ROOT=pathlib.Path('.')
tasks=pd.read_csv(ROOT/'admet_official/submission_template.csv',dtype=str).drop_duplicates('task')
metrics=dict(zip(tasks.task,tasks.metric))
# scores: higher utility for all; MAE negative

def util(metric,y,p):
 if metric=='MAE': return -mean_absolute_error(y,p)
 if metric=='Spearman': return float(spearmanr(y,p).statistic)
 if metric=='AUROC': return roc_auc_score(y,p)
 return average_precision_score(y,p)
def tr(x,mode):
 x=np.asarray(x,float)
 if mode=='raw': return x
 return (rankdata(x,method='average')-.5)/len(x)
def pred_sub(v,t):
 fs=list((ROOT/'admet_submissions'/v).glob('*.csv'))
 if not fs:return None
 d=pd.read_csv(fs[0],dtype=str,keep_default_na=False)
 return d.loc[d.task.eq(t),'y_pred'].astype(float).to_numpy()
def valid_df(t):
 # Prefer v19 (same v16 anchor valid) then v16, v15
 for v in ['v19','v16','v15']:
  p=ROOT/f'admet_model_outputs_{v}'/t/'validation_comparison.csv'
  if p.exists(): return pd.read_csv(p),v
 return None,None

def baseline_valid(t):
 for p in [ROOT/'admet_model_outputs_v19'/t/'validation_comparison.csv',ROOT/'admet_model_outputs_v16'/t/'validation_comparison.csv',ROOT/'admet_release_v15'/t/'validation_predictions.csv']:
  if p.exists():
   d=pd.read_csv(p)
   if 'blend' in d: return d.Y_true.to_numpy(float),d.blend.to_numpy(float)
   return d.Y_true.to_numpy(float),d.Y_pred.to_numpy(float)
 return None,None
# full candidate submissions available
versions=['v15','v16','v17','v18','v19','v20','v21']
# use v20 as anchor; candidate old versions only if full files
out=[]
for t,m in metrics.items():
 base=pred_sub('v20',t)
 if base is None: continue
 yv,bv=baseline_valid(t)
 cand=[]
 for v in versions:
  q=pred_sub(v,t)
  if q is not None and len(q)==len(base) and not np.allclose(q,base): cand.append((v,q))
 # validation candidates from exact comparison columns where available; map names to sources as best possible
 valc=[]
 d,vv=valid_df(t)
 if d is not None and yv is not None:
  for c in d.columns:
   if c in ['Y_true','baseline','blend','cross_meta_diagnostic']: continue
   if len(d[c])==len(yv): valc.append((vv+':'+c,d[c].to_numpy(float)))
 # v20/v21 special valid prediction
 for vv in ['v21']:
  p=ROOT/'admet_model_outputs_v21'/t/'finetune_valid.csv'
  if p.exists() and yv is not None:
   z=pd.read_csv(p)
   if len(z)==len(yv):valc.append((vv,z.Y_pred.to_numpy(float)))
 # evaluate candidates with validation plus oof when same source exists
 base_u=util(m,yv,bv) if yv is not None else None
 res=[]
 for name,q in valc:
  if name.startswith(vv+':') and vv in ['v16','v19']:
   pass
  for mode in (['raw'] if m=='MAE' else ['raw','rank']):
   for a in np.arange(.05,1.0,.05):
    qq=(1-a)*tr(bv,mode)+a*tr(q,mode)
    u=util(m,yv,qq)
    res.append((u-base_u,a,mode,name))
 # use direct full test candidates only when validation name belongs to same version or prior submission
 res=sorted(res,key=lambda x:x[0],reverse=(m!='MAE'))
 best=[]
 for r in res:
  name=r[3].split(':')[0]
  # only candidates with matching full test version; v15/v16 etc
  if name in versions and any(v==name for v,_ in cand): best.append(r)
 # also use all candidate full test with a conservative blend as a fallback; rank search without labels unavailable
 if best:
  choice=best[0]
 else: choice=None
 out.append(dict(task=t,metric=m,valid_n=None if yv is None else len(yv),valid_base=None if base_u is None else float(base_u),top=[dict(delta=float(a),alpha=float(b),mode=c,name=d) for a,b,c,d in res[:12]],candidate_versions=[v for v,_ in cand],choice=None if choice is None else dict(delta=float(choice[0]),alpha=float(choice[1]),mode=choice[2],name=choice[3].split(':')[0])))
print(json.dumps(out,indent=2))
(ROOT/'admet_v22_search.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
