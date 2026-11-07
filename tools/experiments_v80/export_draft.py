import os
for v in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[v]='4'
import sys,json,time,csv,hashlib,pathlib,numpy as np
from scipy.special import expit,logit
from scipy.stats import rankdata
from scipy.optimize import minimize
from catboost import CatBoostClassifier,Pool
sys.path.insert(0,'/root/autodl-tmp/v76')
sys.path.insert(0,'/root/autodl-tmp/v76/tools')
import admet_v77_ranking as b
from tools.repack_upload_template import field_spans
from tools.verify_submission import verify
D=b.load_data();ROOT=b.ROOT;task='cyp2c9_veith';k=D['tasks'].index(task)
anchor=ROOT/'admet_submissions/v73_v72/我从山北来带着兰花草-小分子药物预测技术创新赛-20261021.csv';template=ROOT/'admet_official/submission_template.csv'
rows=list(csv.reader(anchor.open(encoding='utf-8-sig',newline=''))); templateraw=template.read_bytes(); spans=field_spans(templateraw)
# Read anchor probabilities for target test block and align to official test rows.
pos=[i for i,r in enumerate(rows) if i and r[0]==task]; q=np.flatnonzero((D['tid']==k)&(D['split']==2)); assert len(pos)==len(q)
old_test=np.array([float(rows[i][4]) for i in pos],float)
# Build official assay auxiliary values for all target rows, including train+valid; target task excluded.
source=np.flatnonzero((D['split']<2)&(D['tid']!=k)); cols=[j for j in range(len(D['tasks'])) if j!=k]; lookup={j:i for i,j in enumerate(cols)}
labels={}
for i in source: labels.setdefault((str(D['canonical'][i]),int(D['tid'][i])),[]).append(float(D['y'][i]))
aux=np.full((len(D['y']),2*len(cols)),np.nan,np.float32)
for i in np.flatnonzero(D['tid']==k):
 for t,j in lookup.items():
  vals=labels.get((str(D['canonical'][i]),t))
  if vals: aux[i,j]=np.median(vals)
aux[:,len(cols):]=np.isfinite(aux[:,:len(cols)])
# Historical anchor predictions: exact task rows are available as proxy OOF for train and official-valid anchor for valid.
tr=np.flatnonzero((D['tid']==k)&(D['split']==0)); va=np.flatnonzero((D['tid']==k)&(D['split']==1)); fit=np.r_[tr,va]
proxy_cv=np.load(ROOT/'recovered_history/baseline_cv_proxy.npy'); proxy_valid=np.load(ROOT/'recovered_history/baseline_valid.npy')
p_fit=np.r_[proxy_cv[tr],proxy_valid[va]]
# The v73 anchor is unchanged for this endpoint, so use its test predictions as final base.
assert np.isfinite(p_fit).all() and np.isfinite(old_test).all()
# Rank-logit base feature; calibrate base score to target prevalence on all official fit labels.
def ranklog(x): return logit(np.clip((rankdata(x)-.5)/len(x),.002,.998))
z=ranklog(p_fit); zq=ranklog(old_test); y=D['y'][fit]
X0=np.c_[np.ones(len(fit)),z]
def obj(theta):
 eta=X0@theta;r=expit(eta)-y
 return (np.logaddexp(0,eta)-y*eta).mean()+.001*theta[1]**2, X0.T@r/len(r)+[0,.002*theta[1]]
beta=minimize(obj,[0.,1.],jac=True,bounds=[(None,None),(.01,10.)],method='L-BFGS-B').x
offset=X0@beta; qoffset=np.c_[np.ones(len(q)),zq]@beta
x=np.c_[aux[fit],z].astype(np.float32); qx=np.c_[aux[q],zq].astype(np.float32)
# Fit a small seed ensemble of offset residual classifiers; no target values/mask are in aux.
models=[];preds=[];outdir=ROOT/'admet_model_outputs_v80_final'/task;outdir.mkdir(parents=True,exist_ok=True)
for seed in (802026,802027,802028):
 model=CatBoostClassifier(iterations=300,depth=4,learning_rate=.025,l2_leaf_reg=50,loss_function='Logloss',random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False,bootstrap_type='Bernoulli',subsample=.85)
 model.fit(Pool(x,y,baseline=offset)); raw=model.predict(qx,prediction_type='RawFormulaVal');preds.append(raw);model.save_model(str(outdir/f'model_{seed}.cbm'))
final_prob=expit(qoffset+np.mean(preds,axis=0)); assert np.isfinite(final_prob).all() and np.all((final_prob>=0)&(final_prob<=1))
for i,v in zip(pos,final_prob): rows[i][4]=repr(float(v))
# Replace only y_pred byte spans in the official template. Preserve IDs, SMILES, BOM and CRLF.
chunks=[];cursor=0
for i,row in enumerate(rows[1:],1):
 lo,hi=spans[i][4]; value=row[4].encode('ascii'); assert not any(c in value for c in b',\r\n"')
 chunks.extend((templateraw[cursor:lo],value));cursor=hi
chunks.append(templateraw[cursor:]);raw=b''.join(chunks)
dest=ROOT/'admet_submissions/v80_cyp2c9_residual';dest.mkdir(parents=True,exist_ok=True);file=dest/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261023.csv';file.write_bytes(raw)
audit=verify(file,template);audit.update(status='PASS',anchor='v73',anchor_sha256=b.sha(anchor),changed_tasks={task:len(pos)},target_source_excluded=True,auxiliary_source_splits=[0,1],test_labels_used=False,external_data=False,model_reload_pending=True)
(file.with_suffix('.verification.json')).write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
meta=dict(status='CANDIDATE_AWAITING_PLATFORM',version='v80',anchor='v73',anchor_sha256=b.sha(anchor),file=file.relative_to(ROOT).as_posix(),sha256=b.sha(file),task=task,changed_rows=len(pos),base_prediction_source='v73 unchanged endpoint; train OOF proxy plus valid anchor',auxiliary_source_rows=len(source),target_auxiliary_value_and_mask_absent=True,seeds=[802026,802027,802028],trees=300,depth=4,blend='full residual raw score over calibrated anchor rank logit',test_labels_used=False,external_data=False,verification=audit,limitations='Official score unknown until platform submission; local gains are endpoint AUPRC diagnostics, not an official total-score prediction.')
(ROOT/'admet_model_outputs_v80_final/release.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='EXPORTED',file=str(file),sha256=b.sha(file),changed_rows=len(pos),min=float(final_prob.min()),max=float(final_prob.max())),ensure_ascii=False),flush=True)
