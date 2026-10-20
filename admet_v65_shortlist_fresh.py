"""Start frozen small-task fresh checks while remaining task evaluation continues."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import json
import torch
import admet_v24_models as b
import admet_v65_run as v
b.remote_only();torch.backends.cuda.matmul.allow_tf32=False;d=b.build_data();report=json.loads((v.OUT/'analysis.json').read_text());short=[r for r in report['tasks'] if r['selection']]
# Writes an independent checkpoint directory and report; main fresh stage can reuse
# predictions only once this process has completed.
records=[]
for r in short:
    k=d['tasks'].index(r['task']);s=r['selection'];m=r['metric'];import numpy as np
    from catboost import CatBoostClassifier,CatBoostRegressor
    allrows=np.flatnonzero((d['tid']==k)&(d['split']<2));groups=np.unique(d['rg'][allrows]);held=np.random.default_rng(v.P['fresh_seed']+k).permutation(groups)[:max(1,int(.25*len(groups)))];active=np.isin(d['rg'][allrows],held);rr,q=allrows[~active],allrows[active]
    pp=v.fit(d,k,rr,q,[0,1],v.OUT/r['task']/'early_fresh',[s['variant']])[s['variant']];xx,state,audit=v.design(d,k,rr,q,[0,1],[s['variant']]);cls=m in ['AUROC','AUPRC'];tf=b.Target(d['y'][rr],m);model=(CatBoostClassifier if cls else CatBoostRegressor)(iterations=650,depth=5,learning_rate=.04,l2_leaf_reg=15,loss_function='Logloss' if cls else ('MAE' if m=='MAE' else 'RMSE'),task_type='CPU',thread_count=2,verbose=False,allow_writing_files=False,random_seed=v.P['fresh_seed']);model.fit(xx['fit'][s['variant']],tf.encode(d['y'][rr]));ctrl=model.predict_proba(xx['query'][s['variant']])[:,1] if cls else model.predict(xx['query'][s['variant']]);ctrl=tf.decode(ctrl) if m=='MAE' else ctrl;merged=b.blend(m,ctrl,pp,s['weight']);gain=b.delta(m,d['y'][q],ctrl,merged);records.append(dict(task=r['task'],gain=gain,rows=len(q),pass_gate=bool(gain>.001)));b.emit('v65_early_fresh',**records[-1])
b.save(v.OUT/'early_fresh.json',dict(status='COMPLETE',signature=v.signature(),tasks=records))
