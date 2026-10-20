"""Benchmark a small-task CPU tree on the remote host; never changes selection."""
import time
import numpy as np
from catboost import CatBoostClassifier
import admet_v24_models as b
import admet_v33_run as r
b.remote_only();d=b.build_data();k=d['tasks'].index('cyp2d6_substrate_carbonmangels')
rows=np.flatnonzero((d['tid']==k)&(d['split']==0));q=np.flatnonzero((d['tid']==k)&(d['split']==1))
x,_,_,tf,_=r.features(d,k,rows,q,[0],r.other_cache(d,[0]))
start=time.time();m=CatBoostClassifier(iterations=800,depth=7,learning_rate=.04,l2_leaf_reg=14,
    random_seed=330612,loss_function='Logloss',task_type='CPU',thread_count=3,border_count=96,
    allow_writing_files=False,verbose=False)
m.fit(x['fit']['trees'][1],tf.encode(d['y'][rows]));print('REMOTE_CPU_TREE_SECONDS',time.time()-start,flush=True)
