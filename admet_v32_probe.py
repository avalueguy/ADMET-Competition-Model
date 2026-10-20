import json, numpy as np
import admet_v24_models as b
from admet_v25_run import v24_valid

b.remote_only(); d=b.build_data()
print('tasks', len(d['tasks']), 'rows', len(d['y']), 'molecules', len(d['smiles']))
print('task train valid test')
for k,t in enumerate(d['tasks']):
    print(t, int(np.sum((d['tid']==k)&(d['split']==0))), int(np.sum((d['tid']==k)&(d['split']==1))), int(np.sum((d['tid']==k)&(d['split']==2))))
can=d['canonical'];
for k,t in enumerate(d['tasks']):
    tr=np.flatnonzero((d['tid']==k)&(d['split']==0)); va=np.flatnonzero((d['tid']==k)&(d['split']==1)); te=np.flatnonzero((d['tid']==k)&(d['split']==2))
    s=set(can[d['mi'][tr]]); qv=can[d['mi'][va]]; qt=can[d['mi'][te]]
    print('dup',t,'valid_exact',sum(x in s for x in qv),'test_exact',sum(x in s for x in qt),'valid_unique',len(set(qv)))
print('baseline valid available', b.OUT/'baseline_valid.npy', (b.OUT/'baseline_valid.npy').exists())
