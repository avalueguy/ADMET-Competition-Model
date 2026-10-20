import os,json
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import numpy as np,torch
import admet_v24_models as b
b.remote_only();d=b.build_data();f=np.load(b.OUT/'folds.npy')
v=np.load(Path('admet_model_outputs_v60/baseline_valid.npy'));c=np.load(Path('admet_model_outputs_v60/baseline_cv_proxy.npy'))
r=[]
for k,t in enumerate(d['tasks']):
    tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1));te=np.flatnonzero((d['tid']==k)&(d['split']==2));m=d['metrics'][k]
    r.append(dict(task=t,metric=m,rows=[len(tr),len(vi),len(te)],proxy_cv=[b.metric(m,d['y'][ix],c[ix]) for z in range(3) for ix in [tr[f[tr]==z]]],valid=b.metric(m,d['y'][vi],v[vi])))
print(json.dumps(dict(cuda=torch.cuda.get_device_name(0),molecules=len(d['smiles']),tasks=r)),flush=True)
