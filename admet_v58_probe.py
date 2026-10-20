"""Read-only source/provenance and CUDA audit before the v58 experiment."""
import json
from pathlib import Path
import numpy as np
import torch
import admet_v24_models as b
import admet_v51_run as trees

root=Path(__file__).resolve().parent
b.remote_only()
print(json.dumps(dict(cuda=torch.cuda.get_device_name(),memory=torch.cuda.mem_get_info(),
    transformers=__import__('transformers').__version__,torch=torch.__version__)),flush=True)
d=b.build_data();v,c=trees.baseline(d);f=np.load(b.OUT/'folds.npy')
for k,t in enumerate(d['tasks']):
    tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1))
    print(json.dumps(dict(task=t,metric=d['metrics'][k],n=len(tr),valid=b.metric(d['metrics'][k],d['y'][vi],v[vi]),
        cv=[b.metric(d['metrics'][k],d['y'][ix],c[ix]) for j in range(3) for ix in [tr[f[tr]==j]]])),flush=True)
