"""Read-only first-seed fold diagnostics; never selects final release."""
import os,json
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
import numpy as np
from pathlib import Path
import admet_v24_models as b
import admet_v61_run as prior
from admet_v62_run import variants
b.remote_only();d=b.build_data();bv,bc=prior.baseline(d);out=[]
for fold in range(3):
    path=Path(f'admet_model_outputs_v62/cv/fold{fold}/621017/predictions.npz')
    if not path.exists():continue
    z=np.load(path);q=z['query'];p=z['p']
    for k,t in enumerate(d['tasks']):
        active=d['tid'][q]==k;ix=q[active];m=d['metrics'][k];v=variants(p[:,active],m)
        gain=max(b.delta(m,d['y'][ix],bc[ix],b.blend(m,bc[ix],vv,w)) for vv in v for w in [.15,.35,.6,1.])
        out.append(dict(task=t,fold=fold,baseline=b.metric(m,d['y'][ix],bc[ix]),pure_hybrid=b.metric(m,d['y'][ix],v[1]),best_exploratory_blend_gain=gain))
print(json.dumps(out),flush=True)
