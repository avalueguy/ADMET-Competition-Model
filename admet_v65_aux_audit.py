import os,json
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import numpy as np
import admet_v24_models as b
from admet_v31_run import auxiliary
b.remote_only();d=b.build_data();results=[]
for k,t in enumerate(d['tasks']):
    aux,cov,meta=auxiliary(d,[0,1],k);parts=[]
    for split in [0,1,2]:
        ix=np.flatnonzero((d['tid']==k)&(d['split']==split));nt=len(d['tasks'])-1;columns=[j for j in range(len(d['tasks'])) if j!=k];top=[(d['tasks'][columns[j]],float(np.mean(aux[ix,nt+j]>0))) for j in range(nt)];top=sorted(top,key=lambda x:-x[1])[:4];parts.append(dict(split=split,rows=len(ix),coverage=float(np.mean(cov[ix]>0)),top=top))
    results.append(dict(task=t,parts=parts))
print(json.dumps(results),flush=True)
