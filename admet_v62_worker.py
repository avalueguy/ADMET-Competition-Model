"""One disjoint model partition per process; no shared checkpoint writes."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import argparse
import numpy as np,torch
import admet_v24_models as b
import admet_v27_models as graph
from admet_v62_model import OUT,P,fit,signature
p=argparse.ArgumentParser();p.add_argument('partition',choices=['fold1','fold2','valid']);a=p.parse_args();b.remote_only();torch.backends.cuda.matmul.allow_tf32=False;d=b.build_data();g=graph.graphs(d)
if a.partition=='valid':
    q=np.flatnonzero(d['split']==1);rr=np.flatnonzero((d['split']==0)&~np.isin(d['rg'],np.unique(d['rg'][q])));dest=OUT/'valid'
else:
    fold=int(a.partition[-1]);f=np.load(b.OUT/'folds.npy');q=np.flatnonzero(f==fold);rr=np.flatnonzero((f>=0)&(f!=fold));dest=OUT/'cv'/a.partition
for seed in P['seeds']:fit(d,g,rr,q,seed,dest/str(seed))
b.save(OUT/(a.partition+'_worker_completed.json'),dict(status='COMPLETE',signature=signature()))
