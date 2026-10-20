import os,sys
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
sys.path.insert(0,str(Path('admet_runtime/v63_packages').resolve()))
import numpy as np,torch,json
from tabicl import TabICLClassifier
import admet_v24_models as b
import admet_v65_run as v
b.remote_only();d=b.build_data();k=d['tasks'].index('cyp2d6_veith');f=np.load(b.OUT/'folds.npy');tr=np.flatnonzero((d['tid']==k)&(d['split']==0));rr=tr[f[tr]!=0];q=tr[f[tr]==0];xx,_,_=v.design(d,k,rr,q,[0],[0]);file=Path('admet_model_outputs_v65/cyp2d6_veith/fold0/family0.pt');a=TabICLClassifier.load(file,device='cuda')
X=xx['query'][0][:320]
for amp in [True,False]:
    a.use_amp=amp;a.use_fa3=False
    a.inference_config={key:dict(safety_factor=0.0,min_batch_size=size,use_async=False,offload='cpu') for key,size in [('COL_CONFIG',8),('ROW_CONFIG',64),('ICL_CONFIG',1)]}
    a._build_inference_config();torch.backends.cuda.matmul.allow_tf32=False
    p=a.predict_proba(X)[:,1];r=a.predict_proba(X)[:,1]
    temp=Path('admet_model_outputs_v65/fixed_precision_probe.pt');a.save(temp,save_model_weights=False,save_training_data=True,save_kv_cache=False)
    del a;import gc;gc.collect();torch.cuda.empty_cache()
    a=TabICLClassifier.load(temp,device='cuda');q=a.predict_proba(X)[:,1]
    print(json.dumps(dict(amp=amp,reload_max=float(np.max(np.abs(p-q))),repeat_max=float(np.max(np.abs(p-r))),fixed_batches=True)),flush=True)
