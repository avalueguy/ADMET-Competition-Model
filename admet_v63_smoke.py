import os,sys,time,json,inspect
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
sys.path.insert(0,str(Path('admet_runtime/v63_packages').resolve()))
import numpy as np,torch
from tabicl import TabICLRegressor
import admet_v24_models as b
b.remote_only();d=b.build_data();k=d['tasks'].index('half_life_obach');f=np.load(b.OUT/'folds.npy');tr=np.flatnonzero((d['tid']==k)&(d['split']==0)&(f!=0));q=np.flatnonzero((d['tid']==k)&(d['split']==0)&(f==0));pre=b.Preprocessor().fit(d['desc'][d['mi'][tr]]);x=pre.transform(d['desc']);start=time.time()
m=TabICLRegressor(n_estimators=4,model_path='admet_pretrained/tabicl_v2/tabicl-regressor-v2-20260212.ckpt',allow_auto_download=False,device='cuda',n_jobs=2,batch_size=1,use_fa3=False,offload_mode=False,random_state=631017)
m.fit(x[d['mi'][tr]],d['y'][tr]);p=m.predict(x[d['mi'][q]]);print(json.dumps(dict(rows=len(q),spearman=b.metric('Spearman',d['y'][q],p),seconds=time.time()-start)),flush=True)
print('SAVE',inspect.signature(m.save));print('LOAD',inspect.signature(m.load))
