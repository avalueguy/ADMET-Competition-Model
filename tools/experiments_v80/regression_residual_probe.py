import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
import sys,json,pathlib,time,numpy as np
sys.path.insert(0,'/root/autodl-tmp/v76')
import admet_v77_ranking as b
from catboost import CatBoostRegressor
from scipy.optimize import linprog
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import QuantileRegressor, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
D=b.load_data(); f=np.load(b.ROOT/'recovered_history/legacy_v24_folds.npy')
chem=np.load(b.ROOT/'admet_model_outputs_v67/features/official_chemistry.npz')['x']
# descriptors stable intrinsic quantities; all quantile/preprocessing fitted only on fit meta rows
from rdkit.Chem import Descriptors
names=[n for n,fn in Descriptors.descList if n not in {'Ipc','AvgIpc','BalabanJ'}]
select=['MolWt','MolLogP','TPSA','NumHDonors','NumHAcceptors','NumRotatableBonds','RingCount','FractionCSP3','HeavyAtomCount','NHOHCount','NOCount','NumAromaticRings','NumAliphaticRings','NumSaturatedRings','NumHeteroatoms','LabuteASA','BertzCT','MaxPartialCharge','MinPartialCharge','qed']
ix=[names.index(n) for n in select if n in names]
print('features',len(names),chem.shape,[(n,names.index(n)) for n in select if n in names],flush=True)
assert chem.shape[1]-4096==len(names)
res=[]
for k,t in enumerate(D['tasks']):
 if D['metrics'][k]!='MAE' or t=='solubility_aqsoldb':continue
 tr=np.flatnonzero((D['tid']==k)&(D['split']==0)); va=np.flatnonzero((D['tid']==k)&(D['split']==1))
 p=D['proxy_cv'][tr]; vp=D['proxy_valid'][va]; y=D['y'][tr]; vy=D['y'][va]; ff=f[tr]
 features={'p':p[:,None], 'simple':np.c_[p,chem[D['mi'][tr],4096:][:,ix]],'all':np.c_[p,D['views']['physchem'][D['mi'][tr]]]}
 vfeatures={'p':vp[:,None], 'simple':np.c_[vp,chem[D['mi'][va],4096:][:,ix]],'all':np.c_[vp,D['views']['physchem'][D['mi'][va]]]}
 # target residual in original units; fit meta-model on proxies only
 for mode in ['bias','affine','linear','cb_p','cb_simple','cb_all']:
  variants=[0] if mode in ['bias','affine','linear'] else [0,1,2]
  for var in variants:
   z=np.empty(len(tr)); params=[]
   def fit_predict(a,q,x,qx):
    r=y[a]-p[a]
    if mode=='bias':return np.full(len(qx),np.median(r)),{'offset':float(np.median(r))}
    if mode in ['affine','linear']:
     if mode=='affine': xx=p[a,None];qq=qx
     else:xx=x;qq=qx
     net=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),QuantileRegressor(quantile=.5,alpha=.03 if mode=='linear' else .005,solver='highs',solver_options={'threads':4}))
     net.fit(xx,r)
     return net.predict(qq),None
    net=CatBoostRegressor(iterations=[80,160,300][var],depth=2 if var<2 else 3,learning_rate=.025,l2_leaf_reg=50,loss_function='MAE',random_seed=802026,thread_count=4,verbose=False,allow_writing_files=False,bootstrap_type='Bernoulli',subsample=.8)
    net.fit(x,r)
    return net.predict(qx),None
   fam='p' if mode in ['bias','affine','cb_p'] else 'simple' if mode in ['linear','cb_simple'] else 'all'
   for fold in range(3):
    a=ff!=fold;q=ff==fold
    rr,_=fit_predict(a,q,features[fam][a],features[fam][q]); z[q]=p[q]+rr
   a=np.ones(len(tr),bool);rp,_=fit_predict(a,None,features[fam],vfeatures[fam]);vnew=vp+rp
   if t=='ppbr_az':z=z.clip(0,100);vnew=vnew.clip(0,100)
   rec={'task':t,'mode':mode,'variant':var,'fold_gains':[float(mae(y[ff==i],p[ff==i])-mae(y[ff==i],z[ff==i])) for i in range(3)],'valid_gain':float(mae(vy,vp)-mae(vy,vnew)),'base_valid':mae(vy,vp),'new_valid':mae(vy,vnew),'median_delta':float(np.median(rp)), 'delta_q':np.quantile(rp,[0,.1,.5,.9,1]).tolist()}
   print(json.dumps(rec),flush=True);res.append(rec)
pathlib.Path('/tmp/v80_residual_probe.json').write_text(json.dumps(res,indent=2))
