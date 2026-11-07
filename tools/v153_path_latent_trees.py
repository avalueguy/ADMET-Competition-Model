"""Fit-only latent path representations and native multitarget boosted trees."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.decomposition import TruncatedSVD
from catboost import CatBoostClassifier,CatBoostRegressor
import joblib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v146_current_baseline as v
import v147_path_features as features
import v148_experiment_runner as runner
b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v153';SEEDS=(1534026,1534027)
CONFIGS=(dict(depth=5,trees=1000),)
METHOD='Fit-only logcount atom-pair/torsion/path SVD96 per view plus supervised prevalent fragments and augmented chemistry/context; CPU CatBoost. Spearman jointly log, rank and ordinal target regression.'
def sig():return hashlib.sha256((b.sha(__file__)+v.sig()+features.sig()+b.sha(runner.__file__)).encode()).hexdigest()
def key(c):return 'depth'+str(c['depth'])
def design(d,k,part):
    fit,q,x,qx,source=v.old.features(d,k,part,'augmented');folder=OUT/'features'/d['tasks'][k]/part;folder.mkdir(parents=True,exist_ok=True);p=folder/'features.npz'
    if p.exists():
        z=np.load(p);assert str(z['signature'])==sig();np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['query'],q);return fit,q,z['x'],z['qx'],source
    pieces=[x];qq=[qx]
    for j,raw in enumerate(d['path_fingerprints']):
        xx=raw[d['mi'][fit]].astype(float);zz=raw[d['mi'][q]].astype(float);xx.data=np.log1p(xx.data);zz.data=np.log1p(zz.data)
        cols=np.flatnonzero(np.asarray((xx>0).sum(0)).ravel()>=3)
        # Counts and SVD statistics depend only on fitting structures.
        xx,zz=xx[:,cols],zz[:,cols];n=min(96,min(xx.shape)-1)
        svd=TruncatedSVD(n_components=n,n_iter=5,random_state=1534026+j);a=svd.fit_transform(xx);z=svd.transform(zz);joblib.dump(dict(model=svd,columns=cols),folder/f'projection{j}.joblib',compress=3)
        pieces.append(a);qq.append(z)
        # Prevalence-selected dense fragments complement low-rank geometry.
        freq=np.asarray((xx>0).sum(0)).ravel();dense=np.argsort(freq,kind='stable')[-min(256,len(freq)):]
        pieces.append(xx[:,dense].toarray());qq.append(zz[:,dense].toarray());np.save(folder/f'dense_columns{j}.npy',dense)
    x=np.c_[*pieces].astype(np.float32);qx=np.c_[*qq].astype(np.float32);x[~np.isfinite(x)]=np.nan;qx[~np.isfinite(qx)]=np.nan
    np.savez_compressed(p,signature=sig(),fit=fit,query=q,x=x,qx=qx);return fit,q,x,qx,source
def predict(d,k,part,c):
    folder=OUT/'models'/d['tasks'][k]/part/key(c);file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);assert str(z['signature'])==sig();np.testing.assert_array_equal(z['query'],v.split(d,k,part)[1]);return z['p'].copy()
    fit,q,x,qx,source=design(d,k,part);m=d['metrics'][k];y=d['y'][fit];cls=m in ('AUROC','AUPRC');center=float(np.median(y)) if m=='MAE' else 0.;scale=max(float(np.subtract(*np.quantile(y,[.75,.25]))),.05) if m=='MAE' else 1.
    if m=='Spearman':
        thresholds=np.unique(np.quantile(y,[.15,.3,.5,.7,.85]));target=np.c_[r83.ranknorm(y),np.log1p(np.maximum(y,0)),(y[:,None]>thresholds).astype(float)];tc=target.mean(0);ts=target.std(0).clip(.1);target=(target-tc)/ts
    else:target=y if cls else (y-center)/scale;tc=np.array(0.);ts=np.array(1.)
    folder.mkdir(parents=True,exist_ok=True);members=[];start=time.monotonic()
    for seed in SEEDS:
        model=(CatBoostClassifier if cls else CatBoostRegressor)(iterations=c['trees'],depth=c['depth'],learning_rate=.03,l2_leaf_reg=35,random_seed=seed,thread_count=2,verbose=False,allow_writing_files=False,
            bootstrap_type='Bernoulli',subsample=.85,border_count=64,loss_function='Logloss' if cls else 'MAE' if m=='MAE' else 'MultiRMSE')
        model.fit(x,target);raw=model.predict(qx,prediction_type='RawFormulaVal');mp=folder/f'model_{seed}.cbm';model.save_model(str(mp));replay=CatBoostClassifier() if cls else CatBoostRegressor();replay.load_model(str(mp));np.testing.assert_array_equal(raw,replay.predict(qx,prediction_type='RawFormulaVal'))
        if m=='Spearman':raw=np.mean([r83.rankdata(z)/len(z) for z in raw.T],axis=0)
        members.append(raw)
    p=np.mean(members,axis=0).astype(float)*scale+center
    if cls:p=expit(p)
    if d['tasks'][k]=='ppbr_az':p=p.clip(0,100)
    np.savez_compressed(file,signature=sig(),fit=fit,query=q,p=p,center=center,scale=scale,target_center=tc,target_scale=ts)
    print(json.dumps(dict(stage='train',task=d['tasks'][k],part=part,config=c,seconds=round(time.monotonic()-start))),flush=True);return p
def main():runner.run(sys.modules[__name__],features.load(v.load()))
if __name__=='__main__':main()
