"""Fit-pruned atom-pair/torsion/path/circular fragments plus assay context."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='3'
import hashlib,json,sys,time
from pathlib import Path
import numpy as np,xgboost as xgb
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v146_current_baseline as v
import v147_path_features as features
import v148_experiment_runner as runner
import v104_sparse_features as circular
b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v151';SEEDS=(1514026,1514027)
CONFIGS=(dict(depth=3,trees=1500),dict(depth=6,trees=1100))
METHOD='GPU boosting on new unfolded atom-pair/torsion/RDKit paths plus circular counts; columns present in >=3 fit molecules, log counts, augmented chemistry and target-excluded exact/neutral assay context. Strong regularization.'
def sig():return hashlib.sha256((b.sha(__file__)+v.sig()+features.sig()+circular.sig()+b.sha(runner.__file__)).encode()).hexdigest()
def key(c):return 'depth'+str(c['depth'])
def load():
    d=features.load(v.load());a=json.loads((circular.OUT/'audit.json').read_text());assert a['schema']['signature']==circular.sig();xx=[]
    for j in range(2):
        p=circular.OUT/f'view{j}.npz';assert b.sha(p)==a['views'][j]['sha256'];xx.append(sparse.load_npz(p))
    d['all_sparse_fingerprints']=xx+d['path_fingerprints'];return d
def design(d,k,part):
    fit,q,x,qx,source=v.old.features(d,k,part,'augmented');pieces=[];qpieces=[];columns=[]
    for raw in d['all_sparse_fingerprints']:
        xx=raw[d['mi'][fit]];cols=np.flatnonzero(np.asarray((xx>0).sum(0)).ravel()>=3);columns.append(cols)
        a=xx[:,cols].copy();z=raw[d['mi'][q]][:,cols].copy();a.data=np.log1p(a.data);z.data=np.log1p(z.data);pieces.append(a);qpieces.append(z)
    pieces.append(sparse.csr_matrix(x));qpieces.append(sparse.csr_matrix(qx))
    return fit,q,sparse.hstack(pieces,format='csr'),sparse.hstack(qpieces,format='csr'),columns,source
def predict(d,k,part,c):
    folder=OUT/'models'/d['tasks'][k]/part/key(c);file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);assert str(z['signature'])==sig();np.testing.assert_array_equal(z['query'],v.split(d,k,part)[1]);return z['p'].copy()
    fit,q,x,qx,cols,source=design(d,k,part);m=d['metrics'][k];y=d['y'][fit];cls=m in ('AUROC','AUPRC');center=float(np.median(y)) if m=='MAE' else 0.;scale=max(float(np.subtract(*np.quantile(y,[.75,.25]))),.05) if m=='MAE' else 1.
    target=y if cls else (y-center)/scale if m=='MAE' else r83.ranknorm(y);folder.mkdir(parents=True,exist_ok=True);members=[];start=time.monotonic()
    dm=xgb.DMatrix(x,label=target,nthread=3);qm=xgb.DMatrix(qx,nthread=3)
    for seed in SEEDS:
        model=xgb.train(dict(objective='binary:logistic' if cls else 'reg:absoluteerror' if m=='MAE' else 'reg:squarederror',max_depth=c['depth'],eta=.02,reg_lambda=40,reg_alpha=.1,min_child_weight=5,subsample=.85,colsample_bytree=.65,max_bin=64,nthread=3,seed=seed,device='cuda',tree_method='hist',verbosity=0),dm,num_boost_round=c['trees'])
        model.set_param({'device':'cpu'});p=model.predict(qm);mp=folder/f'model_{seed}.ubj';model.save_model(mp);replay=xgb.Booster();replay.load_model(mp);np.testing.assert_array_equal(p,replay.predict(qm));members.append(p)
    p=np.mean(members,axis=0).astype(float)*scale+center
    if d['tasks'][k]=='ppbr_az':p=p.clip(0,100)
    np.savez_compressed(file,signature=sig(),fit=fit,query=q,p=p,center=center,scale=scale,**{f'columns{j}':c for j,c in enumerate(cols)})
    print(json.dumps(dict(stage='train',task=d['tasks'][k],part=part,config=c,seconds=round(time.monotonic()-start))),flush=True);return p
def main():runner.run(sys.modules[__name__],load())
if __name__=='__main__':main()
