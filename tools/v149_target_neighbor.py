"""Target assay neighbor and structural experts with honest out-of-fit profiles."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='3'
import hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch,xgboost as xgb
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v146_current_baseline as v
import v148_experiment_runner as runner
b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v149';SEEDS=(1494026,1494027)
CONFIGS=(dict(depth=3,trees=1200),dict(depth=5,trees=1200))
METHOD='XGBoost plus target-task neighbor labels from actual fit rows only. Training summaries exclude same molecular identity and scaffold; query uses all fit rows. Two targets/log-rank for Spearman. Official structure/context only.'
def sig():return hashlib.sha256((b.sha(__file__)+v.sig()+b.sha(runner.__file__)).encode()).hexdigest()
def key(c):return 'depth'+str(c['depth'])

def summaries(d,k,fit,rows,train):
    m=d['metrics'][k];y=d['y'][fit];out=[];raw=d['views']['multiview'];n=len(fit)
    target=np.c_[r83.ranknorm(y),np.log1p(np.maximum(y,0))] if m=='Spearman' else y[:,None]
    if m=='MAE':target=(target-np.median(y))/max(float(np.subtract(*np.quantile(y,[.75,.25]))),.05)
    gy=torch.as_tensor(target,device='cuda',dtype=torch.float32)
    # Circular and feature counts are separate neighborhoods.
    for start in (0,2048,4096):
        ref=torch.as_tensor(raw[d['mi'][fit],start:start+2048],device='cuda');rn=ref.square().sum(1);blocks=[]
        for lo in range(0,len(rows),256):
            rr=rows[lo:lo+256];x=torch.as_tensor(raw[d['mi'][rr],start:start+2048],device='cuda');dot=x@ref.T
            sim=(dot/(x.square().sum(1)[:,None]+rn[None]-dot).clamp_min(1e-8)).clamp(0,1)
            if train:
                mask=(d['canonical'][rr,None]==d['canonical'][fit][None])|(d['groups'][rr,None]==d['groups'][fit][None]);sim.masked_fill_(torch.as_tensor(mask,device='cuda'),-1)
            values,ix=torch.topk(sim,min(64,n),dim=1);valid=values>=0;values=values.clamp_min(0);yy=gy[ix];features=[]
            for take in (4,16,64):
                t=min(take,n);s=values[:,:t];vv=valid[:,:t];lab=yy[:,:t]
                for power in (2,8):
                    w=s.pow(power)*vv;den=w.sum(1).clamp_min(1e-8);mu=(lab*w[:,:,None]).sum(1)/den[:,None]
                    sd=(((lab-mu[:,None])**2*w[:,:,None]).sum(1)/den[:,None]).sqrt();features.extend((mu,sd))
                features.extend((s[:,0,None],(s.sum(1)/vv.sum(1).clamp_min(1))[:,None]))
            blocks.append(torch.cat(features,dim=1).cpu().numpy())
        out.append(np.concatenate(blocks));del ref
    return np.c_[*out]

def design(d,k,part):
    fit,q,x,qx,source=v.old.features(d,k,part,'augmented');folder=OUT/'features'/d['tasks'][k];folder.mkdir(parents=True,exist_ok=True);file=folder/(part+'.npz')
    if file.exists():
        z=np.load(file);assert str(z['signature'])==sig();np.testing.assert_array_equal(z['fit'],fit);np.testing.assert_array_equal(z['query'],q);tr,te=z['train_neighbors'],z['query_neighbors']
    else:
        tr,te=summaries(d,k,fit,fit,True),summaries(d,k,fit,q,False);np.savez_compressed(file,signature=sig(),fit=fit,query=q,train_neighbors=tr,query_neighbors=te)
    return fit,q,np.c_[x,tr],np.c_[qx,te],source

def predict(d,k,part,c):
    folder=OUT/'models'/d['tasks'][k]/part/key(c);file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);assert str(z['signature'])==sig();np.testing.assert_array_equal(z['query'],v.split(d,k,part)[1]);return z['p'].copy()
    fit,q,x,qx,source=design(d,k,part);m=d['metrics'][k];y=d['y'][fit];cls=m in ('AUROC','AUPRC');center=float(np.median(y)) if m=='MAE' else 0.;scale=max(float(np.subtract(*np.quantile(y,[.75,.25]))),.05) if m=='MAE' else 1.
    targets=[y] if cls else [(y-center)/scale] if m=='MAE' else [r83.ranknorm(y),np.log1p(np.maximum(y,0))]
    folder.mkdir(parents=True,exist_ok=True);members=[];start=time.monotonic()
    for j,target in enumerate(targets):
        dm=xgb.DMatrix(x,label=target);qm=xgb.DMatrix(qx);pp=[]
        for seed in SEEDS:
            model=xgb.train(dict(objective='binary:logistic' if cls else 'reg:absoluteerror' if m=='MAE' else 'reg:squarederror',max_depth=c['depth'],eta=.025,reg_lambda=30,min_child_weight=4,subsample=.85,colsample_bytree=.8,max_bin=64,nthread=3,seed=seed,device='cuda',tree_method='hist',verbosity=0),dm,num_boost_round=c['trees'])
            model.set_param({'device':'cpu'});p=model.predict(qm);mp=folder/f'model_t{j}_{seed}.ubj';model.save_model(mp);replay=xgb.Booster();replay.load_model(mp);np.testing.assert_array_equal(p,replay.predict(qm));pp.append(p)
        p=np.mean(pp,axis=0).astype(float);members.append(r83.rankdata(p)/len(p) if m=='Spearman' else p)
    p=np.mean(members,axis=0)*scale+center
    if d['tasks'][k]=='ppbr_az':p=p.clip(0,100)
    np.savez_compressed(file,signature=sig(),fit=fit,query=q,p=p,center=center,scale=scale)
    print(json.dumps(dict(stage='train',task=d['tasks'][k],part=part,config=c,seconds=round(time.monotonic()-start))),flush=True);return p
def main():
    torch.set_num_threads(3);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');d=v.load();runner.run(sys.modules[__name__],d)
if __name__=='__main__':main()
