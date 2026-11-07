"""Related-task boosted trees with balanced source-task supervision."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='3'
import hashlib,json,sys,time
from pathlib import Path
import numpy as np,xgboost as xgb
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v146_current_baseline as v
import v148_experiment_runner as runner
b,r83=v.b,v.r83
OUT=ROOT/'admet_model_outputs_v173';SEEDS=(1734026,1734027)
CONFIGS=(dict(depth=5,trees=1100,auxiliary=.5),)
METHOD='Task-conditioned related-endpoint XGBoost: all official source observations from a predefined related task group, one-hot endpoint ID, task-balanced weights, fit-only target scales. Query identity and scaffold excluded from all auxiliary training labels. No measured assay values or historical predictions as features.'
GROUPS=(('ames','dili','herg'),('bbb_martins','bioavailability_ma','hia_hou','pgp_broccatelli'),('cyp2c9_substrate_carbonmangels','cyp2d6_substrate_carbonmangels','cyp3a4_substrate_carbonmangels','cyp2c9_veith','cyp2d6_veith','cyp3a4_veith'),('clearance_hepatocyte_az','clearance_microsome_az','half_life_obach','vdss_lombardo'),('caco2_wang','lipophilicity_astrazeneca','solubility_aqsoldb'),('ld50_zhu','ppbr_az'))

def sig():return hashlib.sha256((b.sha(__file__)+v.sig()+b.sha(runner.__file__)).encode()).hexdigest()
def key(c):return 'depth'+str(c['depth'])+'_aux'+str(c['auxiliary'])

def design(d,k,part,c):
    fit,q=v.split(d,k,part);task=d['tasks'][k];group=next(g for g in GROUPS if task in g);others=[d['tasks'].index(t) for t in group if t!=task]
    aux=np.flatnonzero(np.isin(d['tid'],others)&(d['split']<(2 if part=='final' else 1)))
    if part!='final':aux=aux[~np.isin(d['groups'][aux],d['groups'][q])&~np.isin(d['canonical'][aux],d['canonical'][q])]
    source=np.r_[fit,aux];assert not set(source)&set(q) and not np.any(d['tid'][aux]==k)
    if part!='final':assert not set(d['groups'][source])&set(d['groups'][q])
    raw=np.c_[d['views']['multiview'],d['pharmacophore']];x=np.c_[raw[d['mi'][source]],np.eye(len(d['tasks']))[d['tid'][source]]].astype(np.float32);qx=np.c_[raw[d['mi'][q]],np.eye(len(d['tasks']))[d['tid'][q]]].astype(np.float32)
    x[~np.isfinite(x)]=np.nan;qx[~np.isfinite(qx)]=np.nan
    target=np.zeros(len(source));weight=np.zeros(len(source));states={};active=np.unique(d['tid'][aux])
    for j in np.unique(d['tid'][source]):
        mask=d['tid'][source]==j;y=d['y'][source[mask]];m=d['metrics'][j];center=float(np.median(y)) if m=='MAE' else 0.;scale=max(float(np.subtract(*np.quantile(y,[.75,.25]))),.05) if m=='MAE' else 1.
        target[mask]=r83.ranknorm(y) if m=='Spearman' else (y-center)/scale if m=='MAE' else y;weight[mask]=1. if j==k else c['auxiliary']*len(fit)/max(1,len(active))/mask.sum();states[int(j)]=dict(center=center,scale=scale)
    return dict(fit=fit,query=q,source=source,auxiliary=aux,x=x,qx=qx,target=target,weight=weight,states=states)

def predict(d,k,part,c):
    folder=OUT/'models'/d['tasks'][k]/part/key(c);file=folder/'predictions.npz'
    if file.exists():
        z=np.load(file);assert str(z['signature'])==sig();np.testing.assert_array_equal(z['query'],v.split(d,k,part)[1]);return z['p'].copy()
    a=design(d,k,part,c);m=d['metrics'][k];dm=xgb.DMatrix(a['x'],label=a['target'],weight=a['weight'],nthread=3);qm=xgb.DMatrix(a['qx'],nthread=3);folder.mkdir(parents=True,exist_ok=True);members=[];start=time.monotonic()
    for seed in SEEDS:
        model=xgb.train(dict(objective='binary:logistic' if m in ('AUROC','AUPRC') else 'reg:absoluteerror' if m=='MAE' else 'reg:squarederror',max_depth=c['depth'],eta=.025,reg_lambda=35,reg_alpha=.1,min_child_weight=4,subsample=.85,colsample_bytree=.7,max_bin=64,nthread=3,seed=seed,device='cuda',tree_method='hist',verbosity=0),dm,num_boost_round=c['trees'])
        model.set_param({'device':'cpu'});raw=model.predict(qm);mp=folder/f'model_{seed}.ubj';model.save_model(mp);replay=xgb.Booster();replay.load_model(mp);np.testing.assert_array_equal(raw,replay.predict(qm));members.append(raw)
    state=a['states'][k];p=np.mean(members,axis=0).astype(float)*state['scale']+state['center']
    if d['tasks'][k]=='ppbr_az':p=p.clip(0,100)
    np.savez_compressed(file,signature=sig(),fit=a['fit'],query=a['query'],source=a['source'],p=p,center=state['center'],scale=state['scale']);b.save(folder/'target_states.json',dict(signature=sig(),states=a['states']))
    print(json.dumps(dict(stage='train',task=d['tasks'][k],part=part,source_rows=len(a['source']),seconds=round(time.monotonic()-start))),flush=True);return p

def main():runner.run(sys.modules[__name__],v.load())
if __name__=='__main__':main()
