"""Fast GPU asymmetric boosting complements v33 symmetric trees and neural branches."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from xgboost import XGBClassifier,XGBRegressor
import admet_v24_models as b
import admet_v33_run as n
ROOT=n.ROOT;OUT=ROOT/'admet_model_outputs_v35';P=json.loads((ROOT/'admet_v35_protocol.json').read_text())
def signature():return hashlib.sha256(json.dumps(dict(neighbor=n.signature(),files={f:b.sha(ROOT/f) for f in ['admet_v35_run.py','admet_v35_protocol.json']}),sort_keys=True).encode()).hexdigest()
def variants(p,m):
    a=p.copy() if m=='MAE' else np.stack([b.rank(x) for x in p])
    return np.concatenate([p,((a[0]+a[1])*.5)[None]])
def fit(d,k,rows,query,splits,other,dest):
    dest.mkdir(parents=True,exist_ok=True);path=dest/'predictions.npz';sig=signature()
    if path.exists():
        z=np.load(path);assert str(z['signature'])==sig
        np.testing.assert_array_equal(z['fit'],rows);np.testing.assert_array_equal(z['query'],query);return z['p'].copy()
    start=time.time();x,_,_,tf,audit=n.features(d,k,rows,query,splits,other)
    m=d['metrics'][k];cls=m in ['AUROC','AUPRC'];klass=XGBClassifier if cls else XGBRegressor;y=tf.encode(d['y'][rows]);pred=[]
    _,iv,c=np.unique(d['canonical'][d['mi'][rows]],return_inverse=True,return_counts=True);weights=1/np.sqrt(c[iv])
    for family in range(2):
        parts=[]
        for seed,depth,estimators in zip(P['seeds'],P['depths'],P['estimators']):
            params=dict(n_estimators=estimators,max_depth=depth,learning_rate=.035,min_child_weight=5,
                subsample=.85,colsample_bytree=.8,reg_lambda=15.,reg_alpha=.15,
                tree_method='hist',device='cuda',max_bin=128,n_jobs=3,random_state=seed,
                objective='binary:logistic' if cls else ('reg:absoluteerror' if m=='MAE' else 'reg:squarederror'))
            if cls:params['scale_pos_weight']=min(3.,max(1.,np.sqrt((1-y.mean())/max(y.mean(),.01))))
            model=klass(**params);model.fit(x['fit']['trees'][family],y,sample_weight=weights)
            pathm=dest/f'boost{family}_{seed}.ubj';model.save_model(pathm)
            p=model.predict_proba(x['query']['trees'][family])[:,1] if cls else model.predict(x['query']['trees'][family])
            restored=klass();restored.load_model(pathm);pp=restored.predict_proba(x['query']['trees'][family])[:,1] if cls else restored.predict(x['query']['trees'][family])
            np.testing.assert_allclose(p,pp,rtol=1e-6,atol=1e-6);parts.append(p if cls or m=='Spearman' else tf.decode(p))
        pred.append(np.mean(parts,0))
    p=np.stack(pred);assert np.isfinite(p).all();np.savez_compressed(path,p=p,fit=rows,query=query,signature=sig)
    audit.update(signature=sig,task=d['tasks'][k],fit_rows=len(rows),query_rows=len(query),seconds=time.time()-start,
        model_reload_passed=True,test_labels_used=False,local_gpu_training=False)
    b.save(dest/'metadata.json',audit);b.emit('v35_fit',task=d['tasks'][k],partition=str(dest.relative_to(OUT)),seconds=round(time.time()-start));return p
def main():
    b.remote_only();torch.backends.cuda.matmul.allow_tf32=True;OUT.mkdir(parents=True,exist_ok=True);d=b.build_data()
    valid=np.load(n.OUT/'baseline_valid.npy');old=np.load(n.OUT/'baseline_cv_proxy.npy');f=np.load(b.OUT/'folds.npy');other=n.other_cache(d,[0]);records=[]
    for t in n.P['tasks']:
        k=d['tasks'].index(t);m=d['metrics'][k];tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1))
        vp=variants(fit(d,k,tr,vi,[0],other,OUT/t/'valid'),m)
        screen=max(b.delta(m,d['y'][vi],valid[vi],b.blend(m,valid[vi],p,w)) for p in vp for w in P['weights'])
        record=dict(task=t,metric=m,valid_baseline=b.metric(m,d['y'][vi],valid[vi]),valid_rows=len(vi),screen_gain=screen,selection=None,grid=[])
        if screen>2e-4:
            cp=np.full((3,len(d['y'])),np.nan)
            for fold in range(3):
                q=tr[f[tr]==fold];rows=tr[f[tr]!=fold];cp[:,q]=variants(fit(d,k,rows,q,[0],other,OUT/t/f'fold{fold}'),m)
            np.save(OUT/t/'cv_predictions.npy',cp);grid=[]
            for v in range(3):
                for w in P['weights']:
                    gains=[]
                    for fold in range(3):
                        ix=tr[f[tr]==fold];gains.append(b.delta(m,d['y'][ix],old[ix],b.blend(m,old[ix],cp[v,ix],w)))
                    vg=b.delta(m,d['y'][vi],valid[vi],b.blend(m,valid[vi],vp[v],w));obj=float(.55*vg+.45*np.mean(gains)-.04*np.std(gains))
                    grid.append(dict(variant=v,weight=w,cv_gains=gains,cv_mean=float(np.mean(gains)),valid_gain=vg,objective=obj,
                        eligible=bool(vg>2e-4 and np.mean(gains)>0 and obj>5e-4)))
            eligible=[g for g in grid if g['eligible']];record.update(grid=grid,selection=max(eligible,key=lambda g:g['objective']) if eligible else None)
        records.append(record);b.save(OUT/'analysis.json',dict(signature=signature(),status='IN_PROGRESS',tasks=records,limitations=P['limitations']))
        b.emit('v35_selection',task=t,selection=record['selection'])
    b.save(OUT/'analysis.json',dict(signature=signature(),status='EVALUATED',tasks=records,selected_tasks=[r['task'] for r in records if r['selection']],official_score_prediction=None,limitations=P['limitations']))
    compact={};other=n.other_cache(d,[0,1])
    for r in records:
        if r['selection'] is None:continue
        k=d['tasks'].index(r['task']);rows=np.flatnonzero((d['tid']==k)&(d['split']<2));q=np.flatnonzero((d['tid']==k)&(d['split']==2))
        compact[r['task']]=fit(d,k,rows,q,[0,1],other,OUT/r['task']/'final').tolist()
    b.save(OUT/'test_predictions_compact.json',compact);b.save(OUT/'completed.json',dict(signature=signature(),completed=True))
if __name__=='__main__':main()
