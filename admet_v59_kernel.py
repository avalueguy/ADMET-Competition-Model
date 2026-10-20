"""Partition-isolated supervised-graph Gaussian/Tanimoto support-vector experts."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='2'
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch,joblib
from sklearn.svm import SVC,SVR
import admet_v24_models as b
import admet_v43_run as source
import admet_v51_run as old

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'admet_model_outputs_v59'
TASKS=['cyp2c9_substrate_carbonmangels','cyp2d6_substrate_carbonmangels','cyp3a4_substrate_carbonmangels','bioavailability_ma','herg','bbb_martins','caco2_wang','half_life_obach','clearance_hepatocyte_az','clearance_microsome_az','vdss_lombardo','dili','ppbr_az','lipophilicity_astrazeneca']
CONFIGS=[(0,.3),(0,3.),(0,30.),(1,.3),(1,3.),(1,30.)]
WEIGHTS=[.15,.35,.6,.85,1.]

def signature():return hashlib.sha256((b.sha(__file__)+source.signature()).encode()).hexdigest()

def matrices(d,rows,query,partition):
    # representation_audit checks supervised source exclusion for every query group.
    provenance=source.representation_audit(d,rows,query,partition)
    latent=source.representations(d,None,partition)
    a=np.unique(d['mi'][rows]);ix=d['mi'][rows];iq=d['mi'][query]
    pre=b.Preprocessor().fit(latent[a]);x=pre.transform(latent)
    dense=np.concatenate([d['desc'],d['emb']],1);dp=b.Preprocessor().fit(dense[a]);dense=dp.transform(dense)
    def rbf(x,y):
        dist=np.maximum(0,(x*x).sum(1)[:,None]+(y*y).sum(1)[None]-2*x@y.T)/x.shape[1]
        return (.5*np.exp(-dist/2)+.5*np.exp(-dist/8)).astype(np.float64)
    def tan(x,y):
        dot=x@y.T;return (dot/np.maximum((x*x).sum(1)[:,None]+(y*y).sum(1)[None]-dot,1e-8)).astype(np.float64)
    graph=[rbf(x[ix],x[ix]),rbf(x[iq],x[ix])]
    densek=[rbf(dense[ix],dense[ix]),rbf(dense[iq],dense[ix])]
    chem=[]
    for q in [ix,iq]:chem.append(.65*tan(d['fp2'][q],d['fp2'][ix])+.35*tan(d['fcfp'][q],d['fcfp'][ix]))
    families=[[.8*graph[j]+.2*densek[j] for j in range(2)],
              [.45*graph[j]+.4*chem[j]+.15*densek[j] for j in range(2)]]
    for pair in families:pair[0]=(pair[0]+pair[0].T)*.5
    return families,provenance

def fit(d,k,rows,query,partition,dest):
    sig=signature();dest.mkdir(parents=True,exist_ok=True);path=dest/'predictions.npz'
    if path.exists():
        z=np.load(path);assert str(z['signature'])==sig
        np.testing.assert_array_equal(rows,z['fit']);np.testing.assert_array_equal(query,z['query']);return z['p'].copy()
    assert not set(rows)&set(query) and np.all(d['split'][rows]<2)
    if partition!='final':assert not set(d['rg'][rows])&set(d['rg'][query])
    t0=time.time();kernels,audit=matrices(d,rows,query,partition);metric=d['metrics'][k];tf=b.Target(d['y'][rows],metric);y=tf.encode(d['y'][rows]);cls=metric in ['AUROC','AUPRC']
    _,inv,count=np.unique(d['canonical'][d['mi'][rows]],return_inverse=True,return_counts=True);weights=1./count[inv];weights/=weights.mean()
    predictions=[]
    for ci,(family,c) in enumerate(CONFIGS):
        tr,qe=kernels[family]
        model=SVC(C=c,kernel='precomputed',class_weight='balanced',cache_size=256) if cls else SVR(C=c,epsilon=.06,kernel='precomputed',cache_size=256)
        model.fit(tr,y,sample_weight=weights)
        p=model.decision_function(qe) if cls else model.predict(qe)
        if metric=='MAE':p=tf.decode(p)
        joblib.dump(dict(model=model,target=tf,rows=rows,partition=partition,signature=sig),dest/f'config{ci}.joblib',compress=3)
        restored=joblib.load(dest/f'config{ci}.joblib')['model'];q=restored.decision_function(qe) if cls else restored.predict(qe)
        if metric=='MAE':q=tf.decode(q)
        np.testing.assert_allclose(p,q,atol=1e-10,rtol=1e-10);predictions.append(p)
    p=np.stack(predictions);assert np.isfinite(p).all()
    np.savez_compressed(path,p=p,fit=rows,query=query,signature=sig)
    b.save(dest/'metadata.json',dict(**audit,signature=sig,preprocessor_fit_only=True,test_labels_used=False,local_gpu_training=False,model_reload_passed=True,seconds=time.time()-t0))
    b.emit('v59_fit',task=d['tasks'][k],partition=dest.name,seconds=round(time.time()-t0));return p

def main():
    b.remote_only();OUT.mkdir(exist_ok=True);d=b.build_data();basev,basec=old.baseline(d);folds=np.load(b.OUT/'folds.npy');records=[]
    np.save(OUT/'baseline_valid.npy',basev);np.save(OUT/'baseline_cv_proxy.npy',basec)
    for task in TASKS:
        k=d['tasks'].index(task);metric=d['metrics'][k];tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1));cp=np.full((len(CONFIGS),len(d['y'])),np.nan)
        for f in range(3):
            rr=tr[folds[tr]!=f];q=tr[folds[tr]==f];cp[:,q]=fit(d,k,rr,q,'cv/fold'+str(f),OUT/task/f'fold{f}')
        np.save(OUT/task/'cv_predictions.npy',cp);vp=fit(d,k,tr,vi,'valid',OUT/task/'valid');grid=[]
        for v in range(len(CONFIGS)):
            for w in WEIGHTS:
                gains=[b.delta(metric,d['y'][ix],basec[ix],b.blend(metric,basec[ix],cp[v,ix],w)) for f in range(3) for ix in [tr[folds[tr]==f]]]
                vg=b.delta(metric,d['y'][vi],basev[vi],b.blend(metric,basev[vi],vp[v],w));obj=float(.85*np.mean(gains)+.15*vg-.07*np.std(gains))
                grid.append(dict(variant=v,weight=w,cv_gains=gains,cv_mean=float(np.mean(gains)),valid_gain=vg,objective=obj))
        held=[max(grid,key=lambda x:np.mean([g for j,g in enumerate(x['cv_gains']) if j!=f]))['cv_gains'][f] for f in range(3)]
        good=[x for x in grid if x['cv_mean']>.001 and x['objective']>.001 and sum(g>0 for g in x['cv_gains'])>=2]
        selected=max(good,key=lambda x:x['objective']) if good and np.mean(held)>0 else None
        r=dict(task=task,metric=metric,selection=selected,held_gains=held,held_mean=float(np.mean(held)),grid=grid);records.append(r)
        b.save(OUT/'analysis.json',dict(status='IN_PROGRESS',signature=signature(),tasks=records));b.emit('v59_selection',**{k:v for k,v in r.items() if k!='grid'})
    b.save(OUT/'analysis.json',dict(status='FROZEN',signature=signature(),tasks=records,official_score=None))

if __name__=='__main__':main()
