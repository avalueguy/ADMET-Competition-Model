"""Remote complementary small-task kernels with learned experimental neighborhoods."""
import hashlib,json,time
from pathlib import Path
import joblib
import numpy as np
import torch
from scipy.special import expit
from sklearn.svm import SVC,SVR
import admet_v24_models as b
import admet_v33_run as n
ROOT=n.ROOT;OUT=ROOT/'admet_model_outputs_v34'
P=json.loads((ROOT/'admet_v34_protocol.json').read_text())
def signature():
    return hashlib.sha256(json.dumps(dict(neighbor=n.signature(),files={name:b.sha(ROOT/name) for name in ['admet_v34_run.py','admet_v34_protocol.json']}),sort_keys=True).encode()).hexdigest()
def rbf(a,c):
    aa=torch.tensor(a,device='cuda');bb=torch.tensor(c,device='cuda');dist=(aa.square().sum(1)[:,None]+bb.square().sum(1)[None]-2*aa@bb.T).clamp_min(0)/aa.shape[1]
    return (.5*torch.exp(-dist)+.5*torch.exp(-dist/4)).cpu().numpy()
def matrices(d,k,rows,query,splits,other):
    x,pre,_,tf,audit=n.features(d,k,rows,query,splits,other)
    ap=b.Preprocessor().fit(x['fit']['aux']);assay={key:ap.transform(x[key]['aux']) for key in ['fit','query']}
    tr=d['mi'][rows];q=d['mi'][query]
    a=n.similarity(d,tr,tr).cpu().numpy();c=n.similarity(d,q,tr).cpu().numpy()
    structural=[.8*a+.2*a*a,.8*c+.2*c*c]
    full=[.55*a+.15*a*a+.15*rbf(x['fit']['dense'],x['fit']['dense'])+.15*rbf(assay['fit'],assay['fit']),
        .55*c+.15*c*c+.15*rbf(x['query']['dense'],x['fit']['dense'])+.15*rbf(assay['query'],assay['fit'])]
    for matrix in [structural[0],full[0]]:np.testing.assert_allclose(matrix,matrix.T,rtol=2e-5,atol=2e-5)
    return [full,structural],tf,audit
def fit(d,k,rows,query,splits,other,dest):
    dest.mkdir(parents=True,exist_ok=True);path=dest/'predictions.npz';sig=signature()
    if path.exists():
        z=np.load(path);assert str(z['signature'])==sig
        np.testing.assert_array_equal(z['fit'],rows);np.testing.assert_array_equal(z['query'],query);return z['p'].copy()
    t0=time.time();kernels,tf,audit=matrices(d,k,rows,query,splits,other)
    m=d['metrics'][k];cls=m in ['AUROC','AUPRC'];y=tf.encode(d['y'][rows]);pred=[];bundles=[]
    _,inv,count=np.unique(d['canonical'][d['mi'][rows]],return_inverse=True,return_counts=True)
    weights=1/np.sqrt(count[inv]);weights/=weights.mean()
    for family,(train,q) in enumerate(kernels):
        models=[];values=[]
        for c in P['c_values']:
            model=SVC(C=c,kernel='precomputed',class_weight='balanced',cache_size=384) if cls else SVR(C=c,kernel='precomputed',epsilon=.05,cache_size=384)
            model.fit(train.astype(np.float64),y,sample_weight=weights);p=model.decision_function(q.astype(np.float64)) if cls else model.predict(q.astype(np.float64))
            models.append(model);values.append(expit(p) if cls else (p if m=='Spearman' else tf.decode(p)))
        pathm=dest/f'kernel{family}.joblib';joblib.dump(dict(models=models,target=tf,signature=sig),pathm,compress=3)
        reload=joblib.load(pathm)
        for model,value in zip(reload['models'],values):
            pp=model.decision_function(q.astype(np.float64)) if cls else model.predict(q.astype(np.float64))
            np.testing.assert_allclose(value,expit(pp) if cls else (pp if m=='Spearman' else tf.decode(pp)),rtol=1e-10,atol=1e-10)
        pred.append(np.mean(values,0))
    p=np.stack(pred);np.savez_compressed(path,p=p,fit=rows,query=query,signature=sig)
    audit.update(signature=sig,task=d['tasks'][k],fit_rows=len(rows),query_rows=len(query),seconds=time.time()-t0,
        model_reload_passed=True,local_gpu_training=False,test_labels_used=False)
    b.save(dest/'metadata.json',audit);b.emit('v34_fit',task=d['tasks'][k],partition=str(dest.relative_to(OUT)),seconds=round(time.time()-t0));return p
def variants(p,m):
    a=p.copy() if m=='MAE' else np.stack([b.rank(x) for x in p])
    return np.stack([p[0],(a[0]+a[1])*.5])
def main():
    b.remote_only();torch.backends.cuda.matmul.allow_tf32=True;OUT.mkdir(parents=True,exist_ok=True);d=b.build_data()
    # Use already replayed full v32 and CV proxy from the v33 shards.
    base=np.load(n.OUT/'baseline_valid.npy');old=np.load(n.OUT/'baseline_cv_proxy.npy');f=np.load(b.OUT/'folds.npy');other=n.other_cache(d,[0]);records=[]
    for k,t in enumerate(d['tasks']):
        tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1));m=d['metrics'][k]
        if len(tr)>P['maximum_fit_rows']:continue
        vp=variants(fit(d,k,tr,vi,[0],other,OUT/t/'valid'),m)
        screen=max(b.delta(m,d['y'][vi],base[vi],b.blend(m,base[vi],p,w)) for p in vp for w in P['weights'])
        record=dict(task=t,metric=m,valid_rows=len(vi),valid_baseline=b.metric(m,d['y'][vi],base[vi]),screen_gain=screen,selection=None,grid=[])
        if screen>2e-4:
            cp=np.full((2,len(d['y'])),np.nan)
            for fold in range(3):
                query=tr[f[tr]==fold];rows=tr[f[tr]!=fold];cp[:,query]=variants(fit(d,k,rows,query,[0],other,OUT/t/f'fold{fold}'),m)
            np.save(OUT/t/'cv_predictions.npy',cp);grid=[]
            for v in range(2):
                for w in P['weights']:
                    gains=[]
                    for fold in range(3):
                        ix=tr[f[tr]==fold];gains.append(b.delta(m,d['y'][ix],old[ix],b.blend(m,old[ix],cp[v,ix],w)))
                    vg=b.delta(m,d['y'][vi],base[vi],b.blend(m,base[vi],vp[v],w));obj=float(.55*vg+.45*np.mean(gains)-.04*np.std(gains))
                    grid.append(dict(variant=v,weight=w,cv_gains=gains,cv_mean=float(np.mean(gains)),valid_gain=vg,objective=obj,
                        eligible=bool(vg>2e-4 and np.mean(gains)>0 and obj>5e-4)))
            eligible=[g for g in grid if g['eligible']];record.update(grid=grid,selection=max(eligible,key=lambda g:g['objective']) if eligible else None)
        records.append(record);b.save(OUT/'analysis.json',dict(signature=signature(),status='IN_PROGRESS',tasks=records,limitations=P['limitations']))
        b.emit('v34_selection',task=t,selection=record['selection'])
    b.save(OUT/'analysis.json',dict(signature=signature(),status='EVALUATED',tasks=records,selected_tasks=[r['task'] for r in records if r['selection']],
        official_score_prediction=None,limitations=P['limitations']))
    compact={};selected=[r for r in records if r['selection']]
    if selected:
        other=n.other_cache(d,[0,1])
        for r in selected:
            k=d['tasks'].index(r['task']);rows=np.flatnonzero((d['tid']==k)&(d['split']<2));q=np.flatnonzero((d['tid']==k)&(d['split']==2))
            compact[r['task']]=fit(d,k,rows,q,[0,1],other,OUT/r['task']/'final').tolist()
    b.save(OUT/'test_predictions_compact.json',compact);b.save(OUT/'completed.json',dict(signature=signature(),completed=True))
if __name__=='__main__':main()
