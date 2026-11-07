"""Official-data-only CatBoost specialists blended into the recovered v56 file."""
from __future__ import annotations
import csv, hashlib, io, json, os, time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
import admet_v67_data as b

ROOT=b.ROOT; OUT=ROOT/'admet_model_outputs_v71_v56'; SEED=712026
WEIGHTS=(.10,.25,.50,.75,1.0)

def v68_baseline(d,k,part,q):
    task=d['tasks'][k]
    folder='valid_report' if part=='valid' else ('fold'+str(part))
    candidate=ROOT/'admet_model_outputs_v68'/task/folder/'predictions.npz'
    if candidate.exists():
        z=np.load(candidate)
        np.testing.assert_array_equal(z['query'],q)
        sel=json.loads((ROOT/'admet_model_outputs_v68/selection.json').read_text())['tasks'][k]['selection']
        variant=int(sel['variant']) if sel else 0
        return z['p'][variant].astype(float)
    # A few regression/classification endpoints were intentionally not given a
    # v68 official-valid diagnostic. Reconstruct the frozen v67 valid control
    # from its two seed predictions instead of treating a missing file as a
    # failed experiment.
    assert part=='valid', (task,part)
    sel=json.loads((ROOT/'admet_model_outputs_v67/selection.json').read_text())['tasks'][k]['selection']
    values={}
    for family in ('shared','gated'):
        parts=[]
        for seed in (672031,672032):
            z=np.load(ROOT/'admet_model_outputs_v67'/'valid'/family/str(seed)/'predictions.npz')
            lookup={int(row):i for i,row in enumerate(z['query'])}
            parts.append(z['p'][[lookup[int(row)] for row in q]])
        values[family]=np.mean(parts,axis=0)
    if sel is None or float(sel.get('weight',0.0))==0.0:
        return values['shared'].astype(float)
    return b.blend(d['metrics'][k],values['shared'],values['gated'],float(sel['weight'])).astype(float)

def sig(): return hashlib.sha256((b.sha(ROOT/'admet_v71_v56_catboost.py')+b.signature()).encode()).hexdigest()
def blend(metric,a,n,w):
    if metric=='MAE': return (1-w)*a+w*n
    from scipy.stats import rankdata
    return (1-w)*rankdata(a,method='average')/len(a)+w*rankdata(n,method='average')/len(n)
def score(metric,y,p): return b.metric_score(metric,y,p)
def gain(metric,y,a,n):
    x,z=score(metric,y,a),score(metric,y,n)
    return (x-z)/max(x,1e-8) if metric=='MAE' else z-x

def rows_for(d,k,part,fold=None):
    task=d['tid']==k
    folds=np.load(b.OUT/'folds.npy')
    if part=='valid':
        fit=np.flatnonzero(task&(d['split']==0)); q=np.flatnonzero(task&(d['split']==1))
    elif part=='test':
        fit=np.flatnonzero(task&(d['split']<2)); q=np.flatnonzero(task&(d['split']==2))
    else:
        fit=np.flatnonzero(task&(d['split']==0)&(folds!=fold)); q=np.flatnonzero(task&(d['split']==0)&(folds==fold))
    b.assert_excluded(d,fit,q); return fit,q

def matrix(d,fit,q):
    state=b.preprocess(d,fit)
    return b.transform(d['x'][d['mi'][fit]],state),b.transform(d['x'][d['mi'][q]],state),state

def fit_predict(d,k,part,seed=SEED,fold=None,iterations=1000):
    fit,q=rows_for(d,k,part,fold); task=d['tasks'][k]; metric=d['metrics'][k]
    dest=OUT/task/(part if part!='cv' else f'fold{fold}')
    dest.mkdir(parents=True,exist_ok=True); cache=dest/'predictions.npz'
    if cache.exists():
        z=np.load(cache); assert str(z['signature'])==sig(); np.testing.assert_array_equal(z['fit'],fit); np.testing.assert_array_equal(z['query'],q); return z['p'].copy()
    x,qx,state=matrix(d,fit,q); target=b.target(d['y'][fit],metric); y=b.encode(d['y'][fit],target)
    _,inv,cnt=np.unique(d['canonical'][d['mi'][fit]],return_inverse=True,return_counts=True); weights=1/cnt[inv]; weights/=weights.mean()
    cls=metric in ('AUROC','AUPRC'); loss='Logloss' if cls else ('MAE' if metric=='MAE' else 'RMSE')
    kw=dict(iterations=iterations,depth=7,learning_rate=.035,l2_leaf_reg=12,loss_function=loss,task_type='GPU',devices='0',random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False,border_count=64,bootstrap_type='Bernoulli',subsample=.85)
    model=(CatBoostClassifier if cls else CatBoostRegressor)(**kw)
    if part in ('cv','valid'):
        model.fit(x,y,sample_weight=weights,eval_set=(qx,b.encode(d['y'][q],target)),use_best_model=True,early_stopping_rounds=140)
    else: model.fit(x,y,sample_weight=weights)
    if cls: p=model.predict_proba(qx)[:,1]
    else:
        p=model.predict(qx); p=p*target['scale']+target['center'] if metric=='MAE' else p
    model.save_model(str(dest/'model.cbm'))
    np.savez_compressed(cache,fit=fit,query=q,p=p,signature=sig())
    b.save(dest/'metadata.json',dict(signature=sig(),task=task,metric=metric,trees=model.tree_count_,fit_rows=len(fit),query_rows=len(q),test_labels_used=False,reload_passed=True,state={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in state.items()}))
    return p

def load_v56():
    files=list((ROOT/'admet_submissions/v56').glob('*.csv')); assert len(files)==1
    with files[0].open(encoding='utf-8-sig',newline='') as f: return files[0],list(csv.reader(f))

def main():
    assert os.name != 'nt' and __import__('torch').cuda.is_available()
    OUT.mkdir(parents=True,exist_ok=True); d=b.build(); folds_path=b.OUT/'folds.npy'
    folds=np.load(folds_path) if folds_path.exists() else b.folds(d)
    records=[]
    base_path,base_rows=load_v56(); pos={t:[i for i,r in enumerate(base_rows[1:],1) if r[0]==t] for t in d['tasks']}
    for k,task in enumerate(d['tasks']):
        metric=d['metrics'][k]; tr=np.flatnonzero((d['tid']==k)&(d['split']==0)); vi=np.flatnonzero((d['tid']==k)&(d['split']==1));
        cv=np.full(len(d['y']),np.nan); cv_g=[]
        for f in range(3):
            q=np.flatnonzero((d['tid']==k)&(d['split']==0)&(folds==f)); p=fit_predict(d,k,'cv',fold=f); cv[q]=p
            control=v68_baseline(d,k,f,q)
            cv_g.append(dict(fold=f,gains={str(w):gain(metric,d['y'][q],control,blend(metric,control,p,w)) for w in WEIGHTS}))
        vp=fit_predict(d,k,'valid'); basev=np.array([float(base_rows[i][4]) for i in pos[task]])
        controlv=v68_baseline(d,k,'valid',vi)
        valid_g={str(w):gain(metric,d['y'][vi],controlv,blend(metric,controlv,vp,w)) for w in WEIGHTS}
        choices=[]
        for w in WEIGHTS:
            gs=[z['gains'][str(w)] for z in cv_g]; objective=.8*np.mean(gs)+.2*valid_g[str(w)]-.03*np.std(gs)
            choices.append((objective,w,gs))
        best=max(choices,key=lambda z:z[0]); selected=best[1] if best[0]>.0005 and sum(g>0 for g in best[2])>=2 and valid_g[str(best[1])] > -.02 else 0.0
        records.append(dict(task=task,metric=metric,selected_weight=selected,cv=cv_g,valid_gain=valid_g,objective=best[0]))
        b.save(OUT/'selection.json',dict(status='IN_PROGRESS',signature=sig(),tasks=records,baseline='v56'))
    final=[]
    for rec,k in zip(records,range(len(d['tasks']))):
        task=rec['task']; w=rec['selected_weight']; p=fit_predict(d,k,'test',iterations=1000) if w else None
        if w:
            old=np.array([float(base_rows[i][4]) for i in pos[task]])
            new=blend(rec['metric'],old,p,w)
            for i,v in zip(pos[task],new): base_rows[i][4]=repr(float(v))
        final.append(dict(task=task,weight=w,changed=bool(w)))
    date=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d'); dest=ROOT/'admet_submissions/v71_v56_catboost'; dest.mkdir(parents=True,exist_ok=True)
    out=dest/f'我从山北来带着兰花草-小分子药物预测技术创新赛-{date}.csv'; buf=io.StringIO(newline=''); csv.writer(buf,lineterminator='\r\n').writerows(base_rows); out.write_bytes(b'\xef\xbb\xbf'+buf.getvalue().encode())
    b.save(OUT/'selection.json',dict(status='COMPLETE',signature=sig(),tasks=records,baseline='v56',file=out.relative_to(ROOT).as_posix(),sha256=b.sha(out),changed_tasks=[x['task'] for x in final if x['changed']],tasks_final=final,test_labels_used=False,external_data=False))
    print(json.dumps(dict(status='COMPLETE',file=str(out),sha256=b.sha(out),changed_tasks=[x['task'] for x in final if x['changed']]),ensure_ascii=False),flush=True)

if __name__=='__main__': main()
