"""v20 rare-positive ranking models for the AUPRC bottleneck tasks."""
import argparse, csv, json, hashlib, io
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.stats import rankdata
from admet_v16_models import load_task
from admet_v6_models import Preprocess
from admet_v6_runner import remote_only, partition, get_metric
from admet_v20_residual import anchor_oof, anchor_valid, anchor_test, save_json, read_json, sha

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'admet_model_outputs_v20_rare'
REL = ROOT / 'admet_release_v20_rare'
TASKS = ['cyp2c9_substrate_carbonmangels','cyp2c9_veith','cyp2d6_substrate_carbonmangels','cyp2d6_veith','cyp3a4_veith']

def feats(d):
    return np.concatenate([d['molformer'], d['emb'], d['desc'], d['maccs']], axis=1).astype(np.float32)

def model(x, y, seed, pos_weight):
    m = CatBoostClassifier(iterations=650, depth=6, learning_rate=.04, l2_leaf_reg=12,
        random_seed=seed, class_weights=[1.0, float(pos_weight)], loss_function='Logloss',
        eval_metric='AUC', task_type='GPU', devices='0', gpu_ram_part=.12, thread_count=4,
        random_strength=1.0, bootstrap_type='Bayesian', bagging_temperature=.7,
        border_count=64, verbose=False, allow_writing_files=False)
    m.fit(x, y)
    return m

def predict_ensemble(x, y, seeds, weights):
    ratio = max((1-float(y.mean()))/max(float(y.mean()),1e-3), 1.0)
    schemes = [max(1.0, ratio**.5), max(1.0, ratio), max(1.0, ratio**.75)]
    out=[]
    for i,(seed,scale) in enumerate(zip(seeds, schemes)):
        m=model(x,y,seed,scale)
        out.append(m.predict_proba(weights)[:,1])
    return np.mean(out,axis=0)

def train_task(task):
    dest=OUT/task; dest.mkdir(parents=True,exist_ok=True)
    d=load_task(task); metric=get_metric(task); assert metric=='AUPRC'
    n=len(d['frames']['train']); fold=partition(task,d); y=d['frames']['train'].Y.to_numpy(float)
    xraw=feats(d); q=np.full(n,np.nan)
    for f in range(3):
        it=np.flatnonzero(fold!=f); iv=np.flatnonzero(fold==f)
        pre=Preprocess().fit(xraw[it]); x=pre.transform(xraw[it]); z=pre.transform(xraw[iv])
        q[iv]=predict_ensemble(x,y[it],[30000+f*10+i for i in range(3)],z)
    pd.DataFrame(dict(row=np.arange(n),fold=fold,Y_true=y,Y_pred=q)).to_csv(dest/'rare_oof.csv',index=False)
    # official valid prediction from all training labels
    vy=anchor_valid(task)[0]; xtrain=Preprocess().fit(xraw[:n]); xx=xtrain.transform(xraw[:n]); xv=xtrain.transform(xraw[n:n+len(vy)])
    pv=predict_ensemble(xx,y,[30100+i for i in range(3)],xv)
    pd.DataFrame(dict(Y_true=vy,Y_pred=pv)).to_csv(dest/'rare_valid.csv',index=False)
    # fit again on train+valid for the blind test query
    all_y=np.concatenate([y,vy]); pre=Preprocess().fit(xraw[:n+len(vy)]); xx=pre.transform(xraw[:n+len(vy)]); xt=pre.transform(xraw[n+len(vy):])
    pt=predict_ensemble(xx,all_y,[30200+i for i in range(3)],xt)
    pd.DataFrame(dict(Y_pred=pt)).to_csv(dest/'rare_test.csv',index=False)
    print(json.dumps(dict(stage='v20_rare_task',task=task),ensure_ascii=False),flush=True)

def combine(base,cand,w,mode):
    if w==0: return np.asarray(base).copy()
    if mode=='raw': return (1-w)*base+w*cand
    n=len(base); rb=(rankdata(base,method='average')-.5)/n; rc=(rankdata(cand,method='average')-.5)/n
    return (1-w)*rb+w*rc

def evaluate(task):
    dest=OUT/task; metric='AUPRC'; y,fold,base=anchor_oof(task)
    oof=pd.read_csv(dest/'rare_oof.csv'); valid=pd.read_csv(dest/'rare_valid.csv')
    np.testing.assert_array_equal(oof.row,np.arange(len(y)))
    np.testing.assert_array_equal(oof.fold,fold)
    np.testing.assert_array_equal(oof.Y_true,y)
    cand=oof.Y_pred.to_numpy(float)
    vy,vbase=anchor_valid(task); vcand=valid.Y_pred.to_numpy(float)
    np.testing.assert_array_equal(valid.Y_true,vy)
    assert np.isfinite(cand).all() and np.isfinite(vcand).all()
    grid=[]
    for mode in ['raw','rank']:
      for w in [0,.1,.2,.3,.4,.5,.6,.75,1.0]:
        q=np.empty_like(base)
        for f in np.unique(fold):
            mask=fold==f; q[mask]=combine(base[mask],cand[mask],w,mode)
        qs=combine(vbase,vcand,w,mode)
        fg=[average_precision_score(y[fold==f],q[fold==f])-average_precision_score(y[fold==f],base[fold==f]) for f in sorted(np.unique(fold))]
        vg=average_precision_score(vy,qs)-average_precision_score(vy,vbase)
        vals=np.asarray(fg+[vg]); obj=float(vals.mean()-.15*vals.std()-.25*max(0,-vals.min()))
        eligible=(w==0 or (obj>1e-5 and np.mean(fg)>0 and vg>0 and sum(g>=-.002 for g in fg)>=2))
        grid.append(dict(mode=mode,weight=w,objective=obj,oof_gain=float(np.mean(fg)),valid_gain=float(vg),fold_gains=list(map(float,fg)),eligible=bool(eligible)))
    best=max([z for z in grid if z['eligible']],key=lambda z:(z['objective'],-z['weight']))
    save_json(dest/'decision.json',dict(task=task,selection=best,grid=grid,
        foldwise_ranking=True,inputs={p.name:sha(p) for p in [dest/'rare_oof.csv',dest/'rare_valid.csv']},
        selection_caveat='Historical OOF and official valid reused for selection; not independent evidence or a platform score forecast.'))
    print(json.dumps(dict(stage='v20_rare_decision',task=task,**best),ensure_ascii=False),flush=True)

def publish(task):
    d=load_task(task); route=read_json(OUT/task/'decision.json'); sel=route['selection']; base=anchor_test(task); cand=pd.read_csv(OUT/task/'rare_test.csv').Y_pred.to_numpy(float)
    q=combine(base,cand,sel['weight'],sel['mode']); dest=REL/task; dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'Y':q}).to_csv(dest/f'submission_{task}.csv',index=False); save_json(dest/'selection.json',route)

def export():
    base=next((ROOT/'admet_submissions/v19').glob('*.csv')); templ=ROOT/'admet_official/submission_template.csv'; rows=list(csv.reader(io.StringIO(base.read_bytes().decode('utf-8-sig'),newline=''))); tt=list(csv.reader(io.StringIO(templ.read_bytes().decode('utf-8-sig'),newline=''))); changed=[]
    for task in TASKS:
      route=read_json(OUT/task/'decision.json'); sel=route['selection'];
      if sel['weight']<=0: continue
      changed.append(task); pred=pd.read_csv(REL/task/f'submission_{task}.csv').Y.to_numpy(float)
      for r in rows[1:]:
        if r[0]==task: r[4]=repr(float(np.clip(pred[int(r[1])],0,1)))
    assert changed, 'No eligible model-level improvement; keep v19 instead of exporting an identical version.'
    assert rows[0]==tt[0] and len(rows)==len(tt)==16380
    seen=set()
    for a,b in zip(rows[1:],tt[1:]):
        assert len(a)==6 and a[:4]+a[5:]==b[:4]+b[5:]
        assert np.isfinite(float(a[4]))
        if a[5] in {'AUROC','AUPRC'}: assert 0<=float(a[4])<=1
        key=(a[0],a[1]); assert key not in seen; seen.add(key)
    out=ROOT/'admet_submissions/v20'/'我从山北来带着兰花草-小分子药物预测技术创新赛-20260927-v20.csv'; out.parent.mkdir(parents=True,exist_ok=True); buf=io.StringIO(newline=''); csv.writer(buf,lineterminator='\r\n').writerows(rows); out.write_bytes(b'\xef\xbb\xbf'+buf.getvalue().encode('utf-8'))
    result=dict(status='EXPORTED_AWAITING_PLATFORM',submission=str(out.resolve()),sha256=sha(out),bytes=out.stat().st_size,rows=16379,tasks=22,columns=6,encoding='UTF-8 BOM',line_endings='CRLF',baseline='v19',changed=changed,automatic_submission=False); save_json(out.parent/'verification.json',result); print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    remote_only(); ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['train','evaluate','publish','export']); ap.add_argument('--tasks',nargs='*'); a=ap.parse_args(); ts=a.tasks or TASKS
    for t in ts:
      if a.stage=='train': train_task(t)
      elif a.stage=='evaluate': evaluate(t)
      elif a.stage=='publish': publish(t)
    if a.stage=='export': export()
