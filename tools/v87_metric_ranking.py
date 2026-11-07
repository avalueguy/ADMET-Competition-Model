"""Loss-scale-aware MAP/pairwise residual ranking over the v86 anchor.

Frozen task/objective/view selection on legacy fold0/1, confirmation on
fold2 and official valid. Historical baseline predictions remain proxies.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='4'
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import xgboost as xgb
from scipy.special import expit
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
import admet_v77_ranking as b
import v83_residual as r83
import v85_smooth_residual as r85
import v86_metric_ranking as r86

OUT=ROOT/'admet_model_outputs_v87'
ANCHOR=ROOT/'admet_submissions/v86_metric_ranking/我从山北来带着兰花草-小分子药物预测技术创新赛-20261025.csv'
ANCHOR_SHA='8cb01bb2db0e308f1c7b1d215c7eca75436646bc0561a019e971ac883ce759c3'
SEEDS=(874026,874027)
WEIGHTS=(.25,.5,1.)
CONFIGS=[dict(name=f'{view}_{method}_d{depth}_t{trees}',view=view,method=method,depth=depth,trees=trees)
    for view,depth,trees in [('aux',3,200),('compact',3,400),('multiview',3,400)] for method in ('map','pairwise')]


def signature():
    files=[__file__,b.__file__,r83.__file__,r85.__file__,r85.r84.__file__,r86.__file__]
    return hashlib.sha256(''.join(b.sha(p) for p in files).encode()).hexdigest()


def emit(**kw):print(json.dumps(kw,ensure_ascii=False),flush=True)


def load():
    d=b.load_data();d['legacy_folds']=np.load(ROOT/'recovered_history/legacy_v24_folds.npy')
    r83.TASK_NAMES=d['tasks'];assert b.sha(ANCHOR)==ANCHOR_SHA
    return d


def baseline(d,k,part):
    if part=='final':return np.array([float(r['y_pred']) for r in b.table(ANCHOR) if r['task']==d['tasks'][k]])
    if d['tasks'][k]=='cyp2d6_substrate_carbonmangels':
        conf=json.loads((r86.OUT/'confirmation.json').read_text())
        chosen=next(v for v in conf['records'] if v['task']==d['tasks'][k] and v['accepted'])
        return r86.mix(r86.baseline(d,k,part),r86.fit(d,k,part,chosen['config']),chosen['weight'])
    return r86.baseline(d,k,part)


def base_state(d,k,part):
    fit,q=r83.split(d,k,part);base=r83.raw_proxy(d,fit)
    qp=(np.array([float(r['y_pred']) for r in b.table(r85.r84.ORIGINAL) if r['task']==d['tasks'][k]])
        if part=='final' else r83.raw_proxy(d,q))
    z,qz=r83.ranklog(base),r83.ranklog(qp);beta=r83.calibrate(d['y'][fit],z)
    off,qoff=beta[0]+beta[1]*z,beta[0]+beta[1]*qz
    return fit,q,base,qp,z,qz,beta,off,qoff


def design(d,k,part,view):
    fit,q,base,qp,z,qz,beta,off,qoff=base_state(d,k,part)
    aux,cols,source,counts=r83.aux(d,k,part=='final');assert k not in cols
    # XGBoost handles measured-value NaNs; 0/1 masks remain explicit.
    x=np.c_[aux[fit],z];qx=np.c_[aux[q],qz]
    if view in ('compact','multiview'):
        key='physchem' if view=='compact' else 'multiview'
        x=np.c_[x,d['views'][key][d['mi'][fit]]]
        qx=np.c_[qx,d['views'][key][d['mi'][q]]]
    x=x.astype(np.float32);qx=qx.astype(np.float32)
    return dict(fit=fit,query=q,base_fit=base,base_query=qp,z=z,qz=qz,beta=beta,off=off,qoff=qoff,
        x=x,qx=qx,source=source,cols=cols,known_counts=counts[q])


def fit(d,k,part,cfg):
    folder=OUT/'models'/d['tasks'][k]/part/cfg['name'];path=folder/'predictions.npz'
    if path.exists():
        z=np.load(path);assert str(z['signature'])==signature();np.testing.assert_array_equal(z['query'],r83.split(d,k,part)[1])
        return z['p'].copy()
    folder.mkdir(parents=True,exist_ok=True);started=time.monotonic();a=design(d,k,part,cfg['view'])
    y=d['y'][a['fit']];members=[];hashes={}
    for seed in SEEDS:
        objective={'logloss':'binary:logistic','map':'rank:map','pairwise':'rank:pairwise'}[cfg['method']]
        dm=xgb.DMatrix(a['x'],label=y,base_margin=a['off'])
        if cfg['method']!='logloss':dm.set_group([len(y)])
        params=dict(objective=objective,device='cuda',tree_method='hist',max_depth=cfg['depth'],eta=.025,
            reg_lambda=1.,reg_alpha=0.,min_child_weight=.001,subsample=.85,colsample_bytree=.85,
            max_bin=64,nthread=4,seed=seed,verbosity=0)
        if cfg['method']!='logloss':
            params.update(lambdarank_pair_method='mean' if cfg['method']=='pairwise' else 'topk',
                lambdarank_num_pair_per_sample=16 if cfg['method']=='pairwise' else 64)
        model=xgb.train(params,dm,num_boost_round=cfg['trees'])
        query=xgb.DMatrix(a['qx'],base_margin=a['qoff'])
        member=model.predict(query,output_margin=True).astype(float)
        members.append(member);mp=folder/f'model_{seed}.ubj';model.save_model(mp);hashes[mp.name]=b.sha(mp)
    members=np.stack(members);pred=expit(members.mean(0));assert np.isfinite(pred).all()
    np.savez_compressed(path,signature=signature(),p=pred,members=members,seeds=SEEDS,**a)
    b.save(folder/'metadata.json',dict(signature=signature(),task=d['tasks'][k],part=part,config=cfg,
        seeds=SEEDS,model_hashes=hashes,predictions_sha256=b.sha(path),xgboost=xgb.__version__,
        target_assay_excluded=True,ensemble='sigmoid(mean(raw margins including fixed base margin))',
        seconds=time.monotonic()-started))
    return pred


def mix(base,p,w):return (1-w)*rankdata(base)/len(base)+w*rankdata(p)/len(p)


def develop(d):
    OUT.mkdir(exist_ok=True);tasks=[k for k,m in enumerate(d['metrics']) if m in ('AUROC','AUPRC')]
    b.save(OUT/'protocol.json',dict(version=87,signature=signature(),baseline='v86',anchor_sha256=ANCHOR_SHA,
        baseline_official_score_user_reported=None,highest_official_score_user_reported=50.889,configs=CONFIGS,seeds=SEEDS,weights=WEIGHTS,
        selection_parts=['fold0','fold1'],confirmation_parts=['fold2','valid'],
        selection='Max minimum gain; minimum .0005 AUROC or AUPRC on both folds. Freeze one task/config/weight per task.',
        confirmation='Same fixed two-seed formula, all four parts strictly positive. No confirmation fallback tuning.',
        classification_objectives=['LambdaMART MAP','Pairwise ranking'],
        ranking_scale=dict(reg_lambda=1.,reg_alpha=0.,min_child_weight=.001),
        prior_v86_ranking_issue='High min_child_weight=10 and L1=0.1 gave zero splits; fixed here before selection',
        matched_logloss_control='Saved v86 models used appropriate Logloss regularization; baseline includes accepted v86 change',early_stopping=False,
        test_labels_used=False,external_data=False,new_pretrained_weights=False,
        limitations='Historical base proxies not full v86 OOF; development labels reused; auxiliary official other-assay values are conditional. Ranking uses one group, MAP top64 and Pairwise mean16 pairs.'))
    records=[]
    for k in tasks:
        task=d['tasks'][k];metric=d['metrics'][k];bases={part:baseline(d,k,part) for part in ('fold0','fold1')}
        for cfg in CONFIGS:
            candidates={part:fit(d,k,part,cfg) for part in bases}
            for w in WEIGHTS:
                parts=[]
                for part,base in bases.items():
                    q=r83.split(d,k,part)[1];p=mix(base,candidates[part],w)
                    parts.append(dict(part=part,rows=len(q),baseline=r83.score(metric,d['y'][q],base),
                        candidate=r83.score(metric,d['y'][q],p),gain=r83.gain(metric,d['y'][q],base,p)))
                records.append(dict(task=task,metric=metric,config=cfg,weight=w,parts=parts,objective=min(r['gain'] for r in parts)))
        best=max((r for r in records if r['task']==task),key=lambda r:r['objective']);emit(stage='development',task=task,best=best)
        b.save(OUT/'development.json',dict(status='IN_PROGRESS',records=records,signature=signature()))
    selected=[]
    for k in tasks:
        best=max((r for r in records if r['task']==d['tasks'][k]),key=lambda r:r['objective'])
        if best['objective']>.0005:selected.append(best)
    b.save(OUT/'development.json',dict(status='COMPLETE',records=records,signature=signature()))
    b.save(OUT/'frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=signature(),
        selected=selected,confirmation_inspected=False))
    emit(stage='frozen',selected=selected)


def confirm(d):
    frozen=json.loads((OUT/'frozen_selection.json').read_text());assert frozen['signature']==signature();records=[]
    for chosen in frozen['selected']:
        k=d['tasks'].index(chosen['task']);parts=[]
        for part in ('fold0','fold1','fold2','valid'):
            q=r83.split(d,k,part)[1];base=baseline(d,k,part);expert=fit(d,k,part,chosen['config'])
            p=mix(base,expert,chosen['weight']);metric=chosen['metric']
            parts.append(dict(part=part,rows=len(q),baseline=r83.score(metric,d['y'][q],base),
                candidate=r83.score(metric,d['y'][q],p),gain=r83.gain(metric,d['y'][q],base,p)))
        r=dict(**chosen,confirmation=parts,accepted=all(p['gain']>0 for p in parts));records.append(r);emit(stage='confirmation',**r)
        b.save(OUT/'confirmation.json',dict(status='IN_PROGRESS',signature=signature(),records=records,frozen_sha256=b.sha(OUT/'frozen_selection.json')))
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=signature(),records=records,
        accepted=[r['task'] for r in records if r['accepted']],frozen_sha256=b.sha(OUT/'frozen_selection.json')))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['develop','confirm']);args=ap.parse_args();d=load()
    (develop if args.stage=='develop' else confirm)(d)
