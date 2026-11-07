"""Official-only chemical-neighbor assay transfer with frozen confirmation.

The target assay is removed from all measured and neighbor-derived inputs.
Other assay labels may be available on the same molecule, so this evaluates
conditional prediction, not wholly unseen-molecule generalization.
"""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from scipy.special import expit
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import admet_v77_ranking as b
import v83_residual as old

OUT = ROOT / 'admet_model_outputs_v84'
ANCHOR = ROOT / 'admet_submissions/v83_matched_residual/我从山北来带着兰花草-小分子药物预测技术创新赛-20261025.csv'
ANCHOR_SHA = '47d384497619e437d9baa295307cdfc8014b65402833ce75a91978f69c996c4c'
ORIGINAL = ROOT / 'admet_submissions/v73_v72/我从山北来带着兰花草-小分子药物预测技术创新赛-20261021.csv'
SEEDS = (844026, 844027)
CONFIGS = [
    dict(name='neighbors_d2_t200', view='neighbors', depth=2, trees=200),
    dict(name='neighbors_d4_t400', view='neighbors', depth=4, trees=400),
    dict(name='combined_d3_t250', view='combined', depth=3, trees=250),
    dict(name='physical_d4_t400', view='physical', depth=4, trees=400),
]
WEIGHTS = (.25, .5, 1.)
NEIGHBOR_COUNT = 32
PROFILE_FIELDS = ('weighted_mean_power4', 'weighted_mean_power16',
                  'weighted_sd_power4', 'max_tanimoto', 'mean_top32_tanimoto')


def signature():
    return hashlib.sha256((b.sha(__file__) + b.sha(old.__file__) +
                           b.sha(b.__file__)).encode()).hexdigest()


def load():
    d = b.load_data()
    d['legacy_folds'] = np.load(ROOT / 'recovered_history/legacy_v24_folds.npy')
    old.TASK_NAMES = d['tasks']
    assert b.sha(ANCHOR) == ANCHOR_SHA
    return d


def emit(**kw):
    print(json.dumps(kw, ensure_ascii=False), flush=True)


def profiles(d, final=False):
    """Learn no query labels: fixed top32 Tanimoto summaries per source task."""
    stage = 'train_valid' if final else 'train'
    folder = OUT / 'profiles' / stage
    path = folder / 'profiles.npz'
    if path.exists():
        with np.load(path) as z:
            assert str(z['signature']) == signature()
            return z['x'].copy()
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with np.load(ROOT / 'admet_model_outputs_v67/features/official_chemistry.npz') as z:
        x = z['x'][:, :4096].astype(np.float32)
        canonical = z['canonical']
    assert np.isfinite(x).all() and np.min(x) >= 0
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda')
    gx = torch.as_tensor(x, device=device)
    norms = gx.square().sum(1)
    result = np.full((len(x), len(d['tasks']), len(PROFILE_FIELDS)), np.nan, np.float32)
    audit_queries = np.unique([0,len(x)//2,len(x)-1])
    audits = []
    for k, task in enumerate(d['tasks']):
        source = np.flatnonzero((d['tid'] == k) & (d['split'] < (2 if final else 1)))
        assert len(source) and not np.any(d['split'][source] == 2)
        # A molecule measured repeatedly in one assay contributes once.
        unique, inverse = np.unique(d['canonical'][source], return_inverse=True)
        refs = np.array([d['mi'][source[np.flatnonzero(inverse == j)[0]]]
                         for j in range(len(unique))])
        values = np.array([np.median(d['y'][source[inverse == j]])
                           for j in range(len(unique))])
        assert np.array_equal(canonical[refs], unique)
        if d['metrics'][k] == 'MAE':
            center = float(np.median(values))
            scale = max(float(np.quantile(values, .75) - np.quantile(values, .25)), 1e-3)
            labels = (values - center) / scale
        elif d['metrics'][k] == 'Spearman':
            center, scale = 0., 1.
            labels = old.ranknorm(values)
        else:
            center, scale = 0., 1.
            labels = values
        gy = torch.as_tensor(labels, dtype=torch.float32, device=device)
        reference = gx[refs]
        refnorm = norms[refs]
        take = min(NEIGHBOR_COUNT, len(refs))
        sample_ix,sample_top=[],[]
        for start in range(0, len(x), 512):
            stop = min(start + 512, len(x))
            dot = gx[start:stop] @ reference.T
            sim = (dot / (norms[start:stop, None] + refnorm[None] - dot).clamp_min(1e-8)).clamp(0, 1)
            top, ix = torch.topk(sim, take, dim=1, sorted=True)
            for query in audit_queries[(audit_queries>=start)&(audit_queries<stop)]:
                sample_ix.append(ix[int(query)-start].cpu().numpy())
                sample_top.append(top[int(query)-start].cpu().numpy())
            y = gy[ix]
            w4 = top.pow(4).clamp_min(1e-12)
            w16 = top.pow(16).clamp_min(1e-12)
            mean4 = (w4 * y).sum(1) / w4.sum(1)
            mean16 = (w16 * y).sum(1) / w16.sum(1)
            sd = ((w4 * (y - mean4[:, None]).square()).sum(1) / w4.sum(1)).sqrt()
            result[start:stop, k] = torch.stack((mean4, mean16, sd, top[:, 0], top.mean(1)), 1).cpu().numpy()
        audits.append(dict(task=task,source_rows=len(source),source_molecules=len(refs),
                           source_splits=[0,1] if final else [0],center=center,scale=scale,
                           refs_sha256=hashlib.sha256(refs.tobytes()).hexdigest(),
                           values_sha256=hashlib.sha256(values.tobytes()).hexdigest()))
        # Store exact source identities and labels used to enable independent replay.
        np.savez_compressed(folder / f'{task}.npz', source=source, refs=refs,
                            values=values, labels=labels, center=center, scale=scale,
                            audit_queries=audit_queries,audit_ix=np.stack(sample_ix),audit_top=np.stack(sample_top))
        emit(stage='profile',source_task=task,source_rows=len(source),seconds=round(time.monotonic()-started,1))
    assert np.isfinite(result).all()
    np.savez_compressed(path,x=result,signature=signature())
    b.save(folder/'audit.json',dict(signature=signature(),source_splits=[0,1] if final else [0],
        fields=PROFILE_FIELDS,neighbors=NEIGHBOR_COUNT,records=audits,
        query_labels_used=False,external_data=False,test_csv_opened=False,
        profile_sha256=b.sha(path)))
    return result


def baseline(d, k, part):
    if part == 'final':
        return np.array([float(r['y_pred']) for r in b.table(ANCHOR) if r['task']==d['tasks'][k]])
    p = old.baseline(d,k,part)
    if d['tasks'][k] == 'ppbr_az':
        cfg = old.CONFIG_MAP['aux_d3_t250']
        p = .5*p + .5*old.fit(d,k,part,cfg,old.SEEDS,'confirmation')
    return p


def mix(metric, oldp, new, weight):
    if metric == 'MAE':
        return (1-weight)*oldp + weight*new
    return (1-weight)*rankdata(oldp)/len(oldp) + weight*rankdata(new)/len(new)


def design(d,k,rows,z,cfg,profile,aux,cols):
    other = [j for j in range(len(d['tasks'])) if j != k]
    # All five target-task neighbor fields are excluded, not just its label.
    assert k not in other and k not in cols
    pieces = [profile[d['mi'][rows]][:,other].reshape(len(rows),-1)]
    if cfg['view'] in ('combined','physical'):
        pieces.append(aux[rows])
    if cfg['view'] == 'physical':
        pieces.append(d['views']['physchem'][d['mi'][rows]])
    pieces.append(z[:,None])
    return np.concatenate(pieces,axis=1).astype(np.float32)


def fit(d,k,part,cfg,profile):
    task, metric = d['tasks'][k],d['metrics'][k]
    iscls = metric in ('AUROC','AUPRC')
    fitrows,q = old.split(d,k,part)
    folder = OUT/'models'/task/part/cfg['name']
    path = folder/'predictions.npz'
    if path.exists():
        with np.load(path) as z:
            assert str(z['signature'])==signature()
            np.testing.assert_array_equal(z['query'],q)
            assert z['seeds'].tolist()==list(SEEDS)
            return z['p'].copy()
    folder.mkdir(parents=True,exist_ok=True)
    aux,cols,source,counts = old.aux(d,k,part=='final')
    base = old.raw_proxy(d,fitrows)
    qp = (np.array([float(r['y_pred']) for r in b.table(ORIGINAL) if r['task']==task])
          if part=='final' else old.raw_proxy(d,q))
    y = d['y'][fitrows]
    if iscls:
        z,qz = old.ranklog(base),old.ranklog(qp)
        beta = old.calibrate(y,z)
        off,qoff = beta[0]+beta[1]*z,beta[0]+beta[1]*qz
        yt,scale = y,1.
        state = dict(beta=beta.tolist(),scale=scale)
    elif metric=='MAE':
        center = float(np.median(y))
        scale = max(float(np.quantile(y,.75)-np.quantile(y,.25)),1e-3)
        z,qz = (base-center)/scale,(qp-center)/scale
        yt = (y-base)/scale
        off,qoff = np.zeros(len(base)),qp
        state = dict(center=center,scale=scale)
    else:
        z,qz = old.ranknorm(base),old.ranknorm(qp)
        target = old.ranknorm(y)
        xx = np.c_[np.ones(len(z)),z]
        beta = np.linalg.solve(xx.T@xx+np.diag([1e-8,1.]),xx.T@target)
        beta[1] = max(beta[1],.01)
        yt = target-xx@beta
        off,qoff = np.zeros(len(base)),beta[0]+beta[1]*qz
        scale = 1.
        state = dict(beta=beta.tolist(),scale=scale)
    x = design(d,k,fitrows,z,cfg,profile,aux,cols)
    qx = design(d,k,q,qz,cfg,profile,aux,cols)
    started = time.monotonic()
    members,hashes = [],{}
    for seed in SEEDS:
        kw=dict(iterations=cfg['trees'],depth=cfg['depth'],learning_rate=.03,l2_leaf_reg=50,
                random_seed=seed,thread_count=4,verbose=False,allow_writing_files=False,
                bootstrap_type='Bernoulli',subsample=.85,task_type='GPU',devices='0',border_count=64)
        if iscls:
            model=CatBoostClassifier(loss_function='Logloss',**kw)
            model.fit(Pool(x,yt,baseline=off))
        else:
            model=CatBoostRegressor(loss_function='MAE' if metric=='MAE' else 'RMSE',**kw)
            model.fit(x,yt)
        member=qoff+model.predict(qx,prediction_type='RawFormulaVal')*scale
        members.append(member)
        mp=folder/f'model_{seed}.cbm'
        model.save_model(str(mp));hashes[mp.name]=b.sha(mp)
    members=np.stack(members)
    pred=expit(members.mean(0)) if iscls else members.mean(0)
    if task=='ppbr_az':pred=pred.clip(0,100)
    assert np.isfinite(pred).all()
    np.savez_compressed(path,signature=signature(),fit=fitrows,query=q,p=pred,seeds=SEEDS,
                        members=members,x=x,qx=qx,off=off,qoff=qoff,source=source,
                        known_counts=counts[q],base_fit=base,base_query=qp)
    b.save(folder/'metadata.json',dict(signature=signature(),task=task,part=part,metric=metric,
        config=cfg,seeds=SEEDS,state=state,model_hashes=hashes,predictions_sha256=b.sha(path),
        target_assay_value_mask_and_profiles_absent=True,seconds=time.monotonic()-started))
    return pred


def develop(d):
    OUT.mkdir(exist_ok=True)
    b.save(OUT/'protocol.json',dict(version=84,signature=signature(),baseline='v83',
        anchor_sha256=ANCHOR_SHA,baseline_official_score_user_reported=50.870,
        seeds=SEEDS,configs=CONFIGS,weights=WEIGHTS,select_parts=['fold0','fold1'],
        confirm_parts=['fold2','valid'],neighbors=NEIGHBOR_COUNT,fields=PROFILE_FIELDS,
        selection='Select max minimum fold gain; practical margin .003 relative MAE, .0005 AUROC/AUPRC, .001 Spearman. Freeze task/config/weight before confirmation.',
        confirmation='All four parts must improve using the same two-seed ensemble; no fallback tuning after confirmation.',
        source_assays='Official train during development, train+valid final; target assay value, mask and all profile fields excluded.',
        new_external_data=False,test_labels_used=False,historical_labels_reused=True,
        baseline_limitations='Historical proxies and replayed residual layers, not full exact v83 OOF; conditional other-assay information.'))
    profile=profiles(d,False)
    records=[]
    for k,task in enumerate(d['tasks']):
        metric=d['metrics'][k]
        parts={part:(old.split(d,k,part)[1],baseline(d,k,part)) for part in ('fold0','fold1')}
        for cfg in CONFIGS:
            candidates={part:fit(d,k,part,cfg,profile) for part in parts}
            for w in WEIGHTS:
                gains=[]
                for part,(q,base) in parts.items():
                    mixed=mix(metric,base,candidates[part],w)
                    gains.append(dict(part=part,rows=len(q),baseline=old.score(metric,d['y'][q],base),
                        candidate=old.score(metric,d['y'][q],mixed),gain=old.gain(metric,d['y'][q],base,mixed)))
                records.append(dict(task=task,metric=metric,config=cfg,weight=w,parts=gains,
                                    objective=min(r['gain'] for r in gains)))
        best=max((r for r in records if r['task']==task),key=lambda r:r['objective'])
        emit(stage='development',task=task,best=best)
        b.save(OUT/'development.json',dict(status='IN_PROGRESS',signature=signature(),records=records))
    selected=[]
    for task in d['tasks']:
        best=max((r for r in records if r['task']==task),key=lambda r:r['objective'])
        margin=.003 if best['metric']=='MAE' else .001 if best['metric']=='Spearman' else .0005
        if best['objective']>margin:selected.append(best)
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=signature(),records=records))
    b.save(OUT/'frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=signature(),
        selected=selected,confirmation_inspected=False,seeds=SEEDS))
    emit(stage='frozen',selected=selected)


def confirm(d):
    frozen=json.loads((OUT/'frozen_selection.json').read_text())
    assert frozen['signature']==signature()
    profile=profiles(d,False)
    records=[]
    for chosen in frozen['selected']:
        k=d['tasks'].index(chosen['task']);metric=chosen['metric'];parts=[]
        for part in ('fold0','fold1','fold2','valid'):
            q=old.split(d,k,part)[1];base=baseline(d,k,part)
            pred=fit(d,k,part,chosen['config'],profile)
            mixed=mix(metric,base,pred,chosen['weight'])
            parts.append(dict(part=part,rows=len(q),baseline=old.score(metric,d['y'][q],base),
                candidate=old.score(metric,d['y'][q],mixed),gain=old.gain(metric,d['y'][q],base,mixed)))
        record=dict(**chosen,confirmation=parts,accepted=all(r['gain']>0 for r in parts))
        records.append(record);emit(stage='confirmation',**record)
        b.save(OUT/'confirmation.json',dict(status='IN_PROGRESS',records=records,
             frozen_sha256=b.sha(OUT/'frozen_selection.json')))
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=signature(),records=records,
         accepted=[r['task'] for r in records if r['accepted']],frozen_sha256=b.sha(OUT/'frozen_selection.json')))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['profiles','develop','confirm']);ap.add_argument('--final',action='store_true')
    args=ap.parse_args();d=load()
    if args.stage=='profiles':profiles(d,args.final)
    elif args.stage=='develop':develop(d)
    else:confirm(d)
