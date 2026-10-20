"""Remote direct average-precision ranking with scaffold-only inner selection."""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as ap
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRanker

from admet_v16_models import load_task
from admet_v6_models import Preprocess
from admet_v6_runner import remote_only, partition
from admet_v12_models import inner_split
from admet_v20_residual import anchor_oof, anchor_valid, anchor_test, read_json, save_json, sha
from admet_v20_rare import TASKS, combine

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'admet_model_outputs_v20_map'
PROTOCOL = ROOT / 'admet_v20_map_protocol.json'


def raw(d):
    dense = np.concatenate([d['molformer'], d['emb'], d['desc'], d['maccs']], axis=1).astype(np.float32)
    sparse = np.concatenate([np.log1p(d['count']), np.log1p(d['fcfp'])], axis=1).astype(np.float32)
    return np.concatenate([dense, sparse], axis=1), dense.shape[1]


def columns(x, fit, dense, p):
    variance = np.nanvar(x[fit, dense:], axis=0)
    chosen = np.argsort(variance, kind='stable')[-min(p['fingerprint_columns'], len(variance)):]
    return np.r_[np.arange(dense), dense+chosen]


def ranker(p, seed, depth, iterations, inner=False):
    kw = dict(objective='rank:map', eval_metric='map', n_estimators=int(iterations),
              max_depth=depth, learning_rate=p['learning_rate'], reg_lambda=15., reg_alpha=.1,
              min_child_weight=1., subsample=.9, colsample_bytree=.8,
              tree_method='hist', device='cuda', max_bin=128, random_state=seed,
              n_jobs=4, lambdarank_pair_method='topk', lambdarank_num_pair_per_sample=16)
    if inner:
        kw['early_stopping_rounds'] = p['early_stopping_rounds']
    return XGBRanker(**kw)


def materialize(task, d, x, dense, fit, query, stage):
    dest = OUT / task / stage
    dest.mkdir(parents=True, exist_ok=True)
    cache, meta = dest / 'predictions.npz', dest / 'metadata.json'
    p = read_json(PROTOCOL)
    if cache.exists() and meta.exists():
        info = read_json(meta)
        assert info['protocol_sha256'] == sha(PROTOCOL)
        with np.load(cache) as z:
            np.testing.assert_array_equal(z['fit'], fit)
            np.testing.assert_array_equal(z['query'], query)
            return z['prediction'].copy()
    labels = pd.concat([d['frames']['train'], d['frames']['valid']], ignore_index=True).Y.to_numpy(float)
    assert len(fit) and np.max(fit) < len(labels) and not set(fit) & set(query)
    y = labels[fit]
    a, b, split_seed = inner_split(d, fit, y, 'AUPRC')
    cc = columns(x, fit[a], dense, p)
    pp = Preprocess().fit(x[fit[a]][:, cc])
    xt, xv = pp.transform(x[fit[a]][:, cc]), pp.transform(x[fit[b]][:, cc])
    iterations = []
    for seed, depth in zip(p['seeds'], p['depths']):
        m = ranker(p, seed, depth, p['max_iterations'], inner=True)
        m.fit(xt, y[a], group=[len(a)], eval_set=[(xv, y[b])], eval_group=[[len(b)]], verbose=False)
        iterations.append(max(50, int(m.best_iteration)+1))
        del m
    cc = columns(x, fit, dense, p)
    pp = Preprocess().fit(x[fit][:, cc])
    xt, xq = pp.transform(x[fit][:, cc]), pp.transform(x[query][:, cc])
    joblib.dump(dict(preprocess=pp, columns=cc), dest / 'preprocess.joblib', compress=3)
    preds = []
    for i, (seed, depth, niter) in enumerate(zip(p['seeds'], p['depths'], iterations)):
        m = ranker(p, seed, depth, niter)
        m.fit(xt, y, group=[len(fit)], verbose=False)
        q = m.predict(xq)
        model_path = dest / f'model_{i}.ubj'
        m.save_model(str(model_path))
        replay = XGBRanker()
        replay.load_model(str(model_path))
        replayed = replay.predict(xq)
        np.testing.assert_allclose(q, replayed, rtol=1e-6, atol=1e-6)
        q = replayed
        preds.append((rankdata(q, method='average')-.5)/len(q))
        del m, replay
    pred = np.mean(preds, axis=0)
    assert np.isfinite(pred).all()
    np.savez_compressed(cache, fit=fit, query=query, prediction=pred)
    save_json(meta, dict(task=task, stage=stage, fit_rows=len(fit), query_rows=len(query),
                        iterations=iterations, inner_split_seed=split_seed,
                        inner_scaffold_overlap=0, model_replay_passed=True,
                        protocol_sha256=sha(PROTOCOL),
                        files={q.name:sha(q) for q in dest.iterdir() if q != meta}))
    print(json.dumps(dict(stage='map_fit', task=task, partition=stage, iterations=iterations)), flush=True)
    return pred


def train(task):
    d = load_task(task)
    n, nv = len(d['frames']['train']), len(d['frames']['valid'])
    fold = partition(task, d)
    y = d['frames']['train'].Y.to_numpy(float)
    x, dense = raw(d)
    pred = np.empty(n)
    dest = OUT / task
    dest.mkdir(parents=True, exist_ok=True)
    for f in np.unique(fold):
        fit, query = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        assert not set(d['groups'][fit]) & set(d['groups'][query])
        assert not set(d['canonical'][fit]) & set(d['canonical'][query])
        pred[query] = materialize(task, d, x, dense, fit, query, f'fold{f}')
    pd.DataFrame(dict(row=np.arange(n), fold=fold, Y_true=y, Y_pred=pred)).to_csv(dest/'map_oof.csv', index=False)
    pv = materialize(task, d, x, dense, np.arange(n), np.arange(n,n+nv), 'valid')
    pd.DataFrame(dict(Y_true=d['frames']['valid'].Y, Y_pred=pv)).to_csv(dest/'map_valid.csv', index=False)
    evaluate(task)


def evaluate(task):
    dest = OUT / task
    y, fold, base = anchor_oof(task)
    vy, vb = anchor_valid(task)
    oof, valid = pd.read_csv(dest/'map_oof.csv'), pd.read_csv(dest/'map_valid.csv')
    np.testing.assert_array_equal(oof.row, np.arange(len(y)))
    np.testing.assert_array_equal(oof.fold, fold)
    np.testing.assert_array_equal(oof.Y_true, y)
    np.testing.assert_array_equal(valid.Y_true, vy)
    candidate, vc = oof.Y_pred.to_numpy(float), valid.Y_pred.to_numpy(float)
    grid = []
    for w in read_json(PROTOCOL)['weight_grid']:
        gains = []
        for f in np.unique(fold):
            mask = fold == f
            q = combine(base[mask], candidate[mask], w, 'rank')
            gains.append(float(ap(y[mask], q)-ap(y[mask], base[mask])))
        vg = float(ap(vy, combine(vb, vc, w, 'rank'))-ap(vy, vb))
        vals = np.array(gains+[vg])
        objective = float(vals.mean()-.15*vals.std()-.25*max(0, -vals.min()))
        eligible = w == 0 or (objective>1e-5 and np.mean(gains)>0 and vg>0 and sum(g>=-.002 for g in gains)>=2)
        grid.append(dict(mode='rank', weight=w, fold_gains=gains, oof_gain=float(np.mean(gains)), valid_gain=vg,
                         objective=objective, eligible=bool(eligible)))
    best = max([g for g in grid if g['eligible']], key=lambda g:(g['objective'],-g['weight']))
    save_json(dest/'decision.json', dict(task=task, selection=best, grid=grid, protocol_sha256=sha(PROTOCOL)))
    print(json.dumps(dict(stage='map_decision', task=task, **best)), flush=True)


def final(task):
    dest = OUT / task
    sel = read_json(dest/'decision.json')['selection']
    if sel['weight'] == 0:
        return
    d = load_task(task)
    total = len(d['frames']['train'])+len(d['frames']['valid'])
    x, dense = raw(d)
    q = materialize(task, d, x, dense, np.arange(total), np.arange(total,len(d['smiles'])), 'final')
    pd.DataFrame(dict(Y_pred=q)).to_csv(dest/'map_test.csv', index=False)


if __name__ == '__main__':
    remote_only()
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['train','evaluate','final'])
    p.add_argument('--tasks', nargs='*')
    args = p.parse_args()
    for task in args.tasks or TASKS:
        globals()[args.stage](task)
