"""Matched-row XGBoost MAE/Huber versus RMSE, with frozen confirmation.

This experiment uses the v77 audited data loader, with all its limitations.
No inherited baseline is represented as exact v73 OOF. No test Y are opened.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import admet_v77_ranking as data

ROOT = data.ROOT
OUT = ROOT / 'admet_model_outputs_v78'
TASKS = ('caco2_wang', 'ld50_zhu', 'lipophilicity_astrazeneca', 'ppbr_az', 'solubility_aqsoldb')
METHODS = ('rmse', 'mae', 'huber')
VIEWS = ('physchem', 'multiview')
SEEDS = (782026, 782027)
MAX_TREES = 1600
PRIOR = {}


def signature():
    return hashlib.sha256((data.signature() + data.sha(__file__)).encode()).hexdigest()


def prior(d, task, rows):
    # Priors are computed from molecular structure, without experimental labels.
    col = 0 if task == 'lipophilicity_astrazeneca' else 1 if task == 'solubility_aqsoldb' else None
    return np.zeros(len(rows)) if col is None else PRIOR['values'][d['mi'][rows], col].astype(float)


def prepare(d):
    global PRIOR
    with np.load(ROOT / 'admet_model_outputs_v73/features/views.npz') as z:
        PRIOR['values'] = z['priors'].copy()
    audit = dict(version=78, signature=signature(), tasks=TASKS, methods=METHODS, views=VIEWS,
        seeds=SEEDS, max_trees=MAX_TREES, xgboost_version=xgb.__version__,
        outer_folds='Exact reconstructed v24 historical folds; SHA256 verified. Current canonical/scaffold exclusion checked for every fitted task.',
        selection_parts=['fold0', 'fold1'], confirmation_parts=['fold2', 'valid'],
        prediction='Equal-seed mean. Train residual of structure-only LogP/ESOL prior for lipo/solubility; direct standardized targets otherwise.',
        stopping='Inner groups within outer fitting data. Inner RMSE and MAE models both stop on MAE. Outer refit uses fixed inner best iteration.',
        selection='At most one task. Fixed 0.5 expert/anchor blend. Both selection folds must improve vs same-row RMSE and historical proxy by >=1 percent. Freeze before confirmation.',
        confirmation='Positive gain vs RMSE and >1 percent proxy MAE improvement in fold2 and valid. Scaffold bootstrap P(gain>0)>=0.9 on each confirmation partition.',
        full_v73_oof_available=False, test_csv_opened=False, test_labels_used=False,
        external_data=False, pretrained_weights=False, official_score=None,
        limitations='Proxy v57 is not exact v73 OOF. Development labels historically reused. Internal validation cannot prove an official gain.')
    data.save(OUT / 'protocol.json', audit)
    data.save(OUT / 'data_audit.json', dict(input_hashes=d['sources'], signature=signature(),
        test_csv_opened=False, test_labels_used=False, chemistry_cache_regenerated=False))


def fit(d, k, rows, q, view, method, seed, dest, trees=None):
    dest.mkdir(parents=True, exist_ok=True)
    cache = dest / 'predictions.npz'
    if cache.exists():
        z = np.load(cache)
        assert str(z['signature']) == signature()
        np.testing.assert_array_equal(z['fit'], rows)
        np.testing.assert_array_equal(z['query'], q)
        return z['p'].copy()
    started = time.time()
    assert np.all(d['split'][rows] < 2) and not set(rows) & set(q)
    if np.all(d['split'][q] < 2):
        assert not set(d['groups'][rows]) & set(d['groups'][q])
    task = d['tasks'][k]
    raw = d['y'][rows] - prior(d, task, rows)
    center = float(np.median(raw))
    scale = max(float(np.subtract(*np.quantile(raw, [.75, .25]))), .01)
    y = ((raw - center) / scale).astype(np.float32)
    x = d['views'][view][d['mi'][rows]]
    qx = d['views'][view][d['mi'][q]]
    objective = {'rmse': 'reg:squarederror', 'mae': 'reg:absoluteerror', 'huber': 'reg:pseudohubererror'}[method]
    cfg = dict(objective=objective, eval_metric='mae', n_estimators=MAX_TREES if trees is None else int(trees),
        max_depth=5, learning_rate=.035, min_child_weight=4, subsample=.85, colsample_bytree=.65,
        reg_lambda=15., reg_alpha=.02, max_bin=128, tree_method='hist', device='cuda', n_jobs=4,
        random_state=seed, verbosity=0, base_score=0.)
    if method == 'huber':
        cfg['huber_slope'] = .3
    if trees is None:
        cfg['early_stopping_rounds'] = 100
    net = xgb.XGBRegressor(**cfg)
    _, inv, count = np.unique(d['canonical'][rows], return_inverse=True, return_counts=True)
    weight = 1 / count[inv]; weight /= weight.mean()
    if trees is None:
        truth = ((d['y'][q] - prior(d, task, q) - center) / scale).astype(np.float32)
        net.fit(x, y, sample_weight=weight, eval_set=[(qx, truth)], verbose=False)
        actual_trees = net.best_iteration + 1
        stopped_trees = net.get_booster().num_boosted_rounds()
    else:
        net.fit(x, y, sample_weight=weight, verbose=False)
        actual_trees = int(trees)
        stopped_trees = actual_trees
    def predict(model):
        p = model.predict(qx, iteration_range=(0, actual_trees)).astype(float) * scale + center + prior(d, task, q)
        return p.clip(0, 100) if task == 'ppbr_az' else p
    p = predict(net)
    assert np.isfinite(p).all()
    path = dest / 'model.ubj'
    net.save_model(path)
    replay = xgb.XGBRegressor()
    replay.load_model(path)
    np.testing.assert_allclose(p, predict(replay), rtol=1e-7, atol=1e-7)
    np.savez_compressed(cache, signature=signature(), fit=rows, query=q, p=p)
    data.save(dest / 'metadata.json', dict(signature=signature(), task=task, view=view, method=method,
        config=cfg, seed=seed, actual_trees=actual_trees, stopped_trees=stopped_trees,
        target=dict(center=center, scale=scale, prior=task if task in ('lipophilicity_astrazeneca', 'solubility_aqsoldb') else None),
        model_sha256=data.sha(path), predictions_sha256=data.sha(cache), independent_reload_passed=True,
        fit_rows=len(rows), query_rows=len(q), seconds=time.time() - started,
        mae=float(mean_absolute_error(d['y'][q], p)) if np.all(d['split'][q] < 2) else None))
    data.emit(stage='v78_fit', task=task, view=view, method=method, seed=seed,
              folder=str(dest.relative_to(OUT)), trees=actual_trees, seconds=round(time.time() - started))
    return p


def outer(d, k, part, view, method):
    rows, q = data.rows(d, k, part)
    dest = OUT / d['tasks'][k] / part / view / method
    if part == 'final':
        counts = [json.loads((OUT / d['tasks'][k] / p / view / method / 'inner/metadata.json').read_text())['actual_trees']
                  for p in ('fold0', 'fold1', 'fold2', 'valid')]
        trees = int(np.median(counts))
    else:
        tr, va = data.inner(d, rows, k)
        fit(d, k, tr, va, view, method, SEEDS[0], dest / 'inner')
        trees = json.loads((dest / 'inner/metadata.json').read_text())['actual_trees']
    p = np.mean([fit(d, k, rows, q, view, method, s, dest / str(s), trees) for s in SEEDS], axis=0)
    np.savez_compressed(dest / 'ensemble.npz', signature=signature(), query=q, p=p, trees=trees)
    return p


def gain(y, old, p):
    return float((mean_absolute_error(y, old) - mean_absolute_error(y, p)) / mean_absolute_error(y, old))


def develop(d):
    for task in TASKS:
        k = d['tasks'].index(task)
        for part in ('fold0', 'fold1'):
            for view in VIEWS:
                for method in METHODS:
                    p = outer(d, k, part, view, method)
                    q = data.rows(d, k, part)[1]
                    data.emit(stage='v78_score', task=task, part=part, view=view, method=method,
                              mae=float(mean_absolute_error(d['y'][q], p)))
    return select(d)


def select(d):
    records = []
    for task in TASKS:
        k = d['tasks'].index(task)
        for view in VIEWS:
            for method in METHODS[1:]:
                parts = []
                for part in ('fold0', 'fold1'):
                    q = data.rows(d, k, part)[1]
                    root = OUT / task / part / view
                    p = np.load(root / method / 'ensemble.npz')['p']
                    control = np.load(root / 'rmse/ensemble.npz')['p']
                    proxy = data.proxy(d, q, part)
                    mixed = .5 * p + .5 * proxy
                    parts.append(dict(part=part, n=len(q), expert_mae=float(mean_absolute_error(d['y'][q], p)),
                        control_mae=float(mean_absolute_error(d['y'][q], control)),
                        proxy_mae=float(mean_absolute_error(d['y'][q], proxy)),
                        blend_mae=float(mean_absolute_error(d['y'][q], mixed)),
                        gain_control=gain(d['y'][q], control, p), gain_proxy=gain(d['y'][q], proxy, mixed)))
                eligible = all(r['gain_control'] >= .01 and r['gain_proxy'] >= .01 for r in parts)
                records.append(dict(task=task, view=view, method=method, parts=parts,
                    eligible=eligible, objective=min(r['gain_proxy'] for r in parts)))
    eligible = [r for r in records if r['eligible']]
    chosen = max(eligible, key=lambda r: r['objective']) if eligible else None
    result = dict(signature=signature(), status='FROZEN_BEFORE_CONFIRMATION', candidates=records,
                  selected=chosen, full_v73_oof_available=False)
    path = OUT / 'frozen_selection.json'
    if path.exists():
        assert json.loads(path.read_text()) == result
    else:
        data.save(path, result)
    data.emit(stage='v78_selection', selected=chosen)
    return chosen


def bootstrap(y, old, new, groups, seed):
    blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(2000):
        ix = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        draws.append(gain(y[ix], old[ix], new[ix]))
    return dict(percentile_95=np.quantile(draws, [.025, .975]).tolist(),
                probability_gain_positive=float(np.mean(np.array(draws) > 0)))


def confirm(d):
    selected = json.loads((OUT / 'frozen_selection.json').read_text())['selected']
    if selected is None:
        data.save(OUT / 'confirmation.json', dict(status='NO_SELECTED_CANDIDATE', accepted=False))
        return False
    task, view, method = (selected[k] for k in ('task', 'view', 'method'))
    k = d['tasks'].index(task)
    parts = []
    for part in ('fold2', 'valid'):
        control = outer(d, k, part, view, 'rmse')
        p = outer(d, k, part, view, method)
        q = data.rows(d, k, part)[1]
        proxy = data.proxy(d, q, part)
        mixed = .5 * proxy + .5 * p
        y = d['y'][q]
        parts.append(dict(part=part, n=len(q), expert_mae=float(mean_absolute_error(y, p)),
            control_mae=float(mean_absolute_error(y, control)), proxy_mae=float(mean_absolute_error(y, proxy)),
            blend_mae=float(mean_absolute_error(y, mixed)), gain_control=gain(y, control, p),
            gain_proxy=gain(y, proxy, mixed), bootstrap_proxy=bootstrap(y, proxy, mixed, d['groups'][q], 782100 + k)))
    accepted = all(r['gain_control'] > 0 and r['gain_proxy'] >= .01 and
        r['bootstrap_proxy']['probability_gain_positive'] >= .9 for r in parts)
    result = dict(signature=signature(), status='CONFIRMATION_COMPLETE', selected=selected,
        frozen_sha256=data.sha(OUT / 'frozen_selection.json'), parts=parts, accepted=accepted,
        full_v73_oof_available=False, new_platform_score=None)
    data.save(OUT / 'confirmation.json', result)
    data.emit(stage='v78_confirmation', **result)
    return accepted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('run', 'develop', 'select', 'confirm'))
    args = parser.parse_args()
    d = data.load_data()
    fold_path = ROOT / 'recovered_history/legacy_v24_folds.npy'
    assert data.sha(fold_path) == '3a24d8a1359df47c4197bff62a80809bbdc42f53db22c5dd1f478db2a95a4b98'
    d['folds'] = np.load(fold_path)
    d['sources']['recovered_history/legacy_v24_folds.npy'] = data.sha(fold_path)
    # Validate every planned exclusion before starting any model training.
    for task in TASKS:
        for part in ('fold0', 'fold1', 'fold2', 'valid'):
            data.rows(d, d['tasks'].index(task), part)
    prepare(d)
    if args.stage in ('run', 'develop'):
        develop(d)
    if args.stage == 'select':
        select(d)
    if args.stage in ('run', 'confirm'):
        confirm(d)
    data.save(OUT / 'pipeline_status.json', dict(status='COMPLETE', stage=args.stage, signature=signature(),
        official_score=None, pid=os.getpid()))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        data.save(OUT / 'pipeline_status.json', dict(status='FAILED', error=repr(exc), signature=signature()))
        raise
