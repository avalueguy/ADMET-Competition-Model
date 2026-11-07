"""New leaf-wise LightGBM experts, selected before fixed confirmation gates."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v146_current_baseline as v
import v148_experiment_runner as runner
b, r83 = v.b, v.r83
OUT = ROOT / 'admet_model_outputs_v189'
SEEDS = (1894026, 1894027)
CONFIGS = (dict(view='augmented', leaves=7, trees=900, extra=False),
           dict(view='full', leaves=15, trees=900, extra=True))
METHOD = ('Deterministic leaf-wise LightGBM, two-seed ensembles. Seven-leaf shape/'
          'pharmacophore/assay expert and fifteen-leaf randomized-threshold full '
          'fragment expert. Fit-only constant/duplicate-column pruning, native '
          'missing values, L1 regression for MAE and rank-normal regression for '
          'Spearman. All21 improvable endpoints, original v148 gates and blends.')


def sig():
    return hashlib.sha256((b.sha(__file__) + v.sig() + b.sha(runner.__file__) +
                           importlib.metadata.version('lightgbm')).encode()).hexdigest()


def key(c):
    return c['view'] + '_leaves' + str(c['leaves'])


def design(d, k, part, c):
    fit, q, x, qx, source = v.old.features(d, k, part, c['view'])
    x, qx = x.astype(np.float32), qx.astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    qx[~np.isfinite(qx)] = np.nan
    # Only fitting rows decide whether a feature is constant or duplicated.
    keep, seen = [], set()
    for j in range(x.shape[1]):
        a = x[:, j]
        values = a[np.isfinite(a)]
        if not len(values) or (np.ptp(values) == 0 and len(values) == len(a)):
            continue
        h = hashlib.sha256(np.nan_to_num(a, nan=np.inf).tobytes()).digest()
        if h not in seen:
            seen.add(h)
            keep.append(j)
    columns = np.array(keep, dtype=np.int64)
    assert len(columns) and k not in set(d['tid'][source])
    return dict(fit=fit, query=q, source=source, columns=columns,
                x=x[:, columns], qx=qx[:, columns])


def transform(d, k, fit):
    y = d['y'][fit]
    metric = d['metrics'][k]
    center = float(np.median(y)) if metric == 'MAE' else 0.
    scale = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05) if metric == 'MAE' else 1.
    target = ((y - center) / scale if metric == 'MAE' else
              r83.ranknorm(y) if metric == 'Spearman' else y)
    return target, center, scale


def predict(d, k, part, c):
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    file = folder / 'predictions.npz'
    if file.exists():
        z = np.load(file)
        assert str(z['signature']) == sig()
        np.testing.assert_array_equal(z['query'], v.split(d, k, part)[1])
        return z['p'].copy()
    start = time.monotonic()
    a = design(d, k, part, c)
    metric = d['metrics'][k]
    target, center, scale = transform(d, k, a['fit'])
    params = dict(objective='binary' if metric in ('AUROC', 'AUPRC') else
                  'regression_l1' if metric == 'MAE' else 'regression',
                  learning_rate=.025, num_leaves=c['leaves'], max_depth=-1,
                  min_data_in_leaf=max(8, min(60, len(target) // 30)),
                  min_sum_hessian_in_leaf=.5, lambda_l1=.1, lambda_l2=25.,
                  path_smooth=15., max_bin=63, feature_pre_filter=False,
                  feature_fraction=.7 if c['view'] == 'augmented' else .25,
                  bagging_fraction=.85, bagging_freq=1, extra_trees=c['extra'],
                  deterministic=True, force_col_wise=True, num_threads=2, verbosity=-1)
    folder.mkdir(parents=True, exist_ok=True)
    members = []
    for seed in SEEDS:
        model = lgb.train(dict(params, seed=seed, data_random_seed=seed,
                               feature_fraction_seed=seed, bagging_seed=seed,
                               extra_seed=seed), lgb.Dataset(a['x'], label=target),
                          num_boost_round=c['trees'])
        p = model.predict(a['qx'], num_threads=2)
        mp = folder / ('model_' + str(seed) + '.txt')
        model.save_model(str(mp))
        replay = lgb.Booster(model_file=str(mp))
        np.testing.assert_array_equal(p, replay.predict(a['qx'], num_threads=2))
        members.append(p)
    p = np.mean(members, axis=0) * scale + center
    if d['tasks'][k] == 'ppbr_az':
        p = p.clip(0, 100)
    np.savez_compressed(file, signature=sig(), fit=a['fit'], query=a['query'],
                        source=a['source'], columns=a['columns'], p=p,
                        center=center, scale=scale)
    print(json.dumps(dict(stage='train', task=d['tasks'][k], part=part, config=c,
                          seconds=round(time.monotonic() - start))), flush=True)
    return p


def smoke(d):
    k = d['tasks'].index('caco2_wang')
    a = design(d, k, 'fold0', CONFIGS[0])
    saved = d['y'][a['query']].copy()
    try:
        d['y'][a['query']] = 888888.
        d.pop('_v96_assays', None)
        z = design(d, k, 'fold0', CONFIGS[0])
        for name in a:
            np.testing.assert_array_equal(a[name], z[name])
    finally:
        d['y'][a['query']] = saved
        d.pop('_v96_assays', None)
    target, _, _ = transform(d, k, a['fit'])
    model = lgb.train(dict(objective='regression_l1', num_leaves=3, verbosity=-1,
                           num_threads=2, deterministic=True, force_col_wise=True,
                           seed=1894026), lgb.Dataset(a['x'], label=target), num_boost_round=7)
    p = model.predict(a['qx'][:11], num_threads=2)
    replay = lgb.Booster(model_str=model.model_to_string())
    np.testing.assert_array_equal(p, replay.predict(a['qx'][:11], num_threads=2))
    b.save(OUT / 'smoke.json', dict(status='PASS', signature=sig(),
           query_label_perturbation_invariant=True, independent_model_reload=True,
           lightgbm=importlib.metadata.version('lightgbm')))


def main():
    d = v.load()
    smoke(d)
    runner.run(sys.modules[__name__], d)


if __name__ == '__main__':
    smoke(v.load()) if '--smoke' in sys.argv else main()
