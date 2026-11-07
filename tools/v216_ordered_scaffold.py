"""Ordered boosting with fit-only scaffold-balanced training risk."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v146_current_baseline as v
import v148_experiment_runner as runner
b, r83 = v.b, v.r83
OUT = ROOT / 'admet_model_outputs_v216'
SEEDS = (2161029, 2161030)
CONFIGS = (dict(depth=5, trees=700),)
METHOD = ('Ordered CatBoost computes ordered target gradients instead of plain '
          'boosting. Fit-only inverse-square-root scaffold frequencies balance '
          'training risk, clipped[.25,4] before mean normalization. Original full '
          'molecular counts/physchem/shape/pharmacophore and target-excluded '
          'measured context, no normalizedv210 representations or predictions. '
          'One fixed configuration and two seeds; originalv148 gates unchanged.')


def sig():
    return hashlib.sha256((b.sha(__file__) + v.sig() + b.sha(runner.__file__) +
                           importlib.metadata.version('catboost')).encode()).hexdigest()


def key(c):
    return 'depth' + str(c['depth'])


def scaffold_weights(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1. / np.sqrt(counts[inverse].astype(float))
    weights /= weights.mean()
    weights = weights.clip(.25, 4.)
    return weights / weights.mean()


def design(d, k, part):
    fit, query, x, qx, source = v.old.features(d, k, part, 'full')
    extra = np.c_[d['conformer_features'], d['pharmacophore']]
    x = np.c_[x, extra[d['mi'][fit]]].astype(np.float32)
    qx = np.c_[qx, extra[d['mi'][query]]].astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    qx[~np.isfinite(qx)] = np.nan
    metric, y = d['metrics'][k], d['y'][fit]
    center = float(np.median(y)) if metric == 'MAE' else 0.
    scale = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05) if metric == 'MAE' else 1.
    target = ((y - center) / scale if metric == 'MAE' else
              r83.ranknorm(y) if metric == 'Spearman' else y)
    weights = scaffold_weights(d['groups'][fit])
    assert np.isfinite(weights).all() and (weights > 0).all()
    assert not set(fit) & set(query) and k not in set(d['tid'][source])
    state = dict(fit=fit, query=query, source=source, target=target,
                 center=center, scale=scale, weights=weights,
                 fit_scaffolds=d['groups'][fit].copy())
    return dict(x=x, qx=qx, state=state)


def estimator(metric, c, seed):
    return (CatBoostClassifier if metric in ('AUROC', 'AUPRC') else CatBoostRegressor)(
        loss_function='Logloss' if metric in ('AUROC', 'AUPRC') else 'MAE' if metric == 'MAE' else 'RMSE',
        boosting_type='Ordered', iterations=c['trees'], depth=c['depth'],
        learning_rate=.03, l2_leaf_reg=30., random_seed=seed, thread_count=4,
        verbose=False, allow_writing_files=False, bootstrap_type='Bayesian',
        bagging_temperature=1., random_strength=1., rsm=.5,
        border_count=32, task_type='CPU')


def output(metric, raw, state, task):
    p = expit(raw) if metric in ('AUROC', 'AUPRC') else raw * state['scale'] + state['center']
    return p.clip(0, 100) if task == 'ppbr_az' else p


def replay(d, k, part, c):
    from v161_verify_and_assemble import same
    a = design(d, k, part)
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    same(a['state'], joblib.load(folder / 'transform.joblib'))
    hashes = {'transform.joblib': b.sha(folder / 'transform.joblib')}
    members = []
    for seed in SEEDS:
        file = folder / ('model_' + str(seed) + '.cbm')
        net = CatBoostClassifier() if d['metrics'][k] in ('AUROC', 'AUPRC') else CatBoostRegressor()
        net.load_model(str(file))
        members.append(net.predict(a['qx'], prediction_type='RawFormulaVal'))
        hashes[file.name] = b.sha(file)
    return output(d['metrics'][k], np.mean(members, axis=0), a['state'], d['tasks'][k]), hashes


def predict(d, k, part, c):
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    file = folder / 'predictions.npz'
    if file.exists():
        z = np.load(file)
        assert str(z['signature']) == sig()
        np.testing.assert_array_equal(z['query'], v.split(d, k, part)[1])
        return z['p'].copy()
    start = time.monotonic()
    a = design(d, k, part)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(a['state'], folder / 'transform.joblib', compress=3)
    members = []
    for seed in SEEDS:
        net = estimator(d['metrics'][k], c, seed)
        net.fit(a['x'], a['state']['target'], sample_weight=a['state']['weights'])
        raw = net.predict(a['qx'], prediction_type='RawFormulaVal')
        path = folder / ('model_' + str(seed) + '.cbm')
        net.save_model(str(path))
        reloaded = CatBoostClassifier() if d['metrics'][k] in ('AUROC', 'AUPRC') else CatBoostRegressor()
        reloaded.load_model(str(path))
        np.testing.assert_array_equal(raw, reloaded.predict(a['qx'], prediction_type='RawFormulaVal'))
        members.append(raw)
    p = output(d['metrics'][k], np.mean(members, axis=0), a['state'], d['tasks'][k])
    assert np.isfinite(p).all()
    np.savez_compressed(file, signature=sig(), fit=a['state']['fit'], query=a['state']['query'], p=p,
                        center=a['state']['center'], scale=a['state']['scale'])
    print(json.dumps(dict(stage='train', task=d['tasks'][k], part=part,
                          seconds=round(time.monotonic() - start))), flush=True)
    return p


def smoke(d):
    from tempfile import TemporaryDirectory
    from v161_verify_and_assemble import same
    np.testing.assert_allclose(scaffold_weights(np.array([1, 1, 1, 1, 2]))[-1] /
                               scaffold_weights(np.array([1, 1, 1, 1, 2]))[0], 2.)
    np.testing.assert_allclose(scaffold_weights(np.ones(5)), np.ones(5), rtol=1e-14, atol=0)
    checked = []
    for task in ('herg', 'solubility_aqsoldb', 'half_life_obach'):
        k = d['tasks'].index(task)
        a = design(d, k, 'fold0')
        query = a['state']['query']
        labels = d['y'][query].copy()
        try:
            d['y'][query] = 999999.
            d.pop('_v96_assays', None)
            same(a, design(d, k, 'fold0'))
        finally:
            d['y'][query] = labels
            d.pop('_v96_assays', None)
        net = estimator(d['metrics'][k], dict(depth=3, trees=16), SEEDS[0])
        n = min(96, len(a['x']))
        net.fit(a['x'][:n], a['state']['target'][:n], sample_weight=a['state']['weights'][:n])
        p = net.predict(a['qx'][:8], prediction_type='RawFormulaVal')
        np.testing.assert_array_equal(p, net.predict(a['qx'][:8][::-1], prediction_type='RawFormulaVal')[::-1])
        assert np.isfinite(p).all()
        with TemporaryDirectory(dir=OUT) as temp:
            file = Path(temp) / 'smoke.cbm'
            net.save_model(str(file))
            reload = CatBoostClassifier() if d['metrics'][k] in ('AUROC', 'AUPRC') else CatBoostRegressor()
            reload.load_model(str(file))
            np.testing.assert_array_equal(p, reload.predict(a['qx'][:8], prediction_type='RawFormulaVal'))
        checked.append(task)
    b.save(OUT / 'smoke.json', dict(status='PASS', signature=sig(), query_label_isolation=True,
           fit_only_scaffold_weights=True, analytic_weight_reference=True,
           independent_reload_tasks=checked, prediction_row_order=True,
           no_query_scores_used=True))
    print('SMOKE_PASS', flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    d = v.load()
    if '--smoke' in sys.argv:
        smoke(d)
    else:
        assert json.loads((OUT / 'smoke.json').read_text())['signature'] == sig()
        runner.run(sys.modules[__name__], d)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        b.save(OUT / 'failure.json', dict(status='FAILED', traceback=traceback.format_exc()))
        raise
