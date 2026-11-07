"""Small corrections to exact v145 queries, learned from honest inner scaffold OOF.

The inner reference is explicitly a surrogate, not reconstructed v145 OOF.
No historical proxy, in-sample prediction or previous rejected model is used
as a correction-training baseline. Query offsets use immutable v145 controls.
"""
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
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from scipy.special import expit
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v146_current_baseline as v
import v148_experiment_runner as runner
b, r83 = v.b, v.r83
OUT = ROOT / 'admet_model_outputs_v228'
# Predeclared from the official v145 scorecard, not any failed confirmation.
TASKS = ('herg', 'bbb_martins', 'ames', 'clearance_hepatocyte_az',
         'solubility_aqsoldb', 'pgp_broccatelli')
SEEDS = (2281029, 2281030)
REFERENCE_SEED = 2281000
CONFIGS = (dict(inner_folds=3, reference_trees=450, correction_trees=250, depth=3),)
METHOD = ('Targeted incremental correction anchored to exact v145 query predictions. '
          'Only the six predeclared lowest official endpoints train corrections; '
          'remaining tasks return identity controls through the original v148 runner. '
          'Correction fitting offsets come from three newly trained, scaffold-disjoint '
          'inner OOF reference models, never historical proxies or in-sample fits. '
          'The reference is a surrogate, not exact v145 OOF. Small bounded residual '
          'CatBoost updates use base-coordinate, chemistry and target-excluded assays. '
          'One fixed configuration, two correction seeds; original blends/gates unchanged.')


def sig():
    return hashlib.sha256((b.sha(__file__) + v.sig() + b.sha(runner.__file__) +
                           importlib.metadata.version('catboost')).encode()).hexdigest()


def key(c):
    return 'honest_crossfit_residual'


def load():
    return v.load()


def same(a, z):
    if isinstance(a, dict):
        assert set(a) == set(z)
        for name in a:
            same(a[name], z[name])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(z)
        for x, y in zip(a, z):
            same(x, y)
    else:
        np.testing.assert_array_equal(a, z)


def raw_design(d, k, part):
    fit, query, x, qx, source = v.old.features(d, k, part, 'full')
    extra = np.c_[d['conformer_features'], d['pharmacophore']]
    x = np.c_[x, extra[d['mi'][fit]]].astype(np.float32)
    qx = np.c_[qx, extra[d['mi'][query]]].astype(np.float32)
    af, aq, cx, cqx, cs = v.old.features(d, k, part, 'augmented')
    np.testing.assert_array_equal(af, fit)
    np.testing.assert_array_equal(aq, query)
    np.testing.assert_array_equal(cs, source)
    assert not set(fit) & set(query) and np.all(d['tid'][source] != k)
    assert np.all(d['split'][source] < (2 if part == 'final' else 1))
    if part != 'final':
        assert not set(d['groups'][fit]) & set(d['groups'][query])
    return dict(fit=fit, query=query, source=source, x=x, qx=qx,
                cx=cx, cqx=cqx)


def cat(metric, trees, depth, seed, correction=False):
    cls = metric in ('AUROC', 'AUPRC')
    return (CatBoostClassifier if cls else CatBoostRegressor)(
        iterations=trees, depth=depth, learning_rate=.03 if correction else .045,
        l2_leaf_reg=40 if correction else 12, random_seed=seed,
        thread_count=4, verbose=False, allow_writing_files=False,
        task_type='CPU', border_count=64, bootstrap_type='Bernoulli', subsample=.85,
        loss_function='Logloss' if cls else 'MAE' if metric == 'MAE' else 'RMSE')


def read_cat(path, metric):
    cls = metric in ('AUROC', 'AUPRC')
    return (CatBoostClassifier if cls else CatBoostRegressor)().load_model(str(path))


def inner_target(metric, values):
    return r83.ranknorm(values) if metric == 'Spearman' else values


def reference_oof(d, k, a, folder, c, train=False, smoke=False):
    metric = d['metrics'][k]
    oof = np.full(len(a['fit']), np.nan)
    counts = np.zeros(len(oof), dtype=int)
    assignments, hashes = [], {}
    cv = GroupKFold(n_splits=c['inner_folds'])
    for j, (tr, hold) in enumerate(cv.split(a['x'], groups=d['groups'][a['fit']])):
        assert not set(d['groups'][a['fit'][tr]]) & set(d['groups'][a['fit'][hold]])
        assert not set(a['fit'][tr]) & set(a['query'])
        path = folder / ('reference_' + str(j) + '.cbm')
        if not path.exists():
            assert train, 'Missing inner reference checkpoint'
            net = cat(metric, 11 if smoke else c['reference_trees'], 5,
                      REFERENCE_SEED + j)
            net.fit(a['x'][tr], inner_target(metric, d['y'][a['fit'][tr]]))
            p = net.predict(a['x'][hold], prediction_type='RawFormulaVal', thread_count=1)
            net.save_model(str(path))
            z = read_cat(path, metric).predict(a['x'][hold], prediction_type='RawFormulaVal', thread_count=1)
            np.testing.assert_array_equal(p, z)
        else:
            z = read_cat(path, metric).predict(a['x'][hold], prediction_type='RawFormulaVal', thread_count=1)
        oof[hold] = z
        counts[hold] += 1
        assignments.append(dict(train=a['fit'][tr], hold=a['fit'][hold]))
        hashes[path.name] = b.sha(path)
    assert np.all(counts == 1) and np.isfinite(oof).all()
    return oof, assignments, hashes


def design(d, k, part, c=None, folder=None, train=False, smoke=False):
    c = c or CONFIGS[0]
    folder = folder or OUT / 'models' / d['tasks'][k] / part / key(c)
    a = raw_design(d, k, part)
    if train:
        folder.mkdir(parents=True, exist_ok=True)
    ref, assignments, hashes = reference_oof(d, k, a, folder, c, train, smoke)
    # Actual v145 is used only for outer query offsets; it never supplies fit OOF.
    base = v.baseline(d, k, part)
    metric, y = d['metrics'][k], d['y'][a['fit']]
    if metric in ('AUROC', 'AUPRC'):
        z, qz = r83.ranklog(ref), r83.ranklog(base)
        beta = r83.calibrate(y, z)
        off, qoff = beta[0] + beta[1] * z, beta[0] + beta[1] * qz
        target, center, scale, bound = y, 0., 1., .75
    elif metric == 'MAE':
        center = float(np.median(y))
        scale = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05)
        z, qz = (ref - center) / scale, (base - center) / scale
        off, qoff = z, qz
        beta = np.array([0., 1.])
        target, bound = (y - center) / scale - off, .5
    else:
        z, qz = r83.ranknorm(ref), r83.ranknorm(base)
        xx = np.c_[np.ones(len(z)), z]
        beta = np.linalg.solve(xx.T @ xx + np.diag([1e-8, 1.]), xx.T @ r83.ranknorm(y))
        beta[1] = max(beta[1], .01)
        off, qoff = xx @ beta, beta[0] + beta[1] * qz
        target, center, scale, bound = r83.ranknorm(y) - off, 0., 1., .5
    x, qx = np.c_[a['cx'], z].astype(np.float32), np.c_[a['cqx'], qz].astype(np.float32)
    state = dict(fit=a['fit'], query=a['query'], source=a['source'], reference_oof=ref,
                 inner_assignments=assignments, beta=beta, off=off, qoff=qoff,
                 target=target, center=center, scale=scale, bound=bound,
                 base_query=base, reference_hashes=hashes)
    return dict(x=x, qx=qx, state=state)


def output(raw, a, metric, task):
    # Clip only the newly learned correction, preserving the base offset.
    update = np.clip(raw, -a['bound'], a['bound'])
    p = a['qoff'] + update
    if metric in ('AUROC', 'AUPRC'):
        return expit(p)
    p = p * a['scale'] + a['center']
    return p.clip(0, 100) if task == 'ppbr_az' else p


def fit_correction(net, a, metric):
    base = a['state']['off'] if metric in ('AUROC', 'AUPRC') else None
    net.fit(Pool(a['x'], a['state']['target'], baseline=base))


def replay(d, k, part, c):
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    a = design(d, k, part, c)
    same(a['state'], joblib.load(folder / 'transform.joblib'))
    hashes = dict(a['state']['reference_hashes'])
    hashes['transform.joblib'] = b.sha(folder / 'transform.joblib')
    members = []
    for seed in SEEDS:
        path = folder / ('model_' + str(seed) + '.cbm')
        net = read_cat(path, d['metrics'][k])
        members.append(net.predict(a['qx'], prediction_type='RawFormulaVal', thread_count=1))
        hashes[path.name] = b.sha(path)
    return output(np.mean(members, axis=0), a['state'], d['metrics'][k], d['tasks'][k]), hashes


def predict(d, k, part, c):
    if d['tasks'][k] not in TASKS:
        return v.baseline(d, k, part)
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    file = folder / 'predictions.npz'
    if file.exists():
        z = np.load(file)
        assert str(z['signature']) == sig()
        np.testing.assert_array_equal(z['query'], v.split(d, k, part)[1])
        return z['p'].copy()
    started = time.monotonic()
    a = design(d, k, part, c, train=True)
    joblib.dump(a['state'], folder / 'transform.joblib', compress=3)
    members = []
    for seed in SEEDS:
        net = cat(d['metrics'][k], c['correction_trees'], c['depth'], seed, True)
        fit_correction(net, a, d['metrics'][k])
        p = net.predict(a['qx'], prediction_type='RawFormulaVal', thread_count=1)
        path = folder / ('model_' + str(seed) + '.cbm')
        net.save_model(str(path))
        checked = read_cat(path, d['metrics'][k]).predict(a['qx'], prediction_type='RawFormulaVal', thread_count=1)
        np.testing.assert_array_equal(p, checked)
        members.append(checked)
    p = output(np.mean(members, axis=0), a['state'], d['metrics'][k], d['tasks'][k])
    assert np.isfinite(p).all()
    np.savez_compressed(file, signature=sig(), fit=a['state']['fit'], query=a['state']['query'], p=p)
    print(json.dumps(dict(stage='train', task=d['tasks'][k], part=part,
                          seconds=round(time.monotonic() - started), references=3,
                          correction_seeds=2, reference_is_exact_v145=False)), flush=True)
    return p


def smoke(d):
    from tempfile import TemporaryDirectory
    checked = []
    for task in ('herg', 'solubility_aqsoldb', 'clearance_hepatocyte_az'):
        k = d['tasks'].index(task)
        with TemporaryDirectory(dir=OUT) as temp:
            folder = Path(temp)
            a = design(d, k, 'fold0', folder=folder, train=True, smoke=True)
            outside = np.flatnonzero((d['tid'] == k) & ~np.isin(np.arange(len(d['y'])), a['state']['fit']))
            labels = d['y'][outside].copy()
            try:
                d['y'][outside] = 999999.
                d.pop('_v96_assays', None)
                same(a, design(d, k, 'fold0', folder=folder, smoke=True))
            finally:
                d['y'][outside] = labels
                d.pop('_v96_assays', None)
            net = cat(d['metrics'][k], 7, 3, SEEDS[0], True)
            fit_correction(net, a, d['metrics'][k])
            p = net.predict(a['qx'], prediction_type='RawFormulaVal', thread_count=1)
            np.testing.assert_allclose(p, net.predict(a['qx'][::-1], prediction_type='RawFormulaVal', thread_count=1)[::-1], atol=1e-14, rtol=0)
            path = folder / 'correction.cbm'
            net.save_model(str(path))
            np.testing.assert_array_equal(p, read_cat(path, d['metrics'][k]).predict(a['qx'], prediction_type='RawFormulaVal', thread_count=1))
            extreme = output(np.full(len(p), 1e9), a['state'], d['metrics'][k], task)
            expected = output(np.full(len(p), a['state']['bound']), a['state'], d['metrics'][k], task)
            np.testing.assert_array_equal(extreme, expected)
            checked.append(task)
    b.save(OUT / 'smoke.json', dict(status='PASS', signature=sig(), query_label_isolation=True,
           inner_scaffold_disjoint=True, exactly_one_oof_per_fit_row=True,
           reference_is_surrogate=True, historical_proxy_used_for_fit=False,
           independent_reload_tasks=checked, correction_bound_reference=True,
           prediction_row_order=True, no_query_scores_used=True))
    print('SMOKE_PASS', flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    d = load()
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
        b.save(OUT / 'failure.json', dict(status='FAILED', traceback=traceback.format_exc(), retry_permitted=False))
        raise
