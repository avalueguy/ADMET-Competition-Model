"""Optimize leaf scores of exact v145 tree components, with frozen topology.

The archived v127 trees are supervised fitting artifacts. Their leaf maps are
never called OOF on their own training rows. Only original outer query rows are
used for validation by the unchanged v148 selector.
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
from catboost import CatBoostClassifier
from scipy.linalg import cho_factor, cho_solve
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v146_current_baseline as v
import v148_experiment_runner as runner
import v127_low_classification as original

b, r83 = v.b, v.r83
OUT = ROOT / 'admet_model_outputs_v278'
TASKS = ('herg', 'cyp3a4_substrate_carbonmangels')
CONFIGS = (dict(prior_precision=1., max_newton=50, gradient_tolerance=1e-8),)
SEEDS = original.SEEDS
METHOD = ('Direct re-estimation of leaf scores of the exact v127/physchem tree '
          'components carried into v145. Freeze all900trees per seed, splits, '
          'features and original1274026/1274027 checkpoints. Joint leaf indicator '
          'kernel normalized by total trees, logistic MAP with unit isotropic '
          'Gaussian prior and fixed smoothed-fit-prevalence intercept. No tree '
          'growth, new seed, old rejected expert, query-label input, or historical '
          'proxy as fitting offsets. Refit is learned only on original target '
          'fit rows. Candidate replaces the existing component inside its '
          'original .5 rank blend, then originalv148 .2/.45 incremental weights '
          'and all stable gates apply. Twenty endpoints protected;19runner '
          'identity controls, HIA excluded. Single fixed objective, no retries.')


def sig():
    return hashlib.sha256((b.sha(__file__) + v.sig() + b.sha(runner.__file__) +
        original.sig() + importlib.metadata.version('catboost') +
        importlib.metadata.version('scipy')).encode()).hexdigest()


def load():
    threadpool_limits(limits=4)
    return v.load()


def key(c):
    return 'v145_frozen_tree_joint_leaf_map'


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


def design(d, k, part, c=None):
    task = d['tasks'][k]
    assert task in TASKS
    fit, query, x, qx, source = original.v.features(d, k, part, 'physchem')
    np.testing.assert_array_equal(fit, v.split(d, k, part)[0])
    np.testing.assert_array_equal(query, v.split(d, k, part)[1])
    assert np.all(d['tid'][source] != k)
    assert np.all(d['split'][source] < (2 if part == 'final' else 1))
    if part != 'final':
        assert not set(d['groups'][fit]) & set(d['groups'][query])
    train_leaf, query_leaf, hashes, old_members = [], [], {}, []
    for seed in SEEDS:
        path = original.OUT / 'models' / task / part / 'physchem' / ('model_' + str(seed) + '.cbm')
        assert path.exists(), 'Missing carried model: do not silently train it'
        net = CatBoostClassifier().load_model(str(path))
        assert net.tree_count_ == 900
        train_leaf.append(net.calc_leaf_indexes(x, thread_count=1))
        query_leaf.append(net.calc_leaf_indexes(qx, thread_count=1))
        old_members.append(net.predict(qx, prediction_type='RawFormulaVal', thread_count=1))
        hashes[path.relative_to(ROOT).as_posix()] = b.sha(path)
    old = expit(np.mean(old_members, axis=0))
    cached = np.load(original.OUT / 'models' / task / part / 'physchem/predictions.npz')
    assert str(cached['signature']) == original.sig()
    np.testing.assert_array_equal(cached['fit'], fit)
    np.testing.assert_array_equal(cached['query'], query)
    np.testing.assert_allclose(old, cached['p'], atol=1e-12, rtol=0)
    previous = original.v.baseline(d, k, part)
    carried = original.v.mix(d['metrics'][k], previous, old, .5)
    baseline = v.baseline(d, k, part)
    np.testing.assert_allclose(carried, baseline, atol=1e-12, rtol=0)
    return dict(fit=fit, query=query, source=source, x=x, qx=qx,
        train_leaf=np.concatenate(train_leaf, axis=1),
        query_leaf=np.concatenate(query_leaf, axis=1),
        original_component=old, original_previous=previous, original_baseline=baseline,
        original_model_hashes=hashes)


def proximity(train_leaf, query_leaf):
    # Equality is per tree (column), so separate tree leaf IDs cannot collide.
    n, trees = train_leaf.shape
    result = np.empty((len(query_leaf), n))
    for i in range(len(query_leaf)):
        result[i] = np.count_nonzero(train_leaf == query_leaf[i], axis=1) / trees
    return result


def solve(matrix, y, c):
    assert np.isfinite(matrix).all() and np.all((y == 0) | (y == 1))
    eigenvalues, vectors = np.linalg.eigh((matrix + matrix.T) * .5)
    assert eigenvalues.min() > -1e-9
    keep = eigenvalues > 1e-10
    basis = vectors[:, keep] * np.sqrt(eigenvalues[keep])
    projection = vectors[:, keep] / np.sqrt(eigenvalues[keep])
    prevalence = (float(y.sum()) + .5) / (len(y) + 1.)
    offset = float(np.log(prevalence / (1 - prevalence)))
    coef = np.zeros(basis.shape[1])
    precision = c['prior_precision']

    def objective(beta):
        eta = offset + basis @ beta
        probability = expit(eta)
        value = float(np.sum(np.logaddexp(0, eta) - y * eta) + .5 * precision * beta @ beta)
        gradient = basis.T @ (probability - y) + precision * beta
        return value, gradient, probability

    history = []
    for step in range(c['max_newton']):
        value, gradient, probability = objective(coef)
        residual = float(np.abs(gradient).max())
        history.append(dict(step=step, objective=value, gradient_max=residual))
        if residual < c['gradient_tolerance']:
            break
        hessian = (basis.T * (probability * (1 - probability))) @ basis + precision * np.eye(len(coef))
        delta = cho_solve(cho_factor(hessian, lower=True), gradient)
        length = 1.
        while length >= 2. ** -24:
            candidate = coef - length * delta
            if objective(candidate)[0] <= value - 1e-4 * length * float(gradient @ delta):
                coef = candidate
                break
            length *= .5
        else:
            raise RuntimeError('Newton line search failed; preserve evidence without retry')
    else:
        raise RuntimeError('Fixed Newton iteration budget exhausted')
    assert history[-1]['gradient_max'] < c['gradient_tolerance']
    return dict(alpha=projection @ coef, offset=offset, objective=history[-1]['objective'],
                history=history, retained_eigenvalues=eigenvalues[keep], precision=precision)


def train(a, y, c):
    matrix = proximity(a['train_leaf'], a['train_leaf'])
    np.testing.assert_array_equal(matrix, matrix.T)
    net = solve(matrix, y, c)
    return infer(a, net), net


def infer(a, net):
    refitted = expit(net['offset'] + proximity(a['train_leaf'], a['query_leaf']) @ net['alpha'])
    # Keep the historical preceding component and its original blend weight.
    return original.v.mix('AUROC', a['original_previous'], refitted, .5)


def replay(d, k, part, c):
    a = design(d, k, part, c)
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    same(a, joblib.load(folder / 'transform.joblib'))
    net = joblib.load(folder / 'model.joblib')
    matrix = proximity(a['train_leaf'], a['train_leaf'])
    check = solve(matrix, d['y'][a['fit']], c)
    np.testing.assert_allclose(check['alpha'], net['alpha'], atol=1e-10, rtol=0)
    assert check['offset'] == net['offset']
    # Independent stationarity in dual representation: K(alpha+p-y)=0.
    probability = expit(net['offset'] + matrix @ net['alpha'])
    assert np.max(np.abs(matrix @ (c['prior_precision'] * net['alpha'] + probability - d['y'][a['fit']]))) < 1e-6
    return infer(a, net), {name:b.sha(folder / name) for name in ('model.joblib', 'transform.joblib')}


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
    assert not folder.exists(), 'Inspect incomplete leaf refit; no silent retry'
    started = time.monotonic()
    a = design(d, k, part, c)
    folder.mkdir(parents=True)
    joblib.dump(a, folder / 'transform.joblib', compress=3)
    p, net = train(a, d['y'][a['fit']], c)
    joblib.dump(net, folder / 'model.joblib', compress=3)
    np.testing.assert_array_equal(p, infer(a, joblib.load(folder / 'model.joblib')))
    assert np.isfinite(p).all()
    np.savez_compressed(file, signature=sig(), fit=a['fit'], query=a['query'], p=p)
    b.save(folder / 'metadata.json', dict(signature=sig(), fixed_topology=True,
        original_model_hashes=a['original_model_hashes'], new_trees=0, new_seeds=[],
        original_component_weight=.5, solver_history=net['history'],
        seconds=time.monotonic()-started, target_query_labels_used=False))
    print(json.dumps(dict(stage='leaf_refit', task=d['tasks'][k], part=part,
                         seconds=round(time.monotonic()-started))), flush=True)
    return p


def smoke(d):
    from tempfile import TemporaryDirectory
    # Explicit one-hot reference for the normalized leaf equality kernel.
    sample = np.array([[0, 1], [1, 1], [0, 0], [1, 0]])
    explicit = np.c_[sample[:,0] == 0, sample[:,0] == 1,
                     sample[:,1] == 0, sample[:,1] == 1].astype(float) / np.sqrt(2)
    np.testing.assert_allclose(proximity(sample, sample), explicit @ explicit.T, atol=1e-15)
    tiny = solve(proximity(sample, sample), np.array([0.,1.,0.,1.]), CONFIGS[0])
    assert tiny['history'][-1]['gradient_max'] < 1e-8
    checks = []
    for task in TASKS:
        k = d['tasks'].index(task)
        a = design(d, k, 'fold0')
        saved = d['y'][a['query']].copy()
        try:
            d['y'][a['query']] = 999999.
            d.pop('_v96_assays', None)
            same(a, design(d, k, 'fold0'))
        finally:
            d['y'][a['query']] = saved
            d.pop('_v96_assays', None)
        p, net = train(a, d['y'][a['fit']], CONFIGS[0])
        with TemporaryDirectory(dir=OUT) as temp:
            path = Path(temp) / 'model.joblib'
            joblib.dump(net, path)
            np.testing.assert_array_equal(p, infer(a, joblib.load(path)))
        reverse = dict(a, query_leaf=a['query_leaf'][::-1], original_previous=a['original_previous'][::-1])
        np.testing.assert_array_equal(p, infer(reverse, net)[::-1])
        checks.append(dict(task=task, independent_reload=True, original_baseline_replay=True,
                           gradient_max=net['history'][-1]['gradient_max']))
    b.save(OUT / 'smoke.json', dict(status='PASS', signature=sig(),
        kernel_explicit_reference=True, query_label_isolation=True,
        fixed_topology_original_component_replay=True, actual_logistic_optimization=True,
        prediction_row_order=True, independent_reload_tasks=checks,
        scores_computed=False, test_labels_used=False))
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
