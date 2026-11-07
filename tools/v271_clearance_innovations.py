"""Cross-fitted conditional innovations between two clearance measurements.

This tests a conditional observation mechanism, not another architecture or
parameter retry. Molecular reference models deliberately use existing kernels.
The observed other-assay residual conditions the target through a covariance
estimated from scaffold OOF residuals inside the outer fitting partition.
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
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v146_current_baseline as v
import v148_experiment_runner as runner
b, r83 = v.b, v.r83
OUT = ROOT / 'admet_model_outputs_v271'
TASKS = ('clearance_hepatocyte_az', 'clearance_microsome_az')
CONFIGS = (dict(inner_folds=3, ridge=.3, covariance_prior=20.),)
METHOD = ('Two-clearance conditional innovation transfer. Existing structure-only '
          'six fingerprint Tanimoto/physical RBF reference regressors fit log1p '
          'clearance. Three inner scaffold OOF partitions produce target and '
          'other-assay residuals; a shrunk covariance conditions target prediction '
          'on measured other-clearance innovation. Missing other measurements '
          'give zero innovation. Raw observed other assays follow the original '
          'target-excluded permitted context; other-reference labels also exclude '
          'query scaffolds, and all inner held-out scaffolds. No old model, old '
          'prediction or supposed exact v145 training OOF. Deterministic fixed '
          'ridge and covariance prior, no alternate fit after confirmation. '
          'Original v148 choice, blends and confirmation gates unchanged.')


def sig():
    return hashlib.sha256((b.sha(__file__) + v.sig() + b.sha(runner.__file__) +
                           importlib.metadata.version('scipy') +
                           importlib.metadata.version('scikit-learn')).encode()).hexdigest()


def key(c):
    return 'crossfit_clearance_innovations'


def load():
    return v.load()


def same(a, z):
    if isinstance(a, dict):
        assert set(a) == set(z)
        for name in a:
            same(a[name], z[name])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(z)
        for x, y in zip(a, z):
            same(x, y)
    else:
        np.testing.assert_array_equal(a, z)


def design(d, k, part, c=None):
    task = d['tasks'][k]
    assert task in TASKS
    fit, query = v.split(d, k, part)
    other_task = next(t for t in TASKS if t != task)
    other_k = d['tasks'].index(other_task)
    aux, cols, permitted, _, _ = v.old.old.old.assays.assays(d, k, part)
    assert k not in cols and np.all(d['tid'][permitted] != k)
    assert np.all(d['split'][permitted] < (2 if part == 'final' else 1))
    j = list(cols).index(other_k)
    # First channel is the existing exact canonical identity field; no neutral lookup.
    observed = aux[np.r_[fit, query], j].astype(float)
    other = permitted[d['tid'][permitted] == other_k]
    if part != 'final':
        other = other[~np.isin(d['groups'][other], d['groups'][query])]
        assert not set(d['groups'][np.r_[fit, other]]) & set(d['groups'][query])
    assert not set(fit) & set(query) and not set(other) & set(query)
    return dict(fit=fit, query=query, other=other, source=permitted,
                observed_other=observed, other_task=other_task)


def physical(d, rows):
    ids = d['mi'][rows]
    return np.c_[d['views']['physchem'][ids], d['conformer_features'][ids],
                 d['pharmacophore'][ids]].astype(float)


def kernels(d, fit, query, saved=None):
    rows, n = np.r_[fit, query], len(fit)
    matrix = np.zeros((len(rows), n))
    for lo in range(0, 12288, 2048):
        x = sparse.csr_matrix((d['views']['multiview'][d['mi'][rows], lo:lo + 2048] > 0).astype(float))
        norm = np.asarray(x.sum(1)).ravel()
        for start in range(0, len(rows), 256):
            dot = (x[start:start + 256] @ x[:n].T).toarray()
            tan = dot / np.maximum(norm[start:start + len(dot), None] + norm[None, :n] - dot, 1e-12)
            matrix[start:start + len(dot)] += .85 / 6 * (.5 * tan + .5 * tan ** 2)
    raw = physical(d, rows)
    if saved is None:
        median = np.array([np.median(z[np.isfinite(z)]) if np.isfinite(z).any() else 0. for z in raw[:n].T])
        clean = np.where(np.isfinite(raw), raw, median)
        scale = clean[:n].std(0).clip(.05)
        saved = dict(median=median, scale=scale)
    clean = np.where(np.isfinite(raw), raw, saved['median'])
    z = np.c_[np.clip((clean - saved['median']) / saved['scale'], -6., 6.), (~np.isfinite(raw)).astype(float)]
    norm = np.sum(z * z, 1)
    for start in range(0, len(rows), 256):
        xx = z[start:start + 256]
        distance = np.maximum(norm[start:start + len(xx), None] + norm[None, :n] - 2 * xx @ z[:n].T, 0.) / z.shape[1]
        matrix[start:start + len(xx)] += .15 * (.5 * np.exp(-distance / .5) + .5 * np.exp(-distance / 2.))
    matrix[:n] = (matrix[:n] + matrix[:n].T) * .5
    assert np.isfinite(matrix).all()
    return matrix[:n], matrix[n:], saved


def reference(d, fit, query, c):
    assert len(fit) >= 20 and not set(fit) & set(query)
    y = np.log1p(d['y'][fit])
    center, scale = float(np.median(y)), max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05)
    x, qx, state = kernels(d, fit, query)
    covariance = x + np.eye(len(fit)) * c['ridge']
    alpha = cho_solve(cho_factor(covariance, lower=True), (y - center) / scale)
    assert np.max(np.abs(covariance @ alpha - (y - center) / scale)) < 1e-8
    net = dict(fit=fit, alpha=alpha, center=center, scale=scale, physical=state)
    return qx @ alpha * scale + center, net


def reference_replay(d, net, query):
    _, qx, state = kernels(d, net['fit'], query, net['physical'])
    same(state, net['physical'])
    return qx @ net['alpha'] * net['scale'] + net['center']


def innovation_coefficient(target_residual, other_residual, prior):
    assert len(target_residual) == len(other_residual) >= 20
    yc, xc = float(np.mean(target_residual)), float(np.mean(other_residual))
    y, x = target_residual - yc, other_residual - xc
    covariance = float(np.mean(x * y))
    variance = max(float(np.mean(x * x)), 1e-6)
    # Prior equivalent to20 zero-covariance observations; no tuning using query labels.
    beta = covariance / variance * len(x) / (len(x) + prior)
    return dict(beta=beta, other_center=xc, target_residual_center=yc,
                pairs=len(x), covariance=covariance, other_variance=variance)


def train(d, k, part, c, a):
    fit, other = a['fit'], a['other']
    fold_records, target_oof, other_oof = [], np.full(len(fit), np.nan), np.full(len(fit), np.nan)
    assignment = np.full(len(fit), -1, int)
    for fold, (tr, hold) in enumerate(GroupKFold(c['inner_folds']).split(fit, groups=d['groups'][fit])):
        held_groups = d['groups'][fit[hold]]
        other_fit = other[~np.isin(d['groups'][other], held_groups)]
        assert not set(d['groups'][fit[tr]]) & set(held_groups)
        assert not set(d['groups'][other_fit]) & set(held_groups)
        target_oof[hold], own_net = reference(d, fit[tr], fit[hold], c)
        other_oof[hold], other_net = reference(d, other_fit, fit[hold], c)
        assignment[hold] = fold
        fold_records.append(dict(fold=fold, hold=fit[hold], own=own_net, other=other_net))
    assert np.isfinite(target_oof).all() and np.isfinite(other_oof).all() and np.all(assignment >= 0)
    paired = np.isfinite(a['observed_other'][:len(fit)])
    # Each precise duplicate still represents an observation in the task risk.
    ty = np.log1p(d['y'][fit]) - target_oof
    ox = np.log1p(a['observed_other'][:len(fit)][paired]) - other_oof[paired]
    conditional = innovation_coefficient(ty[paired], ox, c['covariance_prior'])
    target_prediction, own = reference(d, fit, a['query'], c)
    other_prediction, auxiliary = reference(d, other, a['query'], c)
    net = dict(own=own, other=auxiliary, conditional=conditional, inner=fold_records,
               inner_assignment=assignment, target_oof=target_oof, other_oof=other_oof)
    return condition(target_prediction, other_prediction, a['observed_other'][len(fit):], conditional), net


def condition(own, auxiliary, observed, conditional):
    innovation = np.zeros(len(own))
    mask = np.isfinite(observed)
    innovation[mask] = np.log1p(observed[mask]) - auxiliary[mask] - conditional['other_center']
    # Conditional mean of a centered Gaussian innovation; missing observations integrate to zero.
    return own + conditional['beta'] * innovation


def infer(d, a, net):
    own = reference_replay(d, net['own'], a['query'])
    auxiliary = reference_replay(d, net['other'], a['query'])
    return condition(own, auxiliary, a['observed_other'][len(a['fit']):], net['conditional'])


def replay(d, k, part, c):
    a = design(d, k, part, c)
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    same(a, joblib.load(folder / 'transform.joblib'))
    net = joblib.load(folder / 'model.joblib')
    # Independently rebuild every structural transform and linear solve,
    # including the inner models that defined the conditional covariance.
    def verify_reference(ref):
        x, _, state = kernels(d, ref['fit'], np.array([], dtype=int))
        same(state, ref['physical'])
        y = np.log1p(d['y'][ref['fit']])
        center = float(np.median(y))
        scale = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05)
        assert center == ref['center'] and scale == ref['scale']
        alpha = np.linalg.solve(x + np.eye(len(x)) * c['ridge'], (y - center) / scale)
        np.testing.assert_allclose(alpha, ref['alpha'], atol=1e-10, rtol=1e-10)
    for ref in (net['own'], net['other']):
        verify_reference(ref)
    target_oof, other_oof = np.full(len(a['fit']), np.nan), np.full(len(a['fit']), np.nan)
    for record in net['inner']:
        for name in ('own', 'other'):
            verify_reference(record[name])
            assert not set(d['groups'][record[name]['fit']]) & set(d['groups'][record['hold']])
        positions = np.searchsorted(a['fit'], record['hold'])
        np.testing.assert_array_equal(a['fit'][positions], record['hold'])
        target_oof[positions] = reference_replay(d, record['own'], record['hold'])
        other_oof[positions] = reference_replay(d, record['other'], record['hold'])
    np.testing.assert_allclose(target_oof, net['target_oof'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(other_oof, net['other_oof'], atol=1e-12, rtol=0)
    paired = np.isfinite(a['observed_other'][:len(a['fit'])])
    recomputed = innovation_coefficient((np.log1p(d['y'][a['fit']]) - target_oof)[paired],
        np.log1p(a['observed_other'][:len(a['fit'])][paired]) - other_oof[paired], c['covariance_prior'])
    for name in recomputed:
        np.testing.assert_allclose(recomputed[name], net['conditional'][name], atol=1e-12, rtol=0)
    return infer(d, a, net), {name: b.sha(folder / name) for name in ('model.joblib', 'transform.joblib')}


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
    assert not folder.exists(), 'Inspect any partial model instead of silently refitting'
    start = time.monotonic()
    a = design(d, k, part, c)
    folder.mkdir(parents=True)
    joblib.dump(a, folder / 'transform.joblib', compress=3)
    p, net = train(d, k, part, c, a)
    joblib.dump(net, folder / 'model.joblib', compress=3)
    np.testing.assert_allclose(p, infer(d, a, joblib.load(folder / 'model.joblib')), atol=1e-12, rtol=0)
    assert np.isfinite(p).all()
    np.savez_compressed(file, signature=sig(), fit=a['fit'], query=a['query'], p=p)
    b.save(folder / 'metadata.json', dict(signature=sig(), conditional=net['conditional'],
        inner_fold_exclusions=True, new_reference_models=8, independent_reload=True,
        query_labels_used=False, seconds=time.monotonic() - start))
    print(json.dumps(dict(stage='train', task=d['tasks'][k], part=part,
                         seconds=round(time.monotonic() - start))), flush=True)
    return p


def smoke(d):
    from tempfile import TemporaryDirectory
    ref = innovation_coefficient(np.arange(25.) * 2 + 3, np.arange(25.), 20.)
    np.testing.assert_allclose(ref['beta'], 2 * 25 / 45, atol=1e-15)
    expected = np.array([1., 2., 3.])
    values = np.array([np.nan, 3., 8.])
    conditional = dict(beta=.5, other_center=0.)
    pred = condition(expected, np.log1p([1., 3., 2.]), values, conditional)
    np.testing.assert_allclose(pred, [1., 2., 3. + .5 * np.log(3.)], atol=1e-15)
    checked = []
    for task in TASKS:
        k = d['tasks'].index(task)
        a = design(d, k, 'fold0')
        outside = np.flatnonzero((d['tid'] == k) & ~np.isin(np.arange(len(d['y'])), a['fit']))
        saved = d['y'][outside].copy()
        try:
            d['y'][outside] = 999999.
            d.pop('_v96_assays', None)
            same(a, design(d, k, 'fold0'))
        finally:
            d['y'][outside] = saved
            d.pop('_v96_assays', None)
        # Actual three-fold model/reload on a small fit subset, no query score.
        sample = dict(a, fit=a['fit'][:96], query=a['query'][:8],
                      observed_other=np.r_[a['observed_other'][:96], a['observed_other'][len(a['fit']):len(a['fit']) + 8]])
        raw, net = train(d, k, 'fold0', CONFIGS[0], sample)
        with TemporaryDirectory(dir=OUT) as temp:
            path = Path(temp) / 'model.joblib'
            joblib.dump(net, path)
            np.testing.assert_allclose(raw, infer(d, sample, joblib.load(path)), atol=1e-12, rtol=0)
        reversed_a = dict(sample, query=sample['query'][::-1],
                          observed_other=np.r_[sample['observed_other'][:96], sample['observed_other'][96:][::-1]])
        np.testing.assert_allclose(raw, infer(d, reversed_a, net)[::-1], atol=1e-12, rtol=0)
        for record in net['inner']:
            for name in ('own', 'other'):
                assert not set(d['groups'][record[name]['fit']]) & set(d['groups'][record['hold']])
        checked.append(task)
    b.save(OUT / 'smoke.json', dict(status='PASS', signature=sig(),
        query_label_isolation=True, inner_reference_scaffold_exclusion=True,
        conditional_reference=True, missing_other_zero_innovation=True,
        real_crossfit_references=True, independent_reload_tasks=checked,
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
