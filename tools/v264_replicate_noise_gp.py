"""Replicate-calibrated observation noise for two v145 endpoints.

Fit-only exact original-SMILES replicates define a variance model, rather than
inverse identity-count risk weights. Raw labels are preserved. Gaussian group
means have their own uncertainty; fit marginal likelihood calibrates the signal
and extra noise. Existing structure/context kernels are deliberately reused.
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
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v146_current_baseline as v
import v148_experiment_runner as runner
b, r83 = v.b, v.r83
OUT = ROOT / 'admet_model_outputs_v264'
TASKS = ('ppbr_az', 'clearance_hepatocyte_az')
CONFIGS = (dict(prior_df=5., max_iterations=60),)
METHOD = ('Exact-original-structure replicate-calibrated Gaussian process for PPBR '
          'and hepatocyte clearance. Fit-only replicate within-group scatter is '
          'shrunk with five pooled-variance degrees of freedom. Group means enter '
          'with estimated observation variance divided by count, not equal '
          'certainty or inverse-count risk weights. Signal amplitude and extra '
          'observation variance are calibrated by bounded marginal likelihood '
          'from a single deterministic initialization. Existing six Tanimoto '
          'views plus fit-transformed physical/context PSD kernels; no new '
          'feature novelty claim, historical predictions or query-label inputs. '
          'Raw PPBR and log1p clearance; posterior median, PPBR legal clipping. '
          'Original v148 choice, blends and all confirmation gates unchanged.')


def sig():
    return hashlib.sha256((b.sha(__file__) + v.sig() + b.sha(runner.__file__) +
                           importlib.metadata.version('scipy') +
                           importlib.metadata.version('numpy')).encode()).hexdigest()


def key(c):
    return 'replicate_noise_gaussian_process'


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


def replicate_state(identity, y, prior_df):
    ids, first, inverse, counts = np.unique(identity, return_index=True,
                                            return_inverse=True, return_counts=True)
    means = np.bincount(inverse, weights=y) / counts
    scatter = np.bincount(inverse, weights=(y - means[inverse]) ** 2)
    degrees = int(np.sum(counts - 1))
    assert degrees > 0, 'This predeclared mechanism requires actual replicates'
    pooled = max(float(scatter.sum() / degrees), 1e-6)
    variance = (prior_df * pooled + scatter) / (prior_df + counts - 1)
    return dict(identity=ids, first=first, inverse=inverse, counts=counts,
                mean=means, scatter=scatter, replicate_degrees=degrees,
                pooled_variance=pooled, observation_variance=variance / counts)


def rbf(raw, n):
    median = np.array([np.median(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.
                       for x in raw[:n].T])
    clean = np.where(np.isfinite(raw), raw, median)
    scale = clean[:n].std(0).clip(.05)
    z = np.clip((clean - median) / scale, -6., 6.)
    z = np.c_[z, (~np.isfinite(raw)).astype(float)]
    norm = np.sum(z * z, 1)
    result = np.empty((len(raw), n))
    for lo in range(0, len(raw), 256):
        xx = z[lo:lo + 256]
        distance = np.maximum(norm[lo:lo + len(xx), None] + norm[None, :n]
                              - 2 * xx @ z[:n].T, 0.) / max(1, z.shape[1])
        result[lo:lo + len(xx)] = .5 * np.exp(-distance / .5) + .5 * np.exp(-distance / 2.)
    return result, dict(median=median, scale=scale)


def design(d, k, part, c=None):
    c = c or CONFIGS[0]
    task = d['tasks'][k]
    assert task in TASKS
    fit, query, x, qx, source = v.old.features(d, k, part, 'augmented')
    raw_y = d['y'][fit]
    transformed = np.log1p(raw_y) if task == 'clearance_hepatocyte_az' else raw_y.copy()
    center = float(np.median(transformed))
    scale = max(float(np.subtract(*np.quantile(transformed, [.75, .25]))), .05)
    target = (transformed - center) / scale
    rep = replicate_state(d['mi'][fit], target, c['prior_df'])
    # Exact SMILES replicate rows have identical structure and measured context.
    np.testing.assert_array_equal(x, x[rep['first']][rep['inverse']])
    representatives = fit[rep['first']]
    rows, n = np.r_[representatives, query], len(representatives)
    matrix = np.zeros((len(rows), n))
    for lo in range(0, 12288, 2048):
        z = sparse.csr_matrix((d['views']['multiview'][d['mi'][rows], lo:lo + 2048] > 0).astype(float))
        norms = np.asarray(z.sum(1)).ravel()
        for start in range(0, len(rows), 256):
            dot = (z[start:start + 256] @ z[:n].T).toarray()
            den = norms[start:start + len(dot), None] + norms[None, :n] - dot
            matrix[start:start + len(dot)] += .75 / 6 * dot / np.maximum(den, 1e-12)
    width = (d['views']['physchem'].shape[1] + d['conformer_features'].shape[1]
             + d['pharmacophore'].shape[1])
    raw = np.r_[x[rep['first']], qx].astype(float)
    physical, ps = rbf(raw[:, -width:], n)
    context, cs = rbf(raw[:, :-width], n)
    matrix += .15 * physical + .10 * context
    matrix[:n] = (matrix[:n] + matrix[:n].T) * .5
    assert not set(fit) & set(query) and np.all(d['tid'][source] != k)
    assert np.all(d['split'][source] < (2 if part == 'final' else 1))
    if part != 'final':
        assert not set(d['groups'][fit]) & set(d['groups'][query])
    assert np.isfinite(matrix).all()
    state = dict(fit=fit, query=query, source=source, representatives=representatives,
                 center=center, scale=scale, target=target, replicates=rep,
                 physical=ps, context=cs, target_log1p=task == 'clearance_hepatocyte_az')
    return dict(x=matrix[:n], qx=matrix[n:], state=state)


def likelihood(theta, kernel, y, noise):
    amplitude, extra = np.exp(theta)
    covariance = amplitude * kernel + np.diag(noise + extra + 1e-8)
    factor = cho_factor(covariance, lower=True, check_finite=False)
    alpha = cho_solve(factor, y, check_finite=False)
    value = .5 * (y @ alpha + 2 * np.log(np.diag(factor[0])).sum()
                  + len(y) * np.log(2 * np.pi))
    inverse = cho_solve(factor, np.eye(len(y)), check_finite=False)
    derivative = inverse - np.outer(alpha, alpha)
    gradient = .5 * np.array([amplitude * np.sum(derivative * kernel),
                              extra * np.trace(derivative)])
    return float(value), gradient


def train(kernel, state, c):
    rep = state['replicates']
    result = minimize(likelihood, np.log([1., .05]),
                      args=(kernel, rep['mean'], rep['observation_variance']),
                      jac=True, method='L-BFGS-B', bounds=[(-5., 3.), (-9., 2.)],
                      options=dict(maxiter=c['max_iterations'], ftol=1e-10, gtol=1e-5))
    assert result.success, dict(message=str(result.message), iterations=result.nit)
    amplitude, extra = np.exp(result.x)
    covariance = amplitude * kernel + np.diag(rep['observation_variance'] + extra + 1e-8)
    alpha = cho_solve(cho_factor(covariance, lower=True), rep['mean'])
    assert np.max(np.abs(covariance @ alpha - rep['mean'])) < 1e-7
    return dict(alpha=alpha, amplitude=amplitude, extra_noise=extra,
                optimizer=dict(success=bool(result.success), iterations=int(result.nit),
                               objective=float(result.fun), log_parameters=result.x,
                               gradient=result.jac, message=str(result.message)))


def infer(net, qx, state):
    # Gaussian posterior mean equals its median; monotone target transforms preserve median.
    p = (net['amplitude'] * qx) @ net['alpha'] * state['scale'] + state['center']
    return np.expm1(p) if state['target_log1p'] else p.clip(0., 100.)


def replay(d, k, part, c):
    a = design(d, k, part, c)
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    same(a['state'], joblib.load(folder / 'transform.joblib'))
    net = joblib.load(folder / 'model.joblib')
    assert net['optimizer']['success']
    return infer(net, a['qx'], a['state']), {
        name: b.sha(folder / name) for name in ('model.joblib', 'transform.joblib')}


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
    assert not folder.exists(), 'Inspect any partial fit instead of blindly retrying'
    started = time.monotonic()
    a = design(d, k, part, c)
    folder.mkdir(parents=True)
    joblib.dump(a['state'], folder / 'transform.joblib', compress=3)
    net = train(a['x'], a['state'], c)
    joblib.dump(net, folder / 'model.joblib', compress=3)
    p = infer(net, a['qx'], a['state'])
    np.testing.assert_array_equal(p, infer(joblib.load(folder / 'model.joblib'), a['qx'], a['state']))
    assert np.isfinite(p).all()
    np.savez_compressed(file, signature=sig(), fit=a['state']['fit'], query=a['state']['query'], p=p)
    b.save(folder / 'metadata.json', dict(signature=sig(), independent_reload=True,
        fit_rows=len(a['state']['fit']), unique_structures=len(net['alpha']),
        replicate_degrees=a['state']['replicates']['replicate_degrees'],
        amplitude=float(net['amplitude']), extra_noise=float(net['extra_noise']),
        optimizer_success=True, optimizer_iterations=net['optimizer']['iterations'],
        seconds=time.monotonic() - started))
    print(json.dumps(dict(stage='train', task=d['tasks'][k], part=part,
                         seconds=round(time.monotonic() - started))), flush=True)
    return p


def smoke(d):
    from tempfile import TemporaryDirectory
    c = CONFIGS[0]
    # Independent analytic variance reference and likelihood finite differences.
    rep = replicate_state(np.array([1, 1, 2, 2, 2, 3]), np.array([0., 2., 1., 2., 6., 4.]), 5.)
    np.testing.assert_allclose(rep['mean'], [1., 3., 4.], atol=1e-15)
    np.testing.assert_allclose(rep['scatter'], [2., 14., 0.], atol=1e-15)
    np.testing.assert_allclose(rep['observation_variance'],
        [(5 * 16 / 3 + 2) / 6 / 2, (5 * 16 / 3 + 14) / 7 / 3, 16 / 3], atol=1e-15)
    kernel = np.array([[1., .4, .2], [.4, 1., .3], [.2, .3, 1.]])
    y, noise, theta = rep['mean'], rep['observation_variance'], np.log([1.2, .1])
    value, gradient = likelihood(theta, kernel, y, noise)
    numerical = []
    for j in range(2):
        step = np.eye(2)[j] * 1e-5
        numerical.append((likelihood(theta + step, kernel, y, noise)[0]
                          - likelihood(theta - step, kernel, y, noise)[0]) / 2e-5)
    np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-8)
    ref = 1.2 * kernel + np.diag(noise + .1 + 1e-8)
    np.testing.assert_allclose(value, .5 * (y @ np.linalg.solve(ref, y)
        + np.linalg.slogdet(ref)[1] + 3 * np.log(2 * np.pi)), atol=1e-12)
    checked = []
    for task in TASKS:
        k = d['tasks'].index(task)
        a = design(d, k, 'fold0', c)
        outside = np.flatnonzero((d['tid'] == k) & ~np.isin(np.arange(len(d['y'])), a['state']['fit']))
        saved = d['y'][outside].copy()
        try:
            d['y'][outside] = 999999.
            d.pop('_v96_assays', None)
            same(a, design(d, k, 'fold0', c))
        finally:
            d['y'][outside] = saved
            d.pop('_v96_assays', None)
        assert np.linalg.eigvalsh(a['x'][:64, :64]).min() > -1e-8
        # Exercise the actual optimizer without fitting a development model or viewing its score.
        small = dict(replicates={name: value[:64] if isinstance(value, np.ndarray)
                               and value.shape == a['state']['replicates']['mean'].shape else value
                               for name, value in a['state']['replicates'].items()})
        net = train(a['x'][:64, :64], small, c)
        pred = infer(net, a['qx'][:8, :64], a['state'])
        np.testing.assert_allclose(pred, infer(net, a['qx'][:8, :64][::-1], a['state'])[::-1], rtol=0, atol=1e-12)
        with TemporaryDirectory(dir=OUT) as temp:
            path = Path(temp) / 'model.joblib'
            joblib.dump(net, path)
            np.testing.assert_array_equal(pred, infer(joblib.load(path), a['qx'][:8, :64], a['state']))
        checked.append(task)
    b.save(OUT / 'smoke.json', dict(status='PASS', signature=sig(),
        query_label_isolation=True, replicate_variance_reference=True,
        likelihood_reference=True, likelihood_gradient_reference=True,
        psd_reference=True, actual_optimizer=True, independent_reload_tasks=checked,
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
