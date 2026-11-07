"""Fit-only cross-scaffold supervised multiple-kernel molecular specialists.

Unlike historical fixed sums/products, the similarity geometry is learned
from target covariance between different fitting scaffolds, under nonnegative
weights. No old model or prediction is used to learn that geometry.
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
from scipy.optimize import nnls
from sklearn.svm import SVC, SVR

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v146_current_baseline as v
import v148_experiment_runner as runner
b, r83 = v.b, v.r83
OUT = ROOT / 'admet_model_outputs_v233'
TASKS = ('herg', 'bbb_martins', 'ames', 'clearance_hepatocyte_az',
         'solubility_aqsoldb', 'pgp_broccatelli')
CONFIGS = (dict(C=3., alignment_ridge=.05, landmarks=256),)
METHOD = ('Cross-scaffold supervised multiple-kernel learning anchored on v145. '
          'Fifteen PSD channels: six fingerprint generalized Tanimoto mixtures, '
          'fit-scaled physical and target-excluded assay RBFs, and seven '
          'chemistry/physical-by-assay Schur products. Nonnegative regularized '
          'alignment weights fit target covariance only between different '
          'training scaffolds; then exact SVC/SVR learns the endpoint. '
          'Six predeclared lowest official v145 endpoints; remaining controls '
          'identity. No old predictions, fixed original v148 blends/gates.')


def sig():
    return hashlib.sha256((b.sha(__file__) + v.sig() + b.sha(runner.__file__) +
                           importlib.metadata.version('scikit-learn') +
                           importlib.metadata.version('scipy')).encode()).hexdigest()


def key(c):
    return 'cross_scaffold_alignment'


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


def rbf(raw, n):
    fit = raw[:n]
    median = np.array([np.median(x[np.isfinite(x)]) if np.isfinite(x).any() else 0. for x in fit.T])
    scale = np.array([np.std(x[np.isfinite(x)]) if np.isfinite(x).any() else 1. for x in fit.T]).clip(.05)
    clean = np.where(np.isfinite(raw), raw, median)
    keep = np.flatnonzero(np.std(clean[:n], axis=0) > .01)
    z = np.clip((clean[:, keep] - median[keep]) / scale[keep], -5., 5.)
    # Explicit missingness contributes distances in a fit-defined feature map.
    z = np.c_[z, (~np.isfinite(raw[:, keep])).astype(float)]
    norm = np.sum(z * z, axis=1)
    matrix = np.empty((len(raw), n), dtype=np.float64)
    for lo in range(0, len(z), 256):
        a = z[lo:lo + 256]
        dist = np.maximum(norm[lo:lo + len(a), None] + norm[None, :n] - 2 * (a @ z[:n].T), 0.) / max(1, z.shape[1])
        matrix[lo:lo + len(a)] = .5 * np.exp(-dist / .5) + .5 * np.exp(-dist / 2.)
    matrix[:n] = (matrix[:n] + matrix[:n].T) * .5
    return matrix, dict(median=median, scale=scale, keep=keep)


def channels(d, k, part):
    fit, query = v.split(d, k, part)
    rows, n = np.r_[fit, query], len(fit)
    blocks, names = [], []
    for start in range(0, 12288, 2048):
        raw = d['views']['multiview'][d['mi'][rows], start:start + 2048].astype(np.float64)
        x = sparse.csr_matrix(raw)
        norm = np.asarray(x.multiply(x).sum(axis=1)).ravel()
        matrix = np.empty((len(rows), n), dtype=np.float64)
        for lo in range(0, len(rows), 256):
            dot = (x[lo:lo + 256] @ x[:n].T).toarray()
            tan = np.clip(dot / np.maximum(norm[lo:lo + len(dot), None] + norm[None, :n] - dot, 1e-12), 0., 1.)
            matrix[lo:lo + len(dot)] = .5 * tan + .5 * tan * tan
        matrix[:n] = (matrix[:n] + matrix[:n].T) * .5
        blocks.append(matrix)
        names.append('fingerprint_' + str(start // 2048))
    fit2, q2, augmented, qaugmented, source = v.old.features(d, k, part, 'augmented')
    np.testing.assert_array_equal(fit, fit2)
    np.testing.assert_array_equal(query, q2)
    physical = np.c_[d['views']['physchem'][d['mi'][rows]],
                     d['conformer_features'][d['mi'][rows]], d['pharmacophore'][d['mi'][rows]]]
    physical_width = physical.shape[1]
    aux = np.r_[augmented[:, :-physical_width], qaugmented[:, :-physical_width]]
    assert aux.shape[1] > 0 and np.all(d['tid'][source] != k)
    assert np.all(d['split'][source] < (2 if part == 'final' else 1))
    pk, ps = rbf(physical.astype(float), n)
    ak, ast = rbf(aux.astype(float), n)
    blocks.extend((pk, ak))
    names.extend(('physical', 'assay'))
    for j in range(7):
        blocks.append(blocks[j] * ak)
        names.append(names[j] + '_times_assay')
    assert len(blocks) == 15
    return fit, query, source, blocks, names, dict(physical=ps, assay=ast)


def alignment(blocks, fit, d, k, c):
    rng = np.random.default_rng(2331029 + k)
    take = np.sort(rng.choice(len(fit), min(c['landmarks'], len(fit)), replace=False))
    y = d['y'][fit][take]
    metric = d['metrics'][k]
    if metric in ('AUROC', 'AUPRC'):
        p = np.clip(np.mean(y), .01, .99)
        target = (y - p) / np.sqrt(p * (1 - p))
    elif metric == 'Spearman':
        target = r83.ranknorm(y)
    else:
        target = (y - np.median(y)) / max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05)
    target = np.clip(target, -3., 3.)
    target -= np.mean(target)
    groups = d['groups'][fit[take]]
    mask = groups[:, None] != groups[None, :]
    assert np.sum(mask) > 50
    cols, scales = [], []
    for raw in blocks:
        z = raw[np.ix_(take, take)]
        centered = z - z.mean(0)[None, :] - z.mean(1)[:, None] + z.mean()
        values = centered[mask]
        scale = max(float(np.sqrt(np.mean(values * values))), 1e-8)
        cols.append(values / scale)
        scales.append(scale)
    x = np.stack(cols, axis=1) / np.sqrt(np.sum(mask))
    yy = np.outer(target, target)[mask] / np.sqrt(np.sum(mask))
    prior = np.full(len(blocks), 1 / len(blocks))
    strength = np.sqrt(c['alignment_ridge'])
    xa = np.r_[x, strength * np.eye(len(blocks))]
    ya = np.r_[yy, strength * prior]
    coef, residual = nnls(xa, ya, maxiter=1000)
    gradient = xa.T @ (xa @ coef - ya)
    assert np.min(coef) >= 0 and np.min(gradient) > -1e-7
    assert np.max(np.abs(coef * gradient)) < 1e-7
    weights = coef / np.array(scales)
    if weights.sum() <= 1e-12:
        weights[:] = 1.
    weights /= weights.sum()
    return dict(landmarks=fit[take], coefficient=coef, scales=np.array(scales),
                weights=weights, residual=float(residual), kkt_gradient=gradient,
                cross_scaffold_pairs=int(np.sum(mask)), target=target)


def design(d, k, part, c=None):
    c = c or CONFIGS[0]
    fit, query, source, blocks, names, normalizers = channels(d, k, part)
    state = alignment(blocks, fit, d, k, c)
    # Positive mixtures and Schur products preserve PSD, no query labels used.
    matrix = sum(w * z for w, z in zip(state['weights'], blocks))
    n = len(fit)
    x, qx = (matrix[:n] + matrix[:n].T) * .5, matrix[n:]
    metric, y = d['metrics'][k], d['y'][fit]
    center = float(np.median(y)) if metric == 'MAE' else 0.
    scale = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05) if metric == 'MAE' else 1.
    target = (y - center) / scale if metric == 'MAE' else r83.ranknorm(y) if metric == 'Spearman' else y
    state.update(fit=fit, query=query, source=source, channel_names=names,
                 normalizers=normalizers, target=target, center=center, scale=scale)
    assert np.isfinite(x).all() and np.isfinite(qx).all()
    return dict(x=x, qx=qx, state=state)


def estimator(metric, c):
    return (SVC(C=c['C'], kernel='precomputed', tol=1e-5, cache_size=1024)
            if metric in ('AUROC', 'AUPRC') else
            SVR(C=c['C'], epsilon=.04, kernel='precomputed', tol=1e-5, cache_size=1024))


def infer(net, x, metric, state):
    p = net.decision_function(x) if metric in ('AUROC', 'AUPRC') else net.predict(x)
    return p * state['scale'] + state['center']


def replay(d, k, part, c):
    a = design(d, k, part, c)
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    same(a['state'], joblib.load(folder / 'transform.joblib'))
    path = folder / 'model.joblib'
    net = joblib.load(path)
    return infer(net, a['qx'], d['metrics'][k], a['state']), {
        'transform.joblib': b.sha(folder / 'transform.joblib'), 'model.joblib': b.sha(path)}


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
    a = design(d, k, part, c)
    net = estimator(d['metrics'][k], c)
    net.fit(a['x'], a['state']['target'])
    assert net.fit_status_ == 0
    p = infer(net, a['qx'], d['metrics'][k], a['state'])
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(a['state'], folder / 'transform.joblib', compress=3)
    path = folder / 'model.joblib'
    joblib.dump(net, path, compress=3)
    np.testing.assert_array_equal(p, infer(joblib.load(path), a['qx'], d['metrics'][k], a['state']))
    np.savez_compressed(file, signature=sig(), fit=a['state']['fit'], query=a['state']['query'], p=p)
    print(json.dumps(dict(stage='train', task=d['tasks'][k], part=part,
                          seconds=round(time.monotonic() - started), weights=a['state']['weights'].tolist())), flush=True)
    return p


def smoke(d):
    from tempfile import TemporaryDirectory
    # Analytical nonnegative quadratic reference, with an interior optimum.
    xx = np.array([[1., .2], [.3, 1.], [.4, .5]])
    yy = xx @ np.array([.7, .3])
    coef, residual = nnls(xx, yy)
    np.testing.assert_allclose(coef, np.linalg.solve(xx.T @ xx, xx.T @ yy), atol=1e-13, rtol=0)
    checked = []
    for task in ('herg', 'solubility_aqsoldb', 'clearance_hepatocyte_az'):
        k = d['tasks'].index(task)
        a = design(d, k, 'fold0')
        outside = np.flatnonzero((d['tid'] == k) & ~np.isin(np.arange(len(d['y'])), a['state']['fit']))
        labels = d['y'][outside].copy()
        try:
            d['y'][outside] = 999999.
            d.pop('_v96_assays', None)
            same(a, design(d, k, 'fold0'))
        finally:
            d['y'][outside] = labels
            d.pop('_v96_assays', None)
        np.testing.assert_allclose(np.diag(a['x']), 1., atol=1e-12, rtol=0)
        assert np.linalg.eigvalsh(a['x'][:64, :64]).min() > -1e-8
        net = estimator(d['metrics'][k], CONFIGS[0])
        net.fit(a['x'][:96, :96], a['state']['target'][:96])
        assert net.fit_status_ == 0
        qx = a['qx'][:8, :96]
        p = infer(net, qx, d['metrics'][k], a['state'])
        np.testing.assert_array_equal(p, infer(net, qx[::-1], d['metrics'][k], a['state'])[::-1])
        with TemporaryDirectory(dir=OUT) as temp:
            path = Path(temp) / 'model.joblib'
            joblib.dump(net, path)
            np.testing.assert_array_equal(p, infer(joblib.load(path), qx, d['metrics'][k], a['state']))
        checked.append(task)
    b.save(OUT / 'smoke.json', dict(status='PASS', signature=sig(), query_label_isolation=True,
           alignment_fit_only_cross_scaffold=True, nnls_closed_form_reference=True,
           actual_alignment_kkt=True, nonnegative_psd_reference=True,
           independent_reload_tasks=checked, prediction_row_order=True, no_query_scores_used=True))
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
