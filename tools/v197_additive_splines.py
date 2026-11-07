"""Smooth additive molecular-response models, fit before frozen confirmation."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.linalg import solve
from scipy.special import expit
from sklearn.preprocessing import SplineTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v146_current_baseline as v
import v148_experiment_runner as runner
b, r83 = v.b, v.r83
OUT = ROOT / 'admet_model_outputs_v197'
CONFIGS = (dict(strength=.003), dict(strength=.03))
METHOD = ('New penalized generalized additive B-spline model: fit-only target '
          'feature screening and correlation pruning, quadratic smooth responses '
          'at four fit quantiles, missingness indicators, second-difference '
          'smoothness penalties. Logistic IRLS for classification, rank-normal '
          'squared regression for Spearman, smooth absolute-error IRLS for MAE. '
          'Augmented molecular/shape/pharmacophore and target-excluded assay '
          'features. Two fixed penalties; original v148 gates/blends, all21 '
          'improvable tasks, no confirmation fallback or historical-prediction inputs.')


def sig():
    return hashlib.sha256((b.sha(__file__) + v.sig() + b.sha(runner.__file__) +
                           importlib.metadata.version('scikit-learn')).encode()).hexdigest()


def key(c):
    return 'strength' + str(c['strength'])


def design(d, k, part):
    fit, query, raw, qr, source = v.old.features(d, k, part, 'augmented')
    raw, qr = raw.astype(float), qr.astype(float)
    median = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.
                       for col in raw.T])
    x = np.where(np.isfinite(raw), raw, median)
    center, scale = x.mean(0), x.std(0).clip(.05)
    x = np.clip((x - center) / scale, -6, 6)
    qx = np.clip((np.where(np.isfinite(qr), qr, median) - center) / scale, -6, 6)
    metric = d['metrics'][k]
    y = d['y'][fit]
    yc = float(np.median(y)) if metric == 'MAE' else 0.
    ys = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05) if metric == 'MAE' else 1.
    target = ((y - yc) / ys if metric == 'MAE' else
              r83.ranknorm(y) if metric == 'Spearman' else y)
    knots = np.quantile(x, [.05, .35, .65, .95], axis=0)
    valid = np.flatnonzero((np.diff(knots, axis=0).min(0) > 1e-5) &
                           (np.isfinite(raw).mean(0) >= .5))
    assert len(valid), 'No admissible continuous descriptor'
    # Screening and redundancy checks only see the current fitting rows.
    yy = target - target.mean()
    xx = x[:, valid] - x[:, valid].mean(0)
    norm = np.sqrt((xx * xx).sum(0)).clip(1e-10)
    corr = np.abs(xx.T @ yy) / norm / max(float(np.linalg.norm(yy)), 1e-10)
    order = np.argsort(-corr, kind='stable')
    selected = []
    maximum = min(96, max(16, len(fit) // 10))
    for j in order:
        if selected:
            pair = np.abs(xx[:, selected].T @ xx[:, j]) / norm[selected] / norm[j]
            if np.max(pair) > .995:
                continue
        selected.append(int(j))
        if len(selected) == maximum:
            break
    columns = valid[selected]
    spline = SplineTransformer(degree=2, knots=knots[:, columns], include_bias=False,
                                extrapolation='constant', sparse_output=False)
    a, z = spline.fit_transform(x[:, columns]), spline.transform(qx[:, columns])
    missing = ~np.isfinite(raw[:, columns])
    qmissing = ~np.isfinite(qr[:, columns])
    active_missing = np.flatnonzero(missing.std(0) > .01)
    matrix = np.c_[np.ones(len(fit)), a, missing[:, active_missing].astype(float)]
    query_matrix = np.c_[np.ones(len(query)), z, qmissing[:, active_missing].astype(float)]
    width = a.shape[1] // len(columns)
    assert width == 4
    penalty = np.eye(matrix.shape[1])
    penalty[0, 0] = 0.
    difference = np.diff(np.eye(width), n=2, axis=0)
    for i in range(len(columns)):
        block = slice(1 + i * width, 1 + (i + 1) * width)
        penalty[block, block] += 10 * difference.T @ difference
    assert np.isfinite(matrix).all() and np.isfinite(query_matrix).all()
    assert not set(fit) & set(query) and k not in set(d['tid'][source])
    state = dict(fit=fit, query=query, source=source, median=median,
                 feature_center=center, feature_scale=scale, columns=columns,
                 knots=knots[:, columns], active_missing=active_missing,
                 bsplines=[dict(t=s.t, c=s.c, k=s.k) for s in spline.bsplines_],
                 target=target, center=yc, scale=ys)
    return dict(x=matrix, qx=query_matrix, penalty=penalty, state=state)


def loss(metric, raw, target):
    if metric in ('AUROC', 'AUPRC'):
        return np.logaddexp(0, raw) - target * raw
    if metric == 'MAE':
        return np.sqrt((raw - target) ** 2 + .03 ** 2)
    return .5 * (raw - target) ** 2


def fit_model(x, target, penalty, metric, strength):
    n, width = x.shape
    coef = np.zeros(width)
    coef[0] = (np.log(np.clip(target.mean(), 1e-5, 1 - 1e-5) /
                      np.clip(1 - target.mean(), 1e-5, 1))
               if metric in ('AUROC', 'AUPRC') else float(np.median(target)))
    regularizer = strength * penalty

    def value(beta):
        return float(loss(metric, x @ beta, target).mean() + .5 * beta @ regularizer @ beta)

    history = [value(coef)]
    converged = False
    for iteration in range(150):
        raw = x @ coef
        if metric in ('AUROC', 'AUPRC'):
            prob = expit(raw)
            weights = (prob * (1 - prob)).clip(1e-7)
            residual = prob - target
        elif metric == 'MAE':
            weights = 1 / np.sqrt((raw - target) ** 2 + .03 ** 2)
            residual = weights * (raw - target)
        else:
            weights = np.ones(n)
            residual = raw - target
        gradient = x.T @ residual / n + regularizer @ coef
        if np.linalg.norm(gradient, ord=np.inf) < 1e-7:
            converged = True
            break
        hessian = x.T @ (x * weights[:, None]) / n + regularizer
        hessian.flat[::width + 1] += 1e-10
        direction = solve(hessian, gradient, assume_a='pos', check_finite=False)
        rate = 1.
        before = history[-1]
        while rate > 2 ** -25:
            next_coef = coef - rate * direction
            after = value(next_coef)
            if after <= before - 1e-4 * rate * float(gradient @ direction):
                break
            rate *= .5
        assert rate > 2 ** -25, 'Optimizer did not find a descent step'
        coef = next_coef
        history.append(after)
        if abs(before - after) < 1e-10 * (1 + abs(before)):
            converged = True
            break
    assert converged and np.isfinite(coef).all(), 'Unconverged model is not eligible'
    assert np.all(np.diff(history) <= 1e-10)
    return dict(coef=coef, iterations=iteration + 1, objective=history[-1],
                objective_history=history, converged=converged)


def output(metric, raw, state, task):
    p = expit(raw) if metric in ('AUROC', 'AUPRC') else raw * state['scale'] + state['center']
    return p.clip(0, 100) if task == 'ppbr_az' else p


def replay(d, k, part, c):
    from v161_verify_and_assemble import same
    a = design(d, k, part)
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    same(a['state'], joblib.load(folder / 'transform.joblib'))
    model = joblib.load(folder / 'model.joblib')
    assert model['signature'] == sig() and model['config'] == c and model['converged']
    p = output(d['metrics'][k], a['qx'] @ model['coef'], a['state'], d['tasks'][k])
    return p, {name: b.sha(folder / name) for name in ('model.joblib', 'transform.joblib')}


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
    state = a['state']
    model = fit_model(a['x'], state['target'], a['penalty'], d['metrics'][k], c['strength'])
    model.update(signature=sig(), config=c)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, folder / 'model.joblib', compress=3)
    joblib.dump(state, folder / 'transform.joblib', compress=3)
    raw = a['qx'] @ model['coef']
    np.testing.assert_array_equal(raw, a['qx'] @ joblib.load(folder / 'model.joblib')['coef'])
    p = output(d['metrics'][k], raw, state, d['tasks'][k])
    assert np.isfinite(p).all()
    np.savez_compressed(file, signature=sig(), fit=state['fit'], query=state['query'],
                        p=p, center=state['center'], scale=state['scale'])
    print(json.dumps(dict(stage='train', task=d['tasks'][k], part=part, config=c,
                          columns=len(state['columns']), iterations=model['iterations'],
                          seconds=round(time.monotonic() - start))), flush=True)
    return p


def smoke(d):
    from v161_verify_and_assemble import same
    checks = []
    for task in ('herg', 'caco2_wang', 'clearance_hepatocyte_az'):
        k = d['tasks'].index(task)
        a = design(d, k, 'fold0')
        saved = d['y'][a['state']['query']].copy()
        try:
            d['y'][a['state']['query']] = 999999.
            d.pop('_v96_assays', None)
            same(a, design(d, k, 'fold0'))
        finally:
            d['y'][a['state']['query']] = saved
            d.pop('_v96_assays', None)
        z = fit_model(a['x'], a['state']['target'], a['penalty'], d['metrics'][k], .03)
        checks.append(dict(task=task, query_label_perturbation_invariant=True,
                           converged=z['converged'], objective_monotonic=True,
                           iterations=z['iterations']))
    # A squared-response solve must equal its closed-form penalized solution.
    x = np.c_[np.ones(30), np.arange(30) / 30]
    y = np.sin(np.arange(30))
    penalty = np.diag([0., 1.])
    z = fit_model(x, y, penalty, 'Spearman', .03)
    expected = solve(x.T @ x / len(x) + .03 * penalty, x.T @ y / len(x))
    np.testing.assert_allclose(z['coef'], expected, atol=1e-8, rtol=1e-8)
    b.save(OUT / 'smoke.json', dict(status='PASS', signature=sig(), checks=checks,
           squared_solver_reference_match=True))
    print('SMOKE_PASS', flush=True)


def main():
    d = v.load()
    OUT.mkdir(exist_ok=True)
    smoke(d)
    if '--smoke' not in sys.argv:
        runner.run(sys.modules[__name__], d)


if __name__ == '__main__':
    main()
