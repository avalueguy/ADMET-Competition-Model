"""Exact-Newton numerical repair of the unfinished additive MAE models."""
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
OUT = ROOT / 'admet_model_outputs_v200'
CONFIGS = (dict(strength=.003), dict(strength=.03))
TASKS = ('solubility_aqsoldb', 'caco2_wang', 'ppbr_az', 'ld50_zhu',
         'lipophilicity_astrazeneca')
METHOD = ('Numerical repair of unfinished v197 MAE additive splines: exact '
          'smooth-absolute Hessian instead of the slowly converging IRLS '
          'majorizer. Identical fit-only design, objectives, penalties, blends '
          'and v148 selection/confirmation gates. Only five MAE endpoints; '
          'no hyperparameter search from confirmation or failed task fallback.')



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
            distance = np.sqrt((raw - target) ** 2 + .03 ** 2)
            weights = .03 ** 2 / distance ** 3
            residual = (raw - target) / distance
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
        # A small loss change alone does not establish stationarity.
        # Keep the original gradient tolerance and150-iteration budget.
    raw = x @ coef
    distance = np.sqrt((raw - target) ** 2 + .03 ** 2)
    final_gradient = x.T @ ((raw - target) / distance) / n + regularizer @ coef
    assert metric == 'MAE'
    assert converged and np.max(np.abs(final_gradient)) < 1e-7 and np.isfinite(coef).all(), 'Unconverged model is not eligible'
    assert np.all(np.diff(history) <= 1e-10)
    return dict(coef=coef, iterations=iteration + 1, objective=history[-1],
                objective_history=history, converged=converged,
                gradient_inf=float(np.max(np.abs(final_gradient))))


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
    from scipy.optimize import minimize
    from v161_verify_and_assemble import same
    rng = np.random.default_rng(2001028)
    x = np.c_[np.ones(120), rng.normal(size=(120, 7))]
    y = x @ rng.normal(size=8) + rng.standard_t(3, 120)
    penalty = np.diag([0.] + [1.] * 7)
    strength = .03
    beta = rng.normal(size=8)
    def value(z):
        r = x @ z - y
        return np.sqrt(r*r + .03**2).mean() + .5*strength*z@penalty@z
    def gradient(z):
        r = x @ z - y
        return x.T@(r/np.sqrt(r*r+.03**2))/len(y) + strength*penalty@z
    r = x @ beta - y
    hessian = x.T@(x*(.03**2/(r*r+.03**2)**1.5)[:, None])/len(y) + strength*penalty
    step = 1e-5
    directions = np.eye(len(beta))
    fd_gradient = np.array([(value(beta+step*h)-value(beta-step*h))/(2*step) for h in directions])
    fd_hessian = np.column_stack([(gradient(beta+step*h)-gradient(beta-step*h))/(2*step) for h in directions])
    np.testing.assert_allclose(gradient(beta), fd_gradient, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(hessian, fd_hessian, atol=2e-6, rtol=1e-5)
    reference = minimize(value, np.zeros(8), jac=gradient, method='BFGS', options=dict(gtol=1e-9, maxiter=150))
    assert np.max(np.abs(gradient(reference.x))) < 1e-7
    result = fit_model(x, y, penalty, 'MAE', strength)
    np.testing.assert_allclose(result['coef'], reference.x, atol=2e-6, rtol=2e-6)
    assert abs(result['objective']-value(reference.x)) < 1e-10
    checks = []
    # Actual previously failing fitting partition; do not score query labels.
    for task, part, strength in (('solubility_aqsoldb','fold1',.003),
                                  ('caco2_wang','fold0',.03)):
        k = d['tasks'].index(task)
        a = design(d,k,part)
        saved = d['y'][a['state']['query']].copy()
        try:
            d['y'][a['state']['query']] = 999999.
            d.pop('_v96_assays',None)
            same(a,design(d,k,part))
        finally:
            d['y'][a['state']['query']] = saved
            d.pop('_v96_assays',None)
        z = fit_model(a['x'],a['state']['target'],a['penalty'],'MAE',strength)
        checks.append(dict(task=task,part=part,query_label_perturbation_invariant=True,
                           converged=True,gradient_inf=z['gradient_inf'],
                           iterations=z['iterations'],objective_monotonic=True,
                           query_metrics_not_examined=True))
    b.save(OUT/'smoke.json',dict(status='PASS',signature=sig(),checks=checks,
           finite_difference_gradient=True,finite_difference_hessian=True,
           independent_BFGS_reference_match=True))
    print('SMOKE_PASS',flush=True)


def run(d):
    # Scope differs from v148.run; all numerical gates call unchanged v148 code.
    OUT.mkdir(exist_ok=True)
    doc=dict(signature=sig(),tasks=TASKS,configs=CONFIGS,weights=runner.WEIGHTS,
        baseline='v145',method=METHOD,
        selection='Max minimum fold0/1 gain; freeze all choices before confirmation; no fallback.',
        confirmation='Original v148.confirm unchanged: all4 positive, practical mean threshold, both scaffold-bootstrap90 lower bounds>0.',
        test_labels_used=False,external_data=False,
        limitations='Public labels reused; historical upstream controls approximate; no official score estimate.')
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==json.loads(json.dumps(doc))
    else:b.save(path,doc)
    dev,selected=[],[]
    for task in TASKS:
        k=d['tasks'].index(task)
        assert d['metrics'][k]=='MAE'
        choices=[]
        for c in CONFIGS:
            for w in runner.WEIGHTS:
                parts=[runner.record(d,k,p,predict,c,w) for p in ('fold0','fold1')]
                choices.append(dict(task=task,metric='MAE',config=c,weight=w,parts=parts,
                                    objective=min(r['gain'] for r in parts)))
        best=max(choices,key=lambda r:r['objective'])
        dev.extend(choices)
        if best['objective']>v.threshold('MAE'):selected.append(best)
        b.save(OUT/'development.json',dict(status='IN_PROGRESS',signature=sig(),records=dev))
        print(json.dumps(dict(stage='development',best=best)),flush=True)
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=dev))
    b.save(OUT/'frozen_selection.json',dict(signature=sig(),selected=selected))
    records=[]
    for ch in selected:
        k=d['tasks'].index(ch['task'])
        result=runner.confirm(d,k,predict,ch)
        records.append(result)
        if result['accepted']:predict(d,k,'final',ch['config'])
        b.save(OUT/'confirmation.json',dict(status='IN_PROGRESS',signature=sig(),records=records))
        print(json.dumps(dict(stage='confirmation',**result)),flush=True)
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,
                                       accepted=[r['task'] for r in records if r['accepted']]))


def main():
    d=v.load()
    OUT.mkdir(exist_ok=True)
    if '--smoke' in sys.argv:smoke(d)
    else:
        assert json.loads((OUT/'smoke.json').read_text())['signature']==sig()
        run(d)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        b.save(OUT/'failure.json',dict(status='FAILED',traceback=traceback.format_exc(),signature=sig()))
        raise
