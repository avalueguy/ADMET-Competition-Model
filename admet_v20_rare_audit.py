"""Remote verification, uncertainty diagnostics and reproducible v20 final models."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score as ap
from sklearn.model_selection import GroupShuffleSplit

from admet_v20_rare import (OUT, ROOT, TASKS, feats, model, combine, predict_ensemble,
                           load_task, Preprocess, partition, remote_only,
                           anchor_oof, anchor_valid, anchor_test, read_json, save_json, sha)


def bootstrap(y, baseline, prediction, groups, seed=20727, repeats=600):
    """Paired scaffold resampling, conditional on trained models and chosen recipe."""
    labels, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(labels))]
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(repeats):
        rows = np.concatenate([members[i] for i in rng.integers(len(labels), size=len(labels))])
        if len(np.unique(y[rows])) < 2:
            continue
        gains.append(float(ap(y[rows], prediction[rows]) - ap(y[rows], baseline[rows])))
    return dict(draws=len(gains), percentile_95=list(map(float, np.quantile(gains, [.025, .975]))),
                positive_fraction=float(np.mean(np.array(gains) > 0)),
                caveat='Conditional paired scaffold bootstrap; excludes model-selection uncertainty and is not an independent test.')


def audit(task):
    dest = OUT / task
    route = read_json(dest / 'decision.json')
    sel = route['selection']
    d = load_task(task)
    y, folds, base = anchor_oof(task)
    vy, vb = anchor_valid(task)
    n, nv = len(y), len(vy)
    np.testing.assert_array_equal(folds, partition(task, d))
    np.testing.assert_array_equal(y, d['frames']['train'].Y.to_numpy(float))
    np.testing.assert_array_equal(vy, d['frames']['valid'].Y.to_numpy(float))
    oo = pd.read_csv(dest / 'rare_oof.csv')
    va = pd.read_csv(dest / 'rare_valid.csv')
    te = pd.read_csv(dest / 'rare_test.csv')
    np.testing.assert_array_equal(oo.row, np.arange(n))
    np.testing.assert_array_equal(oo.fold, folds)
    np.testing.assert_array_equal(oo.Y_true, y)
    np.testing.assert_array_equal(va.Y_true, vy)
    assert len(te) == len(d['frames']['test']) == len(anchor_test(task))
    for frame in [oo, va, te]:
        p = frame.Y_pred.to_numpy(float)
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    checks = []
    for f in np.unique(folds):
        mask = folds == f
        assert not set(d['groups'][:n][mask]) & set(d['groups'][:n][~mask])
        assert not set(d['canonical'][:n][mask]) & set(d['canonical'][:n][~mask])
        p = combine(base[mask], oo.Y_pred.to_numpy()[mask], sel['weight'], sel['mode'])
        checks.append(dict(fold=int(f), n=int(mask.sum()), positives=int(y[mask].sum()),
                           before=float(ap(y[mask], base[mask])), after=float(ap(y[mask], p))))
    vp = combine(vb, va.Y_pred.to_numpy(), sel['weight'], sel['mode'])
    result = dict(task=task, selection=sel, folds=checks, train_n=n, valid_n=nv,
                  test_n=len(te), alignment_passed=True, scaffold_overlap=0,
                  baseline_valid=float(ap(vy, vb)), candidate_valid=float(ap(vy, va.Y_pred)),
                  selected_valid=float(ap(vy, vp)),
                  prediction_hashes={p.name: sha(p) for p in dest.glob('rare_*.csv')})
    if sel['weight'] > 0:
        result['valid_uncertainty'] = bootstrap(vy, vb, vp, d['groups'][n:n+nv])
    save_json(dest / 'audit.json', result)
    print(json.dumps(result), flush=True)


def confirm(task):
    dest = OUT / task
    sel = read_json(dest / 'decision.json')['selection']
    if sel['weight'] == 0:
        return
    d = load_task(task)
    y = d['frames']['train'].Y.to_numpy(float)
    it, iv = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=692509).split(
        np.arange(len(y)), groups=d['groups'][:len(y)]))
    basefile = ROOT / 'admet_model_outputs_v16' / task / 'repeat_comparison.csv'
    route19 = ROOT / 'admet_model_outputs_v19' / task / 'selection.json'
    if route19.exists() and read_json(route19).get('candidate_mass', 0) > 0:
        raise RuntimeError('Need exact v19 repeat anchor reconstruction before confirmation')
    b = pd.read_csv(basefile)
    np.testing.assert_array_equal(b.row, iv)
    np.testing.assert_array_equal(b.Y_true, y[iv])
    assert not set(d['groups'][it]) & set(d['groups'][iv])
    cached = dest / 'rare_repeat.csv'
    if cached.exists():
        c = pd.read_csv(cached)
        np.testing.assert_array_equal(c.row, iv)
        pred = c.Y_pred.to_numpy(float)
    else:
        x = feats(d)
        pre = Preprocess().fit(x[it])
        pred = predict_ensemble(pre.transform(x[it]), y[it], [30300, 30301, 30302], pre.transform(x[iv]))
        pd.DataFrame(dict(row=iv, Y_true=y[iv], Y_pred=pred)).to_csv(cached, index=False)
    bp = b.blend.to_numpy(float)
    qp = combine(bp, pred, sel['weight'], sel['mode'])
    gain = float(ap(y[iv], qp) - ap(y[iv], bp))
    result = dict(task=task, fixed_selection=sel, rows=len(iv), baseline=float(ap(y[iv], bp)),
                  candidate=float(ap(y[iv], pred)), selected=float(ap(y[iv], qp)), gain=gain,
                  uncertainty=bootstrap(y[iv], bp, qp, d['groups'][iv]),
                  selection_changed=False, caveat='Previously used repeat split; diagnostic only, not independent validation.')
    save_json(dest / 'confirmation.json', result)
    print(json.dumps(result), flush=True)


def checkpoint(task):
    dest = OUT / task
    if read_json(dest / 'decision.json')['selection']['weight'] == 0:
        return
    d = load_task(task)
    total = len(d['frames']['train']) + len(d['frames']['valid'])
    y = pd.concat([d['frames']['train'], d['frames']['valid']], ignore_index=True).Y.to_numpy(float)
    x = feats(d)
    target = dest / 'final_models'
    target.mkdir(exist_ok=True)
    pre = Preprocess().fit(x[:total])
    xx, xt = pre.transform(x[:total]), pre.transform(x[total:])
    joblib.dump(pre, target / 'preprocess.joblib', compress=3)
    ratio = max((1-float(y.mean())) / max(float(y.mean()), 1e-3), 1.)
    predictions = []
    for i, power in enumerate([.5, 1., .75]):
        path = target / f'model_{i}.cbm'
        m = model(xx, y, 30200+i, ratio**power)
        p = m.predict_proba(xt)[:, 1]
        m.save_model(str(path))
        loaded = CatBoostClassifier()
        loaded.load_model(str(path))
        np.testing.assert_allclose(loaded.predict_proba(xt)[:, 1], p, rtol=0, atol=1e-12)
        predictions.append(p)
    new = np.mean(predictions, axis=0)
    previous = pd.read_csv(dest / 'rare_test.csv').Y_pred.to_numpy(float)
    prior_hash = sha(dest / 'rare_test.csv')
    # Preserve predictions from the initial uncheckpointed training for provenance.
    backup = dest / 'initial_test_predictions.csv'
    if not backup.exists():
        backup.write_bytes((dest / 'rare_test.csv').read_bytes())
    pd.DataFrame(dict(Y_pred=new)).to_csv(dest / 'rare_test.csv', index=False)
    result = dict(task=task, fit_rows=total, query_rows=len(new), train_plus_valid_only=True,
                  seeds=[30200, 30201, 30202], model_replay_passed=True,
                  prior_prediction_sha256=prior_hash, prediction_sha256=sha(dest / 'rare_test.csv'),
                  max_change_from_gpu_refit=float(np.max(np.abs(new - previous))),
                  mean_change_from_gpu_refit=float(np.mean(np.abs(new - previous))),
                  files={p.name: sha(p) for p in target.iterdir()})
    save_json(dest / 'checkpoint.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    remote_only()
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['audit', 'confirm', 'checkpoint'])
    parser.add_argument('--tasks', nargs='*')
    args = parser.parse_args()
    for task in args.tasks or TASKS:
        globals()[args.stage](task)
