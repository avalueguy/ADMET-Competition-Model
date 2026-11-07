"""Conditional weighted-median forests for MAE endpoints, with fixed gates."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v146_current_baseline as v
import v148_experiment_runner as runner
b, r83 = v.b, v.r83
OUT = ROOT / 'admet_model_outputs_v181'
CONFIGS = (dict(leaf=3, trees=400), dict(leaf=12, trees=400))
METHOD = ('Quantile ExtraTrees: fit target-specific partitions on allowed augmented chemistry '
          'and target-excluded official assay context, then predict the weighted median of '
          'training outcomes under leaf cooccurrence weights. Unlike an average of tree means, '
          'this explicitly targets MAE. Fit-only imputation; fixed two leaf sizes and baseline blends.')


def sig():
    return hashlib.sha256((b.sha(__file__) + v.sig() + b.sha(runner.__file__)).encode()).hexdigest()


def key(c):
    return 'leaf' + str(c['leaf'])


def design(d, k, part):
    fit, q, x, qx, source = v.old.features(d, k, part, 'augmented')
    x, qx = x.astype(float), qx.astype(float)
    median = np.array([np.median(z[np.isfinite(z)]) if np.isfinite(z).any() else 0. for z in x.T])
    xx = np.where(np.isfinite(x), x, median)
    zz = np.where(np.isfinite(qx), qx, median)
    # Missingness remains explicit; all statistics come from fitting rows.
    xx = np.c_[xx, np.isfinite(x)].astype(np.float32)
    zz = np.c_[zz, np.isfinite(qx)].astype(np.float32)
    assert np.isfinite(xx).all() and np.isfinite(zz).all()
    return dict(fit=fit, query=q, source=source, x=xx, qx=zz, median=median)


def incidence(forest, x):
    offsets = np.r_[0, np.cumsum([tree.tree_.node_count for tree in forest.estimators_])]
    leaves = forest.apply(x) + offsets[:-1]
    return sparse.csr_matrix((np.ones(leaves.size),
                             (np.repeat(np.arange(len(x)), leaves.shape[1]), leaves.ravel())),
                            shape=(len(x), int(offsets[-1])))


def quantile(forest, x, qx, y):
    members = incidence(forest, x)
    sizes = np.asarray(members.sum(0)).ravel()
    inv = np.divide(1., sizes, out=np.zeros_like(sizes), where=sizes > 0)
    order = np.argsort(y, kind='stable')
    sorted_y = y[order]
    reference = members[order].T.tocsr()
    result = []
    for lo in range(0, len(qx), 128):
        query = incidence(forest, qx[lo:lo + 128])
        weights = (query.multiply(inv) @ reference).toarray()
        totals = weights.sum(1)
        assert np.allclose(totals, len(forest.estimators_), atol=1e-7, rtol=0)
        cumulative = np.cumsum(weights, axis=1)
        # Choose the lower weighted median deterministically for exact ties.
        ix = (cumulative < totals[:, None] / 2).sum(1)
        result.append(sorted_y[ix])
    return np.concatenate(result).astype(float)


def predict(d, k, part, c):
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    path = folder / 'predictions.npz'
    if path.exists():
        z = np.load(path)
        assert str(z['signature']) == sig()
        np.testing.assert_array_equal(z['query'], v.split(d, k, part)[1])
        return z['p'].copy()
    started = time.monotonic()
    a = design(d, k, part)
    y = d['y'][a['fit']]
    assert d['metrics'][k] == 'MAE' and np.isfinite(y).all()
    forest = ExtraTreesRegressor(n_estimators=c['trees'], min_samples_leaf=c['leaf'],
                                 max_features=.8, max_depth=20, criterion='squared_error',
                                 bootstrap=False, random_state=1814026, n_jobs=2)
    forest.fit(a['x'], y)
    p = quantile(forest, a['x'], a['qx'], y)
    if d['tasks'][k] == 'ppbr_az':
        p = p.clip(0, 100)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(forest, folder / 'forest.joblib', compress=3)
    replay = joblib.load(folder / 'forest.joblib')
    rp = quantile(replay, a['x'], a['qx'], y)
    if d['tasks'][k] == 'ppbr_az':
        rp = rp.clip(0, 100)
    np.testing.assert_array_equal(p, rp)
    np.savez_compressed(path, signature=sig(), fit=a['fit'], query=a['query'],
                        source=a['source'], median=a['median'], p=p)
    print(json.dumps(dict(stage='train', task=d['tasks'][k], part=part, config=c,
                          seconds=round(time.monotonic() - started))), flush=True)
    return p


def smoke(d):
    k = d['tasks'].index('caco2_wang')
    a = design(d, k, 'fold0')
    saved = d['y'][a['query']].copy()
    try:
        d['y'][a['query']] = 888888.
        d.pop('_v96_assays', None)
        z = design(d, k, 'fold0')
        for name in a:
            np.testing.assert_array_equal(a[name], z[name])
    finally:
        d['y'][a['query']] = saved
        d.pop('_v96_assays', None)
    forest = ExtraTreesRegressor(n_estimators=7, min_samples_leaf=3, random_state=1814026, n_jobs=2)
    forest.fit(a['x'], d['y'][a['fit']])
    p = quantile(forest, a['x'], a['qx'][:11], d['y'][a['fit']])
    assert len(p) == 11 and np.isfinite(p).all()
    for i in range(3):
        leaf = forest.apply(a['x'])
        qleaf = forest.apply(a['qx'][i:i+1])[0]
        weights = np.zeros(len(a['fit']))
        for j in range(7):
            mask = leaf[:, j] == qleaf[j]
            weights[mask] += 1 / mask.sum()
        y = d['y'][a['fit']]
        order = np.argsort(y, kind='stable')
        expected = y[order][np.flatnonzero(np.cumsum(weights[order]) >= weights.sum() / 2)[0]]
        assert p[i] == expected
    OUT.mkdir(exist_ok=True)
    b.save(OUT / 'smoke.json', dict(status='PASS', signature=sig(),
           query_label_perturbation_invariant=True, explicit_weighted_median_reference_pass=True,
           external_data=False, test_labels_used=False))


def main():
    d = v.load()
    smoke(d)
    tasks = [t for t in v.tasks(d) if d['metrics'][d['tasks'].index(t)] == 'MAE']
    doc = dict(signature=sig(), tasks=tasks, configs=CONFIGS, weights=runner.WEIGHTS,
               baseline='v145', method=METHOD, acceptance='Unchanged v148 all4/practical mean/both CI90 lower>0; no failed fallback',
               test_labels_used=False, new_external_data=False,
               limitations='Public labels reused; some historical controls approximate. Official gain cannot be guaranteed.')
    path = OUT / 'protocol.json'
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(doc))
    else:
        b.save(path, doc)
    development, selected = [], []
    for task in tasks:
        k = d['tasks'].index(task)
        choices = []
        for config in CONFIGS:
            for weight in runner.WEIGHTS:
                parts = [runner.record(d, k, p, predict, config, weight) for p in ('fold0', 'fold1')]
                choices.append(dict(task=task, metric='MAE', config=config, weight=weight,
                                    parts=parts, objective=min(r['gain'] for r in parts)))
        best = max(choices, key=lambda r: r['objective'])
        development.extend(choices)
        if best['objective'] > v.threshold('MAE'):
            selected.append(best)
        b.save(OUT / 'development.json', dict(status='IN_PROGRESS', signature=sig(), records=development))
        print(json.dumps(dict(stage='development', best=best)), flush=True)
    b.save(OUT / 'development.json', dict(status='COMPLETE', signature=sig(), records=development))
    b.save(OUT / 'frozen_selection.json', dict(signature=sig(), selected=selected))
    records = []
    for choice in selected:
        k = d['tasks'].index(choice['task'])
        record = runner.confirm(d, k, predict, choice)
        records.append(record)
        if record['accepted']:
            predict(d, k, 'final', choice['config'])
        b.save(OUT / 'confirmation.json', dict(status='IN_PROGRESS', signature=sig(), records=records))
        print(json.dumps(dict(stage='confirmation', **record)), flush=True)
    b.save(OUT / 'confirmation.json', dict(status='COMPLETE', signature=sig(), records=records,
                                          accepted=[r['task'] for r in records if r['accepted']]))


if __name__ == '__main__':
    smoke(v.load()) if '--smoke' in sys.argv else main()
