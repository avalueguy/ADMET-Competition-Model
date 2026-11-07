"""Fit-only grouped PCA rotations and diverse nonlinear molecular tree ensembles."""
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
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v146_current_baseline as v
import v148_experiment_runner as runner
b, r83 = v.b, v.r83
OUT = ROOT / 'admet_model_outputs_v224'
SEEDS = (2241029, 2241030)
CONFIGS = (dict(rotations=8, trees=24, group=8),)
METHOD = ('Rotation forest: fit-only standardized molecular fragment counts, '
          'physical/shape/pharmacophore and target-excluded measured context. '
          'Each member randomly partitions features into groups of8 and learns '
          'full-rank orthogonal PCA rotations on75% of actual fitting structures. '
          'Tree ensembles then split learned oblique chemical directions instead '
          'of original feature axes. Eight independent rotations per seed, two '
          'fixed seeds,24 randomized trees per rotation. No old models/predictions, '
          'no query-based early stopping. Originalv148 gates unchanged.')


def sig():
    return hashlib.sha256((b.sha(__file__) + v.sig() + b.sha(runner.__file__) +
                           importlib.metadata.version('scikit-learn')).encode()).hexdigest()


def key(c):
    return 'rot' + str(c['rotations']) + '_trees' + str(c['trees'])


def load():
    return v.load()


def design(d, k, part):
    fit, query, x, qx, source = v.old.features(d, k, part, 'full')
    extra = np.c_[d['conformer_features'], d['pharmacophore']]
    x = np.c_[x, extra[d['mi'][fit]]].astype(float)
    qx = np.c_[qx, extra[d['mi'][query]]].astype(float)
    # The full block prepends assay context, then six2048 count fingerprints,
    # followed by physical descriptors; retain all nonfragment columns.
    auxiliary_width = x.shape[1] - d['views']['multiview'].shape[1] - extra.shape[1]
    assert auxiliary_width >= 0
    first, last = auxiliary_width, auxiliary_width + 12288
    frequency = np.sum(x[:, first:last] > 0, axis=0)
    active = np.flatnonzero(frequency >= 3)
    if len(active) > 1024:
        active = np.sort(active[np.argsort(frequency[active], kind='stable')[-1024:]])
    columns = np.r_[np.arange(first), active + first, np.arange(last, x.shape[1])]
    x, qx = x[:, columns], qx[:, columns]
    finite = np.isfinite(x)
    median = np.array([np.median(z[np.isfinite(z)]) if np.isfinite(z).any() else 0. for z in x.T])
    clean = np.where(finite, x, median)
    center, scale = clean.mean(0), clean.std(0).clip(.05)
    keep = np.flatnonzero((np.ptp(clean, axis=0) > 1e-8) | (finite.sum(0) != len(fit)))
    def transform(mat):
        normalized = np.clip((np.where(np.isfinite(mat), mat, median) - center) / scale, -5., 5.)
        return np.c_[normalized[:, keep], (~np.isfinite(mat[:, keep])).astype(float)].astype(np.float32)
    xx, qq = transform(x), transform(qx)
    # Drop constant missingness columns using fitting rows only.
    varying = np.flatnonzero(np.ptp(xx, axis=0) > 1e-8)
    xx, qq = xx[:, varying], qq[:, varying]
    metric, y = d['metrics'][k], d['y'][fit]
    target_center = float(np.median(y)) if metric == 'MAE' else 0.
    target_scale = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05) if metric == 'MAE' else 1.
    target = ((y - target_center) / target_scale if metric == 'MAE' else
              r83.ranknorm(y) if metric == 'Spearman' else y)
    assert np.isfinite(xx).all() and np.isfinite(qq).all()
    assert not set(fit) & set(query) and np.all(d['tid'][source] != k)
    state = dict(fit=fit, query=query, source=source, columns=columns,
                 median=median, center=center, scale=scale, keep=keep, varying=varying,
                 target=target, target_center=target_center, target_scale=target_scale,
                 auxiliary_width=auxiliary_width)
    return dict(x=xx, qx=qq, state=state)


def rotation(x, seed, index, group):
    rng = np.random.default_rng(seed + 1009 * index)
    order = rng.permutation(x.shape[1])
    rows = np.sort(rng.choice(len(x), max(2, int(.75 * len(x))), replace=False))
    width = int(np.ceil(len(order) / group)) * group
    ref = np.zeros((len(rows), width), dtype=np.float64)
    ref[:, :len(order)] = x[rows][:, order]
    ref = ref.reshape(len(rows), -1, group)
    mean = ref.mean(0)
    centered = ref - mean
    covariance = np.einsum('nbi,nbj->bij', centered, centered, optimize=True) / max(1, len(rows) - 1)
    values, axes = np.linalg.eigh(covariance)
    # Fix the arbitrary eigenvector sign for independent transform rebuilding.
    for block in range(len(axes)):
        for column in range(group):
            j = np.argmax(np.abs(axes[block, :, column]))
            if axes[block, j, column] < 0:
                axes[block, :, column] *= -1
    return dict(order=order, rows=rows, width=width, group=group, mean=mean,
                axes=axes, eigenvalues=values)


def rotate(x, state):
    a = np.zeros((len(x), state['width']), dtype=np.float64)
    a[:, :len(state['order'])] = x[:, state['order']]
    blocks = a.reshape(len(x), -1, state['group']) - state['mean']
    return np.einsum('nbi,bij->nbj', blocks, state['axes'], optimize=True).reshape(len(x), -1).astype(np.float32)


def estimator(metric, trees, seed):
    cls = metric in ('AUROC', 'AUPRC')
    return (ExtraTreesClassifier if cls else ExtraTreesRegressor)(
        n_estimators=trees, max_depth=20, min_samples_leaf=3, max_features=.5,
        bootstrap=False, random_state=seed, n_jobs=4,
        criterion='gini' if cls else 'squared_error')


def member_prediction(net, x, metric):
    # Summation in one fixed tree order makes checkpoint replay bit-identical.
    net.n_jobs = 1
    return net.predict_proba(x)[:, 1] if metric in ('AUROC', 'AUPRC') else net.predict(x)


def output(metric, raw, state, task):
    p = raw * state['target_scale'] + state['target_center']
    return p.clip(0, 100) if task == 'ppbr_az' else p


def replay(d, k, part, c):
    from v161_verify_and_assemble import same
    a = design(d, k, part)
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    same(a['state'], joblib.load(folder / 'transform.joblib'))
    members, hashes = [], {'transform.joblib': b.sha(folder / 'transform.joblib')}
    for seed in SEEDS:
        path = folder / ('model_' + str(seed) + '.joblib')
        bundle = joblib.load(path)
        predictions = []
        for index, item in enumerate(bundle):
            expected = rotation(a['x'], seed, index, c['group'])
            same(expected, item['rotation'])
            predictions.append(member_prediction(item['model'], rotate(a['qx'], expected), d['metrics'][k]))
        members.append(np.mean(predictions, axis=0))
        hashes[path.name] = b.sha(path)
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
        bundle, predictions = [], []
        for index in range(c['rotations']):
            state = rotation(a['x'], seed, index, c['group'])
            xx, qq = rotate(a['x'], state), rotate(a['qx'], state)
            net = estimator(d['metrics'][k], c['trees'], seed + index * 1009)
            net.fit(xx, a['state']['target'])
            predictions.append(member_prediction(net, qq, d['metrics'][k]))
            bundle.append(dict(rotation=state, model=net))
        path = folder / ('model_' + str(seed) + '.joblib')
        joblib.dump(bundle, path, compress=3)
        loaded = joblib.load(path)
        original = np.mean(predictions, axis=0)
        checked = np.mean([member_prediction(z['model'], rotate(a['qx'], z['rotation']), d['metrics'][k])
                           for z in loaded], axis=0)
        np.testing.assert_array_equal(original, checked)
        members.append(checked)
    p = output(d['metrics'][k], np.mean(members, axis=0), a['state'], d['tasks'][k])
    assert np.isfinite(p).all()
    np.savez_compressed(file, signature=sig(), fit=a['state']['fit'], query=a['state']['query'], p=p)
    print(json.dumps(dict(stage='train', task=d['tasks'][k], part=part,
                          seconds=round(time.monotonic() - start), width=a['x'].shape[1])), flush=True)
    return p


def smoke(d):
    from tempfile import TemporaryDirectory
    from v161_verify_and_assemble import same
    checked = []
    for task in ('herg', 'solubility_aqsoldb', 'half_life_obach'):
        k = d['tasks'].index(task)
        a = design(d, k, 'fold0')
        labels = d['y'][a['state']['query']].copy()
        try:
            d['y'][a['state']['query']] = 999999.
            d.pop('_v96_assays', None)
            same(a, design(d, k, 'fold0'))
        finally:
            d['y'][a['state']['query']] = labels
            d.pop('_v96_assays', None)
        state = rotation(a['x'], SEEDS[0], 0, 8)
        np.testing.assert_allclose(np.einsum('bji,bjk->bik', state['axes'], state['axes']),
                                   np.broadcast_to(np.eye(8), state['axes'].shape), atol=1e-12, rtol=0)
        raw = np.zeros((8, state['width']))
        raw[:, :len(state['order'])] = a['qx'][:8, state['order']]
        raw = raw.reshape(8, -1, 8) - state['mean']
        result = rotate(a['qx'][:8], state)
        np.testing.assert_allclose(np.linalg.norm(raw.reshape(8, -1), axis=1),
                                   np.linalg.norm(result, axis=1), atol=1e-4, rtol=1e-6)
        xx, qq = rotate(a['x'][:96], state), rotate(a['qx'][:8], state)
        net = estimator(d['metrics'][k], 5, SEEDS[0])
        net.fit(xx, a['state']['target'][:len(xx)])
        p = member_prediction(net, qq, d['metrics'][k])
        np.testing.assert_allclose(p, member_prediction(net, qq[::-1], d['metrics'][k])[::-1], atol=1e-14, rtol=0)
        with TemporaryDirectory(dir=OUT) as temp:
            path = Path(temp) / 'smoke.joblib'
            joblib.dump(dict(rotation=state, model=net), path)
            z = joblib.load(path)
            same(rotation(a['x'], SEEDS[0], 0, 8), z['rotation'])
            np.testing.assert_allclose(p, member_prediction(z['model'], rotate(a['qx'][:8], z['rotation']), d['metrics'][k]), atol=1e-14, rtol=0)
        checked.append(task)
    b.save(OUT / 'smoke.json', dict(status='PASS', signature=sig(), query_label_isolation=True,
           fit_only_PCA=True, orthogonality_reference=True, distance_preservation=True,
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
        b.save(OUT / 'failure.json', dict(status='FAILED', traceback=traceback.format_exc()))
        raise
