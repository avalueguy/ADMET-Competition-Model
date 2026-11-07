"""Compare direct ranking and regression on identical official scaffold splits.

Only train/valid Y are read. Test structures come from the upload template.
Existing chemistry is loaded as a versioned artifact, without impersonating its
RDKit version or regenerating features. Historical predictions remain proxies.
"""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostRanker, CatBoostRegressor, Pool
from scipy.special import ndtri
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'admet_model_outputs_v77'
TASKS = ('clearance_hepatocyte_az', 'clearance_microsome_az',
         'half_life_obach', 'vdss_lombardo')
SEEDS = (772026, 772027)
METHODS = ('rmse', 'pairlogit', 'yetirank')
VIEWS = ('physchem', 'multiview')
MAX_TREES = 1000
CHECKPOINTS = (100, 200, 400, 650, 1000)
ANCHOR_SHA = 'beab85b972b336526a30c8baa6d181552927f0d004ac6360f42ecdbd96399763'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def emit(**obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def signature():
    return sha(__file__)


def table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def load_data():
    template = ROOT / 'admet_official/submission_template.csv'
    original_protocol = json.loads((ROOT / 'admet_v67_protocol.json').read_text())
    expected_sha = original_protocol['official_template_sha256']
    assert sha(template) == expected_sha
    official = table(template)
    tasks = list(dict.fromkeys(r['task'] for r in official))
    metrics = [next(r['metric'] for r in official if r['task'] == t) for t in tasks]
    sources = {str(template.relative_to(ROOT)): sha(template)}
    smiles, lookup, mi, tid, split, labels = [], {}, [], [], [], []
    manifest = {r['path']: r for r in json.loads((ROOT / original_protocol['data_manifest']).read_text())}
    for k, task in enumerate(tasks):
        for si, part in enumerate(('train', 'valid', 'test')):
            if si == 2:
                rows = [{'Drug': r['smiles']} for r in official if r['task'] == task]
            else:
                path = ROOT / 'admet_competition_data/data_benchmark' / task / (part + '.csv')
                assert sha(path) == manifest[path.relative_to(ROOT).as_posix()]['sha256']
                sources[path.relative_to(ROOT).as_posix()] = sha(path)
                rows = table(path)
            for row in rows:
                sm = row['Drug']
                if sm not in lookup:
                    lookup[sm] = len(smiles)
                    smiles.append(sm)
                mi.append(lookup[sm]); tid.append(k); split.append(si)
                labels.append(float(row['Y']) if si < 2 else np.nan)
    chemistry = ROOT / 'admet_model_outputs_v67/features/official_chemistry.npz'
    meta = json.loads(chemistry.with_suffix('.json').read_text())
    assert meta['smiles_sha256'] == hashlib.sha256(json.dumps(smiles).encode()).hexdigest()
    with np.load(chemistry, allow_pickle=False) as z:
        x, canonical, groups = z['x'], z['canonical'], z['groups']
    with np.load(ROOT / 'admet_model_outputs_v73/features/views.npz') as z:
        desc = np.concatenate([x[:, 4096:], z['parent'], z['priors']], axis=1)
        multi = np.concatenate([x[:, :4096], z['extra'], desc], axis=1)
    mi, tid, split, y = map(np.asarray, (mi, tid, split, labels))
    assert not np.isfinite(y[split == 2]).any()
    for name in ('admet_model_outputs_v67/folds.npy',
                 'recovered_history/baseline_cv_proxy.npy',
                 'recovered_history/baseline_valid.npy',
                 'admet_model_outputs_v67/features/official_chemistry.npz',
                 'admet_model_outputs_v73/features/views.npz'):
        sources[name] = sha(ROOT / name)
    return dict(tasks=tasks, metrics=metrics, mi=mi, tid=tid, split=split, y=y,
                groups=groups[mi], canonical=canonical[mi], template=official,
                views={'physchem': desc.astype(np.float32), 'multiview': multi.astype(np.float32)},
                folds=np.load(ROOT / 'admet_model_outputs_v67/folds.npy'),
                proxy_cv=np.load(ROOT / 'recovered_history/baseline_cv_proxy.npy'),
                proxy_valid=np.load(ROOT / 'recovered_history/baseline_valid.npy'),
                sources=sources, chemistry_producer=meta)


def audit(d):
    protocol = dict(version=77, signature=signature(), tasks=TASKS, methods=METHODS,
        views=VIEWS, seeds=SEEDS, max_trees=MAX_TREES, checkpoints=CHECKPOINTS,
        anchor='v73', anchor_sha256=ANCHOR_SHA, anchor_score_user_reported=50.640,
        selection_parts=['fold0', 'fold1'], confirmation_parts=['fold2', 'valid'],
        stopping='Inner scaffold split of each outer fit, seed 771009+task index. Outer labels never stop models.',
        selection='Choose one task/method/view on folds0/1. Require positive gain vs same-row RMSE and historical proxy in both selection folds. Freeze before confirmation.',
        confirmation='Fixed equal-seed ensemble must beat same-row regression and proxy in fold2 and valid; paired scaffold bootstrap on fixed predictions, multiplicity limits disclosed.',
        submission='At most one task; replace with 50 percent direct-ranking expert, 50 percent original v73 rank. No conditional assays or old expert reapplication.',
        full_v73_oof_available=False, labels_previously_reused=True,
        test_csv_opened=False, test_labels_used=False, external_data=False, pretrained_weights=False,
        limitations='Historical arrays are v57 proxies, not full v73 OOF. Repartitioned development labels are not new independent data. Gains cannot be translated to platform points.')
    save(OUT / 'protocol.json', protocol)
    save(OUT / 'data_audit.json', dict(signature=signature(), input_hashes=d['sources'],
        chemistry_producer=d['chemistry_producer'], rdkit_cache_regenerated=False,
        test_csv_opened=False, test_labels_used=False, external_data=False))


def rows(d, k, part):
    active = d['tid'] == k
    if part == 'final':
        fit = np.flatnonzero(active & (d['split'] < 2))
        query = np.flatnonzero(active & (d['split'] == 2))
    elif part == 'valid':
        fit = np.flatnonzero(active & (d['split'] == 0))
        query = np.flatnonzero(active & (d['split'] == 1))
    else:
        f = int(part[-1])
        fit = np.flatnonzero(active & (d['split'] == 0) & (d['folds'] != f))
        query = np.flatnonzero(active & (d['split'] == 0) & (d['folds'] == f))
    assert not set(fit) & set(query) and np.all(d['split'][fit] < 2)
    if part != 'final':
        assert not set(d['groups'][fit]) & set(d['groups'][query])
        assert not set(d['canonical'][fit]) & set(d['canonical'][query])
    return fit, query


def inner(d, fit, k):
    groups = np.unique(d['groups'][fit])
    held = np.random.default_rng(771009 + k).permutation(groups)[:max(1, round(len(groups) * .2))]
    mask = np.isin(d['groups'][fit], held)
    tr, va = fit[~mask], fit[mask]
    assert len(tr) > 10 and len(va) > 10
    assert not set(d['groups'][tr]) & set(d['groups'][va])
    return tr, va


def score(y, p):
    v = float(spearmanr(y, p).statistic)
    return v if np.isfinite(v) else 0.


def rank(p):
    return rankdata(p) / len(p)


def fit_model(d, k, fit, q, method, view, seed, folder, trees=None):
    folder.mkdir(parents=True, exist_ok=True)
    cache = folder / 'predictions.npz'
    if cache.exists():
        with np.load(cache) as z:
            assert str(z['signature']) == signature()
            np.testing.assert_array_equal(z['fit'], fit)
            np.testing.assert_array_equal(z['query'], q)
            return z['p'].copy()
    started = time.time()
    assert np.all(d['split'][fit] < 2) and not set(fit) & set(q)
    x = d['views'][view][d['mi'][fit]]
    qx = d['views'][view][d['mi'][q]]
    y = d['y'][fit]
    if method == 'rmse':
        yt = ndtri(np.clip((rankdata(y) - .5) / len(y), 1e-5, 1 - 1e-5))
    else:
        yt = (rankdata(y) - .5) / len(y)
    cfg = dict(iterations=MAX_TREES if trees is None else trees, depth=5,
        learning_rate=.035, l2_leaf_reg=15, task_type='GPU', devices='0',
        random_seed=seed, thread_count=4, verbose=False, allow_writing_files=False,
        border_count=128, bootstrap_type='Bernoulli', subsample=.85,
        nan_mode='Min', boosting_type='Plain')
    pairs = None
    if method == 'pairlogit':
        rng = np.random.default_rng(seed)
        a, b = rng.integers(0, len(y), (2, len(y) * 40))
        ok = y[a] != y[b]
        a, b = a[ok], b[ok]
        pairs = np.stack([np.where(y[a] > y[b], a, b), np.where(y[a] > y[b], b, a)], axis=1)
        cfg.update(loss_function='PairLogit', eval_metric='PairLogit')
    elif method == 'yetirank':
        cfg.update(loss_function='YetiRank', eval_metric='QueryRMSE')
    else:
        cfg.update(loss_function='RMSE', eval_metric='RMSE')
    cls = CatBoostRegressor if method == 'rmse' else CatBoostRanker
    net = cls(**cfg)
    if method == 'rmse':
        _, inv, count = np.unique(d['canonical'][fit], return_inverse=True, return_counts=True)
        weight = 1 / count[inv]; weight /= weight.mean()
        pool = Pool(x, yt, weight=weight)
    else:
        pool = Pool(x, yt, group_id=np.zeros(len(fit), dtype=int), pairs=pairs)
    net.fit(pool)
    chosen = trees
    curve = []
    if trees is None:
        for checkpoint in CHECKPOINTS:
            p = net.predict(qx, ntree_end=checkpoint)
            curve.append(dict(trees=checkpoint, spearman=score(d['y'][q], p)))
        chosen = max(curve, key=lambda r: r['spearman'])['trees']
        net.shrink(chosen)
    p = net.predict(qx)
    assert np.isfinite(p).all()
    path = folder / 'model.cbm'
    net.save_model(str(path))
    replay = cls()
    replay.load_model(str(path))
    np.testing.assert_allclose(replay.predict(qx), p, atol=1e-12, rtol=1e-12)
    np.savez_compressed(cache, signature=signature(), p=p, fit=fit, query=q)
    save(folder / 'metadata.json', dict(signature=signature(), task=d['tasks'][k],
        method=method, view=view, seed=seed, config=cfg, actual_trees=int(chosen),
        inner_curve=curve, fit_rows=len(fit), query_rows=len(q), pairs=0 if pairs is None else len(pairs),
        model_sha256=sha(path), predictions_sha256=sha(cache), independent_reload_passed=True,
        seconds=time.time() - started, target_transform='normal ranks' if method == 'rmse' else 'uniform ranks',
        score=score(d['y'][q], p) if np.all(d['split'][q] < 2) else None))
    emit(task=d['tasks'][k], method=method, view=view, seed=seed,
         folder=str(folder.relative_to(OUT)), trees=int(chosen), seconds=round(time.time() - started))
    return p


def outer(d, k, part, method, view):
    fit, q = rows(d, k, part)
    folder = OUT / d['tasks'][k] / part / view / method
    if part == 'final':
        counts = [json.loads((OUT / d['tasks'][k] / p / view / method / 'inner/metadata.json').read_text())['actual_trees']
                  for p in ('fold0', 'fold1', 'fold2', 'valid')]
        trees = int(np.median(counts))
    else:
        tr, va = inner(d, fit, k)
        fit_model(d, k, tr, va, method, view, SEEDS[0], folder / 'inner')
        trees = json.loads((folder / 'inner/metadata.json').read_text())['actual_trees']
    preds = [fit_model(d, k, fit, q, method, view, seed, folder / str(seed), trees) for seed in SEEDS]
    # An identical rank aggregation is used in development and final inference.
    p = np.mean([rank(pp) for pp in preds], axis=0)
    np.savez_compressed(folder / 'ensemble.npz', signature=signature(), p=p, query=q, trees=trees)
    return p


def proxy(d, q, part):
    p = d['proxy_valid' if part == 'valid' else 'proxy_cv'][q]
    assert np.isfinite(p).all()
    return p


def develop(d):
    audit(d)
    for task in TASKS:
        k = d['tasks'].index(task)
        for part in ('fold0', 'fold1'):
            for view in VIEWS:
                for method in METHODS:
                    p = outer(d, k, part, method, view)
                    q = rows(d, k, part)[1]
                    emit(stage='development_score', task=task, part=part, method=method,
                         view=view, score=score(d['y'][q], p), proxy=score(d['y'][q], proxy(d, q, part)))
    return select(d)


def select(d):
    records = []
    for task in TASKS:
        k = d['tasks'].index(task)
        for view in VIEWS:
            for method in METHODS[1:]:
                parts = []
                for part in ('fold0', 'fold1'):
                    q = rows(d, k, part)[1]
                    path = OUT / task / part / view
                    p = np.load(path / method / 'ensemble.npz')['p']
                    c = np.load(path / 'rmse/ensemble.npz')['p']
                    base = proxy(d, q, part)
                    ys = d['y'][q]
                    mixed = .5 * rank(base) + .5 * rank(p)
                    parts.append(dict(part=part, n=len(q), score=score(ys, p), control=score(ys, c),
                        proxy=score(ys, base), gain_control=score(ys, p) - score(ys, c),
                        gain_proxy=score(ys, mixed) - score(ys, base)))
                eligible = all(r['gain_control'] > 0 and r['gain_proxy'] > 0 for r in parts)
                objective = min(r['gain_proxy'] for r in parts)
                records.append(dict(task=task, method=method, view=view, parts=parts,
                                    eligible=eligible, objective=objective))
    eligible = [r for r in records if r['eligible'] and r['objective'] >= .005]
    selected = max(eligible, key=lambda r: r['objective']) if eligible else None
    result = dict(signature=signature(), stage='FROZEN_BEFORE_CONFIRMATION', candidates=records,
                  selected=selected, confirmation_inspected=False, full_v73_oof_available=False)
    frozen = OUT / 'frozen_selection.json'
    if frozen.exists():
        assert json.loads(frozen.read_text()) == result
    else:
        save(frozen, result)
    emit(stage='selection', selected=selected)
    return result


def bootstrap(y, base, new, groups, seed):
    blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(2000):
        ix = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        draws.append(score(y[ix], new[ix]) - score(y[ix], base[ix]))
    return dict(percentile_95=np.quantile(draws, [.025, .975]).tolist(),
                probability_gain_positive=float(np.mean(np.asarray(draws) > 0)), draws=len(draws))


def confirm(d):
    frozen = json.loads((OUT / 'frozen_selection.json').read_text())
    assert frozen['signature'] == signature()
    selected = frozen['selected']
    if selected is None:
        save(OUT / 'confirmation.json', dict(status='NO_SELECTION_CANDIDATE', accepted=False))
        return False
    task, method, view = (selected[key] for key in ('task', 'method', 'view'))
    k = d['tasks'].index(task)
    parts = []
    for part in ('fold2', 'valid'):
        c = outer(d, k, part, 'rmse', view)
        p = outer(d, k, part, method, view)
        q = rows(d, k, part)[1]
        base = proxy(d, q, part)
        mixed = .5 * rank(base) + .5 * rank(p)
        y = d['y'][q]
        parts.append(dict(part=part, n=len(q), score=score(y, p), control=score(y, c),
            proxy=score(y, base), gain_control=score(y, p) - score(y, c),
            gain_proxy=score(y, mixed) - score(y, base),
            bootstrap_proxy=bootstrap(y, base, mixed, d['groups'][q], 772100 + k),
            bootstrap_control=bootstrap(y, c, p, d['groups'][q], 772200 + k)))
    accepted = all(r['gain_control'] > 0 and r['gain_proxy'] > .005 for r in parts)
    # A point estimate alone is insufficient: neither paired interval may
    # strongly favor regression or the historical proxy.
    accepted = accepted and all(r['bootstrap_proxy']['probability_gain_positive'] >= .9 for r in parts)
    result = dict(signature=signature(), status='CONFIRMATION_COMPLETE', selected=selected,
        frozen_sha256=sha(OUT / 'frozen_selection.json'), parts=parts, accepted=accepted,
        full_v73_oof_available=False, limitations='Paired bootstrap conditional on selected fixed models, on historically reused data; no unbiased significance or official-score claim.')
    save(OUT / 'confirmation.json', result)
    emit(stage='confirmation', **result)
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('develop', 'select', 'confirm', 'run'))
    args = parser.parse_args()
    d = load_data()
    if args.stage in ('develop', 'run'):
        develop(d)
    if args.stage == 'select':
        select(d)
    if args.stage in ('confirm', 'run'):
        confirm(d)
    save(OUT / 'pipeline_status.json', dict(status='COMPLETE', stage=args.stage, signature=signature(),
         pid=os.getpid(), official_score=None))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save(OUT / 'pipeline_status.json', dict(status='FAILED', error=repr(exc), signature=signature()))
        raise
