"""Metric-aware upgrade with strict two-metric gates on valid and inner scaffold holdout.

Both sets participate in selection and cannot establish hidden-test guarantees.
Original baseline artifacts are never overwritten. No external labels are used.
"""
from __future__ import annotations
import argparse
import json
import shutil
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier, XGBRegressor
from admet_features_v2 import features, DescriptorPreprocessor, SCHEMA
from admet_competition_train import fit_candidates, predict_model


def metrics(kind, y, p):
    if not np.isfinite(p).all():
        raise ValueError('Non-finite predictions')
    if kind == 'classification':
        return dict(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)))
    rho = float(spearmanr(y, p).statistic)
    return dict(mae=float(mean_absolute_error(y, p)), spearman=rho if np.isfinite(rho) else 0.)


def dominates(kind, old, new):
    if kind == 'classification':
        return new['auroc'] >= old['auroc'] and new['auprc'] >= old['auprc']
    return new['mae'] <= old['mae'] and new['spearman'] >= old['spearman']


def utility(metric, mm):
    return -mm['mae'] if metric == 'MAE' else mm[metric.lower()]


def tanimoto_neighbour(a, y, b, k=24):
    a = a[:, :2048].astype(np.float32)
    out, counts = [], a.sum(1)[:, None]
    for start in range(0, len(b), 256):
        chunk = b[start:start+256, :2048].astype(np.float32)
        inter = a @ chunk.T
        sim = inter / np.maximum(counts + chunk.sum(1)[None, :] - inter, 1.)
        kk = min(k, len(a))
        ix = np.argpartition(sim, len(a)-kk, axis=0)[-kk:]
        w = np.take_along_axis(sim, ix, axis=0)**3 + 1e-9
        out.extend((np.sum(w*y[ix], axis=0)/w.sum(0)).tolist())
    return np.asarray(out)


def model_factory(kind, name):
    if name == 'xgb':
        cls = XGBClassifier if kind == 'classification' else XGBRegressor
        return cls(n_estimators=550, max_depth=4, learning_rate=.035,
                   min_child_weight=2, subsample=.9, colsample_bytree=.65,
                   reg_alpha=.05, reg_lambda=4, tree_method='hist', n_jobs=6, random_state=42)
    if name == 'extra':
        cls = ExtraTreesClassifier if kind == 'classification' else ExtraTreesRegressor
        kw = {'class_weight': 'balanced'} if kind == 'classification' else {}
        return cls(n_estimators=350, max_features=.28, min_samples_leaf=1,
                   random_state=42, n_jobs=6, **kw)
    if name == 'linear':
        return (LogisticRegression(C=.15, solver='liblinear', max_iter=2000,
                                   class_weight='balanced', random_state=42)
                if kind == 'classification' else Ridge(alpha=50))
    raise KeyError(name)


def train_heads(x, y, kind, metric, targets):
    pre = DescriptorPreprocessor().fit(x)
    xx = pre.transform(x)
    xt = {k: pre.transform(v) for k, v in targets.items()}
    use_log = kind == 'regression' and metric == 'Spearman' and y.min() >= 0
    yy = np.log1p(y) if use_log else y
    predictions, models = {k: {} for k in targets}, {}
    for name in ['xgb', 'extra', 'linear']:
        model = model_factory(kind, name)
        model.fit(xx[:, 4096:] if name == 'linear' else xx, yy)
        models[name] = model
        for k, arr in xt.items():
            p = predict_model(model, arr[:, 4096:] if name == 'linear' else arr, kind)
            predictions[k][name] = np.expm1(np.clip(p, -30, 30)) if use_log else p
    models['knn'] = {'x': xx[:, :2048], 'y': yy}
    for k, arr in xt.items():
        p = tanimoto_neighbour(xx, yy, arr)
        predictions[k]['knn'] = np.expm1(np.clip(p, -30, 30)) if use_log else p
    return predictions, {'preprocess': pre, 'models': models, 'log_target': use_log}


def get_metric(task):
    d = pd.read_csv('admet_official/submission_template.csv')
    values = d.loc[d.task == task, 'metric'].unique()
    assert len(values) == 1
    return values[0]


def choose(kind, metric, y, base, pred, ys, bases, preds):
    old, olds = metrics(kind, y, base), metrics(kind, ys, bases)
    best = ('baseline', 0., base, bases, old, olds)
    best_u = utility(metric, old)
    pred['equal'] = np.mean([pred[n] for n in ['xgb', 'extra', 'linear', 'knn']], axis=0)
    preds['equal'] = np.mean([preds[n] for n in ['xgb', 'extra', 'linear', 'knn']], axis=0)
    for name in pred:
        for alpha in [.25, .5, .75, 1.]:
            p = (1-alpha)*base + alpha*pred[name]
            ps = (1-alpha)*bases + alpha*preds[name]
            mm, ms = metrics(kind, y, p), metrics(kind, ys, ps)
            u = utility(metric, mm)
            if dominates(kind, old, mm) and dominates(kind, olds, ms) and u > best_u + 1e-8:
                best, best_u = (name, alpha, p, ps, mm, ms), u
    return best


def run_task(task, root, baseline, out):
    start = time.time()
    frames = {s: pd.read_csv(root/task/(s+'.csv')) for s in ['train', 'valid', 'test']}
    n, nv = len(frames['train']), len(frames['valid'])
    y, yv = frames['train'].Y.to_numpy(float), frames['valid'].Y.to_numpy(float)
    kind = 'classification' if set(np.unique(y)).issubset({0, 1}) else 'regression'
    metric = get_metric(task)
    all_smi = pd.concat([frames[s].Drug for s in frames]).tolist()
    x, canon, groups = features(all_smi, out/'feature_cache_v3')
    tr, va, te = x[:n], x[n:n+nv], x[n+nv:]
    it, ih = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=20260924).split(tr, y, groups[:n]))
    assert not set(np.asarray(canon[:n])[it]) & set(np.asarray(canon[:n])[ih])
    assert not set(np.asarray(groups[:n])[it]) & set(np.asarray(groups[:n])[ih])
    folder = out/task
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'row': range(n), 'partition': np.where(np.isin(np.arange(n), ih), 'holdout', 'fit'),
                  'canonical': canon[:n], 'scaffold': groups[:n]}).to_csv(folder/'internal_split.csv', index=False)
    with np.load(baseline/'feature_cache'/(task+'.npz'), allow_pickle=True) as z:
        xb = z['x'][:n]
    oldbundle = joblib.load(baseline/task/'model_bundle.joblib')
    old_valid = pd.read_csv(baseline/task/'validation_predictions.csv')
    assert np.allclose(old_valid.Y_true.to_numpy(), yv)
    pv, bpred = old_valid.Y_pred.to_numpy(float), np.zeros(len(ih))
    for name, w in oldbundle['weights'].items():
        if w < 1e-10:
            continue
        model = fit_candidates(kind, y[it])[name]
        if hasattr(model, 'n_jobs'):
            model.set_params(n_jobs=6)
        model.fit(xb[it], y[it])
        bpred += w*predict_model(model, xb[ih], kind)
    stress, _ = train_heads(tr[it], y[it], kind, metric, {'stress': tr[ih]})
    vp, _ = train_heads(tr, y, kind, metric, {'valid': va})
    name, alpha, selected, selected_s, mm, ms = choose(
        kind, metric, yv, pv, vp['valid'], y[ih], bpred, stress['stress'])
    accepted = name != 'baseline'
    pd.DataFrame({'Y_true': y[ih], 'baseline': bpred, 'selected': selected_s,
                  **stress['stress']}).to_csv(folder/'scaffold_predictions.csv', index=False)
    pd.DataFrame({'Y_true': yv, 'baseline': pv, 'selected': selected,
                  **vp['valid']}).to_csv(folder/'candidate_predictions.csv', index=False)
    if accepted:
        testpred, bundle = train_heads(np.vstack([tr, va]), np.concatenate([y, yv]),
                                      kind, metric, {'test': te})
        tp = testpred['test']
        tp['equal'] = np.mean(list(tp.values()), axis=0)
        p0 = pd.read_csv(baseline/task/f'submission_{task}.csv').Y.to_numpy(float)
        ptest = (1-alpha)*p0 + alpha*tp[name]
        bundle.update(kind=kind, task=task, schema=SCHEMA, chosen=name, alpha=alpha,
                      baseline=oldbundle)
        joblib.dump(bundle, folder/'model_bundle.joblib', compress=3)
        pd.DataFrame({'Drug_ID': frames['test'].Drug_ID, 'Y': ptest}).to_csv(folder/f'submission_{task}.csv', index=False)
        pd.DataFrame({'Drug_ID': frames['valid'].Drug_ID, 'Y_true': yv, 'Y_pred': selected}).to_csv(folder/'validation_predictions.csv', index=False)
    else:
        for file in ['model_bundle.joblib', f'submission_{task}.csv', 'validation_predictions.csv']:
            shutil.copy2(baseline/task/file, folder/file)
    row = dict(task=task, kind=kind, official_metric=metric, accepted=accepted,
               selected=name, alpha=alpha, elapsed_s=round(time.time()-start, 1))
    for prefix, m in [('baseline', metrics(kind, yv, pv)), ('selected', mm),
                      ('scaffold_baseline', metrics(kind, y[ih], bpred)), ('scaffold_selected', ms)]:
        row.update({prefix+'_'+k: v for k, v in m.items()})
    row['selection_note'] = 'valid and scaffold holdout used for selection; no untouched final test'
    (folder/'selection.json').write_text(json.dumps(row, indent=2), encoding='utf8')
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='*')
    ap.add_argument('--output-dir', type=Path, default=Path('admet_model_outputs_v3'))
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    root, baseline, out = Path('admet_competition_data/data_benchmark'), Path('admet_model_outputs'), a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    names = a.tasks or sorted(p.name for p in root.iterdir() if p.is_dir())
    with threadpool_limits(limits=6):
        for i, name in enumerate(names):
            if a.resume and (out/name/'selection.json').exists():
                continue
            print(f'[{i+1}/{len(names)}] {name}', flush=True)
            row = run_task(name, root, baseline, out)
            print(json.dumps(row), flush=True)
            rows = [json.loads(p.read_text()) for p in out.glob('*/selection.json')]
            pd.DataFrame(rows).to_csv(out/'upgrade_metrics.csv', index=False)


if __name__ == '__main__':
    main()
