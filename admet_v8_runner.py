"""Fixed-pool v8 training with matched v7 baselines and per-task safety gates."""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

from admet_v6_runner import remote_only, partition, metrics, kind, deltas, secondary_ok, save, emit, sha, get_metric
from admet_v7_runner import baseline as baseline6, repeat_baseline as repeat6, subgroup
from admet_v8_models import load_task, fit, predict

OUT = Path('admet_model_outputs_v8')
PFILE = Path('admet_v8_protocol.json')


def protocol():
    return json.loads(PFILE.read_text())


def route7(task):
    return json.loads((Path('admet_release_v7') / task / 'selection.json').read_text())


def baseline(task, d):
    dest = OUT / task; dest.mkdir(parents=True, exist_ok=True)
    path = dest / 'baseline_oof.csv'
    r = route7(task)
    b = baseline6(task, d).copy()
    if r['accepted']:
        s = r['selection']
        c = pd.read_csv(Path('admet_model_outputs_v7') / task / (s['family'] + '_oof.csv'))
        np.testing.assert_allclose(c.Y_true, b.Y_true)
        np.testing.assert_array_equal(c.fold, b.fold)
        b['Y_pred'] = (1-s['alpha'])*b.Y_pred + s['alpha']*c.Y_pred
    if path.exists():
        old = pd.read_csv(path)
        np.testing.assert_allclose(old.Y_pred, b.Y_pred, rtol=1e-12, atol=1e-12)
        return old
    b.to_csv(path, index=False)
    save(dest / 'baseline_provenance.json', dict(version=7,
         route_sha256=sha(Path('admet_release_v7') / task / 'selection.json'),
         reconstructed_v7_accepted_candidate=bool(r['accepted'])))
    return b


def repeat_baseline(task, d, it, iv):
    r = route7(task)
    y = d['frames']['train'].Y.to_numpy(float)
    if r['accepted']:
        b = pd.read_csv(Path('admet_model_outputs_v7') / task / 'repeat_comparison.csv')
        np.testing.assert_array_equal(b.row, iv)
        np.testing.assert_allclose(b.Y_true, y[iv])
        return b.blend.to_numpy()
    return repeat6(task, d, it, iv)


def expected(task, p):
    return p['families_all'] + (['gated_transfer'] if task in p['mtl_tasks'] else [])


def train(task, family):
    p = protocol(); assert family in expected(task, p)
    dest = OUT/task; dest.mkdir(parents=True, exist_ok=True)
    file = dest/(family+'_oof.csv')
    if file.exists():
        assert json.loads((dest/(family+'_cv.json')).read_text())['protocol_sha256'] == sha(PFILE)
        return
    d = load_task(task); y = d['frames']['train'].Y.to_numpy(float)
    fold = partition(task, d); prediction = np.full(len(y), np.nan)
    rows = []; start = time.time()
    for f in range(3):
        it = np.flatnonzero(fold != f); iv = np.flatnonzero(fold == f)
        assert not set(d['groups'][it]) & set(d['groups'][iv])
        assert not set(d['canonical'][it]) & set(d['canonical'][iv])
        cache = dest/(family+f'_fold{f}.npz')
        if cache.exists():
            with np.load(cache) as z:
                assert str(z['protocol_sha256']) == sha(PFILE)
                np.testing.assert_array_equal(z['rows'], iv)
                prediction[iv] = z['p']
        else:
            model = fit(family, d, it, y[it], get_metric(task), p)
            prediction[iv] = predict(model, d, iv)
            if family == 'gated_transfer':
                save(dest/(family+f'_fold{f}_source_audit.json'), model['source_audit'])
            np.savez_compressed(cache, rows=iv, p=prediction[iv], protocol_sha256=sha(PFILE))
            del model
        m = metrics(kind(get_metric(task)), y[iv], prediction[iv])
        rows.append(dict(fold=f, **m))
        emit(stage='v8_train', task=task, family=family, fold=f, elapsed_s=round(time.time()-start), **m)
    assert np.isfinite(prediction).all()
    pd.DataFrame(dict(row=np.arange(len(y)), fold=fold, Y_true=y, Y_pred=prediction)).to_csv(file, index=False)
    save(dest/(family+'_cv.json'), dict(folds=rows, protocol_sha256=sha(PFILE), feature_key=d['v8_feature_key']))


def select(task, d):
    p = protocol(); dest = OUT/task; locked = dest/'selection.json'
    if locked.exists():
        result = json.loads(locked.read_text()); assert result['protocol_sha256'] == sha(PFILE)
        return result
    b = baseline(task, d); y = b.Y_true.to_numpy(); base = b.Y_pred.to_numpy(); fold = b.fold.to_numpy()
    metric = get_metric(task); kk = kind(metric); pool = {}
    for name in expected(task, p):
        df = pd.read_csv(dest/(name+'_oof.csv'))
        np.testing.assert_allclose(df.Y_true, y); np.testing.assert_array_equal(df.fold, fold)
        assert df.row.tolist() == list(range(len(y)))
        assert json.loads((dest/(name+'_cv.json')).read_text())['protocol_sha256'] == sha(PFILE)
        pool[name] = df.Y_pred.to_numpy()
    grid = []; best = None
    for family, pred in pool.items():
        for alpha in p['alphas']:
            q = (1-alpha)*base + alpha*pred; checks = []; gains = []
            for f in range(3):
                mask = fold == f
                before = metrics(kk, y[mask], base[mask]); after = metrics(kk, y[mask], q[mask])
                gain = deltas(metric, before, after); gains.append(gain)
                checks.append(dict(fold=f, before=before, after=after,
                    passed=bool(gain >= (-.01 if metric == 'MAE' else -.005) and secondary_ok(metric, before, after))))
            objective = float(np.mean(gains) - .5*np.std(gains))
            passed = all(c['passed'] for c in checks) and np.mean(gains) > 0 and objective > 0
            row = dict(task=task, family=family, alpha=alpha, objective=objective, mean_gain=float(np.mean(gains)),
                       worst_gain=float(min(gains)), folds=checks, passed=bool(passed))
            grid.append(row)
            if passed and (best is None or objective > best['objective']):
                best = row
    save(dest/'selection_grid.json', grid)
    result = dict(best or dict(task=task, family='baseline', alpha=0, passed=False))
    result.update(metric=metric, kind=kk, pool=expected(task, p), protocol_sha256=sha(PFILE),
                  selection_source='historically reused scaffold OOF; one lock before official valid')
    save(locked, result); emit(stage='v8_lock', task=task, family=result['family'], alpha=result['alpha'])
    return result


def evaluate(task):
    p = protocol(); dest = OUT/task; path = dest/'decision.json'
    if path.exists():
        return
    d = load_task(task); s = select(task, d)
    if s['family'] == 'baseline':
        save(path, dict(task=task, accepted=False, reason='no robust OOF candidate', selection=s)); return
    n = len(d['frames']['train']); nv = len(d['frames']['valid'])
    y = d['frames']['train'].Y.to_numpy(float); metric = s['metric']; kk = s['kind']; alpha = s['alpha']
    baseo = baseline(task, d); cand = pd.read_csv(dest/(s['family']+'_oof.csv')).Y_pred.to_numpy()
    domains = []
    for f in range(3):
        it = np.flatnonzero(baseo.fold != f); iv = np.flatnonzero(baseo.fold == f)
        b = baseo.Y_pred.to_numpy()[iv]; q = (1-alpha)*b + alpha*cand[iv]
        domains.append(dict(split='fold_'+str(f), **subgroup(d, it, iv, y[iv], b, q, metric)))
    if not all(c['passed'] for c in domains):
        save(path, dict(task=task, accepted=False, selection=s, domains=domains, reason='distant OOF subgroup regression')); return
    it = np.arange(n); iv = np.arange(n, n+nv); yv = d['frames']['valid'].Y.to_numpy(float)
    model = fit(s['family'], d, it, y, metric, p); candidate = predict(model, d, iv)
    joblib.dump(model, dest/'validation_model.joblib', compress=3)
    if s['family'] == 'gated_transfer':
        save(dest/'validation_source_audit.json', model['source_audit'])
    del model
    old = pd.read_csv(Path('admet_release_v7')/task/'validation_predictions.csv')
    np.testing.assert_allclose(old.Y_true, yv); b = old.Y_pred.to_numpy(); q = (1-alpha)*b+alpha*candidate
    before = metrics(kk, yv, b); after = metrics(kk, yv, q)
    domains.append(dict(split='valid', **subgroup(d, it, iv, yv, b, q, metric)))
    pd.DataFrame(dict(Y_true=yv, baseline=b, candidate=candidate, blend=q)).to_csv(dest/'validation_comparison.csv', index=False)
    decision = dict(task=task, accepted=False, before=before, after=after, selection=s, domains=domains)
    valid = deltas(metric, before, after) >= 0 and secondary_ok(metric, before, after) and domains[-1]['passed']
    decision.update(valid_pass=bool(valid), reason='valid gate failed')
    emit(stage='v8_valid', task=task, passed=bool(valid), before=before, after=after)
    if not valid:
        save(path, decision); return
    it, iv = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=692509).split(np.arange(n), y, d['groups'][:n]))
    assert not set(d['groups'][it]) & set(d['groups'][iv])
    if kk == 'classification' and (len(np.unique(y[it])) < 2 or len(np.unique(y[iv])) < 2):
        decision['reason'] = 'repeat lacks class coverage'; save(path, decision); return
    b = repeat_baseline(task, d, it, iv)
    model = fit(s['family'], d, it, y[it], metric, p); c = predict(model, d, iv)
    if s['family'] == 'gated_transfer':
        save(dest/'repeat_source_audit.json', model['source_audit'])
    del model
    q = (1-alpha)*b + alpha*c
    before = metrics(kk, y[iv], b); after = metrics(kk, y[iv], q)
    domains.append(dict(split='repeat', **subgroup(d, it, iv, y[iv], b, q, metric)))
    passed = deltas(metric, before, after) >= (-.01 if metric == 'MAE' else -.005) and secondary_ok(metric, before, after) and domains[-1]['passed']
    pd.DataFrame(dict(row=iv, Y_true=y[iv], baseline=b, candidate=c, blend=q)).to_csv(dest/'repeat_comparison.csv', index=False)
    decision.update(accepted=bool(passed), repeat=dict(before=before, after=after, passed=bool(passed), seed=692509),
                    reason='passed all observed gates' if passed else 'repeat gate failed')
    save(path, decision); emit(stage='v8_decision', task=task, accepted=bool(passed), repeat=decision['repeat'])


def main():
    remote_only()
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['train', 'evaluate'])
    parser.add_argument('--family'); parser.add_argument('--tasks', nargs='*'); args = parser.parse_args()
    names = args.tasks or sorted(p.name for p in Path('admet_competition_data/data_benchmark').iterdir() if p.is_dir())
    with threadpool_limits(limits=4):
        for task in names:
            train(task, args.family) if args.stage == 'train' else evaluate(task)


if __name__ == '__main__':
    main()
