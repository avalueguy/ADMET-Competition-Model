"""Official-only, scaffold-cross-fitted property teachers and median residuals.

New teachers use no target-task labels.  This is a bounded experiment, not an
estimate of a leaderboard gain.  All model fitting is restricted to Linux CUDA.
"""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '4')
import hashlib
import json
import time
import joblib
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
PREV = ROOT.parent / 'regression9_v387'
OOF = ROOT.parent / 'regression9_v386'
TASKS = ('ppbr_az', 'solubility_aqsoldb', 'lipophilicity_astrazeneca',
         'caco2_wang', 'ld50_zhu')
PARTS = ('fold0', 'fold1', 'fold2', 'valid')
BASES = (dict(name='residual_aux_d2', depth=2, auxiliary=True),
         dict(name='residual_aux_d4', depth=4, auxiliary=True),
         dict(name='residual_structure_d4', depth=4, auxiliary=False))
HORIZONS = (512, 1536, 3072)
SEEDS = (3901106, 3901107)
WEIGHTS = (.1, .25, .5, .75, 1.)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def emit(stage, **kwargs):
    print(json.dumps(dict(stage=stage, **kwargs)), flush=True)


def load():
    torch.set_num_threads(4)
    assert os.name == 'posix' and torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    d = dict(np.load(PREV / 'official_rows.npz'))
    d.update(json.loads((PREV / 'data_manifest.json').read_text()))
    d.update(dict(np.load(PREV / 'expert_features.npz')))
    d['rg'] = d['groups'][d['mi']]
    return d


def control(task, part):
    c = dict(np.load(PREV / 'controls' / task / (part + '.npz')))
    if task == 'lipophilicity_astrazeneca':
        extra = np.load(PREV / 'outputs/tasks' / task / part /
                        'gaussian_full_C3_r2/predictions.npz')
        for k in ('fit', 'query'):
            np.testing.assert_array_equal(c[k], extra[k])
        c['p'] = .9 * c['p'] + .1 * extra['p']
    return c


def scale_fit(x, q):
    med = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
    med = np.nan_to_num(med)
    clean = np.where(np.isfinite(x), x, med)
    spread = np.subtract(*np.quantile(clean, [.75, .25], axis=0)).clip(.1)
    def apply(z):
        return np.clip((np.where(np.isfinite(z), z, med)-med)/spread, -8, 8)
    return apply(x), apply(q), dict(median=med, scale=spread)


def teacher_features(d, source, query):
    """Fit-only column selection and scaling; no labels enter this transform."""
    sf, sq = d['mi'][source], d['mi'][query]
    train_ids = np.unique(sf)
    raw = d['fingerprints']
    columns = []
    for j in range(6):
        start = j * 2048
        var = raw[train_ids, start:start+2048].var(0)
        columns.extend((start + np.argsort(-var, kind='stable')[:48]).tolist())
    columns = np.asarray(columns)
    x = np.c_[raw[sf][:, columns], d['descriptor'][sf], d['physical'][sf]]
    qx = np.c_[raw[sq][:, columns], d['descriptor'][sq], d['physical'][sq]]
    x, qx, scaling = scale_fit(x, qx)
    # An intercept is unpenalized below. Each block is in comparable units.
    return (np.c_[np.ones(len(x)), x].astype(np.float32),
            np.c_[np.ones(len(qx)), qx].astype(np.float32), columns, scaling)


def teacher_panel(d, task, part, tag, query):
    path = ROOT / 'teachers' / task / part / (tag + '.joblib')
    if path.exists():
        s = joblib.load(path)
        np.testing.assert_array_equal(query, s['query'])
        return s['panel']
    target = d['tasks'].index(task)
    # Exclude every queried scaffold across all auxiliary endpoints as well.
    blocked = np.unique(d['rg'][query])
    source = np.flatnonzero((d['tid'] != target) &
        (d['split'] < (2 if part == 'final' else 1)) &
        ~np.isin(d['rg'], blocked))
    assert np.isfinite(d['y'][source]).all()
    assert not np.intersect1d(d['rg'][source], blocked).size
    x, qx, columns, scaling = teacher_features(d, source, query)
    xt = torch.as_tensor(x, device='cuda')
    qt = torch.as_tensor(qx, device='cuda')
    predictions, states = [], []
    for tid in range(22):
        if tid == target:
            continue
        mask = d['tid'][source] == tid
        rows = source[mask]
        y = d['y'][rows]
        if not len(y):
            predictions.append(np.zeros(len(query)))
            states.append(dict(task=tid, source=rows, empty=True))
            continue
        if d['metrics'][tid] in ('MAE', 'Spearman'):
            center = float(np.median(y))
            spread = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05)
            yy = np.arcsinh((y-center)/spread)
        else:
            center, spread = .5, .5
            yy = (y-center)/spread
        xx = xt[torch.as_tensor(mask, device='cuda')]
        yt = torch.as_tensor(yy.astype(np.float32), device='cuda')
        gram = xx.T @ xx
        ridge = torch.eye(xx.shape[1], device='cuda') * 20.
        ridge[0, 0] = .0001
        coef = torch.linalg.solve(gram + ridge, xx.T @ yt)
        predictions.append((qt @ coef).cpu().numpy())
        states.append(dict(task=tid, source=rows, coefficient=coef.cpu().numpy(),
                           center=center, scale=spread, empty=False))
    panel = np.column_stack(predictions)
    assert panel.shape == (len(query), 21) and np.isfinite(panel).all()
    s = dict(query=query, source=source, panel=panel, columns=columns,
             scaling=scaling, models=states)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(s, path, compress=3)
    emit('teacher_done', task=task, part=part, tag=tag, source=len(source))
    return panel


def design(d, task, part):
    c = control(task, part)
    fit, query = c['fit'], c['query']
    a = dict(np.load(PREV / 'design' / task / (part + '.npz')))
    for k in ('fit', 'query'):
        np.testing.assert_array_equal(a[k], c[k])
    assert np.isfinite(d['y'][fit]).all()
    assert not np.intersect1d(fit, query).size
    if part != 'final':
        assert not np.intersect1d(d['rg'][fit], d['rg'][query]).size
    # Genuine scaffold OOF anchor predictions from the earlier, frozen expert.
    anchor = joblib.load(OOF / 'outputs/oof' / task / part / 'state.joblib')
    for k in ('fit', 'query'):
        np.testing.assert_array_equal(anchor[k], c[k])
    panel = np.full((len(fit), 21), np.nan)
    split_records = []
    for j, (tr, ho) in enumerate(GroupKFold(3).split(fit, groups=d['rg'][fit])):
        panel[ho] = teacher_panel(d, task, part, 'inner'+str(j), fit[ho])
        split_records.append(dict(train=fit[tr], held=fit[ho]))
    qpanel = teacher_panel(d, task, part, 'full', query)
    raw = d['fingerprints']
    var = raw[d['mi'][fit]].var(0)
    columns = np.sort(np.argsort(-var, kind='stable')[:512])
    def features(rows, context, base):
        mi = d['mi'][rows]
        return np.c_[d['descriptor'][mi], d['physical'][mi], context[:, :105],
                     raw[mi][:, columns], base].astype(np.float32)
    x = features(fit, a['x'], anchor['oof'])
    qx = features(query, a['qx'], anchor['full'])
    med = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
    med = np.nan_to_num(med)
    x = np.where(np.isfinite(x), x, med)
    qx = np.where(np.isfinite(qx), qx, med)
    assert np.isfinite(panel).all()
    return dict(fit=fit, query=query, x=x, qx=qx, panel=panel, qpanel=qpanel,
                anchor=anchor['oof'][:, :2].mean(1),
                qanchor=anchor['full'][:, :2].mean(1),
                columns=columns, missing_median=med, splits=split_records,
                baseline=c['p'])


def clip(task, p):
    return np.clip(p, 0, 100) if task == 'ppbr_az' else p


def curve(d, task, part, b, a):
    folder = ROOT / 'curves' / task / part / b['name']
    path = folder / 'state.joblib'
    if path.exists():
        return joblib.load(path)
    started = time.monotonic()
    y = d['y'][a['fit']]
    spread = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05)
    residual = (y - a['anchor'])/spread
    x = np.c_[a['x'], a['panel']] if b['auxiliary'] else a['x']
    qx = np.c_[a['qx'], a['qpanel']] if b['auxiliary'] else a['qx']
    models, preds = [], {h: [] for h in HORIZONS}
    for seed in SEEDS:
        net = XGBRegressor(objective='reg:quantileerror', quantile_alpha=.5,
            n_estimators=max(HORIZONS), max_depth=b['depth'], learning_rate=.025,
            min_child_weight=6, reg_lambda=30., reg_alpha=.1,
            subsample=.85, colsample_bytree=.8, max_bin=64,
            tree_method='hist', device='cuda', n_jobs=4, random_state=seed)
        net.fit(x, residual, verbose=False)
        models.append(net)
        for h in HORIZONS:
            p = a['qanchor'] + spread * net.predict(qx, iteration_range=(0, h))
            preds[h].append(clip(task, p))
        emit('seed_done', task=task, part=part, config=b['name'], seed=seed)
    preds = {h: np.mean(v, axis=0) for h, v in preds.items()}
    state = dict(fit=a['fit'], query=a['query'], models=models, config=b,
                 scale=spread, predictions=preds)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, path, compress=3)
    save(folder/'metadata.json', dict(model_sha256=sha(path),
         seconds=time.monotonic()-started, horizons=HORIZONS, seeds=SEEDS))
    return state


def evaluate(d, a, pred, weight):
    y = d['y'][a['query']]
    candidate = (1-weight)*a['baseline'] + weight*pred
    base = float(abs(y-a['baseline']).mean())
    mae = float(abs(y-candidate).mean())
    return dict(baseline=base, candidate=mae, gain=(base-mae)/base)


def bootstrap(d, a, pred, weight):
    y = d['y'][a['query']]
    err = abs(y-a['baseline'])
    changed = abs(y-((1-weight)*a['baseline']+weight*pred))
    groups = d['rg'][a['query']]
    chunks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(3901106)
    gains = []
    for _ in range(1000):
        rows = np.concatenate([chunks[j] for j in rng.integers(len(chunks), size=len(chunks))])
        gains.append(float((err[rows].mean()-changed[rows].mean())/err[rows].mean()))
    return dict(ci90=np.quantile(gains, [.05, .95]).tolist(), groups=len(chunks))


def replay(d, task, part, b, horizon, a):
    s = joblib.load(ROOT/'curves'/task/part/b['name']/'state.joblib')
    for k in ('fit', 'query'):
        np.testing.assert_array_equal(s[k], a[k])
    qx = np.c_[a['qx'], a['qpanel']] if b['auxiliary'] else a['qx']
    return np.mean([clip(task, a['qanchor']+s['scale']*m.predict(qx, iteration_range=(0, horizon)))
                    for m in s['models']], axis=0)


def main():
    started = time.time()
    ROOT.mkdir(parents=True, exist_ok=True)
    protocol = dict(version='v390', baseline='v387', tasks=TASKS, bases=BASES,
        horizons=HORIZONS, seeds=SEEDS, weights=WEIGHTS,
        hypothesis='Scaffold OOF prediction of 21 other official properties supplies a dense auxiliary panel; native median residual trees learn task-specific corrections.',
        teachers='Fit-only top48 columns of each six fingerprint views plus RDKit descriptors; ridge20, intercept unpenalized; asinh/IQR continuous targets. Source excludes target task and every queried scaffold across all auxiliary tasks.',
        main='Mean of two frozen native median anchors; true inner scaffold OOF anchor and teacher features for fitting; two GPU quantile tree seeds; auxiliary-free depth4 ablation.',
        full_development_budget=60, trees_per_fit=3072,
        selection='Complete fold0/1 curves first; unique max-min configuration/horizon/blend; freeze before fold2/valid; no fallback.',
        acceptance='All four relative gains positive, development minimum >=1%, four-part mean>=1%, both scaffold bootstrap90 lower bounds positive. Otherwise no replacement CSV.',
        external_data=False, test_labels=False, all_fitting_remote=True,
        limitations='Historical baseline OOF remains partly proxy; public partitions reused; no claimed leaderboard score or new blind validation.',
        source_sha256=sha(__file__))
    if (ROOT/'protocol.json').exists():
        assert json.loads((ROOT/'protocol.json').read_text()) == json.loads(json.dumps(protocol))
    else:
        save(ROOT/'protocol.json', protocol)
    d = load()
    development, selections = [], []
    for task in TASKS:
        save(ROOT/'status.json', dict(stage='DEVELOPMENT', task=task, pid=os.getpid()))
        grids = {}
        for part in PARTS[:2]:
            a = design(d, task, part)
            for b in BASES:
                s = curve(d, task, part, b, a)
                for h in HORIZONS:
                    for weight in WEIGHTS:
                        key = (b['name'], h, weight)
                        grids.setdefault(key, []).append(evaluate(d, a, s['predictions'][h], weight))
        rows = [dict(task=task, base=k[0], horizon=k[1], weight=k[2],
                     parts=v, objective=min(r['gain'] for r in v)) for k, v in grids.items()]
        best = max(rows, key=lambda v: (v['objective'], -v['weight'], -v['horizon'], v['base']))
        development.append(dict(task=task, choices=rows, best=best))
        selections.append(best)
        save(ROOT/'development.json', dict(records=development))
        emit('task_development_done', **best)
    save(ROOT/'frozen_selection.json', dict(selected=selections, no_fallback=True))
    results = []
    for selected in selections:
        task = selected['task']
        b = next(b for b in BASES if b['name'] == selected['base'])
        h, weight = selected['horizon'], selected['weight']
        record = dict(**selected, confirmation=[], bootstrap=[], accepted=False)
        # Do not spend confirmation fitting when fixed development criterion fails.
        if selected['objective'] >= .01:
            for part in PARTS[2:]:
                save(ROOT/'status.json', dict(stage='CONFIRMATION', task=task, part=part, pid=os.getpid()))
                a = design(d, task, part)
                s = curve(d, task, part, b, a)
                record['confirmation'].append(dict(part=part, **evaluate(d, a, s['predictions'][h], weight)))
                record['bootstrap'].append(dict(part=part, **bootstrap(d, a, s['predictions'][h], weight)))
                np.testing.assert_array_equal(s['predictions'][h], replay(d, task, part, b, h, a))
            gains = [v['gain'] for v in selected['parts'] + record['confirmation']]
            record['accepted'] = bool(min(gains)>0 and np.mean(gains)>=.01 and
                all(v['ci90'][0]>0 for v in record['bootstrap']))
        if record['accepted']:
            a = design(d, task, 'final')
            assert np.isnan(d['y'][a['query']]).all()
            s = curve(d, task, 'final', b, a)
            pred = replay(d, task, 'final', b, h, a)
            np.testing.assert_array_equal(pred, s['predictions'][h])
            np.savez_compressed(ROOT/(task+'_final.npz'), fit=a['fit'], query=a['query'],
                p=(1-weight)*a['baseline']+weight*pred)
        results.append(record)
        save(ROOT/'confirmation.json', dict(records=results))
        emit('task_result', **record)
    save(ROOT/'status.json', dict(stage='COMPLETE', seconds=time.time()-started,
        accepted=[r['task'] for r in results if r['accepted']], pid=os.getpid()))
    emit('complete', seconds=time.time()-started,
         accepted=[r['task'] for r in results if r['accepted']])


if __name__ == '__main__':
    main()
