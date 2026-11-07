"""Conformer-resolved solvent exposure experts for three v145 MAE endpoints.

New geometry is separate from earlier global shape descriptors: exposed polar
and charged surface, distant donor/acceptor contacts and conformer variability.
No historical predictor is a training input. Original v148 gates are unchanged.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '4'
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostRegressor
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v146_current_baseline as v
import v148_experiment_runner as runner
import v255_surface_feasibility as geometry
b, r83 = v.b, v.r83
OUT = ROOT / 'admet_model_outputs_v256'
TASKS = ('caco2_wang', 'ppbr_az', 'lipophilicity_astrazeneca')
SEEDS = (2561029, 2561030)
CONFIGS = (dict(trees=700, depth=4),)
FEATURE_NAMES = tuple(prefix + '_' + name for prefix in ('mean', 'sd', 'minimum', 'maximum')
                      for name in geometry.NAMES) + ('valid', 'conformers', 'convergence_fraction',
                                                    'energy_range', 'heavy', 'MMFF94', 'UFF')
METHOD = ('Conformer-resolved solvent exposure: original full graph, canonical structure-derived '
          'ETKDG seed,6conformers/MMFF94-or-UFF150steps. Atom solvent-accessible surface with '
          'RDKit FreeSASA and van der Waals radii; polar/donor/acceptor/charged exposed areas, '
          'long-range donor/acceptor distance contacts and centroid-relative charge moment. '
          'Energy-weighted mean/sd and min/max, explicit missingness for unsupported structures. '
          'Small plain CatBoost MAE expert on new exposure plus original physicochemistry and '
          'target-excluded measured context; no old predictions or global-shape input. '
          'Only Caco2/PPBR/lipophilicity,19protected tasks. One fixed config,2seeds, '
          'unchanged original v148 development/blends/confirmation.')


def sig():
    return hashlib.sha256((b.sha(__file__) + b.sha(geometry.__file__) + v.sig() +
                           b.sha(runner.__file__) + rdBase.rdkitVersion +
                           importlib.metadata.version('catboost')).encode()).hexdigest()


def key(c):
    return 'conformer_exposed_polarity'


def same(a, z):
    if isinstance(a, dict):
        assert set(a) == set(z)
        for name in a:
            same(a[name], z[name])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(z)
        for x, y in zip(a, z):
            same(x, y)
    else:
        np.testing.assert_array_equal(a, z)


def structures(d):
    result, cursor = {}, 0
    for task in d['tasks']:
        for part in ('train', 'valid', 'test'):
            rows = ([dict(Drug=x['smiles']) for x in d['template'] if x['task'] == task]
                    if part == 'test' else b.table(ROOT / 'admet_competition_data/data_benchmark' / task / (part + '.csv')))
            for row in rows:
                if task in TASKS:
                    mi = int(d['mi'][cursor])
                    assert result.get(mi, row['Drug']) == row['Drug']
                    result[mi] = row['Drug']
                cursor += 1
    assert cursor == len(d['y'])
    return result


def feature_vector(record):
    if record['status'] != 'PASS':
        return np.r_[np.full(4 * len(geometry.NAMES), np.nan),
                     0., 0., 0., np.nan, record.get('heavy', np.nan), 0., 0.].astype(np.float32)
    x = np.r_[*[record[name] for name in ('mean', 'sd', 'minimum', 'maximum')],
              1., record['conformers'], record['convergence_fraction'], record['energy_range'],
              record['heavy'], record['forcefield'] == 'MMFF94', record['forcefield'] == 'UFF'].astype(np.float32)
    assert len(x) == len(FEATURE_NAMES) and np.isfinite(x).all()
    return x


def load():
    d = v.load()
    d['surface_smiles'] = structures(d)
    source = hashlib.sha256(json.dumps(sorted(d['surface_smiles'].items()), separators=(',', ':')).encode()).hexdigest()
    folder = OUT / 'features'
    audit = json.loads((folder / 'audit.json').read_text())
    assert audit['status'] == 'PASS' and audit['signature'] == sig() and audit['smiles_sha256'] == source
    assert b.sha(folder / 'surface.npz') == audit['features_sha256']
    assert b.sha(folder / 'records.joblib') == audit['records_sha256']
    z = np.load(folder / 'surface.npz')
    np.testing.assert_array_equal(z['mi'], sorted(d['surface_smiles']))
    assert str(z['signature']) == sig() and z['x'].shape == (len(z['mi']), len(FEATURE_NAMES))
    d['surface_features'] = dict(zip(z['mi'].tolist(), z['x']))
    d['surface_records'] = joblib.load(folder / 'records.joblib')
    assert set(d['surface_records']) == set(d['surface_smiles'])
    return d


def design(d, k, part, c=None):
    fit, query, augmented, qaugmented, source = v.old.features(d, k, part, 'augmented')
    width = d['views']['physchem'].shape[1] + d['conformer_features'].shape[1] + d['pharmacophore'].shape[1]
    # Keep measured context and original 2D physical descriptors, then new surface.
    def assemble(rows, x):
        return np.c_[x[:, :-width], d['views']['physchem'][d['mi'][rows]],
                     np.stack([d['surface_features'][int(mi)] for mi in d['mi'][rows]])].astype(np.float32)
    x, qx = assemble(fit, augmented), assemble(query, qaugmented)
    x[~np.isfinite(x)] = np.nan
    qx[~np.isfinite(qx)] = np.nan
    assert d['metrics'][k] == 'MAE' and d['tasks'][k] in TASKS
    y = d['y'][fit]
    center = float(np.median(y))
    scale = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05)
    assert not set(fit) & set(query) and np.all(d['tid'][source] != k)
    assert np.all(d['split'][source] < (2 if part == 'final' else 1))
    if part != 'final':
        assert not set(d['groups'][fit]) & set(d['groups'][query])
    return dict(x=x, qx=qx, state=dict(fit=fit, query=query, source=source,
                target=(y - center) / scale, center=center, scale=scale))


def estimator(c, seed):
    return CatBoostRegressor(loss_function='MAE', boosting_type='Plain',
        iterations=c['trees'], depth=c['depth'], learning_rate=.03,
        l2_leaf_reg=30., random_seed=seed, thread_count=4, verbose=False,
        allow_writing_files=False, bootstrap_type='Bayesian', bagging_temperature=1.,
        random_strength=1., border_count=64, task_type='CPU')


def output(raw, state, task):
    p = np.asarray(raw) * state['scale'] + state['center']
    return p.clip(0., 100.) if task == 'ppbr_az' else p


def replay(d, k, part, c):
    a = design(d, k, part, c)
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    same(a['state'], joblib.load(folder / 'transform.joblib'))
    hashes = {'transform.joblib': b.sha(folder / 'transform.joblib')}
    members = []
    for seed in SEEDS:
        file = folder / ('model_' + str(seed) + '.cbm')
        net = CatBoostRegressor().load_model(str(file))
        members.append(net.predict(a['qx'], prediction_type='RawFormulaVal', thread_count=1))
        hashes[file.name] = b.sha(file)
    return output(np.mean(members, axis=0), a['state'], d['tasks'][k]), hashes


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
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(a['state'], folder / 'transform.joblib', compress=3)
    members = []
    for seed in SEEDS:
        net = estimator(c, seed)
        net.fit(a['x'], a['state']['target'])
        p = net.predict(a['qx'], prediction_type='RawFormulaVal', thread_count=1)
        path = folder / ('model_' + str(seed) + '.cbm')
        net.save_model(str(path))
        replay_net = CatBoostRegressor().load_model(str(path))
        np.testing.assert_array_equal(p, replay_net.predict(a['qx'], prediction_type='RawFormulaVal', thread_count=1))
        members.append(p)
    p = output(np.mean(members, axis=0), a['state'], d['tasks'][k])
    assert np.isfinite(p).all()
    np.savez_compressed(file, signature=sig(), fit=a['state']['fit'], query=a['state']['query'], p=p)
    print(json.dumps(dict(stage='train', task=d['tasks'][k], part=part,
                          seconds=round(time.monotonic() - started))), flush=True)
    return p


def smoke(d):
    from tempfile import TemporaryDirectory
    checked = []
    for task in TASKS:
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
        net = estimator(dict(trees=16, depth=3), SEEDS[0])
        net.fit(a['x'][:96], a['state']['target'][:96])
        p = net.predict(a['qx'][:8], thread_count=1)
        np.testing.assert_array_equal(p, net.predict(a['qx'][:8][::-1], thread_count=1)[::-1])
        with TemporaryDirectory(dir=OUT) as temp:
            path = Path(temp) / 'smoke.cbm'
            net.save_model(str(path))
            np.testing.assert_array_equal(p, CatBoostRegressor().load_model(str(path)).predict(a['qx'][:8], thread_count=1))
        checked.append(task)
    feature_audit = json.loads((OUT / 'features/audit.json').read_text())
    # Exact deterministic feature reconstruction on synthetic molecules and actual task rows.
    same(feature_vector(geometry.exposure('CC(O)N')), feature_vector(geometry.exposure('NC(O)C')))
    for task in TASKS:
        k = d['tasks'].index(task)
        mi = int(d['mi'][v.split(d, k, 'fold0')[0][0]])
        rebuilt = geometry.exposure(d['surface_smiles'][mi])
        same(feature_vector(rebuilt), d['surface_features'][mi])
    b.save(OUT / 'smoke.json', dict(status='PASS', signature=sig(), query_label_isolation=True,
        independent_reload_tasks=checked, prediction_row_order=True,
        surface_feature_reconstruction=True, structure_notation_invariant=True,
        feature_audit=feature_audit, no_query_scores_used=True))
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
