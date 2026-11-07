"""Repair stereochemical assay pooling in the actual v145 v127 components.

Keep the original estimator, seeds, training rows, physical features and blend.
Only other-assay identity keys retain stereochemistry. No target label or mask
is an input, and historical proxy predictions are never fitting offsets.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize as standard
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import admet_v67_data as chemistry
import v146_current_baseline as v
import v148_experiment_runner as runner
import v127_low_classification as original
import v210_standardized_chemistry as strings

b, r83 = v.b, v.r83
OUT = ROOT / 'admet_model_outputs_v284'
TASKS = ('herg', 'cyp3a4_substrate_carbonmangels')
CONFIGS = (dict(identity='isomeric_exact_and_neutral'),)
SEEDS = original.SEEDS
METHOD = ('Direct input-identity repair of the actual v145 v127/physchem components. '
    'Other-assay exact and neutral-parent identities now retain stereochemistry; '
    'separate medians, masks and neutral ranges, unchanged column order. Original '
    '900depth4 CatBoost Logloss, seeds1274026/1274027, lr.025,l2=30, Bernoulli.85, '
    'border64,GPU, physical features and .5 component rank blend preserved. '
    'Fit-only audit supports hERG/CYP3A4 substrate; CYP2D6 excluded by declared '
    'coverage rule. No new architecture, sweep, proxy fitting offsets, failed '
    'expert predictions, or altered labels/splits. Other20endpoints protected; '
    '19runner identity controls/HIA excluded. Unchanged v148 gates and weights.')


def sig():
    packages = '|'.join(importlib.metadata.version(x) for x in ('rdkit', 'catboost', 'numpy'))
    return hashlib.sha256((b.sha(__file__) + v.sig() + original.sig() +
        b.sha(chemistry.__file__) + b.sha(strings.__file__) + b.sha(runner.__file__) + packages).encode()).hexdigest()


def key(c):
    return 'v145_stereo_specific_context'


def same(a, z):
    assert set(a) == set(z)
    for name in a:
        if isinstance(a[name], dict):
            assert a[name] == z[name]
        else:
            np.testing.assert_array_equal(a[name], z[name])


def identity(sm):
    mol, mode = chemistry.parse_molecule(sm)
    exact = Chem.MolToSmiles(mol, isomericSmiles=True)
    if mode == 'full_sanitization':
        parent = standard.LargestFragmentChooser().choose(mol)
        parent = standard.Uncharger().uncharge(parent)
    else:
        parent = mol
    return exact, Chem.MolToSmiles(parent, isomericSmiles=True)


def identities(d, independent=False):
    if '_v284_keys' in d and not independent:
        return d['_v284_keys']
    RDLogger.DisableLog('rdApp.*')
    smiles, sha = strings.originals(d)
    file = OUT / 'features/identities.npz'
    report = file.with_suffix('.json')
    if independent or not file.exists():
        keys = np.array([identity(s) for s in smiles])
        if file.exists():
            np.testing.assert_array_equal(keys, np.load(file)['keys'])
        else:
            file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(file, keys=keys, signature=sig(), smiles_sha256=sha)
            b.save(report, dict(status='PASS', signature=sig(), smiles_sha256=sha,
                               molecules=len(keys), sha256=b.sha(file), labels_used=False))
    else:
        meta = json.loads(report.read_text())
        assert meta['signature'] == sig() and meta['smiles_sha256'] == sha
        assert b.sha(file) == meta['sha256']
        z = np.load(file)
        assert str(z['signature']) == sig() and str(z['smiles_sha256']) == sha
        keys = z['keys']
    d['_v284_keys'] = keys
    return keys


def load():
    threadpool_limits(limits=4)
    return v.load()


def context(d, k, part):
    keys = identities(d)[d['mi']]
    cols = np.array([j for j in range(len(d['tasks'])) if j != k])
    source = np.flatnonzero((d['tid'] != k) & (d['split'] < (2 if part == 'final' else 1)))
    assert k not in cols and np.all(d['tid'][source] != k)
    lookup = [{}, {}]
    for row in source:
        for j in (0, 1):
            lookup[j].setdefault((str(keys[row, j]), int(d['tid'][row])), []).append(float(d['y'][row]))
    n = len(cols)
    result = np.full((len(d['y']), 5*n), np.nan, dtype=np.float32)
    for row in np.flatnonzero(d['tid'] == k):
        for j, t in enumerate(cols):
            exact = lookup[0].get((str(keys[row, 0]), int(t)))
            neutral = lookup[1].get((str(keys[row, 1]), int(t)))
            if exact:
                result[row, j] = np.median(exact)
            if neutral:
                result[row, 2*n+j] = np.median(neutral)
                result[row, 4*n+j] = max(neutral)-min(neutral)
    result[:, n:2*n] = np.isfinite(result[:, :n])
    result[:, 3*n:4*n] = np.isfinite(result[:, 2*n:3*n])
    return result, cols, source


def design(d, k, part, c=None):
    assert d['tasks'][k] in TASKS
    fit, query, x, qx, source = original.v.features(d, k, part, 'physchem')
    new, cols, new_source = context(d, k, part)
    np.testing.assert_array_equal(source, new_source)
    # original.v.features starts with its 2*n non-stereo context columns;
    # retain the complete physical descriptor block after replacing only that
    # context prefix.
    width = 2*len(cols)
    return dict(fit=fit, query=query, source=source, cols=cols,
        x=np.c_[new[fit], x[:, width:]].astype(np.float32),
        qx=np.c_[new[query], qx[:, width:]].astype(np.float32))


def original_replay(d, k, part):
    fit, q, x, qx, source = original.v.features(d, k, part, 'physchem')
    hashes, members = {}, []
    for seed in SEEDS:
        path = original.OUT / 'models' / d['tasks'][k] / part / 'physchem' / ('model_'+str(seed)+'.cbm')
        assert path.exists(), 'Original checkpoint missing; never silently train it'
        net = CatBoostClassifier().load_model(str(path))
        assert net.tree_count_ == 900
        members.append(net.predict(qx, prediction_type='RawFormulaVal', thread_count=1))
        hashes[path.relative_to(ROOT).as_posix()] = b.sha(path)
    p = expit(np.mean(members, axis=0))
    z = np.load(path.parent / 'predictions.npz')
    assert str(z['signature']) == original.sig()
    np.testing.assert_array_equal(z['fit'], fit)
    np.testing.assert_array_equal(z['query'], q)
    np.testing.assert_allclose(p, z['p'], atol=1e-12, rtol=0)
    carried = original.v.mix('AUROC', original.v.baseline(d, k, part), p, .5)
    np.testing.assert_allclose(carried, v.baseline(d, k, part), atol=1e-12, rtol=0)
    return hashes


def estimator(seed, smoke=False):
    return CatBoostClassifier(loss_function='Logloss', iterations=5 if smoke else 900,
        depth=4, learning_rate=.025, l2_leaf_reg=30, random_seed=seed,
        thread_count=3, verbose=False, allow_writing_files=False,
        bootstrap_type='Bernoulli', subsample=.85, border_count=64,
        task_type='GPU', devices='0')


def replay(d, k, part, c):
    a = design(d, k, part, c)
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    z = np.load(folder / 'predictions.npz')
    for name in a:
        np.testing.assert_array_equal(a[name], z[name])
    members, hashes = [], {}
    for seed in SEEDS:
        path = folder / ('model_'+str(seed)+'.cbm')
        net = CatBoostClassifier().load_model(str(path))
        assert net.tree_count_ == 900
        members.append(net.predict(a['qx'], prediction_type='RawFormulaVal', thread_count=1))
        hashes[path.name] = b.sha(path)
    p = original.v.mix('AUROC', original.v.baseline(d, k, part), expit(np.mean(members, axis=0)), .5)
    np.testing.assert_array_equal(p, z['p'])
    return p, hashes


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
    assert not folder.exists(), 'Inspect incomplete model; no silent retry'
    started = time.monotonic()
    a = design(d, k, part, c)
    hashes = original_replay(d, k, part)
    folder.mkdir(parents=True)
    members = []
    for seed in SEEDS:
        net = estimator(seed)
        net.fit(a['x'], d['y'][a['fit']])
        path = folder / ('model_'+str(seed)+'.cbm')
        net.save_model(str(path))
        raw = net.predict(a['qx'], prediction_type='RawFormulaVal', thread_count=1)
        reload = CatBoostClassifier().load_model(str(path))
        np.testing.assert_array_equal(raw, reload.predict(a['qx'], prediction_type='RawFormulaVal', thread_count=1))
        members.append(raw)
    component = expit(np.mean(members, axis=0))
    p = original.v.mix('AUROC', original.v.baseline(d, k, part), component, .5)
    assert np.isfinite(p).all()
    np.savez_compressed(file, signature=sig(), p=p, component=component, **a)
    b.save(folder / 'metadata.json', dict(signature=sig(), original_model_hashes=hashes,
        original_seeds=SEEDS, fixed_original_parameters=True, original_component_weight=.5,
        target_query_labels_used=False, seconds=time.monotonic()-started))
    print(json.dumps(dict(stage='stereo_context_fit', task=d['tasks'][k], part=part,
                         seconds=round(time.monotonic()-started))), flush=True)
    return p


def smoke(d):
    from tempfile import TemporaryDirectory
    # Known enantiomers must not pool; alternate equivalent atom order must.
    assert identity('N[C@@H](C)C(=O)O')[0] != identity('N[C@H](C)C(=O)O')[0]
    mol, _ = chemistry.parse_molecule('N[C@@H](C)C(=O)O')
    alternate = Chem.MolToSmiles(Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms())))), canonical=False)
    assert identity(alternate) == identity('N[C@@H](C)C(=O)O')
    identities(d)
    checks = []
    for task in TASKS:
        k = d['tasks'].index(task)
        a = design(d, k, 'fold0')
        original_replay(d, k, 'fold0')
        saved = d['y'].copy()
        try:
            mask = (d['tid'] == k) & ~np.isin(np.arange(len(d['y'])), a['fit'])
            d['y'][mask] = 999999.
            d.pop('_v96_assays', None)
            same(a, design(d, k, 'fold0'))
        finally:
            d['y'][:] = saved
            d.pop('_v96_assays', None)
        net = estimator(SEEDS[0], smoke=True)
        net.fit(a['x'], d['y'][a['fit']])
        raw = net.predict(a['qx'], prediction_type='RawFormulaVal', thread_count=1)
        with TemporaryDirectory(dir=OUT) as temp:
            path = Path(temp) / 'model.cbm'
            net.save_model(str(path))
            other = CatBoostClassifier().load_model(str(path))
            np.testing.assert_array_equal(raw, other.predict(a['qx'], prediction_type='RawFormulaVal', thread_count=1))
            np.testing.assert_array_equal(raw, other.predict(a['qx'][::-1], prediction_type='RawFormulaVal', thread_count=1)[::-1])
        checks.append(dict(task=task, actual_cuda_training=True, save_reload=True, row_order=True))
    b.save(OUT / 'smoke.json', dict(status='PASS', signature=sig(), scores_computed=False,
        target_nonfit_label_isolation=True, enantiomer_separation=True,
        atom_order_invariance=True, original_component_replay=True, checks=checks))
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
