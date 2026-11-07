"""New neutral-parent/tautomer representations with frozen endpoint experts."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'
import hashlib
import importlib.metadata
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator as fg
from rdkit.Chem.MolStandardize import rdMolStandardize as standard
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import admet_v67_data as chemistry
import v146_current_baseline as v
import v148_experiment_runner as runner
b, r83 = v.b, v.r83
OUT = ROOT / 'admet_model_outputs_v210'
SEEDS = (2101028, 2101029)
CONFIGS = (dict(depth=6, trees=1000),)
DESCRIPTORS = [(n, f) for n, f in Descriptors.descList if n not in ('Ipc', 'AvgIpc', 'BalabanJ')]
METHOD = ('New molecular representations: neutral largest organic parent and bounded '
          'canonical tautomer, each with2048 chirality-aware radius2 Morgan counts, '
          '1024 feature-Morgan radius3 counts and214 RDKit descriptors. Original '
          'unmodified molecular/shape/pharmacophore and target-excluded measured '
          'context remain available. Two-seed1000tree depth6 CatBoost; no old '
          'model predictions, outside data or weights. All21 improvable endpoints, '
          'originalv148 selection/blends/confirmation, HIA protected.')


def sig():
    packages = '|'.join(importlib.metadata.version(p) for p in ('numpy', 'rdkit', 'catboost', 'joblib'))
    return hashlib.sha256((b.sha(__file__) + b.sha(chemistry.__file__) + v.sig() +
                           b.sha(runner.__file__) + packages).encode()).hexdigest()


def key(c):
    return 'depth' + str(c['depth'])


def originals(d):
    values = [None] * len(d['views']['physchem'])
    cursor = 0
    for task in d['tasks']:
        for part in ('train', 'valid', 'test'):
            records = ([dict(Drug=r['smiles']) for r in d['template'] if r['task'] == task]
                       if part == 'test' else b.table(ROOT / 'admet_competition_data/data_benchmark' / task / (part + '.csv')))
            for row in records:
                index = int(d['mi'][cursor])
                cursor += 1
                assert values[index] in (None, row['Drug'])
                values[index] = row['Drug']
    assert cursor == len(d['y']) and all(s is not None for s in values)
    sha = hashlib.sha256(json.dumps(values).encode()).hexdigest()
    expected = json.loads((ROOT / 'admet_model_outputs_v67/features/official_chemistry.json').read_text())['smiles_sha256']
    assert sha == expected
    return values, sha


def normalized(smiles):
    RDLogger.DisableLog('rdApp.*')
    mol, mode = chemistry.parse_molecule(smiles)
    # The documented unusual-valence source graphs must not be silently repaired.
    if mode != 'full_sanitization':
        return mol, mol, dict(parse_mode=mode, normalization='original_valence_exception',
                              tautomer_status='not_enumerated', tautomers=1)
    parent = standard.LargestFragmentChooser(preferOrganic=True).choose(mol)
    neutral = standard.Uncharger().uncharge(parent)
    enumerator = standard.TautomerEnumerator()
    enumerator.SetMaxTautomers(256)
    enumerator.SetMaxTransforms(1024)
    result = enumerator.Enumerate(neutral)
    assert len(result) > 0
    tautomer = enumerator.PickCanonical(result)
    status = str(result.status)
    return neutral, tautomer, dict(parse_mode=mode, normalization='neutral_organic_parent',
                                  tautomer_status=status, tautomers=len(result))


def one(smiles):
    neutral, tautomer, audit = normalized(smiles)
    generators = (fg.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True),
                  fg.GetMorganGenerator(radius=3, fpSize=1024,
                                        atomInvariantsGenerator=fg.GetMorganFeatureAtomInvGen()))
    blocks, identities, descriptor_exceptions = [], [], []
    for j, mol in enumerate((neutral, tautomer)):
        fp = np.r_[*[np.log1p(g.GetCountFingerprintAsNumPy(mol).astype(np.float32)) for g in generators]]
        values = []
        for name, function in DESCRIPTORS:
            try:
                value = float(function(mol))
            except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
                value = np.nan
                descriptor_exceptions.append([j, name])
            values.append(value if np.isfinite(value) else np.nan)
        blocks.append(np.r_[fp, np.asarray(values, dtype=np.float32)])
        identities.append(Chem.MolToSmiles(mol, isomericSmiles=True))
    audit.update(identities=identities, descriptor_exceptions=descriptor_exceptions)
    # Explicit flags prevent incomplete enumeration/graph exceptions being hidden.
    flags = np.array([audit['normalization'] == 'original_valence_exception',
                      audit['tautomer_status'] not in ('Completed', 'not_enumerated'),
                      audit['tautomers'], len(Chem.GetMolFrags(chemistry.parse_molecule(smiles)[0]))], dtype=np.float32)
    row = np.r_[*blocks, flags].astype(np.float32)
    assert not np.isinf(row).any() and np.isfinite(row[:3072]).all()
    return row, audit


def features(d):
    if '_v210_features' in d:
        return d['_v210_features']
    folder = OUT / 'features'
    folder.mkdir(parents=True, exist_ok=True)
    file, report = folder / 'normalized.npz', folder / 'audit.json'
    strings, sha = originals(d)
    if file.exists():
        meta = json.loads(report.read_text())
        assert meta['signature'] == sig() and meta['smiles_sha256'] == sha
        assert meta['sha256'] == b.sha(file)
        x = np.load(file)['x']
        assert list(x.shape) == meta['shape'] and len(x) == len(strings)
    else:
        rows, records, start = [], [], time.monotonic()
        with ProcessPoolExecutor(max_workers=6) as pool:
            for i, (row, audit) in enumerate(pool.map(one, strings, chunksize=16)):
                rows.append(row)
                records.append(audit)
                if i % 3000 == 0:
                    print(json.dumps(dict(stage='standardized_features', molecules=i,
                                          total=len(strings), seconds=round(time.monotonic() - start))), flush=True)
        x = np.stack(rows)
        assert x.shape == (len(strings), 2 * (3072 + len(DESCRIPTORS)) + 4)
        np.savez_compressed(file, x=x, signature=sig())
        joblib.dump(records, folder / 'representations.joblib', compress=3)
        b.save(report, dict(status='PASS', signature=sig(), sha256=b.sha(file),
               representations_sha256=b.sha(folder / 'representations.joblib'),
               smiles_sha256=sha, shape=list(x.shape), rdkit=rdBase.rdkitVersion,
               descriptors=[n for n, _ in DESCRIPTORS], structure_only=True,
               original_graph_exceptions=[i for i, r in enumerate(records) if r['normalization'] == 'original_valence_exception'],
               bounded_enumeration_rows=[i for i, r in enumerate(records) if r['tautomer_status'] not in ('Completed', 'not_enumerated')],
               descriptor_exception_rows=[i for i, r in enumerate(records) if r['descriptor_exceptions']],
               source='Official structures only, deterministic per-molecule transform; no labels or fit/query coupling.'))
    d['_v210_features'] = x
    return x


def verify_features(d):
    strings, _ = originals(d)
    expected = features(d)
    report = json.loads((OUT / 'features/audit.json').read_text())
    records = joblib.load(OUT / 'features/representations.joblib')
    assert b.sha(OUT / 'features/representations.joblib') == report['representations_sha256']
    with ProcessPoolExecutor(max_workers=6) as pool:
        for i, (row, audit) in enumerate(pool.map(one, strings, chunksize=16)):
            np.testing.assert_array_equal(row, expected[i])
            assert audit == records[i]
    b.save(OUT / 'features/independent_verification.json', dict(status='PASS',
           signature=sig(), all_structures_regenerated=True, molecules=len(strings),
           feature_sha256=b.sha(OUT / 'features/normalized.npz'),
           representations_sha256=report['representations_sha256']))


def design(d, k, part):
    fit, query, x, qx, source = v.old.features(d, k, part, 'augmented')
    raw = features(d)
    x = np.c_[x, raw[d['mi'][fit]]].astype(np.float32)
    qx = np.c_[qx, raw[d['mi'][query]]].astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    qx[~np.isfinite(qx)] = np.nan
    metric, y = d['metrics'][k], d['y'][fit]
    center = float(np.median(y)) if metric == 'MAE' else 0.
    scale = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05) if metric == 'MAE' else 1.
    target = ((y - center) / scale if metric == 'MAE' else r83.ranknorm(y) if metric == 'Spearman' else y)
    assert not set(fit) & set(query) and k not in set(d['tid'][source])
    state = dict(fit=fit, query=query, source=source, target=target, center=center, scale=scale,
                 feature_sha256=b.sha(OUT / 'features/normalized.npz'))
    return dict(x=x, qx=qx, state=state)


def output(metric, raw, state, task):
    p = expit(raw) if metric in ('AUROC', 'AUPRC') else raw * state['scale'] + state['center']
    return p.clip(0, 100) if task == 'ppbr_az' else p


def estimator(metric, c, seed):
    return (CatBoostClassifier if metric in ('AUROC', 'AUPRC') else CatBoostRegressor)(
        loss_function='Logloss' if metric in ('AUROC', 'AUPRC') else 'MAE' if metric == 'MAE' else 'RMSE',
        iterations=c['trees'], depth=c['depth'], learning_rate=.03, l2_leaf_reg=30,
        random_seed=seed, thread_count=4, verbose=False, allow_writing_files=False,
        bootstrap_type='Bernoulli', subsample=.85, border_count=64, task_type='CPU')


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
    metric = d['metrics'][k]
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(a['state'], folder / 'transform.joblib', compress=3)
    members = []
    for seed in SEEDS:
        net = estimator(metric, c, seed)
        net.fit(a['x'], a['state']['target'])
        raw = net.predict(a['qx'], prediction_type='RawFormulaVal')
        path = folder / ('model_' + str(seed) + '.cbm')
        net.save_model(str(path))
        reloaded = CatBoostClassifier() if metric in ('AUROC', 'AUPRC') else CatBoostRegressor()
        reloaded.load_model(str(path))
        np.testing.assert_array_equal(raw, reloaded.predict(a['qx'], prediction_type='RawFormulaVal'))
        members.append(raw)
    p = output(metric, np.mean(members, axis=0), a['state'], d['tasks'][k])
    assert np.isfinite(p).all()
    np.savez_compressed(file, signature=sig(), fit=a['state']['fit'], query=a['state']['query'], p=p,
                        center=a['state']['center'], scale=a['state']['scale'])
    print(json.dumps(dict(stage='train', task=d['tasks'][k], part=part,
                          seconds=round(time.monotonic() - start))), flush=True)
    return p


def replay(d, k, part, c):
    from v161_verify_and_assemble import same
    a = design(d, k, part)
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    same(a['state'], joblib.load(folder / 'transform.joblib'))
    hashes = {'transform.joblib': b.sha(folder / 'transform.joblib')}
    members = []
    for seed in SEEDS:
        file = folder / ('model_' + str(seed) + '.cbm')
        net = CatBoostClassifier() if d['metrics'][k] in ('AUROC', 'AUPRC') else CatBoostRegressor()
        net.load_model(str(file))
        members.append(net.predict(a['qx'], prediction_type='RawFormulaVal'))
        hashes[file.name] = b.sha(file)
    return output(d['metrics'][k], np.mean(members, axis=0), a['state'], d['tasks'][k]), hashes


def smoke(d):
    strings, digest = originals(d)
    labels = d['y'].copy()
    try:
        d['y'][:] = 999999.
        assert originals(d) == (strings, digest)
    finally:
        d['y'][:] = labels
    # Known counterion/charge and keto/enol equivalence, no scores consulted.
    neutral = one('CC(=O)O')[0]
    sodium = one('CC(=O)[O-].[Na+]')[0]
    np.testing.assert_array_equal(neutral[:-4], sodium[:-4])
    a, z = one('CC(=O)C'), one('C=C(O)C')
    assert a[1]['identities'][1] == z[1]['identities'][1]
    np.testing.assert_array_equal(a[0][3072 + len(DESCRIPTORS): -4],
                                  z[0][3072 + len(DESCRIPTORS): -4])
    indices = np.linspace(0, len(strings) - 1, 24, dtype=int)
    first = [one(strings[i]) for i in indices]
    reverse = [one(strings[i]) for i in indices[::-1]][::-1]
    for x, y in zip(first, reverse):
        np.testing.assert_array_equal(x[0], y[0])
        assert x[1] == y[1]
    # Actual training/reload for classification, MAE and rank-normal objectives,
    # using bounded real endpoint subsets and only a16-tree smoke estimator.
    from tempfile import TemporaryDirectory
    reloads = []
    for task in ('herg', 'solubility_aqsoldb', 'half_life_obach'):
        k = d['tasks'].index(task)
        fit, query, x, qx, source = v.old.features(d, k, 'fold0', 'augmented')
        fit, query, x, qx = fit[:32], query[:8], x[:32], qx[:8]
        rows = {int(i): one(strings[int(i)])[0] for i in np.unique(d['mi'][np.r_[fit, query]])}
        x = np.c_[x, np.stack([rows[int(i)] for i in d['mi'][fit]])]
        qx = np.c_[qx, np.stack([rows[int(i)] for i in d['mi'][query]])]
        complete_query = v.split(d, k, 'fold0')[1]
        old = d['y'][complete_query].copy()
        try:
            d['y'][complete_query] = 999999.
            d.pop('_v96_assays', None)
            f, q, xx, zz, ss = v.old.features(d, k, 'fold0', 'augmented')
            np.testing.assert_array_equal(x[:, :xx.shape[1]], xx[:32])
            np.testing.assert_array_equal(qx[:, :zz.shape[1]], zz[:8])
            np.testing.assert_array_equal(source, ss)
        finally:
            d['y'][complete_query] = old
            d.pop('_v96_assays', None)
        metric, y = d['metrics'][k], d['y'][fit]
        target = r83.ranknorm(y) if metric == 'Spearman' else y
        net = estimator(metric, dict(depth=3, trees=16), SEEDS[0])
        net.fit(x, target)
        p = net.predict(qx, prediction_type='RawFormulaVal')
        with TemporaryDirectory(dir=OUT) as folder:
            file = Path(folder) / 'smoke.cbm'
            net.save_model(str(file))
            reloaded = CatBoostClassifier() if metric in ('AUROC', 'AUPRC') else CatBoostRegressor()
            reloaded.load_model(str(file))
            np.testing.assert_array_equal(p, reloaded.predict(qx, prediction_type='RawFormulaVal'))
        assert np.isfinite(p).all() and k not in set(d['tid'][source])
        reloads.append(task)
    b.save(OUT / 'smoke.json', dict(status='PASS', signature=sig(),
           all_endpoint_label_perturbation_invariant=True, query_label_isolation=True,
           counterion_neutral_equivalence=True, keto_enol_tautomer_equivalence=True,
           deterministic_per_molecule_row_order=True, independent_reload_tasks=reloads,
           no_query_scores_used=True, source_smiles_sha256=digest))
    print('SMOKE_PASS', flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    d = v.load()
    if '--smoke' in sys.argv:
        smoke(d)
    else:
        assert json.loads((OUT / 'smoke.json').read_text())['signature'] == sig()
        features(d)
        runner.run(sys.modules[__name__], d)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        b.save(OUT / 'failure.json', dict(status='FAILED', traceback=traceback.format_exc()))
        raise
