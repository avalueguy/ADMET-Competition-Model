"""Salt/charge parent molecular views complement original chemistry; remote only."""
import hashlib
import json
import time
import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetMorganFeatureAtomInvGen
from xgboost import XGBClassifier, XGBRegressor
import admet_v24_models as b
import admet_v33_run as n
import admet_v40_run as runner
import admet_v39_run as previous

ROOT = runner.ROOT
OUT = ROOT / 'admet_model_outputs_v41'
P = json.loads((ROOT / 'admet_v41_protocol.json').read_text())


def signature():
    names = ['admet_v41_run.py', 'admet_v41_protocol.json', 'admet_v40_run.py', 'admet_model_outputs_v39/analysis.json']
    return hashlib.sha256(json.dumps(dict(previous=previous.signature(),
        files={name: b.sha(ROOT / name) for name in names}), sort_keys=True).encode()).hexdigest()


def chemistry(d):
    dest = OUT / 'features'
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / 'parents.npz'
    meta = dict(source_sha=b.sha(__file__), rdkit=rdBase.rdkitVersion,
        smiles_sha=hashlib.sha256(json.dumps(d['smiles']).encode()).hexdigest(), standardization=P['standardization'])
    if path.exists():
        assert json.loads((dest / 'parents.json').read_text()) == meta
        z = np.load(path)
        return {key: z[key].copy() for key in z.files}
    descs = [(name, fn) for name, fn in Descriptors.descList if name not in {'Ipc', 'AvgIpc', 'BalabanJ'}]
    assert len(descs) == d['desc'].shape[1]
    fps = np.zeros((len(d['smiles']), 4096), np.float32)
    desc = np.full((len(d['smiles']), len(descs)), np.nan, np.float32)
    flags = np.zeros((len(d['smiles']), 5), np.float32)
    canonical = []
    changed = 0
    fallback = 0
    generators = [GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True),
        GetMorganGenerator(radius=3, fpSize=2048, includeChirality=True, atomInvariantsGenerator=GetMorganFeatureAtomInvGen())]
    uncharger = rdMolStandardize.Uncharger()
    for i, smiles in enumerate(d['smiles']):
        original = Chem.MolFromSmiles(smiles)
        if original is None:
            canonical.append('INVALID:'+smiles)
            continue
        try:
            clean = rdMolStandardize.Cleanup(original)
            parent = rdMolStandardize.FragmentParent(clean, skipStandardize=True)
            parent = uncharger.uncharge(parent)
            Chem.SanitizeMol(parent)
        except Exception:
            parent = original
            fallback += 1
        old = Chem.MolToSmiles(original, isomericSmiles=False)
        can = Chem.MolToSmiles(parent, isomericSmiles=False)
        canonical.append(can)
        changed += int(can != old)
        flags[i] = [len(Chem.GetMolFrags(original)), Chem.GetFormalCharge(original),
            Chem.GetFormalCharge(parent), original.GetNumHeavyAtoms()-parent.GetNumHeavyAtoms(), int(can != old)]
        fps[i] = np.log1p(np.concatenate([g.GetCountFingerprintAsNumPy(parent) for g in generators])).clip(0, 6)
        for j, (_, fn) in enumerate(descs):
            try:
                value = float(fn(parent))
                if np.isfinite(value) and abs(value) < 1e10:
                    desc[i, j] = value
            except Exception:
                pass
        if i % 10000 == 0:
            b.emit('v41_parent_features', done=i, total=len(fps))
    z = dict(fp=fps, desc=desc, flags=flags, canonical=np.array(canonical))
    np.savez_compressed(path, **z)
    b.save(dest / 'parents.json', meta)
    b.save(dest / 'parents_audit.json', dict(**meta, changed_molecules=changed, total_molecules=len(fps),
        changed_fraction=changed/len(fps), failed_standardizations=fallback, labels_used=False))
    return z


def features(d, k, rows, query, splits, other, chem):
    x, _, _, tf, audit = n.features(d, k, rows, query, splits, other)
    unique = np.unique(d['mi'][rows])
    raw = np.concatenate([chem['desc'], chem['desc']-d['desc'], chem['flags']], axis=1)
    pre = b.Preprocessor().fit(raw[unique])
    normalized = pre.transform(raw)
    fps = np.concatenate([d['fp2'], d['fcfp'], chem['fp']], axis=1)
    var = np.var(fps[unique], axis=0)
    selected = np.argsort(var, kind='stable')[-P['fingerprint_columns']:]
    parts = {}
    for name, rr in [('fit', rows), ('query', query)]:
        mol = d['mi'][rr]
        structural = np.concatenate([x[name]['dense'], normalized[mol], fps[mol][:, selected]], axis=1)
        parts[name] = [structural, np.concatenate([structural, x[name]['aux']], axis=1)]
    # No identity matching or label aggregation uses the standardized identity.
    audit.update(parent_structure_label_free=True, parent_descriptor_preprocessor_fit_only=True,
        fingerprint_columns_fit_only=True, fingerprint_columns=selected.tolist(),
        parent_identity_used_for_assay_lookup=False,
        parent_identity_overlap_with_query=int(len(set(chem['canonical'][d['mi'][rows]]) & set(chem['canonical'][d['mi'][query]]))))
    return parts, tf, audit


def fit(d, k, rows, query, splits, other, chemical, dest):
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / 'predictions.npz'
    sig = signature()
    if path.exists():
        z = np.load(path)
        assert str(z['signature']) == sig
        np.testing.assert_array_equal(z['fit'], rows)
        np.testing.assert_array_equal(z['query'], query)
        return z['p'].copy()
    start = time.time()
    x, target, audit = features(d, k, rows, query, splits, other, chemical)
    metric = d['metrics'][k]
    cls = metric in ['AUROC', 'AUPRC']
    y = target.encode(d['y'][rows])
    klass = XGBClassifier if cls else XGBRegressor
    _, inv, count = np.unique(d['canonical'][d['mi'][rows]], return_inverse=True, return_counts=True)
    weights = 1/np.sqrt(count[inv])
    parts = []
    for family in range(2):
        seeds = []
        for seed, depth, trees in zip(P['seeds'], P['depths'], P['estimators']):
            model = klass(n_estimators=trees, max_depth=depth, learning_rate=.035, min_child_weight=4,
                subsample=.85, colsample_bytree=.8, reg_lambda=18., reg_alpha=.1, tree_method='hist',
                device='cuda', max_bin=128, n_jobs=3, random_state=seed,
                objective='binary:logistic' if cls else ('reg:absoluteerror' if metric == 'MAE' else 'reg:squarederror'))
            model.fit(x['fit'][family], y, sample_weight=weights)
            pathm = dest / f'boost{family}_{seed}.ubj'
            model.save_model(pathm)
            def predict(m):
                pred = m.predict_proba(x['query'][family])[:, 1] if cls else m.predict(x['query'][family])
                return target.decode(pred) if metric == 'MAE' else pred
            p = predict(model)
            restored = klass()
            restored.load_model(pathm)
            restored.set_params(device='cuda', n_jobs=3)
            np.testing.assert_allclose(p, predict(restored), rtol=2e-6, atol=2e-6)
            seeds.append(p)
        parts.append(np.mean(seeds, axis=0))
    p = np.stack(parts)
    assert np.isfinite(p).all()
    np.savez_compressed(path, p=p, fit=rows, query=query, signature=sig)
    audit.update(signature=sig, task=d['tasks'][k], fit_rows=len(rows), query_rows=len(query),
        seconds=time.time()-start, model_reload_passed=True, local_gpu_training=False, test_labels_used=False)
    b.save(dest / 'metadata.json', audit)
    b.emit('v41_fit', task=d['tasks'][k], partition=str(dest.relative_to(OUT)), seconds=round(time.time()-start))
    return p


def variants(p, metric):
    return runner.variants(p, metric)


def main():
    b.remote_only()
    torch.backends.cuda.matmul.allow_tf32 = True
    OUT.mkdir(parents=True, exist_ok=True)
    # Reuse the frozen evaluation orchestration, without changing its source files.
    runner.OUT = OUT
    runner.P = P
    runner.signature = signature
    runner.fit = fit
    d = b.build_data()
    chem = chemistry(d)
    other = n.other_cache(d, [0])
    runner.screen(d, chem, other)
    runner.evaluate(d, chem, other)
    runner.final(d, chem)
    b.save(OUT / 'completed.json', dict(signature=signature(), completed=True))


if __name__ == '__main__':
    main()
