"""Read-only audit of actual scaffold mass and the scored ensemble comparator."""
import json
from collections import Counter
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import admet_v24_models as b
import admet_v40_run as older
import admet_v42_run as previous

ROOT = older.ROOT
OUT = ROOT / 'admet_model_outputs_v43'


def baseline(d):
    valid = np.load(older.OUT / 'baseline_valid.npy').copy()
    cv = np.load(older.OUT / 'baseline_cv_proxy.npy').copy()
    folds = np.load(b.OUT / 'folds.npy')
    report = json.loads((previous.OUT / 'analysis.json').read_text())
    assert report['signature'] == previous.signature()
    for r in report['tasks']:
        s = r['selection']
        if s is None:
            continue
        k = d['tasks'].index(r['task'])
        m = r['metric']
        vi = np.flatnonzero((d['tid'] == k) & (d['split'] == 1))
        tr = np.flatnonzero((d['tid'] == k) & (d['split'] == 0))
        vv = np.zeros(len(vi))
        cc = np.zeros(len(tr))
        for source in s['sources']:
            v = previous.source_predictions(r['task'], source['version'], source['variant'], m, 'valid')
            c = np.load(previous.SOURCES[source['version']].OUT / r['task'] / 'cv_predictions.npy')[source['variant'], tr].copy()
            if s['name'] == 'complement_equal' and m != 'MAE':
                v = b.rank(v)
                for fold in range(3):
                    active = folds[tr] == fold
                    c[active] = b.rank(c[active])
            vv += source['weight'] * v
            cc += source['weight'] * c
        p = b.blend(m, valid[vi], vv, s['weight'])
        np.testing.assert_allclose(b.delta(m, d['y'][vi], valid[vi], p), s['valid_gain'], rtol=1e-8, atol=1e-10)
        valid[vi] = p
        for fold in range(3):
            active = folds[tr] == fold
            ix = tr[active]
            cv[ix] = b.blend(m, cv[ix], cc[active], s['weight'])
    np.save(OUT / 'baseline_valid.npy', valid)
    np.save(OUT / 'baseline_cv_proxy.npy', cv)
    return valid, cv


def main():
    b.remote_only()
    OUT.mkdir(parents=True, exist_ok=True)
    d = b.build_data()
    base, _ = baseline(d)
    raw = []
    for sm in d['smiles']:
        mol = Chem.MolFromSmiles(sm)
        can = Chem.MolToSmiles(mol, isomericSmiles=False)
        raw.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or 'ACYCLIC:' + can)
    raw = np.array(raw)
    records = []
    for k, task in enumerate(d['tasks']):
        parts = {}
        for si, name in enumerate(['train', 'valid', 'test']):
            rows = np.flatnonzero((d['tid'] == k) & (d['split'] == si))
            mass = Counter(d['rg'][rows])
            parts[name] = dict(rows=len(rows), raw_scaffolds=len(set(raw[d['mi'][rows]])), merged_groups=len(mass),
                top_groups=[dict(group=str(g), rows=c) for g, c in mass.most_common(3)])
        tr = np.flatnonzero((d['tid'] == k) & (d['split'] == 0))
        vi = np.flatnonzero((d['tid'] == k) & (d['split'] == 1))
        te = np.flatnonzero((d['tid'] == k) & (d['split'] == 2))
        records.append(dict(task=task, metric=d['metrics'][k], partitions=parts,
            train_valid_group_overlap=len(set(d['rg'][tr]) & set(d['rg'][vi])),
            train_valid_identity_overlap=len(set(d['canonical'][d['mi'][tr]]) & set(d['canonical'][d['mi'][vi]])),
            test_fraction_group_seen_in_train=float(np.mean(np.isin(d['rg'][te], d['rg'][tr]))),
            test_fraction_group_seen_in_valid=float(np.mean(np.isin(d['rg'][te], d['rg'][vi]))),
            baseline_valid=b.metric(d['metrics'][k], d['y'][vi], base[vi])))
    result = dict(raw_unique_groups=len(set(raw)), cached_unique_groups=len(set(d['groups'])),
        raw_and_cached_groups_identical=bool(np.array_equal(raw, d['groups'])), tasks=records,
        note='Scaffolds computed label-free; test labels not read; historical valid and CV are reused.')
    b.save(OUT / 'scaffold_audit.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
