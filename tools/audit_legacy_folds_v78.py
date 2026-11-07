"""Reconstruct published v24 folds and measure v67 alignment, no model fitting."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import admet_v77_ranking as data


def main():
    d = data.load_data()
    z = np.load(ROOT / 'recovered_history/v24_group_identity.npz')
    rg = z['groups'][d['mi']]
    train = np.flatnonzero(d['split'] == 0)
    groups, inv = np.unique(rg[train], return_inverse=True)
    counts = np.zeros((len(groups), len(d['tasks'])), int)
    np.add.at(counts, (inv, d['tid'][train]), 1)
    rel = counts / np.maximum(counts.sum(0), 1)
    rng = np.random.default_rng(240929)
    order = np.lexsort((rng.random(len(groups)), -np.max(rel, 1)))
    mass = np.zeros((3, len(d['tasks'])))
    assigned = np.empty(len(groups), int)
    for g in order:
        costs = []
        for f in range(3):
            tmp = mass.copy(); tmp[f] += rel[g]
            costs.append(np.square(tmp).sum())
        f = int(np.argmin(np.array(costs) + rng.random(3) * 1e-12))
        mass[f] += rel[g]; assigned[g] = f
    out = np.full(len(d['y']), -1)
    out[train] = assigned[inv]
    path = ROOT / 'recovered_history/legacy_v24_folds.npy'
    np.save(path, out)
    expected = '3a24d8a1359df47c4197bff62a80809bbdc42f53db22c5dd1f478db2a95a4b98'
    assert data.sha(path) == expected
    old = d['folds']
    cross = [[int(sum((out[train] == i) & (old[train] == j))) for j in range(3)] for i in range(3)]
    records = []
    for k, task in enumerate(d['tasks']):
        active = (d['tid'] == k) & (d['split'] == 0)
        by_fold = []
        for f in range(3):
            q = np.flatnonzero(active & (out == f))
            fit = np.flatnonzero(active & (out != f))
            by_fold.append(dict(fold=f, n=len(q),
                new_scaffold_overlap=len(set(d['groups'][q]) & set(d['groups'][fit])),
                canonical_overlap=len(set(d['canonical'][q]) & set(d['canonical'][fit]))))
        records.append(dict(task=task, label= d['metrics'][k],
            same_numeric_fold_fraction=float(np.mean(old[active] == out[active])), folds=by_fold))
    result = dict(status='AUDITED', exact_legacy_sha256=data.sha(path), v67_sha256=data.sha(ROOT / 'admet_model_outputs_v67/folds.npy'),
        cross_tab_legacy_rows_current_columns=cross, tasks=records,
        proxy_cv_same_fit_rows_as_current_v67=False,
        issue='Historical predictions cannot be treated as same-fold OOF controls under v67 folds.',
        test_labels_used=False)
    data.save(ROOT / 'docs/v78_fold_alignment_audit.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
