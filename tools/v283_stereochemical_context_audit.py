"""Fit-only audit of stereochemical pooling in actual v145 components.

No target query labels or prediction scores are read. The existing fixed
splits, original structures, labels and checkpoints are not changed.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize as standard

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v146_current_baseline as v
import v210_standardized_chemistry as structures

TASKS = ('herg', 'cyp3a4_substrate_carbonmangels', 'cyp2d6_veith')
MINIMUM_CHANGED_FRACTION = .05
MINIMUM_CHANGED_ROWS = 10


def main():
    RDLogger.DisableLog('rdApp.*')
    d = v.load()
    smiles, smiles_sha = structures.originals(d)
    stereo, neutral = [], []
    for sm in smiles:
        mol, mode = structures.chemistry.parse_molecule(sm)
        stereo.append(Chem.MolToSmiles(mol, isomericSmiles=True))
        if mode == 'full_sanitization':
            parent = standard.LargestFragmentChooser().choose(mol)
            parent = standard.Uncharger().uncharge(parent)
        else:
            # Original documented unusual-valence graph: do not normalize it.
            parent = mol
        neutral.append(Chem.MolToSmiles(parent, isomericSmiles=True))
    stereo, neutral = np.array(stereo)[d['mi']], np.array(neutral)[d['mi']]
    records = []
    for task in TASKS:
        k = d['tasks'].index(task)
        parts = []
        for part in ('fold0', 'fold1'):
            fit, _ = v.split(d, k, part)
            old, cols, source, _, _ = v.old.old.old.assays.assays(d, k, part)
            lookups = []
            for keys in (stereo, neutral):
                lookup = {}
                for row in source:
                    lookup.setdefault((str(keys[row]), int(d['tid'][row])), []).append(float(d['y'][row]))
                lookups.append(lookup)
            new = np.full((len(fit), old.shape[1]), np.nan, dtype=np.float32)
            n = len(cols)
            for i, row in enumerate(fit):
                for j, t in enumerate(cols):
                    exact = lookups[0].get((str(stereo[row]), int(t)))
                    value = lookups[1].get((str(neutral[row]), int(t)))
                    if exact:
                        new[i, j] = np.median(exact)
                    if value:
                        new[i, 2*n+j] = np.median(value)
                        new[i, 4*n+j] = max(value)-min(value)
                new[i, n:2*n] = np.isfinite(new[i, :n])
                new[i, 3*n:4*n] = np.isfinite(new[i, 2*n:3*n])
            previous = old[fit]
            equal = (previous == new) | (np.isnan(previous) & np.isnan(new))
            changed = ~equal.all(1)
            # Count other-assay nonstereo identities containing genuinely
            # different isomeric structures and disagreeing measurements.
            pooled = {}
            for row in source:
                key = (str(d['canonical'][row]), int(d['tid'][row]))
                pooled.setdefault(key, []).append(int(row))
            mixed = [rows for rows in pooled.values()
                     if len(set(stereo[rows])) > 1 and np.ptp(d['y'][rows]) > 0]
            supported = bool(changed.sum() >= MINIMUM_CHANGED_ROWS and changed.mean() >= MINIMUM_CHANGED_FRACTION)
            parts.append(dict(part=part, fitting_rows=len(fit), changed_rows=int(changed.sum()),
                changed_fraction=float(changed.mean()), mixed_other_assay_identity_groups=len(mixed),
                mixed_source_rows=sum(map(len, mixed)), supported=supported))
        records.append(dict(task=task, parts=parts, supported=all(p['supported'] for p in parts)))
    report = dict(status='FIT_ONLY_CONTEXT_AUDIT_COMPLETE', at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=v.b.sha(__file__), original_smiles_sha256=smiles_sha,
        tasks_predeclared=TASKS, minimum_changed_fraction=MINIMUM_CHANGED_FRACTION,
        minimum_changed_rows=MINIMUM_CHANGED_ROWS, records=records,
        query_labels_used=False, query_predictions_computed=False, target_labels_changed=False,
        qualification='Pooling differences support a possible input-identity correction, not evidence of gain.',
        next_step='Only supported actual carried components may be considered; preserve original model settings and seeds, do not tune on audit.')
    v.b.save(ROOT / 'docs/v283_stereochemical_context_audit.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
