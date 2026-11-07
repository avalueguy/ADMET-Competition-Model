"""Read-only, development-only chemical strata of v145 MAE errors.

SMARTS are broad structural annotations, not measured pKa or assay pH.
No model is fitted and no confirmation residual is inspected.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Crippen, Descriptors
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tools')]
import admet_v67_data as parser
import v146_current_baseline as v

TASKS = ('caco2_wang', 'ppbr_az', 'lipophilicity_astrazeneca')
PATTERNS = dict(carboxyl='[CX3](=[OX1])[O;H1,-1]',
                phenol='[c][O;H1,-1]',
                sulfonyl_acid='[SX4](=[OX1])(=[OX1])[O;H1,-1]',
                aromatic_base='[nH0;+0]',
                aliphatic_amine='[N;X3;+0;!$(N-C=O);!$(N-S(=O)=O);!$(N-C=N);!$(N-a)]')
QUERIES = {k: Chem.MolFromSmarts(s) for k, s in PATTERNS.items()}
assert all(q is not None for q in QUERIES.values())


def annotate(smiles):
    mol, mode = parser.parse_molecule(smiles)
    counts = {name: len(mol.GetSubstructMatches(query)) for name, query in QUERIES.items()}
    charge = Chem.GetFormalCharge(mol)
    acid = any(counts[name] for name in ('carboxyl', 'phenol', 'sulfonyl_acid'))
    base = any(counts[name] for name in ('aromatic_base', 'aliphatic_amine'))
    return dict(**counts, formal_charge=charge, acid_like=acid, base_like=base,
                both=acid and base, neither=not acid and not base,
                charged=charge != 0, logp=float(Crippen.MolLogP(mol)),
                molecular_weight=float(Descriptors.MolWt(mol)), parse_mode=mode)


def stats(y, p, groups):
    signed = p-y
    unique = np.unique(groups)
    samples = [np.flatnonzero(groups == g) for g in unique]
    rng = np.random.default_rng(2491029)
    intervals = []
    if len(unique) > 1:
        for _ in range(300):
            ix = np.concatenate([samples[i] for i in rng.integers(len(samples), size=len(samples))])
            intervals.append(float(np.mean(signed[ix])))
    return dict(rows=len(y), groups=len(unique), mae=float(np.abs(signed).mean()),
                mean_signed_error=float(signed.mean()),
                median_signed_error=float(np.median(signed)),
                mean_signed_scaffold_ci90=np.quantile(intervals, [.05, .95]).tolist() if intervals else None)


def main():
    RDLogger.DisableLog('rdApp.*')
    d = v.load()
    records = []
    cache = {}
    for task in TASKS:
        k = d['tasks'].index(task)
        train = v.b.table(ROOT/'admet_competition_data/data_benchmark'/task/'train.csv')
        rows = np.flatnonzero((d['tid'] == k) & (d['split'] == 0))
        assert len(rows) == len(train)
        annotation = {}
        for i, row in zip(rows, train):
            if row['Drug'] not in cache:
                cache[row['Drug']] = annotate(row['Drug'])
            annotation[int(i)] = cache[row['Drug']]
        parts = []
        for part in ('fold0', 'fold1'):
            _, q = v.split(d, k, part)
            y, p = d['y'][q], v.baseline(d, k, part)
            z = [annotation[int(i)] for i in q]
            strata = {}
            for name in ('acid_like', 'base_like', 'both', 'neither', 'charged', *PATTERNS):
                keep = np.array([bool(a[name]) for a in z])
                if keep.sum() >= 15:
                    strata[name] = stats(y[keep], p[keep], d['groups'][q][keep])
            logp = np.array([a['logp'] for a in z])
            parts.append(dict(part=part, overall=stats(y,p,d['groups'][q]), strata=strata,
                residual_logp_spearman=float(spearmanr(p-y, logp).statistic),
                residual_molecular_weight_spearman=float(spearmanr(p-y,[a['molecular_weight'] for a in z]).statistic)))
        consistent = []
        for name in set(parts[0]['strata']) & set(parts[1]['strata']):
            a,z=(p['strata'][name]['mean_signed_scaffold_ci90'] for p in parts)
            if a and z and (min(a[0],z[0])>0 or max(a[1],z[1])<0):
                consistent.append(name)
        records.append(dict(task=task, parts=parts,
            same_direction_development_strata_with_ci_support=sorted(consistent)))
    report = dict(status='COMPLETE', checked_at_utc=datetime.now(timezone.utc).isoformat(),
        baseline='v145', baseline_signature=v.sig(), source_sha256=v.b.sha(__file__),
        parser_sha256=v.b.sha(parser.__file__), rdkit_version=rdBase.rdkitVersion,
        tasks=TASKS, structural_patterns=PATTERNS, records=records,
        model_trained=False, confirmation_residuals_used=False, test_labels_used=False,
        thresholds_or_frozen_splits_changed=False,
        interpretation='Exploratory diagnostics on repeatedly reused development labels. SMARTS are broad chemistry categories, not measured pKa/pH. Consistent bias is a hypothesis for new fit-only modelling, not permission to edit predictions per stratum or proof of test improvement.')
    v.b.save(ROOT/'docs/v249_ionization_error_audit.json',report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
