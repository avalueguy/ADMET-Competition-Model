"""Fitting-fold-only support for predeclared physiological assay relationships.

No query targets, prediction, model fit, or confirmation scores are used.
Collapse target repeats only for this descriptive correlation, never relabel
the training table. Other-task inputs follow the existing permitted loader.
"""
import json
import hashlib
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v146_current_baseline as v

PAIRS = {
    'caco2_wang': ['lipophilicity_astrazeneca', 'solubility_aqsoldb'],
    'ppbr_az': ['lipophilicity_astrazeneca'],
    'clearance_hepatocyte_az': ['clearance_microsome_az', 'half_life_obach'],
    'clearance_microsome_az': ['clearance_hepatocyte_az', 'half_life_obach'],
    'vdss_lombardo': ['ppbr_az', 'lipophilicity_astrazeneca'],
    'solubility_aqsoldb': ['lipophilicity_astrazeneca'],
    'herg': ['lipophilicity_astrazeneca'],
}


def main():
    path = ROOT / 'docs/v270_assay_dependence_audit.json'
    assert not path.exists(), 'Preserve the diagnostic; no repeated data search'
    d = v.load()
    records = []
    for task, paired in PAIRS.items():
        k = d['tasks'].index(task)
        for other in paired:
            parts = []
            for part in ('fold0', 'fold1'):
                fit, query = v.split(d, k, part)
                aux, cols, source, _, _ = v.old.old.old.assays.assays(d, k, part)
                # v131 -> v125 -> v117 -> v96 is the immutable context loader.
                assert k not in cols and np.all(d['tid'][source] != k)
                assert np.all(d['split'][source] == 0)
                j = list(cols).index(d['tasks'].index(other))
                unique = {}
                for row in fit:
                    if np.isfinite(aux[row, j]):
                        unique.setdefault(str(d['canonical'][row]), []).append(int(row))
                identity_rows = list(unique.values())
                target = np.array([np.median(d['y'][rows]) for rows in identity_rows])
                value = np.array([aux[rows[0], j] for rows in identity_rows])
                rho = float(spearmanr(target, value).statistic) if len(target) >= 4 and len(np.unique(target)) > 1 and len(np.unique(value)) > 1 else None
                parts.append(dict(part=part, fit_rows=len(fit), paired_rows=sum(map(len, identity_rows)),
                    unique_paired_identities=len(target), distinct_other_values=len(np.unique(value)),
                    rho=rho, target_binary=d['metrics'][k] in ('AUROC', 'AUPRC')))
            supported = (all(p['unique_paired_identities'] >= 40 and p['distinct_other_values'] >= 10
                             and p['rho'] is not None and abs(p['rho']) >= .35 for p in parts)
                         and parts[0]['rho'] * parts[1]['rho'] > 0)
            records.append(dict(task=task, other=other, parts=parts, supported=supported))
    report = dict(status='COMPLETE_FIT_ONLY_DIAGNOSTIC', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        baseline_signature=v.sig(), predeclared_pairs=PAIRS, records=records,
        support_rule='Both fitting folds at least40 distinct target identities and10 other-assay values; absSpearman>=.35 and same sign. Fixed before diagnostic.',
        model_fitted=False, predictions_created=False, query_targets_read_by_diagnostic=False,
        confirmation_scores_computed=False, labels_changed=False,
        limitation='Correlation among observed assays may reflect selection or unrecorded assay conditions. Support is a rationale for a distinct observation model, not evidence of predictive improvement. Other endpoint official training measurements are permitted context, target/query labels are not.')
    v.b.save(path, report)
    print(json.dumps(dict(status=report['status'], records=records)), flush=True)


if __name__ == '__main__':
    main()
