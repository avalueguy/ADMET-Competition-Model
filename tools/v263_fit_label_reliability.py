"""Training-partition-only replicate and boundary-label audit, without prediction.

Conflicting repeated chemical identities could motivate an observation-noise
model. A rounded value alone is not evidence of censoring or permission to
change a label. No fold2/valid scores, new model, or CSV are produced.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v146_current_baseline as v


def replicate_statistics(keys, y, metric):
    values = {}
    for key, value in zip(keys, y):
        values.setdefault(str(key), []).append(float(value))
    repeated = [np.array(x) for x in values.values() if len(x) > 1]
    conflicting = [x for x in repeated if np.ptp(x) > 1e-8]
    scale = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05)
    return dict(unique_identities=len(values), repeated_identities=len(repeated),
                repeated_rows=sum(map(len, repeated)), conflicting_identities=len(conflicting),
                conflicting_rows=sum(map(len, conflicting)),
                conflicting_row_fraction=sum(map(len, conflicting)) / len(y),
                median_disagreement_iqr=float(np.median([np.ptp(x) / scale for x in conflicting])) if conflicting else 0.,
                binary_conflicting_identities=len(conflicting) if metric in ('AUROC', 'AUPRC') else None)


def main():
    dest = ROOT / 'docs/v263_fit_label_reliability.json'
    assert not dest.exists(), 'Preserve diagnostic; no blind retry'
    d = v.load()
    records = []
    for task in v.tasks(d):
        k = d['tasks'].index(task)
        metric = d['metrics'][k]
        parts = []
        for part in ('fold0', 'fold1'):
            fit, query = v.split(d, k, part)
            assert np.all(d['split'][fit] == 0) and not set(fit) & set(query)
            y = d['y'][fit]
            unique, counts = np.unique(y, return_counts=True)
            top = np.argsort(-counts, kind='stable')[:8]
            row = dict(part=part, fit_rows=len(fit), unique_labels=len(unique),
                       label_quantiles=np.quantile(y, [0, .01, .1, .5, .9, .99, 1]).tolist(),
                       most_frequent_values=[dict(value=float(unique[i]), count=int(counts[i]),
                                                  fraction=float(counts[i] / len(y))) for i in top],
                       exact_structure=replicate_statistics(d['mi'][fit], y, metric),
                       nonstereo_canonical=replicate_statistics(d['canonical'][fit], y, metric),
                       neutral_parent=replicate_statistics(d['neutral_identity'][fit], y, metric))
            if task == 'ppbr_az':
                row['bounded_observations'] = dict(zero=int(np.sum(y == 0)), hundred=int(np.sum(y == 100)),
                                                   outside_0_100=int(np.sum((y < 0) | (y > 100))))
            parts.append(row)
        records.append(dict(task=task, metric=metric, parts=parts))
    report = dict(status='COMPLETE_FIT_ONLY_DIAGNOSTIC', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  baseline_signature=v.sig(), parts=['fold0_fit', 'fold1_fit'], records=records,
                  model_fitted=False, new_predictions=False, confirmation_labels_used_by_diagnostic=False,
                  shared_loader_contains_public_labels=True,
                  confirmation_scores_computed=False, labels_changed=False,
                  limitations='Canonical and neutral equivalence may merge distinct stereochemistry/salts. Label repetitions and boundary masses do not establish assay censoring, measurement error, or justified relabeling.')
    v.b.save(dest, report)
    print(json.dumps(dict(status=report['status'], tasks=len(records))), flush=True)


if __name__ == '__main__':
    main()
