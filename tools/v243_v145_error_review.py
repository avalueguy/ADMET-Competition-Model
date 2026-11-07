"""Read-only v145 error and verification feasibility review, no model selection."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v146_current_baseline as v


def main():
    d = v.load()
    records = []
    for task in v.tasks(d):
        k = d['tasks'].index(task)
        parts = []
        for part in ('fold0', 'fold1', 'fold2', 'valid', 'final'):
            fit, q = v.split(d, k, part)
            _, count = np.unique(d['groups'][q], return_counts=True)
            record = dict(part=part, fit=len(fit), query=len(q), groups=len(count),
                          largest_group_fraction=float(count.max() / len(q)))
            # Already-frozen development errors only. No new confirmation scoring.
            if part in ('fold0', 'fold1') and d['metrics'][k] == 'MAE':
                y, p = d['y'][q], v.baseline(d, k, part)
                limits = np.quantile(d['y'][fit], [.1, .5, .9])
                bands = np.searchsorted(limits[[0, 2]], y)
                record.update(mae=float(np.abs(p - y).mean()),
                              median_signed_error=float(np.median(p - y)),
                              label_quantiles=np.quantile(y, [0, .1, .5, .9, 1]).tolist(),
                              prediction_quantiles=np.quantile(p, [0, .1, .5, .9, 1]).tolist(),
                              absolute_error_quantiles=np.quantile(np.abs(p-y), [.5, .9, .95]).tolist(),
                              error_by_fit_label_band=[dict(band=j, rows=int((bands == j).sum()),
                                  mae=float(np.abs(p[bands == j]-y[bands == j]).mean()) if (bands == j).any() else None,
                                  signed_mean=float((p[bands == j]-y[bands == j]).mean()) if (bands == j).any() else None)
                                  for j in range(3)])
            parts.append(record)
        records.append(dict(task=task, metric=d['metrics'][k], parts=parts,
            original_confirmation_feasible=all(r['groups'] > 1 for r in parts if r['part'] in ('fold2', 'valid'))))
    result = dict(status='COMPLETE', at_utc=datetime.now(timezone.utc).isoformat(),
        baseline='v145', official_best=51.166, baseline_signature=v.sig(),
        runner_sha256=v.b.sha(ROOT/'tools/v148_experiment_runner.py'),
        baseline_csv_sha256=v.b.sha(v.ANCHOR), records=records,
        model_trained=False, gates_changed=False, test_labels_used=False,
        new_confirmation_scores_computed=False,
        interpretation=[
            'An official normalized score near 50 does not by itself measure attainable raw-metric headroom.',
            'AMES valid has one frozen exclusion component, so original scaffold confirmation rejects it by construction.',
            'MAE tasks offer larger normalized-score sensitivity; tails and median bias need fit-only distribution modelling, not tuning on confirmation.',
            'Historical controls have approximate upstream OOF and labels have been reused. These diagnostics are not unbiased estimates or proof of hidden-test gain.'])
    v.b.save(ROOT/'docs/v243_v145_error_review.json', result)
    print(json.dumps(dict(status=result['status'],
        infeasible_tasks=[r['task'] for r in records if not r['original_confirmation_feasible']],
        mae_development=[r for r in records if r['metric']=='MAE'])), flush=True)


if __name__ == '__main__':
    main()
