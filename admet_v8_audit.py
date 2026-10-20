"""Summarize fixed-pool evidence without changing any candidate or release decision."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

from admet_v6_runner import remote_only, save, metrics, kind, deltas, secondary_ok, get_metric
from admet_v8_runner import OUT, baseline, protocol
from admet_v8_models import load_task


def main():
    remote_only(); rows = []; source_rows = []; exposures = []; p = protocol()
    for path in sorted(OUT.glob('*/multiscale_kernel_cv.json')):
        task = path.parent.name; d = load_task(task); base = baseline(task, d)
        metric = get_metric(task); kk = kind(metric)
        for family in ['multiscale_kernel', 'gated_transfer']:
            file = path.parent/(family+'_oof.csv')
            if not file.exists():
                continue
            candidate = pd.read_csv(file); gains = []
            for fold in range(3):
                mask = base.fold.eq(fold); y = base.loc[mask, 'Y_true']
                before = metrics(kk, y, base.loc[mask, 'Y_pred']); after = metrics(kk, y, candidate.loc[mask, 'Y_pred'])
                gain = deltas(metric, before, after); gains.append(gain)
            rows.append(dict(task=task, metric=metric, family=family,
                standalone_mean_primary_gain=float(np.mean(gains)), standalone_worst_primary_gain=float(min(gains)),
                note='Standalone vs v7 recipe; does not replace the fixed blend selection'))
        for file in path.parent.glob('*source_audit.json'):
            for r in json.loads(file.read_text()):
                assert r['heldout_scaffold_overlap'] == r['heldout_molecule_overlap'] == 0
                source_rows.append(dict(task=task, audit_file=file.name, **r))
        if (path.parent/'gated_transfer_cv.json').exists():
            n = len(d['frames']['train'])
            draws = p['mtl_epochs']*p['mtl_steps_per_epoch']*p['mtl_target_batch']
            fold = base.fold.to_numpy()
            exposures.append(dict(task=task, target_draws_per_seed=draws,
                nominal_equivalent_passes_by_fold=[draws/int((fold!=f).sum()) for f in range(3)]))
    save('admet_diagnostics/v8_candidate_audit.json', dict(completed_candidate_comparisons=rows,
        source_exclusions=source_rows, training_exposure=exposures, all_observed_auxiliary_scaffold_overlaps_zero=True,
        limitation='No reselection; all folds reused historically; no platform score projection'))
    print(json.dumps(dict(candidate_count=len(rows), audited_source_splits=len(source_rows),
        standalone_improving=[r for r in rows if r['standalone_mean_primary_gain']>0],
        all_observed_auxiliary_scaffold_overlaps_zero=True), indent=2))


if __name__=='__main__':
    main()
