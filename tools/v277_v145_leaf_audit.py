"""Fit-only feasibility audit of immutable tree components carried by v145.

No query labels, development prediction scores, confirmation scores, or updated
models are used. This only checks whether fixed-tree leaf recalibration is a
distinct, reproducible optimization of actual v145 components.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '3'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v146_current_baseline as v
import v127_low_classification as component

TASKS = ('herg', 'cyp3a4_substrate_carbonmangels')


def main():
    d = v.load()
    carried = json.loads((ROOT / 'admet_model_outputs_v130/evidence.json').read_text())['selected']
    records = []
    for task in TASKS:
        assert carried[task]['version'] == 'v127'
        assert carried[task]['record']['view'] == 'physchem' and carried[task]['weight'] == .5
        assert task not in ('ames', 'cyp2c9_veith', 'cyp2d6_veith')
        k = d['tasks'].index(task)
        parts = []
        for part in ('fold0', 'fold1'):
            fit, query, x, _, source = component.v.features(d, k, part, 'physchem')
            np.testing.assert_array_equal(fit, v.split(d, k, part)[0])
            np.testing.assert_array_equal(query, v.split(d, k, part)[1])
            assert not set(d['groups'][fit]) & set(d['groups'][query])
            seeds = []
            for seed in component.SEEDS:
                path = component.OUT / 'models' / task / part / 'physchem' / ('model_' + str(seed) + '.cbm')
                net = CatBoostClassifier().load_model(str(path))
                leaf = net.calc_leaf_indexes(x, thread_count=1)
                count = net.get_tree_leaf_counts()
                offsets = np.r_[0, np.cumsum(count)]
                values = net.get_leaf_values()
                scale, bias = net.get_scale_and_bias()
                reconstructed = scale * values[offsets[:-1] + leaf].sum(1) + bias
                raw = net.predict(x, prediction_type='RawFormulaVal', thread_count=1)
                np.testing.assert_allclose(reconstructed, raw, atol=1e-12, rtol=0)
                residual = expit(raw) - d['y'][fit]
                gradients, occupancy = [], []
                for j in range(leaf.shape[1]):
                    gradients.extend(np.bincount(leaf[:, j], weights=residual, minlength=count[j]))
                    occupancy.extend(np.bincount(leaf[:, j], minlength=count[j]))
                gradients, occupancy = np.asarray(gradients), np.asarray(occupancy)
                assert np.isfinite(gradients).all()
                seeds.append(dict(seed=seed, model_sha256=v.b.sha(path), trees=net.tree_count_,
                    leaves=len(values), occupied_leaves=int((occupancy > 0).sum()),
                    occupied_leaf_fraction=float((occupancy > 0).mean()),
                    small_leaf_fraction=float(((occupancy > 0) & (occupancy < 5)).sum() / (occupancy > 0).sum()),
                    gradient_rms=float(np.sqrt(np.mean(gradients ** 2))),
                    maximum_absolute_gradient=float(np.abs(gradients).max()),
                    logloss=float((np.logaddexp(0, raw) - d['y'][fit] * raw).mean()),
                    topology_reconstruction='PASS'))
            parts.append(dict(part=part, fit_rows=len(fit), fit_groups=len(np.unique(d['groups'][fit])),
                              other_assay_source_rows=len(source), seeds=seeds))
        records.append(dict(task=task, carried_component='v127/physchem', carried_weight=.5, parts=parts))
    result = dict(status='FIT_ONLY_COMPONENT_AUDIT_COMPLETE', at_utc=datetime.now(timezone.utc).isoformat(),
        baseline='v145', source_sha256=v.b.sha(__file__), records=records,
        query_labels_used=False, query_prediction_scores_computed=False,
        confirmation_scores_computed=False, updated_model_trained=False,
        qualification='Training gradients/occupancy are mechanism diagnostics, not evidence of generalization improvement.',
        feasibility='Existing exact v145 component checkpoints expose reconstructible leaf topology and values. Fit-only regularized leaf updates can optimize those components without new splits or arbitrary architecture changes.')
    v.b.save(ROOT / 'docs/v277_v145_leaf_audit.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
