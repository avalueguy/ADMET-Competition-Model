"""Reload parent-chemistry boosting models and prove hidden labels do not enter inputs."""
import json
import numpy as np
import torch
from xgboost import XGBClassifier, XGBRegressor
import admet_v24_models as b
import admet_v33_run as n
import admet_v41_run as v


def main():
    b.remote_only()
    torch.backends.cuda.matmul.allow_tf32 = True
    d = b.build_data()
    chem = v.chemistry(d)
    other = n.other_cache(d, [0, 1])
    report = json.loads((v.OUT / 'analysis.json').read_text())
    assert report['status'] == 'EVALUATED' and report['signature'] == v.signature()
    compact = json.loads((v.OUT / 'test_predictions_compact.json').read_text())
    records = []
    for r in report['tasks']:
        if r['selection'] is None:
            continue
        k = d['tasks'].index(r['task'])
        metric = r['metric']
        cls = metric in ['AUROC', 'AUPRC']
        rows = np.flatnonzero((d['tid'] == k) & (d['split'] < 2))
        q = np.flatnonzero((d['tid'] == k) & (d['split'] == 2))
        dest = v.OUT / r['task'] / 'final'
        z = np.load(dest / 'predictions.npz')
        assert str(z['signature']) == v.signature()
        np.testing.assert_array_equal(z['fit'], rows)
        np.testing.assert_array_equal(z['query'], q)
        x, tf, audit = v.features(d, k, rows, q, [0, 1], other, chem)
        parts = []
        for family in range(2):
            seeds = []
            for seed in v.P['seeds']:
                model = (XGBClassifier if cls else XGBRegressor)()
                model.load_model(dest / f'boost{family}_{seed}.ubj')
                model.set_params(device='cuda', n_jobs=3)
                p = model.predict_proba(x['query'][family])[:, 1] if cls else model.predict(x['query'][family])
                seeds.append(tf.decode(p) if metric == 'MAE' else p)
            parts.append(np.mean(seeds, axis=0))
        pred = np.stack(parts)
        np.testing.assert_allclose(pred, z['p'], rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(pred, compact[r['task']], rtol=2e-5, atol=2e-5)
        probe = dict(d)
        probe['y'] = d['y'].copy()
        probe['y'][d['split'] == 2] = 1e9
        poisoned = other.copy()
        poisoned[:, k, :] = -1e8
        px, _, _ = v.features(probe, k, rows, q, [0, 1], poisoned, chem)
        for name in ['fit', 'query']:
            for family in range(2):
                np.testing.assert_array_equal(x[name][family], px[name][family])
        records.append(dict(task=r['task'], model_reload_passed=True, hidden_target_labels_invariant=True,
            target_auxiliary_block_invariant=True, parent_descriptor_fit_only=True,
            parent_identity_used_for_assay_lookup=False))
        b.emit('v41_replay', **records[-1])
    b.save(v.OUT / 'model_replay.json', dict(status='FINAL_PARENT_AND_ISOLATION_REPLAY_PASSED', signature=v.signature(),
        tasks=records, no_selected_candidate=not bool(records), local_gpu_training=False))


if __name__ == '__main__':
    main()
