"""Replay final models, rebuilt features and label-isolation probes on remote GPU."""
import json
import numpy as np
import torch
import admet_v24_models as b
import admet_v33_run as n
import admet_v37_run as v


def main():
    b.remote_only()
    torch.backends.cuda.matmul.allow_tf32 = True
    d = b.build_data()
    chem = v.chemistry(d)
    other = n.other_cache(d, [0, 1])
    report = json.loads((v.OUT / 'analysis.json').read_text())
    assert report['status'] == 'EVALUATED' and report['signature'] == v.signature()
    records = []
    for r in report['tasks']:
        s = r['selection']
        if s is None:
            continue
        k = d['tasks'].index(r['task'])
        rows = np.flatnonzero((d['tid'] == k) & (d['split'] < 2))
        q = np.flatnonzero((d['tid'] == k) & (d['split'] == 2))
        dest = v.OUT / r['task'] / 'final'
        z = np.load(dest / 'predictions.npz')
        assert str(z['signature']) == v.signature()
        np.testing.assert_array_equal(z['fit'], rows)
        np.testing.assert_array_equal(z['query'], q)
        x, tf, audit = v.matrices(d, k, rows, q, [0, 1], other, chem)
        metadata = json.loads((dest / 'metadata.json').read_text())
        assert audit['expanded_fingerprint_columns'] == metadata['expanded_fingerprint_columns']
        for family in s['families']:
            values = []
            for seed in v.P['seeds']:
                model = v.model_type(r['metric'], family)()
                model.load_model(dest / f'family{family}_{seed}.ubj')
                model.set_params(device='cuda', n_jobs=3)
                p = v.predict_model(model, x['query'][family], r['metric'], family, tf)
                values.append(p)
            replay = np.mean(values, axis=0) if r['metric'] == 'MAE' else np.mean([b.rank(a) for a in values], axis=0)
            np.testing.assert_allclose(replay, z['p'][family], rtol=2e-5, atol=2e-5)
        # Other-task evidence must be unchanged even if this entire target's label block is poisoned.
        probe = dict(d)
        probe['y'] = d['y'].copy()
        target_rows = np.flatnonzero(d['tid'] == k)
        probe['y'][target_rows] = 1e9
        ax, _, _ = n.auxiliary(d, [0, 1], k)
        poisoned, _, _ = n.auxiliary(probe, [0, 1], k)
        np.testing.assert_array_equal(ax, poisoned)
        # Poison all query labels and the complete target neighbor block, then rebuild features.
        probe['y'] = d['y'].copy()
        probe['y'][q] = -1e9
        poisoned_other = other.copy()
        poisoned_other[:, k, :] = 1e8
        xp, _, ap = v.matrices(probe, k, rows, q, [0, 1], poisoned_other, chem)
        for name in ['fit', 'query']:
            for family in range(3):
                np.testing.assert_array_equal(x[name][family], xp[name][family])
        assert ap['other_neighbor_target_block_removed'] and ap['own_neighbors_same_scaffold_excluded']
        records.append(dict(task=r['task'], families=s['families'], model_reload_passed=True,
                            target_assay_block_poison_invariant=True, query_label_poison_invariant=True,
                            fit_only_fingerprint_columns_replayed=True))
        b.emit('v37_remote_replay', **records[-1])
    assert records
    b.save(v.OUT / 'model_replay.json', dict(status='FINAL_MODEL_AND_ISOLATION_REPLAY_PASSED',
        signature=v.signature(), tasks=records, local_gpu_training=False))


if __name__ == '__main__':
    main()
