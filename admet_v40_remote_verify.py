"""Independent neural checkpoint inference and hidden-label isolation probes."""
import json
import numpy as np
import torch
import admet_v24_models as b
import admet_v33_run as n
import admet_v40_run as v


def main():
    b.remote_only()
    torch.backends.cuda.matmul.allow_tf32 = True
    d = b.build_data()
    chem = v.chemistry_source.chemistry(d)
    other = n.other_cache(d, [0, 1])
    report = json.loads((v.OUT / 'analysis.json').read_text())
    assert report['signature'] == v.signature() and report['status'] == 'EVALUATED'
    records = []
    for r in report['tasks']:
        if r['selection'] is None:
            continue
        k = d['tasks'].index(r['task'])
        rows = np.flatnonzero((d['tid'] == k) & (d['split'] < 2))
        q = np.flatnonzero((d['tid'] == k) & (d['split'] == 2))
        dest = v.OUT / r['task'] / 'final'
        z = np.load(dest / 'predictions.npz')
        assert str(z['signature']) == v.signature()
        np.testing.assert_array_equal(z['fit'], rows)
        np.testing.assert_array_equal(z['query'], q)
        x, pre, ap, target, audit = v.features(d, k, rows, q, [0, 1], other, chem)
        query = [torch.tensor(a, device='cuda') for a in x['query']]
        seeds = []
        for seed in v.P['seeds']:
            snapshots = []
            for epoch in v.P['snapshots']:
                ck = torch.load(dest / f'net_{seed}_{epoch}.pt', map_location='cpu', weights_only=False)
                assert ck['signature'] == v.signature()
                for a, c in [(pre, ck['preprocessor']), (ap, ck['assay_preprocessor'])]:
                    for name in ['med', 'mu', 'sd']:
                        np.testing.assert_array_equal(getattr(a, name), getattr(c, name))
                net = v.ChemicalNet(ck['dims']).cuda().eval()
                net.load_state_dict(ck['state'])
                snapshots.append(v.decode(v.infer(net, query), ck['target'], r['metric']))
                del net
            seeds.append(np.mean(snapshots, axis=0))
        np.testing.assert_allclose(np.mean(seeds, axis=0), z['p'], rtol=2e-5, atol=2e-5)
        probe = dict(d)
        probe['y'] = d['y'].copy()
        probe['y'][d['split'] == 2] = 1e9
        poisoned = other.copy()
        poisoned[:, k, :] = -1e8
        xx, _, _, _, paudit = v.features(probe, k, rows, q, [0, 1], poisoned, chem)
        for j in range(5):
            np.testing.assert_array_equal(x['fit'][j], xx['fit'][j])
            np.testing.assert_array_equal(x['query'][j], xx['query'][j])
        assert paudit['other_neighbor_target_block_removed'] and paudit['own_neighbors_same_scaffold_excluded']
        records.append(dict(task=r['task'], model_reload_passed=True, hidden_target_labels_invariant=True,
            target_auxiliary_block_invariant=True, saved_fit_preprocessors_match=True))
        b.emit('v40_replay', **records[-1])
        del query
        torch.cuda.empty_cache()
    if records:
        # Neural batching and head isolation are bounded numerically under TF32.
        torch.manual_seed(400899)
        dims = [12, 14, 10, 9, 11]
        net = v.ChemicalNet(dims).cuda().eval()
        x = [torch.randn(9, dim, device='cuda') for dim in dims]
        with torch.inference_mode():
            p = net(x).cpu().numpy()
            xx = [a.clone() for a in x]
            xx[4].fill_(12345.)
            paux = net(xx).cpu().numpy()
            pfirst = net([a[:1] for a in x]).cpu().numpy()
        np.testing.assert_array_equal(p[0], paux[0])
        np.testing.assert_allclose(p[:, :1], pfirst, rtol=5e-4, atol=2e-4)
    b.save(v.OUT / 'model_replay.json', dict(status='FINAL_NEURAL_AND_ISOLATION_REPLAY_PASSED', signature=v.signature(),
        tasks=records, structural_head_auxiliary_invariant=bool(records), singleton_tf32_absolute_tolerance=2e-4,
        no_selected_candidate=not bool(records), local_gpu_training=False))


if __name__ == '__main__':
    main()
