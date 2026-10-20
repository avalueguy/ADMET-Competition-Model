"""Independent graph extraction, model reload, and hidden-label isolation replay."""
import json
import numpy as np
import torch
import admet_v24_models as b
import admet_v27_models as graph
import admet_v33_run as n
import admet_v41_run as parent
import admet_v43_run as v


def main():
    b.remote_only()
    torch.backends.cuda.matmul.allow_tf32 = False
    d = b.build_data()
    g = graph.graphs(d)
    chem = parent.chemistry(d)
    other = n.other_cache(d, [0, 1])
    report = json.loads((v.OUT / 'analysis.json').read_text())
    assert report['signature'] == v.signature() and report['status'] == 'EVALUATED'
    selected = [r for r in report['tasks'] if r['selection']]
    if not selected:
        b.save(v.OUT / 'model_replay.json', dict(status='NO_SELECTED_CANDIDATE', signature=v.signature(), tasks=[]))
        return
    latent = v.representations(d, g, 'final')
    # Reconstruct queried embeddings from checkpoints rather than trusting the cache.
    query = np.flatnonzero((d['split'] == 2) & np.isin(d['tid'], [d['tasks'].index(r['task']) for r in selected]))
    molecules = np.unique(d['mi'][query])
    sources = []
    for seed in v.P['graph_seeds']:
        path = graph.OUT / 'final' / str(seed) / 'checkpoint_5500.pt'
        ck = torch.load(path, map_location='cpu', weights_only=False)
        assert ck['signature'] == graph.signature()
        net = graph.BondNet(*ck['dimensions']).cuda().eval()
        net.load_state_dict(ck['state'])
        parts = []
        with torch.inference_mode():
            for start in range(0, len(molecules), 96):
                mm = molecules[start:start+96]
                parts.append(v.latent_forward(net, graph.pack(g, mm)).cpu().numpy())
            # Check permutation, batch, and isolated-molecule invariance.
            mm = molecules[:3]
            a = v.latent_forward(net, graph.pack(g, mm)).cpu().numpy()
            c = v.latent_forward(net, graph.pack(g, mm[::-1])).cpu().numpy()
            q = v.latent_forward(net, graph.pack(g, mm[:1])).cpu().numpy()
        np.testing.assert_allclose(a[::-1], c, rtol=3e-5, atol=3e-5)
        np.testing.assert_allclose(a[:1], q, rtol=3e-5, atol=3e-5)
        sources.append(np.concatenate(parts))
        del net
    np.testing.assert_allclose(latent[molecules], np.concatenate(sources, axis=1), rtol=3e-5, atol=3e-5)
    records = []
    for r in selected:
        k = d['tasks'].index(r['task'])
        rows = np.flatnonzero((d['tid'] == k) & (d['split'] < 2))
        q = np.flatnonzero((d['tid'] == k) & (d['split'] == 2))
        dest = v.OUT / r['task'] / 'final'
        z = np.load(dest / 'predictions.npz')
        assert str(z['signature']) == v.signature()
        np.testing.assert_array_equal(z['fit'], rows)
        np.testing.assert_array_equal(z['query'], q)
        x, target, meta, pre = v.features(d, k, rows, q, [0, 1], other, chem, latent)
        ck = torch.load(dest / 'preprocessing.pt', map_location='cpu', weights_only=False)
        assert ck['signature'] == v.signature()
        for a, c in zip(pre, [ck['dense_pre'], ck['graph_pre']]):
            for name in ['med', 'mu', 'sd']:
                np.testing.assert_array_equal(getattr(a, name), getattr(c, name))
        parts = []
        for family in range(3):
            seeds = []
            for seed, depth, trees in zip(v.P['seeds'], v.P['depths'], v.P['estimators']):
                model = v.make_model(r['metric'], seed, depth, trees)
                model.load_model(dest / f'family{family}_{seed}.ubj')
                model.set_params(device='cuda', n_jobs=3)
                seeds.append(v.predict_model(model, x['query'][family], r['metric'], ck['target']))
            parts.append(np.mean(seeds, axis=0))
        np.testing.assert_allclose(np.stack(parts), z['p'], rtol=2e-6, atol=2e-6)
        probe = dict(d)
        probe['y'] = d['y'].copy()
        probe['y'][d['split'] == 2] = 1e9
        poison = other.copy()
        poison[:, k] = -1e8
        xx, _, _, _ = v.features(probe, k, rows, q, [0, 1], poison, chem, latent)
        for family in range(3):
            np.testing.assert_array_equal(x['fit'][family], xx['fit'][family])
            np.testing.assert_array_equal(x['query'][family], xx['query'][family])
        iso = v.representation_audit(d, rows, q, 'final')
        source = graph.OUT / 'final' / str(v.P['graph_seeds'][0]) / 'predictions.npz'
        assert not set(np.load(source)['fit']) & set(q)
        records.append(dict(task=r['task'], model_reload_passed=True,
            graph_checkpoint_query_extraction_passed=True, graph_query_batch_invariant=True,
            hidden_target_labels_invariant=True, target_auxiliary_block_invariant=True,
            saved_fit_preprocessors_match=True, source_graph_target_test_labels_absent=True))
        b.emit('v43_replay', **records[-1])
    b.save(v.OUT / 'model_replay.json', dict(status='FINAL_GRAPH_AND_BOOST_ISOLATION_REPLAY_PASSED',
        signature=v.signature(), tasks=records, graph_sources_recomputed=True,
        local_gpu_training=False, test_labels_used=False))


if __name__ == '__main__':
    main()
