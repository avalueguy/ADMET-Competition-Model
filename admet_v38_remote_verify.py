"""Rebuild kernels, reload final coefficients and probe hidden-label invariance."""
import json
import numpy as np
import torch
import admet_v24_models as b
import admet_v37_run as v37
import admet_v38_run as v


def main():
    b.remote_only()
    torch.backends.cuda.matmul.allow_tf32 = False
    d = b.build_data()
    chem = v37.chemistry(d)
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
        c = np.load(dest / 'coefficients.npz')
        assert str(z['signature']) == str(c['signature']) == v.signature()
        np.testing.assert_array_equal(z['fit'], rows)
        np.testing.assert_array_equal(z['query'], q)
        train, queries = v.matrices(d, rows, q, chem)
        tf = b.Target(d['y'][rows], r['metric'])
        parts = []
        for family in range(2):
            values = []
            for j in range(len(v.P['alphas'])):
                a = torch.tensor(c['a'][family*len(v.P['alphas'])+j], device='cuda')
                p = (queries[family]@a).cpu().numpy()
                values.append(tf.decode(p) if r['metric'] == 'MAE' else p)
            parts.append(np.mean(values, axis=0))
        np.testing.assert_allclose(np.stack(parts), z['p'], rtol=1e-8, atol=1e-8)
        probe = dict(d)
        probe['y'] = d['y'].copy()
        probe['y'][d['split'] == 2] = 1e9
        _, pq = v.matrices(probe, rows, q[:7], chem)
        for a, c in zip(queries, pq):
            np.testing.assert_allclose(a[:7].cpu().numpy(), c.cpu().numpy(), rtol=2e-6, atol=2e-6)
        records.append(dict(task=r['task'], model_reload_passed=True, hidden_label_poison_invariant=True))
        b.emit('v38_replay', **records[-1])
        del train, queries, pq
        torch.cuda.empty_cache()
    b.save(v.OUT / 'model_replay.json', dict(status='FINAL_KERNEL_REPLAY_PASSED', signature=v.signature(),
                                            tasks=records, local_gpu_training=False))


if __name__ == '__main__':
    main()
