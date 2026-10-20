"""Remote robust chemical kernel fits complement tree partitions on continuous tasks."""
import hashlib
import json
import time
import numpy as np
import torch
import admet_v24_models as b
import admet_v37_run as v37

ROOT = v37.ROOT
OUT = ROOT / 'admet_model_outputs_v38'
P = json.loads((ROOT / 'admet_v38_protocol.json').read_text())


def signature():
    return hashlib.sha256(json.dumps(dict(base=v37.signature(), files={name: b.sha(ROOT / name) for name in
        ['admet_v38_run.py', 'admet_v38_protocol.json']}), sort_keys=True).encode()).hexdigest()


def tanimoto(a, c):
    aa = torch.tensor(a, device='cuda')
    cc = torch.tensor(c, device='cuda')
    parts = []
    norm = cc.square().sum(1)
    for start in range(0, len(aa), 384):
        x = aa[start:start+384]
        dot = x @ cc.T
        parts.append(dot / (x.square().sum(1)[:, None] + norm[None] - dot).clamp_min(1e-9))
    return torch.cat(parts)


def rbf(a, c):
    aa = torch.tensor(a, device='cuda')
    cc = torch.tensor(c, device='cuda')
    dist = (aa.square().sum(1)[:, None] + cc.square().sum(1)[None] - 2*aa@cc.T).clamp_min(0) / aa.shape[1]
    return .5*torch.exp(-dist) + .5*torch.exp(-dist/4)


def matrices(d, rows, query, chem):
    tr = d['mi'][rows]
    q = d['mi'][query]
    views = [(d['fp2'], .30), (d['fcfp'], .15), (chem[:, :4096], .25),
             (chem[:, 4096:6144], .10), (chem[:, 6144:7168], .10), (chem[:, 7168:], .10)]
    kk = torch.zeros((len(rows), len(rows)), device='cuda')
    qk = torch.zeros((len(query), len(rows)), device='cuda')
    for x, weight in views:
        kk += weight*tanimoto(x[tr], x[tr])
        qk += weight*tanimoto(x[q], x[tr])
    raw = np.concatenate([d['desc'], d['emb']], axis=1)
    pre = b.Preprocessor().fit(raw[np.unique(tr)])
    xx = pre.transform(raw)
    full = [.85*kk + .15*rbf(xx[tr], xx[tr]), .85*qk + .15*rbf(xx[q], xx[tr])]
    # Float64 solver and exact symmetric averaging avoid TF32 factorization sensitivity.
    train = [(kk+kk.T).double()*.5, (full[0]+full[0].T).double()*.5]
    return train, [qk.double(), full[1].double()]


def fit(d, k, rows, query, chem, dest):
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / 'predictions.npz'
    sig = signature()
    if path.exists():
        z = np.load(path)
        assert str(z['signature']) == sig
        np.testing.assert_array_equal(z['fit'], rows)
        np.testing.assert_array_equal(z['query'], query)
        return z['p'].copy()
    assert not set(rows) & set(query) and np.all(d['split'][rows] < 2)
    if np.any(d['split'][query] < 2):
        assert not set(d['rg'][rows]) & set(d['rg'][query])
    t0 = time.time()
    metric = d['metrics'][k]
    tf = b.Target(d['y'][rows], metric)
    y = torch.tensor(tf.encode(d['y'][rows]), dtype=torch.float64, device='cuda')
    kernels, queries = matrices(d, rows, query, chem)
    _, inv, count = np.unique(d['canonical'][d['mi'][rows]], return_inverse=True, return_counts=True)
    base = torch.tensor(1/np.sqrt(count[inv]), dtype=torch.float64, device='cuda')
    base /= base.mean()
    predictions = []
    coefficients = []
    for family, (kernel, qk) in enumerate(zip(kernels, queries)):
        values = []
        for alpha in P['alphas']:
            weights = base.clone()
            for iteration in range(P['irls_iterations'] if metric == 'MAE' else 1):
                system = kernel.clone()
                system.diagonal().add_(alpha / weights.clamp_min(.05))
                coeff = torch.linalg.solve(system, y)
                if metric == 'MAE':
                    residual = kernel @ coeff - y
                    weights = base / torch.sqrt(residual.square() + P['mae_smoothing']**2)
                    weights /= weights.mean()
                    weights.clamp_(.05, 20.)
            pred = (qk@coeff).cpu().numpy()
            values.append(tf.decode(pred) if metric == 'MAE' else pred)
            coefficients.append(coeff.cpu().numpy())
        predictions.append(np.mean(values, axis=0))
        b.emit('v38_fit_family', task=d['tasks'][k], family=family, partition=str(dest.relative_to(OUT)), seconds=round(time.time()-t0))
    p = np.stack(predictions)
    assert np.isfinite(p).all()
    np.savez_compressed(dest / 'coefficients.npz', a=np.stack(coefficients), fit=rows, signature=sig)
    np.savez_compressed(path, p=p, fit=rows, query=query, signature=sig)
    b.save(dest / 'metadata.json', dict(signature=sig, task=d['tasks'][k], fit_rows=len(rows), query_rows=len(query),
        seconds=time.time()-t0, target_transform_fit_only=True, dense_preprocessor_fit_only=True,
        test_labels_used=False, other_task_labels_used=False, local_gpu_training=False))
    del kernels, queries
    torch.cuda.empty_cache()
    return p


def variants(p, metric):
    x = p.copy() if metric == 'MAE' else np.stack([b.rank(a) for a in p])
    return np.concatenate([p, np.mean(x, axis=0)[None]], axis=0)


def main():
    b.remote_only()
    torch.backends.cuda.matmul.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)
    d = b.build_data()
    chem = v37.chemistry(d)
    base, old = v37.baseline(d)
    folds = np.load(b.OUT / 'folds.npy')
    records = []
    for task in P['tasks']:
        k = d['tasks'].index(task)
        m = d['metrics'][k]
        tr = np.flatnonzero((d['tid'] == k) & (d['split'] == 0))
        vi = np.flatnonzero((d['tid'] == k) & (d['split'] == 1))
        vp = variants(fit(d, k, tr, vi, chem, OUT / task / 'valid'), m)
        screen = max(b.delta(m, d['y'][vi], base[vi], b.blend(m, base[vi], p, w)) for p in vp for w in P['weights'])
        r = dict(task=task, metric=m, valid_baseline=b.metric(m, d['y'][vi], base[vi]), screen_gain=screen, selection=None, grid=[])
        if screen > .0002:
            cp = np.full((3, len(d['y'])), np.nan)
            for fold in range(3):
                q = tr[folds[tr] == fold]
                rows = tr[folds[tr] != fold]
                cp[:, q] = variants(fit(d, k, rows, q, chem, OUT / task / f'fold{fold}'), m)
            np.save(OUT / task / 'cv_predictions.npy', cp)
            grid = []
            for v in range(3):
                for w in P['weights']:
                    gains = []
                    for fold in range(3):
                        ix = tr[folds[tr] == fold]
                        gains.append(b.delta(m, d['y'][ix], old[ix], b.blend(m, old[ix], cp[v, ix], w)))
                    vg = b.delta(m, d['y'][vi], base[vi], b.blend(m, base[vi], vp[v], w))
                    obj = float(.45*vg + .55*np.mean(gains) - .04*np.std(gains))
                    grid.append(dict(variant=v, weight=w, valid_gain=vg, cv_gains=gains, cv_mean=float(np.mean(gains)), objective=obj,
                                     eligible=bool(vg > .0002 and np.mean(gains) > 0 and obj > .0005)))
            eligible = [g for g in grid if g['eligible']]
            r.update(grid=grid, selection=max(eligible, key=lambda a: a['objective']) if eligible else None)
        records.append(r)
        b.save(OUT / 'analysis.json', dict(signature=signature(), status='IN_PROGRESS', tasks=records))
        b.emit('v38_selection', task=task, screen=screen, selection=r['selection'])
    b.save(OUT / 'analysis.json', dict(signature=signature(), status='EVALUATED', baseline='v36', baseline_score=P['baseline_score'],
        tasks=records, selected_tasks=[r['task'] for r in records if r['selection']], official_score_prediction=None, limitations=P['limitations']))
    for r in records:
        if r['selection'] is None:
            continue
        k = d['tasks'].index(r['task'])
        rows = np.flatnonzero((d['tid'] == k) & (d['split'] < 2))
        q = np.flatnonzero((d['tid'] == k) & (d['split'] == 2))
        fit(d, k, rows, q, chem, OUT / r['task'] / 'final')
    b.save(OUT / 'completed.json', dict(signature=signature(), completed=True))


if __name__ == '__main__':
    main()
