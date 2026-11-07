"""Clamped molecular-neighborhood diffusion with held-out labels excluded."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '3'
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v146_current_baseline as v
import v148_experiment_runner as runner
b, r83 = v.b, v.r83
OUT = ROOT / 'admet_model_outputs_v204'
CONFIGS = (dict(alpha=.8), dict(alpha=.95))
NEIGHBORS = 32
METHOD = ('Clamped label diffusion on a symmetric32-neighbor molecular graph. '
          'Six count-Tanimoto views plus fit-scaled augmented molecular/context '
          'RBF; fourth-power affinities. Fitting labels are boundary values, '
          'query labels are absent. Query structures jointly define unlabeled '
          'graph nodes; restart prior and target scaling use fitting labels only. '
          'Two fixed diffusion strengths, unchanged v148 gates/blends, all21 '
          'improvable tasks. No pretrained weights or historical predictions.')


def sig():
    packages = '|'.join(importlib.metadata.version(p) for p in ('numpy', 'scipy', 'torch'))
    return hashlib.sha256((b.sha(__file__) + v.sig() + b.sha(runner.__file__) + packages).encode()).hexdigest()


def key(c):
    return 'alpha' + str(c['alpha'])


def digest(matrix):
    h = hashlib.sha256()
    for value in (np.array(matrix.shape, np.int64), matrix.indptr, matrix.indices, matrix.data):
        h.update(value.tobytes())
    return h.hexdigest()


def transition(fp, desc):
    """Create graph using structures/covariates only, independently of labels."""
    count = len(fp)
    take = min(NEIGHBORS, count - 1)
    assert take > 0 and fp.shape[1] == 12288
    chem = torch.as_tensor(fp, dtype=torch.float32, device='cuda')
    physical = torch.as_tensor(desc, dtype=torch.float32, device='cuda')
    rows, columns, values = [], [], []
    physical_norm = physical.square().sum(1)
    for lo in range(0, count, 256):
        hi = min(count, lo + 256)
        affinity = torch.zeros((hi - lo, count), dtype=torch.float32, device='cuda')
        for start in range(0, 12288, 2048):
            ref = chem[:, start:start + 2048]
            x = ref[lo:hi]
            dot = x @ ref.T
            union = x.square().sum(1)[:, None] + ref.square().sum(1)[None] - dot
            affinity += (dot / union.clamp_min(1e-8)).clamp(0, 1) * (.75 / 6)
        x = physical[lo:hi]
        distance = (physical_norm[lo:hi, None] + physical_norm[None] - 2 * x @ physical.T).clamp_min(0)
        affinity += .25 * torch.exp(-distance / max(1, physical.shape[1]))
        block = affinity.cpu().numpy().astype(float)
        block[np.arange(hi - lo), np.arange(lo, hi)] = -1.
        # Full stable sorting gives deterministic index-order tie breaking.
        nearest = np.argsort(-block, axis=1, kind='stable')[:, :take]
        weight = np.take_along_axis(block, nearest, axis=1).clip(0, 1) ** 4
        rows.append(np.repeat(np.arange(lo, hi), take))
        columns.append(nearest.ravel())
        values.append(weight.ravel())
    graph = sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                              shape=(count, count)).tocsr()
    graph = ((graph + graph.T) * .5).tocsr()
    graph.eliminate_zeros()
    graph.sort_indices()
    degree = np.asarray(graph.sum(1)).ravel()
    inverse = np.divide(1., degree, out=np.zeros_like(degree), where=degree > 0)
    markov = (sparse.diags(inverse) @ graph).tocsr()
    markov.sort_indices()
    assert np.isfinite(markov.data).all() and np.min(markov.data, initial=0) >= 0
    np.testing.assert_allclose(np.asarray(markov.sum(1)).ravel()[degree > 0], 1., atol=1e-12, rtol=0)
    del chem, physical
    torch.cuda.empty_cache()
    return markov, degree


def design(d, k, part):
    fit, query, raw, qr, source = v.old.features(d, k, part, 'augmented')
    raw, qr = raw.astype(float), qr.astype(float)
    median = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.
                       for col in raw.T])
    x = np.where(np.isfinite(raw), raw, median)
    center, scale = x.mean(0), x.std(0).clip(.05)
    active = np.flatnonzero(x.std(0) > .01)
    qx = np.where(np.isfinite(qr), qr, median)
    # Missing indicators are additional fit-defined coordinates.
    missing = np.flatnonzero((~np.isfinite(raw)).std(0) > .01)
    desc = np.c_[np.clip((np.r_[x[:, active], qx[:, active]] - center[active]) / scale[active], -6, 6),
                 (~np.isfinite(np.r_[raw[:, missing], qr[:, missing]])).astype(float)].astype(np.float32)
    rows = np.r_[fit, query]
    fp = d['views']['multiview'][d['mi'][rows], :12288]
    assert np.isfinite(fp).all() and np.min(fp) >= 0 and np.isfinite(desc).all()
    assert not set(fit) & set(query) and k not in set(d['tid'][source])
    if part != 'final':
        assert not set(d['groups'][fit]) & set(d['groups'][query])
    metric = d['metrics'][k]
    y = d['y'][fit]
    yc = float(np.median(y)) if metric == 'MAE' else 0.
    ys = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05) if metric == 'MAE' else 1.
    target = ((y - yc) / ys if metric == 'MAE' else r83.ranknorm(y) if metric == 'Spearman' else y)
    prior = float(np.median(target)) if metric == 'MAE' else float(target.mean())
    graph, degree = transition(fp, desc)
    state = dict(fit=fit, query=query, source=source, median=median,
                 feature_center=center, feature_scale=scale, active=active,
                 missing=missing, center=yc, scale=ys, target=target, prior=prior,
                 degree=degree, graph_sha256=digest(graph))
    return dict(graph=graph, state=state)


def solve_graph(graph, target, prior, alpha):
    n = len(target)
    count = graph.shape[0] - n
    assert count > 0 and 0 < alpha < 1
    system = sparse.eye(count, format='csc') - alpha * graph[n:, n:].tocsc()
    # Isolated nodes receive the fit-only prior, preserving the maximum principle.
    row_mass = np.asarray(graph[n:].sum(1)).ravel()
    rhs = alpha * (graph[n:, :n] @ target) + (1 - alpha * row_mass) * prior
    value = np.asarray(spsolve(system, rhs))
    residual = float(np.max(np.abs(system @ value - rhs)))
    assert np.isfinite(value).all() and residual < 1e-10 * (1 + np.max(np.abs(rhs)))
    assert value.min() >= min(target.min(), prior) - 1e-9
    assert value.max() <= max(target.max(), prior) + 1e-9
    return value, residual


def output(metric, raw, state, task):
    p = raw * state['scale'] + state['center']
    if metric in ('AUROC', 'AUPRC'):
        p = p.clip(0, 1)
    return p.clip(0, 100) if task == 'ppbr_az' else p


def predict(d, k, part, c):
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    file = folder / 'predictions.npz'
    if file.exists():
        z = np.load(file)
        assert str(z['signature']) == sig()
        np.testing.assert_array_equal(z['query'], v.split(d, k, part)[1])
        return z['p'].copy()
    start = time.monotonic()
    # Keep only the last fitting graph; two fixed alphas share structural work.
    cache = (k, part, sig())
    if d.get('_v204_design_key') != cache:
        d['_v204_design'] = design(d, k, part)
        d['_v204_design_key'] = cache
    a = d['_v204_design']
    state = a['state']
    raw, residual = solve_graph(a['graph'], state['target'], state['prior'], c['alpha'])
    p = output(d['metrics'][k], raw, state, d['tasks'][k])
    assert len(np.unique(p)) > 1, 'Degenerate diffusion expert'
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, folder / 'transform.joblib', compress=3)
    joblib.dump(dict(signature=sig(), config=c, transition=a['graph'],
                     target=state['target'], prior=state['prior']), folder / 'model.joblib', compress=3)
    loaded = joblib.load(folder / 'model.joblib')
    reloaded, _ = solve_graph(loaded['transition'], loaded['target'], loaded['prior'], loaded['config']['alpha'])
    np.testing.assert_array_equal(raw, reloaded)
    np.savez_compressed(file, signature=sig(), fit=state['fit'], query=state['query'], p=p,
                        source=state['source'], center=state['center'], scale=state['scale'])
    print(json.dumps(dict(stage='train', task=d['tasks'][k], part=part, config=c,
                          graph_edges=a['graph'].nnz, solve_residual=residual,
                          seconds=round(time.monotonic() - start))), flush=True)
    return p


def replay(d, k, part, c):
    from v161_verify_and_assemble import same
    a = design(d, k, part)
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    same(a['state'], joblib.load(folder / 'transform.joblib'))
    model = joblib.load(folder / 'model.joblib')
    assert model['signature'] == sig() and model['config'] == c
    same(a['graph'], model['transition'])
    same(a['state']['target'], model['target'])
    assert a['state']['prior'] == model['prior']
    p, _ = solve_graph(model['transition'], model['target'], model['prior'], c['alpha'])
    return output(d['metrics'][k], p, a['state'], d['tasks'][k]), {
        name: b.sha(folder / name) for name in ('model.joblib', 'transform.joblib')}


def configure():
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')


def smoke(d):
    from v161_verify_and_assemble import same
    # Closed-form two-query example, including unlabeled-to-unlabeled flow.
    graph = sparse.csr_matrix([[0., .4, .6], [.3, 0., .7], [.5, .5, 0.]])
    target = np.array([1.])
    z, residual = solve_graph(graph, target, .25, .8)
    system = np.eye(2) - .8 * graph.toarray()[1:, 1:]
    rhs = .8 * graph.toarray()[1:, :1] @ target + .2 * .25
    np.testing.assert_allclose(z, np.linalg.solve(system, rhs), atol=1e-14, rtol=0)
    isolated, _ = solve_graph(sparse.csr_matrix((3, 3)), target, .25, .95)
    np.testing.assert_array_equal(isolated, [.25, .25])
    checks = []
    for task in ('herg', 'caco2_wang', 'clearance_hepatocyte_az'):
        k = d['tasks'].index(task)
        a = design(d, k, 'fold0')
        query = a['state']['query']
        labels = d['y'][query].copy()
        try:
            d['y'][query] = 999999.
            d.pop('_v96_assays', None)
            same(a, design(d, k, 'fold0'))
        finally:
            d['y'][query] = labels
            d.pop('_v96_assays', None)
        p, error = solve_graph(a['graph'], a['state']['target'], a['state']['prior'], .95)
        assert np.isfinite(p).all() and len(np.unique(p)) > 1
        checks.append(dict(task=task, query_label_perturbation_invariant=True,
                           graph_edges=a['graph'].nnz, maximum_principle=True,
                           linear_solve_residual=error))
    b.save(OUT / 'smoke.json', dict(status='PASS', signature=sig(), checks=checks,
           dense_solve_reference_match=True, isolated_node_prior=True,
           only_fit_labels_as_boundary=True, query_structures_used_without_labels=True))
    print('SMOKE_PASS', flush=True)


def main():
    configure()
    d = v.load()
    OUT.mkdir(exist_ok=True)
    if '--smoke' in sys.argv:
        smoke(d)
    else:
        assert json.loads((OUT / 'smoke.json').read_text())['signature'] == sig()
        runner.run(sys.modules[__name__], d)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        b.save(OUT / 'failure.json', dict(status='FAILED', traceback=traceback.format_exc()))
        raise
