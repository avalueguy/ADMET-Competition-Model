"""Supervised low-rank pairwise interactions between unfolded molecular fragments."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'
import hashlib
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from scipy.special import expit
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v104_sparse_features as fragments
import v146_current_baseline as v
import v148_experiment_runner as runner
b, r83 = v.b, v.r83
OUT = ROOT / 'admet_model_outputs_v221'
SEEDS = (2211029, 2211030)
CONFIGS = (dict(rank=32, epochs=100),)
METHOD = ('Factorization machine learns explicit low-rank pairwise fragment/fragment '
          'and fragment/context interactions, with an independently fitted linear term. '
          'Two unfolded circular fragment vocabularies; fit-only prevalence selection, '
          'IDF and molecular row normalization. Fit-scaled augmented chemistry and '
          'target-excluded measured context. No old predictions, pretrained weights, '
          'tree models or normalizedv210 features. Two fixed seeds and one fixed '
          'configuration; originalv148 selection and confirmation unchanged.')
torch.set_num_threads(4)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sig():
    return hashlib.sha256((b.sha(__file__) + v.sig() + fragments.sig() +
                           b.sha(runner.__file__) + importlib.metadata.version('torch')).encode()).hexdigest()


def key(c):
    return 'rank' + str(c['rank']) + '_epochs' + str(c['epochs'])


def load():
    d = v.load()
    audit = json.loads((fragments.OUT / 'audit.json').read_text())
    assert audit['schema']['signature'] == fragments.sig()
    raw = []
    for j in range(2):
        path = fragments.OUT / ('view' + str(j) + '.npz')
        vocab = fragments.OUT / ('vocabulary' + str(j) + '.npy')
        assert b.sha(path) == audit['views'][j]['sha256']
        assert b.sha(vocab) == audit['views'][j]['vocabulary_sha256']
        raw.append(sparse.load_npz(path))
    d['v221_fragments'] = raw
    return d


def design(d, k, part):
    fit, query, x, qx, source = v.old.features(d, k, part, 'augmented')
    blocks, queries, states = [], [], []
    for raw in d['v221_fragments']:
        a = raw[d['mi'][fit]].astype(np.float64)
        z = raw[d['mi'][query]].astype(np.float64)
        freq = np.asarray((a > 0).sum(0)).ravel()
        cols = np.flatnonzero(freq >= 3)
        if len(cols) > 30000:
            cols = np.sort(cols[np.argsort(freq[cols], kind='stable')[-30000:]])
        a, z = a[:, cols].tocsr(), z[:, cols].tocsr()
        idf = np.log((1. + len(fit)) / (1. + freq[cols])) + 1.
        for mat in (a, z):
            mat.data = np.log1p(mat.data)
        a = a.multiply(idf).tocsr()
        z = z.multiply(idf).tocsr()
        for mat in (a, z):
            norm = np.sqrt(np.asarray(mat.multiply(mat).sum(1)).ravel()).clip(1e-12)
            mat.data /= np.repeat(norm * np.sqrt(2.), np.diff(mat.indptr))
        blocks.append(a)
        queries.append(z)
        states.append(dict(columns=cols, idf=idf))
    x, qx = x.astype(np.float64), qx.astype(np.float64)
    median = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.
                       for col in x.T])
    clean = np.where(np.isfinite(x), x, median)
    center, scale = clean.mean(0), clean.std(0).clip(.05)
    def dense(mat):
        clean = np.where(np.isfinite(mat), mat, median)
        return np.c_[np.clip((clean - center) / scale, -5., 5.),
                     ~np.isfinite(mat)] / np.sqrt(mat.shape[1])
    blocks.append(sparse.csr_matrix(dense(x)))
    queries.append(sparse.csr_matrix(dense(qx)))
    xx = sparse.hstack(blocks, format='csr', dtype=np.float32)
    qq = sparse.hstack(queries, format='csr', dtype=np.float32)
    xx.eliminate_zeros()
    qq.eliminate_zeros()
    y, metric = d['y'][fit], d['metrics'][k]
    target_center = float(np.median(y)) if metric == 'MAE' else 0.
    target_scale = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05) if metric == 'MAE' else 1.
    target = ((y - target_center) / target_scale if metric == 'MAE' else
              r83.ranknorm(y) if metric == 'Spearman' else y)
    assert not set(fit) & set(query) and np.all(d['tid'][source] != k)
    assert np.isfinite(xx.data).all() and np.isfinite(qq.data).all()
    state = dict(fit=fit, query=query, source=source, fragment_states=states,
                 median=median, center=center, scale=scale, target=target,
                 target_center=target_center, target_scale=target_scale,
                 width=xx.shape[1], fragment_audit_sha256=b.sha(fragments.OUT / 'audit.json'))
    return dict(x=xx, qx=qq, state=state)


class PairwiseFM(nn.Module):
    def __init__(self, width, rank):
        super().__init__()
        # Last column is the linear coefficient, all preceding columns are factors.
        self.weights = nn.Parameter(torch.empty(width, rank + 1))
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.weights[:, :-1], std=.1)
        nn.init.zeros_(self.weights[:, -1])

    def forward(self, x):
        pooled = torch.sparse.mm(x, self.weights)
        square = torch.sparse_coo_tensor(x.indices(), x.values().square(), x.shape,
                                        device=x.device, is_coalesced=True)
        diagonal = torch.sparse.mm(square, self.weights[:, :-1].square())
        return self.bias + pooled[:, -1] + .5 * (pooled[:, :-1].square() - diagonal).sum(1)


def tensor(x, device=DEVICE):
    coo = x.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64)).to(device)
    values = torch.from_numpy(coo.data.astype(np.float32)).to(device)
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device).coalesce()


def inference(net, x):
    # GPU sparse reductions use atomic summation; CPU gives exactly replayable
    # outputs from the same saved tensors without weakening reload assertions.
    previous = next(net.parameters()).device
    net.to('cpu')
    net.eval()
    with torch.no_grad():
        result = np.concatenate([net(tensor(x[i:i + 512], 'cpu')).numpy()
                                 for i in range(0, x.shape[0], 512)]).astype(float)
    net.to(previous)
    return result


def train(x, target, metric, c, seed, task, part):
    torch.manual_seed(seed)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    net = PairwiseFM(x.shape[1], c['rank']).to(DEVICE)
    initial = (float(np.log((target.mean() + 1e-5) / (1. - target.mean() + 1e-5)))
               if metric in ('AUROC', 'AUPRC') else float(target.mean()))
    with torch.no_grad():
        net.bias.fill_(initial)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.005, weight_decay=.05)
    log = []
    for epoch in range(c['epochs']):
        # Fixed cosine schedule and fixed epochs; no query-based early stopping.
        rate = .005 * (.1 + .9 * .5 * (1. + math.cos(math.pi * epoch / c['epochs'])))
        for group in optimizer.param_groups:
            group['lr'] = rate
        order = rng.permutation(len(target))
        total = 0.
        net.train()
        for first in range(0, len(target), 256):
            ix = order[first:first + 256]
            raw = tensor(x[ix])
            # Independent feature dropout, kept unbiased, on fitting rows only.
            keep = (torch.rand_like(raw.values()) >= .1).to(raw.values().dtype) / .9
            dropped = torch.sparse_coo_tensor(raw.indices(), raw.values() * keep,
                                              raw.shape, device=DEVICE, is_coalesced=True)
            y = torch.as_tensor(target[ix], dtype=torch.float32, device=DEVICE)
            prediction = net(dropped)
            loss = (F.binary_cross_entropy_with_logits(prediction, y)
                    if metric in ('AUROC', 'AUPRC') else
                    F.smooth_l1_loss(prediction, y, beta=.1) if metric == 'MAE' else
                    F.mse_loss(prediction, y))
            assert torch.isfinite(loss), (task, part, seed, epoch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(ix)
        if epoch % 25 == 0 or epoch == c['epochs'] - 1:
            row = dict(epoch=epoch + 1, fit_loss=total / len(target), gradient_norm=float(norm))
            log.append(row)
            print(json.dumps(dict(stage='fit', task=task, part=part, seed=seed, **row)), flush=True)
    return net, log


def reload(path, width, rank):
    net = PairwiseFM(width, rank).to(DEVICE)
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    net.load_state_dict(state)
    return net


def output(metric, raw, state, task):
    p = expit(raw) if metric in ('AUROC', 'AUPRC') else raw * state['target_scale'] + state['target_center']
    return p.clip(0, 100) if task == 'ppbr_az' else p


def replay(d, k, part, c):
    from v161_verify_and_assemble import same
    a = design(d, k, part)
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    same(a['state'], joblib.load(folder / 'transform.joblib'))
    hashes = {'transform.joblib': b.sha(folder / 'transform.joblib')}
    members = []
    for seed in SEEDS:
        path = folder / ('model_' + str(seed) + '.pt')
        net = reload(path, a['state']['width'], c['rank'])
        members.append(inference(net, a['qx']))
        hashes[path.name] = b.sha(path)
    return output(d['metrics'][k], np.mean(members, axis=0), a['state'], d['tasks'][k]), hashes


def predict(d, k, part, c):
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    file = folder / 'predictions.npz'
    if file.exists():
        z = np.load(file)
        assert str(z['signature']) == sig()
        np.testing.assert_array_equal(z['query'], v.split(d, k, part)[1])
        return z['p'].copy()
    start = time.monotonic()
    a = design(d, k, part)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(a['state'], folder / 'transform.joblib', compress=3)
    members, histories = [], []
    for seed in SEEDS:
        net, history = train(a['x'], a['state']['target'], d['metrics'][k], c,
                             seed, d['tasks'][k], part)
        raw = inference(net, a['qx'])
        path = folder / ('model_' + str(seed) + '.pt')
        torch.save({name: value.cpu() for name, value in net.state_dict().items()}, path)
        reloaded = reload(path, a['state']['width'], c['rank'])
        np.testing.assert_array_equal(raw, inference(reloaded, a['qx']))
        members.append(raw)
        histories.append(dict(seed=seed, history=history))
    p = output(d['metrics'][k], np.mean(members, axis=0), a['state'], d['tasks'][k])
    assert np.isfinite(p).all()
    np.savez_compressed(file, signature=sig(), fit=a['state']['fit'], query=a['state']['query'], p=p)
    b.save(folder / 'training.json', dict(task=d['tasks'][k], part=part, histories=histories,
           device=str(DEVICE), seconds=round(time.monotonic() - start), no_query_scores_used=True))
    print(json.dumps(dict(stage='train', task=d['tasks'][k], part=part,
                          seconds=round(time.monotonic() - start))), flush=True)
    return p


def smoke(d):
    from tempfile import TemporaryDirectory
    from v161_verify_and_assemble import same
    torch.manual_seed(SEEDS[0])
    net = PairwiseFM(5, 3).to(DEVICE)
    reference = np.array([[1., 2., 0., -1., .3], [0., 1., 1., 0., .2]], np.float32)
    observed = net(tensor(sparse.csr_matrix(reference))).detach().cpu().numpy()
    weights = net.weights.detach().cpu().numpy()
    expected = reference @ weights[:, -1]
    for i in range(5):
        for j in range(i + 1, 5):
            expected += reference[:, i] * reference[:, j] * np.dot(weights[i, :-1], weights[j, :-1])
    np.testing.assert_allclose(observed, expected, atol=1e-6, rtol=1e-6)
    checked = []
    for task in ('herg', 'solubility_aqsoldb', 'half_life_obach'):
        k = d['tasks'].index(task)
        a = design(d, k, 'fold0')
        labels = d['y'][a['state']['query']].copy()
        try:
            d['y'][a['state']['query']] = 999999.
            d.pop('_v96_assays', None)
            z = design(d, k, 'fold0')
            same(a['state'], z['state'])
            assert (a['x'] != z['x']).nnz == (a['qx'] != z['qx']).nnz == 0
        finally:
            d['y'][a['state']['query']] = labels
            d.pop('_v96_assays', None)
        c = dict(rank=8, epochs=2)
        n = min(96, a['x'].shape[0])
        net, _ = train(a['x'][:n], a['state']['target'][:n], d['metrics'][k], c, SEEDS[0], task, 'smoke')
        p = inference(net, a['qx'][:8])
        np.testing.assert_allclose(p, inference(net, a['qx'][:8][::-1])[::-1], atol=1e-6, rtol=1e-6)
        with TemporaryDirectory(dir=OUT) as temp:
            path = Path(temp) / 'model.pt'
            torch.save({name: value.cpu() for name, value in net.state_dict().items()}, path)
            np.testing.assert_array_equal(p, inference(reload(path, a['state']['width'], c['rank']), a['qx'][:8]))
        checked.append(task)
    b.save(OUT / 'smoke.json', dict(status='PASS', signature=sig(), query_label_isolation=True,
           analytical_pairwise_reference=True, finite_gradients=True,
           independent_reload_tasks=checked, prediction_row_order=True, no_query_scores_used=True))
    print('SMOKE_PASS', flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    d = load()
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
