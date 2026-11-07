"""Endpoint-specific attention between chemical views, an incremental v145 expert."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import copy
import hashlib
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v146_current_baseline as v
import v148_experiment_runner as runner
b, r83 = v.b, v.r83
OUT = ROOT / 'admet_model_outputs_v238'
TASKS = ('herg', 'bbb_martins', 'ames', 'clearance_hepatocyte_az',
         'solubility_aqsoldb', 'pgp_broccatelli')
SEEDS = (2381029, 2381030)
CONFIGS = (dict(width=64, heads=4, layers=2, epochs=100, minimum_steps=900, maximum_steps=3000),)
METHOD = ('Endpoint-isolated molecular-view Transformer from scratch. Six count '
          'fingerprint tokens, physical/shape/pharmacophore token and target-excluded '
          'measured-assay token interact through two four-head attention layers '
          'and a learned readout token. Fixed two seeds and step count, no early '
          'stopping/query scoring, no old models or prediction inputs. Six lowest '
          'official v145 endpoints only, other controls identity. Final blend with '
          'v145 and original v148 frozen selection/confirmation unchanged.')


def initialize():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)


def sig():
    return hashlib.sha256((b.sha(__file__) + v.sig() + b.sha(runner.__file__) +
                           importlib.metadata.version('torch')).encode()).hexdigest()


def key(c):
    return 'molecular_view_attention'


def load():
    initialize()
    return v.load()


def same(a, z):
    if isinstance(a, dict):
        assert set(a) == set(z)
        for name in a:
            same(a[name], z[name])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(z)
        for x, y in zip(a, z):
            same(x, y)
    else:
        np.testing.assert_array_equal(a, z)


def design(d, k, part):
    fit, query, x, qx, source = v.old.features(d, k, part, 'augmented')
    raw = np.r_[x, qx].astype(np.float64)
    finite = np.isfinite(raw[:len(fit)])
    med = np.array([np.median(z[np.isfinite(z)]) if np.isfinite(z).any() else 0. for z in raw[:len(fit)].T])
    clean = np.where(np.isfinite(raw), raw, med)
    center = clean[:len(fit)].mean(0)
    scale = clean[:len(fit)].std(0).clip(.05)
    z = np.clip((clean - center) / scale, -5., 5.).astype(np.float32)
    width = d['views']['physchem'].shape[1] + d['conformer_features'].shape[1] + d['pharmacophore'].shape[1]
    assert 0 < width < z.shape[1]
    physical = np.c_[z[:, -width:], (~np.isfinite(raw[:, -width:])).astype(np.float32)]
    assay = np.c_[z[:, :-width], (~np.isfinite(raw[:, :-width])).astype(np.float32)]
    rows = np.r_[fit, query]
    fp = np.sqrt(np.maximum(d['views']['multiview'][d['mi'][rows], :12288], 0.)).reshape(len(rows), 6, 2048).astype(np.float32)
    # Per-molecule normalization has no learned query statistics.
    fp /= np.linalg.norm(fp, axis=2, keepdims=True).clip(1e-6)
    y, metric = d['y'][fit], d['metrics'][k]
    yc = float(np.median(y)) if metric == 'MAE' else 0.
    ys = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05) if metric == 'MAE' else 1.
    target = (y - yc) / ys if metric == 'MAE' else r83.ranknorm(y) if metric == 'Spearman' else y
    assert np.all(d['tid'][source] != k) and np.all(d['split'][source] < (2 if part == 'final' else 1))
    assert not set(fit) & set(query)
    if part != 'final':
        assert not set(d['groups'][fit]) & set(d['groups'][query])
    state = dict(fit=fit, query=query, source=source, median=med, feature_center=center,
                 feature_scale=scale, physical_width=width, center=yc, scale=ys,
                 target=target.astype(np.float32))
    return dict(fp=fp, physical=physical.astype(np.float32), assay=assay.astype(np.float32), state=state)


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=.15, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(.15),
                                nn.Linear(width * 2, width), nn.Dropout(.15))

    def forward(self, x):
        z = self.norm1(x)
        update, _ = self.attention(z, z, z, need_weights=True)
        x = x + update
        return x + self.ff(self.norm2(x))


class Net(nn.Module):
    def __init__(self, physical_dim, assay_dim, c):
        super().__init__()
        width = c['width']
        self.views = nn.ModuleList(nn.Sequential(nn.Linear(2048, width), nn.LayerNorm(width), nn.GELU()) for _ in range(6))
        self.physical = nn.Sequential(nn.Linear(physical_dim, width), nn.LayerNorm(width), nn.GELU())
        self.assay = nn.Sequential(nn.Linear(assay_dim, width), nn.LayerNorm(width), nn.GELU())
        self.readout = nn.Parameter(torch.zeros(1, 1, width))
        self.view_identity = nn.Parameter(torch.randn(1, 9, width) * .02)
        self.blocks = nn.Sequential(*(Block(width, c['heads']) for _ in range(c['layers'])))
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Dropout(.2), nn.Linear(64, 1))

    def forward(self, fp, physical, assay):
        tokens = [self.readout.expand(len(fp), -1, -1)]
        tokens.extend(layer(fp[:, j]).unsqueeze(1) for j, layer in enumerate(self.views))
        tokens.extend((self.physical(physical).unsqueeze(1), self.assay(assay).unsqueeze(1)))
        x = torch.cat(tokens, dim=1) + self.view_identity
        return self.head(self.blocks(x)[:, 0]).squeeze(1)


def objective(metric, p, y):
    if metric in ('AUROC', 'AUPRC'):
        value = nn.functional.binary_cross_entropy_with_logits(p, y)
        pos, neg = p[y > .5], p[y < .5]
        if len(pos) and len(neg):
            value = value + .2 * nn.functional.softplus(neg[:, None] - pos[None]).mean()
    elif metric == 'MAE':
        value = (p - y).abs().mean()
    else:
        delta = y[:, None] - y[None]
        keep = delta.abs() > .1
        value = .5 * (p - y).square().mean()
        if keep.any():
            value = value + .3 * nn.functional.softplus(-delta.sign() * (p[:, None] - p[None]))[keep].mean()
    return value + .001 * p.square().mean()


def infer_cpu(net, a, positions=None):
    # Fixed CPU float64 path avoids GPU atomic/order changes in delivered values.
    net = net.cpu().double().eval()
    if positions is None:
        positions = np.arange(len(a['state']['fit']), len(a['fp']))
    parts = []
    with torch.inference_mode():
        for lo in range(0, len(positions), 128):
            ix = positions[lo:lo + 128]
            args = [torch.as_tensor(a[name][ix], dtype=torch.float64) for name in ('fp', 'physical', 'assay')]
            parts.append(net(*args).numpy())
    return np.concatenate(parts)


def make_loaded(path, a, c):
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert state['signature'] == sig() and state['config'] == c
    net = Net(a['physical'].shape[1], a['assay'].shape[1], c)
    net.load_state_dict(state['weights'])
    return net


def output(raw, a, metric):
    return expit(raw) if metric in ('AUROC', 'AUPRC') else raw * a['scale'] + a['center']


def replay(d, k, part, c):
    a = design(d, k, part)
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    same(a['state'], joblib.load(folder / 'transform.joblib'))
    hashes = {'transform.joblib': b.sha(folder / 'transform.joblib')}
    members = []
    for seed in SEEDS:
        path = folder / ('model_' + str(seed) + '.pt')
        members.append(infer_cpu(make_loaded(path, a, c), a))
        hashes[path.name] = b.sha(path)
    return output(np.mean(members, axis=0), a['state'], d['metrics'][k]), hashes


def predict(d, k, part, c):
    if d['tasks'][k] not in TASKS:
        return v.baseline(d, k, part)
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    file = folder / 'predictions.npz'
    if file.exists():
        z = np.load(file)
        assert str(z['signature']) == sig()
        np.testing.assert_array_equal(z['query'], v.split(d, k, part)[1])
        return z['p'].copy()
    started = time.monotonic()
    a = design(d, k, part)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(a['state'], folder / 'transform.joblib', compress=3)
    assert torch.cuda.is_available()
    arrays = [torch.as_tensor(a[name][:len(a['state']['fit'])], device='cuda') for name in ('fp', 'physical', 'assay')]
    target = torch.as_tensor(a['state']['target'], device='cuda')
    steps = max(c['minimum_steps'], min(c['maximum_steps'], math.ceil(len(target) / 128) * c['epochs']))
    members = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        net = Net(a['physical'].shape[1], a['assay'].shape[1], c).cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=.0005, weight_decay=.03)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=.00002)
        for step in range(steps):
            net.train()
            ix = rng.choice(len(target), min(128, len(target)), replace=False)
            p = net(*(raw[ix] for raw in arrays))
            loss = objective(d['metrics'][k], p, target[ix])
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.)
            opt.step()
            schedule.step()
            if step % 500 == 0:
                print(json.dumps(dict(stage='updates', task=d['tasks'][k], part=part,
                                      seed=seed, step=step, steps=steps, loss=float(loss.detach()))), flush=True)
        path = folder / ('model_' + str(seed) + '.pt')
        torch.save(dict(signature=sig(), config=c, steps=steps,
                        weights={name: x.detach().cpu() for name, x in net.state_dict().items()}), path)
        expected = infer_cpu(net, a)
        actual = infer_cpu(make_loaded(path, a, c), a)
        np.testing.assert_array_equal(expected, actual)
        members.append(actual)
        del net, opt
    p = output(np.mean(members, axis=0), a['state'], d['metrics'][k])
    assert np.isfinite(p).all()
    np.savez_compressed(file, signature=sig(), fit=a['state']['fit'], query=a['state']['query'], p=p)
    del arrays, target
    torch.cuda.empty_cache()
    print(json.dumps(dict(stage='train', task=d['tasks'][k], part=part,
                          steps_per_seed=steps, seconds=round(time.monotonic() - started))), flush=True)
    return p


def smoke(d):
    from tempfile import TemporaryDirectory
    checked = []
    c = CONFIGS[0]
    for task in ('herg', 'solubility_aqsoldb', 'clearance_hepatocyte_az'):
        k = d['tasks'].index(task)
        a = design(d, k, 'fold0')
        outside = np.flatnonzero((d['tid'] == k) & ~np.isin(np.arange(len(d['y'])), a['state']['fit']))
        labels = d['y'][outside].copy()
        try:
            d['y'][outside] = 999999.
            d.pop('_v96_assays', None)
            same(a, design(d, k, 'fold0'))
        finally:
            d['y'][outside] = labels
            d.pop('_v96_assays', None)
        torch.manual_seed(SEEDS[0])
        net = Net(a['physical'].shape[1], a['assay'].shape[1], c).cuda()
        args = [torch.as_tensor(a[name][:32], device='cuda') for name in ('fp', 'physical', 'assay')]
        loss = objective(d['metrics'][k], net(*args), torch.as_tensor(a['state']['target'][:32], device='cuda'))
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        for p in net.parameters():
            if p.grad is not None:
                p.data.add_(p.grad, alpha=-1e-4)
        ix = np.arange(len(a['state']['fit']), len(a['state']['fit']) + 8)
        raw = infer_cpu(net, a, ix)
        np.testing.assert_allclose(raw, infer_cpu(net, a, ix[::-1])[::-1], atol=1e-12, rtol=0)
        with TemporaryDirectory(dir=OUT) as temp:
            path = Path(temp) / 'model.pt'
            # Save float32 training weights, as predict does before float64 inference.
            weights = {name: x.detach().float().cpu() for name, x in net.state_dict().items()}
            torch.save(dict(signature=sig(), config=c, weights=weights), path)
            restored = make_loaded(path, a, c)
            expected = infer_cpu(copy.deepcopy(restored), a, ix)
            np.testing.assert_array_equal(expected, infer_cpu(make_loaded(path, a, c), a, ix))
        checked.append(task)
    b.save(OUT / 'smoke.json', dict(status='PASS', signature=sig(), query_label_isolation=True,
           finite_attention_gradients=True, actual_cuda_training_step=True,
           independent_reload_tasks=checked, prediction_row_order=True,
           deterministic_cpu_float64_inference=True, no_query_scores_used=True))
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
        b.save(OUT / 'failure.json', dict(status='FAILED', traceback=traceback.format_exc(), retry_permitted=False))
        raise
