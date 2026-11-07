"""Fit-only conditional logistic densities; median of the ensemble for MAE."""
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
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v146_current_baseline as v
import v148_experiment_runner as runner
b, r83 = v.b, v.r83
OUT = ROOT / 'admet_model_outputs_v244'
TASKS = ('caco2_wang', 'ppbr_az', 'lipophilicity_astrazeneca')
SEEDS = (2441029, 2441030)
CONFIGS = (dict(components=3, epochs=120, minimum_steps=1000, maximum_steps=3500),)
METHOD = ('Conditional heteroscedastic three-logistic mixture density experts for '
          'three MAE tasks with viable original scaffold confirmation. Each of two '
          'fixed-seed networks learns probabilities, locations and scales by '
          'negative log likelihood. The delivered expert is the median of the '
          'six-component ensemble CDF, the MAE decision rule; individual medians '
          'are never averaged. No historical predictions enter training. Fit-only '
          'chemistry/context transforms, from-scratch models, original v148 '
          'development/confirmation/blend gates unchanged; other19 tasks protected.')


def initialize():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)


def sig():
    return hashlib.sha256((b.sha(__file__) + v.sig() + b.sha(runner.__file__) +
                           importlib.metadata.version('torch')).encode()).hexdigest()


def key(c):
    return 'conditional_logistic_mixture'


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
    assert d['metrics'][k] == 'MAE'
    fit, query, x, qx, source = v.old.features(d, k, part, 'augmented')
    raw = np.r_[x, qx].astype(np.float64)
    med = np.array([np.median(z[np.isfinite(z)]) if np.isfinite(z).any() else 0.
                    for z in raw[:len(fit)].T])
    clean = np.where(np.isfinite(raw), raw, med)
    center, scale = clean[:len(fit)].mean(0), clean[:len(fit)].std(0).clip(.05)
    z = np.clip((clean-center)/scale, -6., 6.).astype(np.float32)
    width = d['views']['physchem'].shape[1] + d['conformer_features'].shape[1] + d['pharmacophore'].shape[1]
    physical = np.c_[z[:, -width:], (~np.isfinite(raw[:, -width:])).astype(np.float32)]
    assay = np.c_[z[:, :-width], (~np.isfinite(raw[:, :-width])).astype(np.float32)]
    rows = np.r_[fit, query]
    fp = d['views']['multiview'][d['mi'][rows], :12288].reshape(len(rows), 6, 2048).copy()
    fp /= np.linalg.norm(fp, axis=2, keepdims=True).clip(1e-6)
    yc = float(np.median(d['y'][fit]))
    ys = max(float(np.subtract(*np.quantile(d['y'][fit], [.75, .25]))), .05)
    state = dict(fit=fit, query=query, source=source, median=med,
                 feature_center=center, feature_scale=scale, physical_width=width,
                 center=yc, scale=ys, target=((d['y'][fit]-yc)/ys).astype(np.float32))
    assert not set(fit) & set(query)
    assert np.all(d['tid'][source] != k) and np.all(d['split'][source] < (2 if part == 'final' else 1))
    if part != 'final':
        assert not set(d['groups'][fit]) & set(d['groups'][query])
    assert all(np.isfinite(a).all() for a in (fp, physical, assay, state['target']))
    return dict(fp=fp.astype(np.float32), physical=physical.astype(np.float32),
                assay=assay.astype(np.float32), state=state)


class Net(nn.Module):
    def __init__(self, physical_dim, assay_dim, c):
        super().__init__()
        self.views = nn.ModuleList(nn.Sequential(nn.Linear(2048, 24), nn.LayerNorm(24), nn.SiLU()) for _ in range(6))
        self.physical = nn.Sequential(nn.Linear(physical_dim, 96), nn.LayerNorm(96), nn.SiLU(), nn.Dropout(.1))
        self.assay = nn.Sequential(nn.Linear(assay_dim, 32), nn.LayerNorm(32), nn.SiLU())
        self.body = nn.Sequential(nn.Linear(272, 128), nn.LayerNorm(128), nn.SiLU(),
                                  nn.Dropout(.1), nn.Linear(128, 64), nn.SiLU())
        self.head = nn.Linear(64, 3*c['components'])
        self.components = c['components']
        with torch.no_grad():
            self.head.weight.mul_(.05)
            self.head.bias.zero_()
            self.head.bias[c['components']:2*c['components']] = torch.linspace(-.75, .75, c['components'])
            self.head.bias[2*c['components']:] = -.7

    def forward(self, fp, physical, assay):
        x = torch.cat([*[layer(fp[:, j]) for j, layer in enumerate(self.views)],
                       self.physical(physical), self.assay(assay)], dim=1)
        raw = self.head(self.body(x))
        logits, locations, raw_scales = raw.chunk(3, dim=1)
        return logits, locations, .03 + F.softplus(raw_scales)


def objective(params, y):
    logits, locations, scales = params
    z = (y[:, None]-locations)/scales
    logpdf = F.log_softmax(logits, dim=1)-scales.log()-F.softplus(z)-F.softplus(-z)
    # Weak fixed scale penalty prevents variance explosion on sparse chemistry.
    return -torch.logsumexp(logpdf, dim=1).mean() + .0001*scales.square().mean()


def parameters_cpu(net, a, positions=None):
    net = net.cpu().double().eval()
    if positions is None:
        positions = np.arange(len(a['state']['fit']), len(a['fp']))
    parts = []
    with torch.inference_mode():
        for lo in range(0, len(positions), 128):
            ix = positions[lo:lo+128]
            args = [torch.as_tensor(a[name][ix], dtype=torch.float64) for name in ('fp', 'physical', 'assay')]
            logits, locations, scales = net(*args)
            parts.append(np.stack((logits.softmax(1).numpy(), locations.numpy(), scales.numpy()), axis=1))
    result = np.concatenate(parts)
    assert np.isfinite(result).all() and (result[:, 2] > 0).all()
    return result


def median_of_densities(members):
    assert members
    weights = np.concatenate([x[:, 0]/len(members) for x in members], axis=1)
    locations = np.concatenate([x[:, 1] for x in members], axis=1)
    scales = np.concatenate([x[:, 2] for x in members], axis=1)
    np.testing.assert_allclose(weights.sum(1), 1., rtol=0, atol=1e-12)
    lo = np.min(locations-40*scales, axis=1)
    hi = np.max(locations+40*scales, axis=1)
    for _ in range(64):
        mid = .5*(lo+hi)
        cdf = (weights*expit((mid[:, None]-locations)/scales)).sum(1)
        lo, hi = np.where(cdf < .5, mid, lo), np.where(cdf >= .5, mid, hi)
    p = .5*(lo+hi)
    np.testing.assert_allclose((weights*expit((p[:, None]-locations)/scales)).sum(1), .5, rtol=0, atol=1e-12)
    return p


def make_loaded(path, a, c):
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert state['signature'] == sig() and state['config'] == c
    net = Net(a['physical'].shape[1], a['assay'].shape[1], c)
    net.load_state_dict(state['weights'])
    return net


def replay(d, k, part, c):
    a = design(d, k, part)
    folder = OUT/'models'/d['tasks'][k]/part/key(c)
    same(a['state'], joblib.load(folder/'transform.joblib'))
    hashes = {'transform.joblib': b.sha(folder/'transform.joblib')}
    members = []
    for seed in SEEDS:
        path = folder/('model_'+str(seed)+'.pt')
        members.append(parameters_cpu(make_loaded(path, a, c), a))
        hashes[path.name] = b.sha(path)
    p = median_of_densities(members)*a['state']['scale']+a['state']['center']
    return p, hashes


def predict(d, k, part, c):
    if d['tasks'][k] not in TASKS:
        return v.baseline(d, k, part)
    folder = OUT/'models'/d['tasks'][k]/part/key(c)
    file = folder/'predictions.npz'
    if file.exists():
        z = np.load(file)
        assert str(z['signature']) == sig()
        np.testing.assert_array_equal(z['query'], v.split(d, k, part)[1])
        return z['p'].copy()
    assert torch.cuda.is_available()
    started = time.monotonic()
    a = design(d, k, part)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(a['state'], folder/'transform.joblib', compress=3)
    n = len(a['state']['fit'])
    arrays = [torch.as_tensor(a[name][:n], device='cuda') for name in ('fp', 'physical', 'assay')]
    y = torch.as_tensor(a['state']['target'], device='cuda')
    steps = max(c['minimum_steps'], min(c['maximum_steps'], math.ceil(n/128)*c['epochs']))
    members = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        net = Net(a['physical'].shape[1], a['assay'].shape[1], c).cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=.0008, weight_decay=.02)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=.00003)
        for step in range(steps):
            net.train()
            ix = rng.choice(n, min(128, n), replace=False)
            loss = objective(net(*(raw[ix] for raw in arrays)), y[ix])
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.)
            opt.step()
            schedule.step()
            if step % 500 == 0:
                print(json.dumps(dict(stage='updates', task=d['tasks'][k], part=part,
                    seed=seed, step=step, steps=steps, nll=float(loss.detach()))), flush=True)
        path = folder/('model_'+str(seed)+'.pt')
        torch.save(dict(signature=sig(), config=c, steps=steps,
                   weights={name: x.detach().cpu() for name, x in net.state_dict().items()}), path)
        expected = parameters_cpu(net, a)
        actual = parameters_cpu(make_loaded(path, a, c), a)
        np.testing.assert_array_equal(expected, actual)
        members.append(actual)
        del net, opt
    p = median_of_densities(members)*a['state']['scale']+a['state']['center']
    assert np.isfinite(p).all()
    np.savez_compressed(file, signature=sig(), fit=a['state']['fit'], query=a['state']['query'], p=p)
    del arrays, y
    torch.cuda.empty_cache()
    print(json.dumps(dict(stage='train', task=d['tasks'][k], part=part,
        steps_per_seed=steps, seconds=round(time.monotonic()-started))), flush=True)
    return p


def smoke(d):
    from tempfile import TemporaryDirectory
    from scipy.optimize import brentq
    c = CONFIGS[0]
    reference = np.array([[[.2, .3, .5], [-2., 0., 3.], [.4, 1., .7]]])
    median = median_of_densities([reference])
    expected = brentq(lambda t: float((reference[0, 0]*expit((t-reference[0, 1])/reference[0, 2])).sum()-.5), -40., 40.)
    np.testing.assert_allclose(median, expected, rtol=0, atol=1e-12)
    shifted = reference.copy()
    shifted[:, 1] = 5.+2.*shifted[:, 1]
    shifted[:, 2] *= 2.
    np.testing.assert_allclose(median_of_densities([shifted]), 5.+2.*median, rtol=0, atol=1e-12)
    checked = []
    for task in TASKS:
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
        loss = objective(net(*args), torch.as_tensor(a['state']['target'][:32], device='cuda'))
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        with torch.no_grad():
            for p in net.parameters():
                if p.grad is not None:
                    p.add_(p.grad, alpha=-1e-4)
        ix = np.arange(len(a['state']['fit']), len(a['state']['fit'])+8)
        raw = parameters_cpu(net, a, ix)
        np.testing.assert_allclose(raw, parameters_cpu(net, a, ix[::-1])[::-1], rtol=0, atol=1e-12)
        with TemporaryDirectory(dir=OUT) as temp:
            path = Path(temp)/'model.pt'
            torch.save(dict(signature=sig(), config=c,
                weights={name: x.detach().float().cpu() for name, x in net.state_dict().items()}), path)
            restored = make_loaded(path, a, c)
            expected = parameters_cpu(copy.deepcopy(restored), a, ix)
            np.testing.assert_array_equal(expected, parameters_cpu(make_loaded(path, a, c), a, ix))
            assert np.isfinite(median_of_densities([expected, expected])).all()
        checked.append(task)
    b.save(OUT/'smoke.json', dict(status='PASS', signature=sig(), query_label_isolation=True,
        actual_cuda_training_step=True, finite_density_gradients=True,
        cdf_median_independent_reference=True, affine_equivariance=True,
        independent_reload_tasks=checked, prediction_row_order=True,
        deterministic_cpu_float64_inference=True, no_query_scores_used=True))
    print('SMOKE_PASS', flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    d = load()
    if '--smoke' in sys.argv:
        smoke(d)
    else:
        assert json.loads((OUT/'smoke.json').read_text())['signature'] == sig()
        runner.run(sys.modules[__name__], d)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        b.save(OUT/'failure.json', dict(status='FAILED', traceback=traceback.format_exc(), retry_permitted=False))
        raise
