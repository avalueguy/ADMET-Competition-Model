"""Label-free molecular denoising pretraining followed by frozen task experts."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '3'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v146_current_baseline as v
import v148_experiment_runner as runner
b, r83 = v.b, v.r83
OUT = ROOT / 'admet_model_outputs_v207'
SEEDS = (2071028, 2071029)
PRETRAIN_STEPS = 6000
CONFIGS = (dict(depth=5, trees=900),)
METHOD = ('Label-free denoising molecular encoder trained from scratch on unique '
          'official training structures, without any endpoint labels. Six count '
          'fingerprint views and physicochemical descriptors,128D bottleneck, '
          'masked weighted reconstruction, two fixed seeds. Frozen256D embeddings '
          'augment target-excluded chemical/context features for two-seed CatBoost '
          'experts. Official validation/test structures only receive inference; '
          'training-structure overlaps disclosed. Original v148 gates/blends, '
          'all21 improvable endpoints, HIA protected, no external weights.')


def sig():
    packages = '|'.join(importlib.metadata.version(p) for p in ('numpy', 'torch', 'catboost'))
    return hashlib.sha256((b.sha(__file__) + v.sig() + b.sha(runner.__file__) + packages).encode()).hexdigest()


def key(c):
    return 'depth' + str(c['depth'])


def configure():
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)


def pretraining_design(d):
    # Pretraining uses structure columns and official split markers; never d['y'].
    unique = np.unique(d['mi'][d['split'] == 0])
    raw = d['views']['physchem'].astype(float)
    fit = raw[unique]
    median = np.array([np.median(z[np.isfinite(z)]) if np.isfinite(z).any() else 0.
                       for z in fit.T])
    clean = np.where(np.isfinite(raw), raw, median)
    center = clean[unique].mean(0)
    scale = clean[unique].std(0).clip(.05)
    active = np.flatnonzero(clean[unique].std(0) > .01)
    desc = np.clip((clean[:, active] - center[active]) / scale[active], -6, 6).astype(np.float32)
    fp = np.log1p(np.maximum(d['views']['multiview'][:, :12288], 0)).clip(0, 4).astype(np.float32)
    frequency = (fp[unique] > 0).mean(0)
    positive_weight = np.sqrt(1 / np.maximum(frequency, 1 / len(unique))).clip(1, 20).astype(np.float32)
    assert np.isfinite(fp).all() and np.isfinite(desc).all() and len(unique) > 100
    state = dict(unique=unique, median=median, center=center, scale=scale, active=active,
                 positive_weight=positive_weight,
                 fingerprint_sha256=hashlib.sha256(fp.tobytes()).hexdigest(),
                 descriptor_sha256=hashlib.sha256(desc.tobytes()).hexdigest())
    return dict(fp=fp, desc=desc, state=state)


class Encoder(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.views = nn.ModuleList(nn.Sequential(nn.Linear(2048, 64), nn.LayerNorm(64), nn.SiLU())
                                   for _ in range(6))
        self.desc = nn.Sequential(nn.Linear(dimensions, 64), nn.LayerNorm(64), nn.SiLU())
        self.encoder = nn.Sequential(nn.Linear(448, 256), nn.LayerNorm(256), nn.SiLU(),
                                      nn.Linear(256, 128), nn.LayerNorm(128), nn.SiLU())
        self.decoder = nn.Sequential(nn.Linear(128, 256), nn.SiLU())
        self.fp_decoder = nn.Linear(256, 12288)
        self.desc_decoder = nn.Linear(256, dimensions)

    def encode(self, fp, desc):
        blocks = [layer(fp[:, j * 2048:(j + 1) * 2048]) for j, layer in enumerate(self.views)]
        return self.encoder(torch.cat(blocks + [self.desc(desc)], 1))

    def forward(self, fp, desc):
        z = self.decoder(self.encode(fp, desc))
        return self.fp_decoder(z), self.desc_decoder(z)


def denoising_loss(net, fp, desc, weights):
    masked = torch.rand_like(fp) < .25
    desc_mask = torch.rand_like(desc) < .25
    f, z = net(fp.masked_fill(masked, 0), desc.masked_fill(desc_mask, 0))
    weight = torch.where(fp > 0, weights[None], 1.)
    # Missing coordinates dominate loss; visible coordinates retain count scale.
    selected = torch.where(masked, 1., .2)
    loss = nn.functional.smooth_l1_loss(f, fp, reduction='none') * weight * selected
    physical = (z - desc).square() * torch.where(desc_mask, 1., .2)
    return loss.sum() / (weight * selected).sum() + .1 * physical.mean()


def encode(net, fp, desc):
    net.eval()
    result = []
    with torch.inference_mode():
        for lo in range(0, len(fp), 512):
            result.append(net.encode(fp[lo:lo + 512], desc[lo:lo + 512]).cpu().numpy())
    return np.concatenate(result)


def pretrained(d):
    folder = OUT / 'pretraining'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'embeddings.npz'
    if path.exists():
        z = np.load(path)
        assert str(z['signature']) == sig()
        for seed in SEEDS:
            assert str(z['model_' + str(seed)]) == b.sha(folder / ('model_' + str(seed) + '.pt'))
        assert z['x'].shape == (len(d['views']['physchem']), 256)
        return z['x'].copy()
    a = pretraining_design(d)
    joblib.dump(a['state'], folder / 'transform.joblib', compress=3)
    fp = torch.as_tensor(a['fp'], device='cuda')
    desc = torch.as_tensor(a['desc'], device='cuda')
    weight = torch.as_tensor(a['state']['positive_weight'], device='cuda')
    unique = a['state']['unique']
    members, hashes = [], {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        net = Encoder(desc.shape[1]).cuda()
        opt = torch.optim.AdamW(net.parameters(), lr=.0008, weight_decay=.01)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, PRETRAIN_STEPS, eta_min=.00003)
        trace, start = [], time.monotonic()
        for step in range(PRETRAIN_STEPS):
            net.train()
            ix = rng.choice(unique, 256, replace=len(unique) < 256)
            loss = denoising_loss(net, fp[ix], desc[ix], weight)
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
            nn.utils.clip_grad_norm_(net.parameters(), 3.)
            opt.step()
            schedule.step()
            trace.append(float(loss.detach()))
            if step % 1000 == 0:
                print(json.dumps(dict(stage='label_free_pretraining', seed=seed, step=step,
                                      loss=trace[-1], seconds=round(time.monotonic() - start))), flush=True)
        assert np.mean(trace[-100:]) < np.mean(trace[:100]), 'Pretraining loss did not decrease'
        raw = encode(net, fp, desc)
        assert np.isfinite(raw).all() and np.min(raw.std(0)) > 1e-5
        file = folder / ('model_' + str(seed) + '.pt')
        torch.save(dict(signature=sig(), dimensions=desc.shape[1], seed=seed,
                         weights={name: p.detach().cpu() for name, p in net.state_dict().items()}), file)
        replay = Encoder(desc.shape[1]).cuda()
        replay.load_state_dict(torch.load(file, weights_only=True)['weights'])
        np.testing.assert_array_equal(raw, encode(replay, fp, desc))
        members.append(raw)
        hashes['model_' + str(seed)] = b.sha(file)
        b.save(folder / ('trace_' + str(seed) + '.json'), dict(loss=trace, signature=sig(),
               fitting_structures=len(unique), endpoint_labels_used=False))
        del net, replay, opt, schedule
    x = np.c_[*members]
    np.savez_compressed(path, signature=sig(), x=x, **hashes)
    del fp, desc, weight
    torch.cuda.empty_cache()
    return x


def verify_pretraining(d):
    from v161_verify_and_assemble import same
    a = pretraining_design(d)
    folder = OUT / 'pretraining'
    same(a['state'], joblib.load(folder / 'transform.joblib'))
    saved = d['y'].copy()
    try:
        d['y'][:] = 999999.
        same(a, pretraining_design(d))
    finally:
        d['y'][:] = saved
    fp = torch.as_tensor(a['fp'], device='cuda')
    desc = torch.as_tensor(a['desc'], device='cuda')
    members, hashes = [], {}
    for seed in SEEDS:
        file = folder / ('model_' + str(seed) + '.pt')
        state = torch.load(file, weights_only=True)
        assert state['signature'] == sig() and state['seed'] == seed
        net = Encoder(state['dimensions']).cuda()
        net.load_state_dict(state['weights'])
        members.append(encode(net, fp, desc))
        hashes[file.name] = b.sha(file)
        del net
    np.testing.assert_array_equal(np.c_[*members], pretrained(d))
    del fp, desc
    torch.cuda.empty_cache()
    b.save(OUT / 'pretraining_independent_verification.json', dict(status='PASS', signature=sig(),
           model_hashes=hashes, all_endpoint_label_perturbation_invariant=True,
           complete_embedding_reload=True, fitting_structures=len(a['state']['unique'])))


def design(d, k, part):
    fit, query, x, qx, source = v.old.features(d, k, part, 'augmented')
    if '_v207_embedding' not in d:
        d['_v207_embedding'] = pretrained(d)
    embedding = d['_v207_embedding']
    x = np.c_[x, embedding[d['mi'][fit]]].astype(np.float32)
    qx = np.c_[qx, embedding[d['mi'][query]]].astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    qx[~np.isfinite(qx)] = np.nan
    metric, y = d['metrics'][k], d['y'][fit]
    center = float(np.median(y)) if metric == 'MAE' else 0.
    scale = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05) if metric == 'MAE' else 1.
    target = ((y - center) / scale if metric == 'MAE' else r83.ranknorm(y) if metric == 'Spearman' else y)
    assert not set(fit) & set(query) and k not in set(d['tid'][source])
    state = dict(fit=fit, query=query, source=source, target=target, center=center, scale=scale,
                 embedding_sha256=b.sha(OUT / 'pretraining/embeddings.npz'))
    return dict(x=x, qx=qx, state=state)


def output(metric, raw, state, task):
    p = expit(raw) if metric in ('AUROC', 'AUPRC') else raw * state['scale'] + state['center']
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
    a = design(d, k, part)
    metric = d['metrics'][k]
    cls = metric in ('AUROC', 'AUPRC')
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(a['state'], folder / 'transform.joblib', compress=3)
    members = []
    for seed in SEEDS:
        net = (CatBoostClassifier if cls else CatBoostRegressor)(
            loss_function='Logloss' if cls else 'MAE' if metric == 'MAE' else 'RMSE',
            iterations=c['trees'], depth=c['depth'], learning_rate=.035, l2_leaf_reg=25,
            random_seed=seed, thread_count=3, verbose=False, allow_writing_files=False,
            bootstrap_type='Bernoulli', subsample=.85, border_count=64, task_type='CPU')
        net.fit(a['x'], a['state']['target'])
        raw = net.predict(a['qx'], prediction_type='RawFormulaVal')
        path = folder / ('model_' + str(seed) + '.cbm')
        net.save_model(str(path))
        reloaded = CatBoostClassifier() if cls else CatBoostRegressor()
        reloaded.load_model(str(path))
        np.testing.assert_array_equal(raw, reloaded.predict(a['qx'], prediction_type='RawFormulaVal'))
        members.append(raw)
    p = output(metric, np.mean(members, axis=0), a['state'], d['tasks'][k])
    assert np.isfinite(p).all()
    np.savez_compressed(file, signature=sig(), fit=a['state']['fit'], query=a['state']['query'], p=p,
                        center=a['state']['center'], scale=a['state']['scale'])
    print(json.dumps(dict(stage='train', task=d['tasks'][k], part=part,
                          seconds=round(time.monotonic() - start))), flush=True)
    return p


def replay(d, k, part, c):
    from v161_verify_and_assemble import same
    a = design(d, k, part)
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    same(a['state'], joblib.load(folder / 'transform.joblib'))
    hashes = {'transform.joblib': b.sha(folder / 'transform.joblib')}
    members = []
    for seed in SEEDS:
        file = folder / ('model_' + str(seed) + '.cbm')
        net = CatBoostClassifier() if d['metrics'][k] in ('AUROC', 'AUPRC') else CatBoostRegressor()
        net.load_model(str(file))
        members.append(net.predict(a['qx'], prediction_type='RawFormulaVal'))
        hashes[file.name] = b.sha(file)
    return output(d['metrics'][k], np.mean(members, axis=0), a['state'], d['tasks'][k]), hashes


def smoke(d):
    from v161_verify_and_assemble import same
    a = pretraining_design(d)
    labels = d['y'].copy()
    try:
        d['y'][:] = 999999.
        same(a, pretraining_design(d))
    finally:
        d['y'][:] = labels
    torch.manual_seed(SEEDS[0])
    net = Encoder(a['desc'].shape[1]).cuda()
    fp = torch.as_tensor(a['fp'][:16], device='cuda')
    desc = torch.as_tensor(a['desc'][:16], device='cuda')
    weights = torch.as_tensor(a['state']['positive_weight'], device='cuda')
    loss = denoising_loss(net, fp, desc, weights)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    torch.manual_seed(SEEDS[0])
    z = denoising_loss(net, fp, desc, weights)
    assert torch.isfinite(z)
    with torch.inference_mode():
        p = net.encode(fp, desc)
        perm = torch.arange(15, -1, -1, device='cuda')
        np.testing.assert_allclose(p.cpu().numpy(), net.encode(fp[perm], desc[perm])[perm].cpu().numpy(), atol=2e-6, rtol=1e-5)
    b.save(OUT / 'smoke.json', dict(status='PASS', signature=sig(),
           all_endpoint_label_perturbation_invariant=True, finite_reconstruction_gradient=True,
           encoder_row_order_invariant=True, pretraining_unique_structures=len(a['state']['unique']),
           official_training_structures_only=True, no_endpoint_labels_used=True))
    del net, fp, desc, weights
    torch.cuda.empty_cache()
    print('SMOKE_PASS', flush=True)


def main():
    configure()
    d = v.load()
    OUT.mkdir(exist_ok=True)
    if '--smoke' in sys.argv:
        smoke(d)
    else:
        assert json.loads((OUT / 'smoke.json').read_text())['signature'] == sig()
        d['_v207_embedding'] = pretrained(d)
        verify_pretraining(d)
        runner.run(sys.modules[__name__], d)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        b.save(OUT / 'failure.json', dict(status='FAILED', traceback=traceback.format_exc()))
        raise
