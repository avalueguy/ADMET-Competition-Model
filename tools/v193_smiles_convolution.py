"""Fit-only randomized isomeric SMILES convolution, with frozen confirmation."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from rdkit import Chem
from scipy.special import expit
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import admet_v67_data as chemistry
import v146_current_baseline as v
import v148_experiment_runner as runner
b, r83 = v.b, v.r83
OUT = ROOT / 'admet_model_outputs_v193'
SEEDS = (1934026, 1934027)
CONFIGS = (dict(embedding=48, channels=96, epochs=40, minimum_steps=700, maximum_steps=2400),)
METHOD = ('Endpoint-specific SMILES convolution trained from scratch, four deterministic '
          'isomeric random enumerations per official original molecule. Fit-only token '
          'vocabulary and physchem normalization; masked max/mean sequence pooling; '
          'random enumeration augmentation and four-enumeration prediction averaging. '
          'Two seeds and one fixed architecture/duration rule, all21 improvable tasks. '
          'No measured assays, historical predictions, external data or pretrained weights. '
          'Original v148 frozen development/confirmation gates, no rejection fallback.')
TOKEN = re.compile(r'(\[[^\]]+\]|Br|Cl|%[0-9]{2}|.)')


def sig():
    return hashlib.sha256((b.sha(__file__) + v.sig() + b.sha(runner.__file__) +
                           b.sha(chemistry.__file__) + torch.__version__).encode()).hexdigest()


def key(c):
    return 'e' + str(c['embedding']) + '_c' + str(c['channels']) + '_epochs' + str(c['epochs'])


def initialize():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    assert torch.cuda.is_available(), 'Use the authorized remote GPU instance'


def sequences(d):
    if '_v193_sequences' in d:
        return d['_v193_sequences']
    cache = OUT / 'enumerations.joblib'
    audit = OUT / 'enumerations.json'
    OUT.mkdir(exist_ok=True)
    if cache.exists():
        meta = json.loads(audit.read_text())
        assert meta['signature'] == sig() and meta['sha256'] == b.sha(cache)
        values = joblib.load(cache)
        assert len(values) == len(d['views']['physchem'])
        d['_v193_sequences'] = values
        return values
    originals = [None] * len(d['views']['physchem'])
    cursor = 0
    for task in d['tasks']:
        for part in ('train', 'valid', 'test'):
            records = ([dict(Drug=r['smiles']) for r in d['template'] if r['task'] == task]
                       if part == 'test' else b.table(ROOT / 'admet_competition_data/data_benchmark' / task / (part + '.csv')))
            for row in records:
                index = int(d['mi'][cursor])
                cursor += 1
                assert originals[index] in (None, row['Drug'])
                originals[index] = row['Drug']
    assert cursor == len(d['y']) and all(s is not None for s in originals)
    original_sha = hashlib.sha256(json.dumps(originals).encode()).hexdigest()
    expected = json.loads((ROOT / 'admet_model_outputs_v67/features/official_chemistry.json').read_text())['smiles_sha256']
    assert original_sha == expected
    values, exceptions = [], []
    for i, text in enumerate(originals):
        mol, mode = chemistry.parse_molecule(text)
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        # RNG is local to RDKit's seeded call and never depends on labels.
        alternatives = Chem.MolToRandomSmilesVect(mol, 3, randomSeed=1934026 + i,
                                                 isomericSmiles=True)
        strings = [canonical] + list(alternatives)
        assert len(strings) == 4
        tokenized = [TOKEN.findall(s) for s in strings]
        assert all(''.join(t) == s for t, s in zip(tokenized, strings))
        values.append(tokenized)
        if mode != 'full_sanitization':
            exceptions.append(dict(index=i, mode=mode))
    joblib.dump(values, cache, compress=3)
    b.save(audit, dict(signature=sig(), sha256=b.sha(cache), molecules=len(values),
                       original_structure_sha256=original_sha, enumerations=4,
                       original_graph_exceptions=exceptions, labels_used=False))
    d['_v193_sequences'] = values
    return values


def design(d, k, part):
    values = sequences(d)
    fit, query = v.split(d, k, part)
    unique = np.unique(d['mi'][fit])
    vocabulary = sorted({t for i in unique for seq in values[i] for t in seq})
    vocabulary = {t: i + 2 for i, t in enumerate(vocabulary)}
    raw = d['views']['physchem'].astype(float)
    median = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.
                       for col in raw[unique].T])
    a = np.where(np.isfinite(raw[unique]), raw[unique], median)
    center = a.mean(0)
    scale = a.std(0).clip(.05)
    x = np.clip((np.where(np.isfinite(raw), raw, median) - center) / scale, -6, 6).astype(np.float32)
    metric = d['metrics'][k]
    y = d['y'][fit]
    yc = float(np.median(y)) if metric == 'MAE' else 0.
    ys = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05) if metric == 'MAE' else 1.
    target = ((y - yc) / ys if metric == 'MAE' else
              r83.ranknorm(y) if metric == 'Spearman' else y).astype(np.float32)
    return dict(fit=fit, query=query, vocabulary=vocabulary, median=median,
                feature_center=center, feature_scale=scale, descriptors=x,
                target=target, center=yc, scale=ys)


def metadata(a):
    return {k: x for k, x in a.items() if k != 'descriptors'}


class Net(nn.Module):
    def __init__(self, vocab, desc, c):
        super().__init__()
        h = c['channels']
        self.embedding = nn.Embedding(vocab, c['embedding'], padding_idx=0)
        self.stem = nn.Conv1d(c['embedding'], h, 5, padding=2)
        self.layers = nn.ModuleList(nn.Conv1d(h, h, 3, padding=1) for _ in range(2))
        self.norms = nn.ModuleList(nn.LayerNorm(h) for _ in range(3))
        self.desc = nn.Sequential(nn.Linear(desc, 64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(.15))
        self.head = nn.Sequential(nn.Linear(2 * h + 64, 128), nn.LayerNorm(128),
                                  nn.SiLU(), nn.Dropout(.3), nn.Linear(128, 1))

    def forward(self, tokens, desc):
        mask = tokens.ne(0).unsqueeze(1)
        h = self.embedding(tokens).transpose(1, 2)
        h = torch.nn.functional.silu(self.norms[0](self.stem(h).transpose(1, 2))).transpose(1, 2) * mask
        for layer, norm in zip(self.layers, self.norms[1:]):
            z = torch.nn.functional.silu(norm(layer(h).transpose(1, 2))).transpose(1, 2)
            h = (h + torch.nn.functional.dropout(z, .15, training=self.training)) * mask
        mean = h.sum(2) / mask.sum(2).clamp_min(1)
        maximum = h.masked_fill(~mask, -1e9).amax(2)
        return self.head(torch.cat([mean, maximum, self.desc(desc)], 1)).squeeze(1)


def batch(values, vocabulary, ids, enumerations, device):
    tokenized = [values[int(i)][int(e)] for i, e in zip(ids, enumerations)]
    width = max(len(s) for s in tokenized)
    array = np.zeros((len(ids), width), dtype=np.int64)
    for row, seq in enumerate(tokenized):
        array[row, :len(seq)] = [vocabulary.get(t, 1) for t in seq]
    return torch.as_tensor(array, device=device)


def infer(net, values, a, d, desc, device):
    net.eval()
    allp = []
    with torch.inference_mode():
        for enum in range(4):
            parts = []
            for lo in range(0, len(a['query']), 64):
                ids = d['mi'][a['query'][lo:lo + 64]]
                tokens = batch(values, a['vocabulary'], ids, np.full(len(ids), enum), device)
                parts.append(net(tokens, desc[ids]).cpu().numpy().astype(float))
            allp.append(np.concatenate(parts))
    return np.mean(allp, axis=0)


def objective(metric, p, y):
    if metric in ('AUROC', 'AUPRC'):
        loss = nn.functional.binary_cross_entropy_with_logits(p, y)
        pos, neg = p[y > .5], p[y < .5]
        if len(pos) and len(neg):
            loss = loss + .2 * nn.functional.softplus(neg[:, None] - pos[None]).mean()
    elif metric == 'MAE':
        loss = (p - y).abs().mean()
    else:
        delta = y[:, None] - y[None]
        active = delta.abs() > .1
        loss = .5 * (p - y).square().mean()
        if active.any():
            loss = loss + .3 * nn.functional.softplus(-delta.sign() * (p[:, None] - p[None]))[active].mean()
    return loss + .001 * p.square().mean()


def replay(d, k, part, c):
    a = design(d, k, part)
    values = sequences(d)
    folder = OUT / 'models' / d['tasks'][k] / part / key(c)
    saved = joblib.load(folder / 'transform.joblib')
    from v161_verify_and_assemble import same
    same(metadata(a), saved)
    device = 'cuda'
    desc = torch.as_tensor(a['descriptors'], device=device)
    members, hashes = [], {}
    for seed in SEEDS:
        path = folder / ('model_' + str(seed) + '.pt')
        state = torch.load(path, map_location=device, weights_only=True)
        assert state['signature'] == sig() and state['config'] == c
        net = Net(len(a['vocabulary']) + 2, desc.shape[1], c).to(device)
        net.load_state_dict(state['weights'])
        members.append(infer(net, values, a, d, desc, device))
        hashes[path.name] = b.sha(path)
        del net
    p = np.mean(members, axis=0)
    p = expit(p) if d['metrics'][k] in ('AUROC', 'AUPRC') else p * a['scale'] + a['center']
    if d['tasks'][k] == 'ppbr_az':
        p = p.clip(0, 100)
    return p, hashes


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
    values = sequences(d)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(metadata(a), folder / 'transform.joblib', compress=3)
    device = 'cuda'
    desc = torch.as_tensor(a['descriptors'], device=device)
    target = torch.as_tensor(a['target'], device=device)
    steps = max(c['minimum_steps'], min(c['maximum_steps'], math.ceil(len(a['fit']) / 64) * c['epochs']))
    members = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        net = Net(len(a['vocabulary']) + 2, desc.shape[1], c).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=.0008, weight_decay=.025)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=.00003)
        for step in range(steps):
            net.train()
            ix = rng.choice(len(a['fit']), min(64, len(a['fit'])), replace=False)
            ids = d['mi'][a['fit'][ix]]
            tokens = batch(values, a['vocabulary'], ids, rng.integers(0, 4, len(ix)), device)
            z = objective(d['metrics'][k], net(tokens, desc[ids]), target[ix])
            assert torch.isfinite(z)
            opt.zero_grad(set_to_none=True)
            z.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.)
            opt.step()
            schedule.step()
        raw = infer(net, values, a, d, desc, device)
        path = folder / ('model_' + str(seed) + '.pt')
        torch.save(dict(signature=sig(), config=c, steps=steps,
                        weights={n: z.detach().cpu() for n, z in net.state_dict().items()}), path)
        state = torch.load(path, map_location=device, weights_only=True)
        copy = Net(len(a['vocabulary']) + 2, desc.shape[1], c).to(device)
        copy.load_state_dict(state['weights'])
        np.testing.assert_array_equal(raw, infer(copy, values, a, d, desc, device))
        members.append(raw)
        del net, copy, opt
    p = np.mean(members, axis=0)
    p = expit(p) if d['metrics'][k] in ('AUROC', 'AUPRC') else p * a['scale'] + a['center']
    if d['tasks'][k] == 'ppbr_az':
        p = p.clip(0, 100)
    assert np.isfinite(p).all()
    np.savez_compressed(file, signature=sig(), fit=a['fit'], query=a['query'], p=p,
                        center=a['center'], scale=a['scale'])
    del desc, target
    torch.cuda.empty_cache()
    print(json.dumps(dict(stage='train', task=d['tasks'][k], part=part, steps=steps,
                          seconds=round(time.monotonic() - start))), flush=True)
    return p


def smoke(d):
    from v161_verify_and_assemble import same
    checks = []
    for task in ('herg', 'caco2_wang', 'clearance_hepatocyte_az'):
        k = d['tasks'].index(task)
        a = design(d, k, 'fold0')
        labels = d['y'][a['query']].copy()
        try:
            d['y'][a['query']] = 987654.
            same(a, design(d, k, 'fold0'))
        finally:
            d['y'][a['query']] = labels
        assert not set(a['fit']) & set(a['query'])
        checks.append(dict(task=task, query_label_perturbation_invariant=True,
                           fit_only_vocabulary=True, fit_only_normalization=True))
    torch.manual_seed(SEEDS[0])
    net = Net(len(a['vocabulary']) + 2, a['descriptors'].shape[1], CONFIGS[0]).cuda()
    ids = d['mi'][a['fit'][:7]]
    tokens = batch(sequences(d), a['vocabulary'], ids, np.zeros(len(ids), int), 'cuda')
    desc = torch.as_tensor(a['descriptors'][ids], device='cuda')
    z = objective('Spearman', net(tokens, desc), torch.as_tensor(a['target'][:7], device='cuda'))
    z.backward()
    net.eval()
    with torch.inference_mode():
        before = net(tokens, desc)
        # Padding must not alter a molecule's pooled representation.
        after = net(nn.functional.pad(tokens, (0, 17)), desc)
        torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-6)
    b.save(OUT / 'smoke.json', dict(status='PASS', signature=sig(), checks=checks,
                                  finite_training_gradient=True, padding_invariance=True))
    print('SMOKE_PASS', flush=True)


def main():
    initialize()
    d = v.load()
    smoke(d)
    if '--smoke' not in sys.argv:
        runner.run(sys.modules[__name__], d)


if __name__ == '__main__':
    main()
