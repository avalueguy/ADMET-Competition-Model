"""Official-only, jointly trained directed graph/fingerprint task experts.

Whole query scaffold groups are absent from *every* task's training labels.
Fixed-duration training and checkpoints never inspect outer validation labels.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '4'
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.special import expit, ndtri
from scipy.stats import rankdata
from rdkit import Chem, rdBase, RDLogger
import admet_v77_ranking as data
import admet_v67_data as chemistry

ROOT = data.ROOT
OUT = ROOT / 'admet_model_outputs_v79'
SEED = 792026
STEPS = 4500
SNAPSHOTS = (3500, 4500)
BATCH = 128
HIDDEN = 128
VARS = ('graph', 'fingerprint', 'hybrid', 'task_gate')


def signature():
    return hashlib.sha256((data.signature() + data.sha(__file__)).encode()).hexdigest()


def onehot(value, choices):
    return [float(value == x) for x in choices] + [float(value not in choices)]


def atom(a):
    return (onehot(a.GetAtomicNum(), [1,5,6,7,8,9,11,12,14,15,16,17,19,20,26,30,35,53])
        + onehot(a.GetTotalDegree(), list(range(6))) + onehot(a.GetFormalCharge(), [-2,-1,0,1,2])
        + onehot(int(a.GetChiralTag()), list(range(4))) + onehot(a.GetTotalNumHs(), list(range(5)))
        + onehot(int(a.GetHybridization()), [2,3,4,6,7])
        + [float(a.GetIsAromatic()), float(a.IsInRing()), a.GetMass() * .01])


def bond(e):
    return ([float(e.GetBondType() == x) for x in [Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
             Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]]
        + [float(e.GetIsConjugated()), float(e.IsInRing())] + onehot(int(e.GetStereo()), list(range(6))))


def graphs(d):
    folder = OUT / 'features'
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / 'graphs.pt'
    if file.exists():
        saved = torch.load(file, weights_only=False)
        assert saved['source_signature'] == signature()
        return saved['graphs']
    structures = [None] * len(d['views']['physchem'])
    cursor = 0
    for task in d['tasks']:
        for part in ('train', 'valid', 'test'):
            rows = ([{'Drug': r['smiles']} for r in d['template'] if r['task'] == task] if part == 'test'
                else data.table(ROOT / 'admet_competition_data/data_benchmark' / task / (part + '.csv')))
            for row in rows:
                index = int(d['mi'][cursor]); cursor += 1
                assert structures[index] in (None, row['Drug'])
                structures[index] = row['Drug']
    assert cursor == len(d['y']) and all(s is not None for s in structures)
    expected = json.loads((ROOT / 'admet_model_outputs_v67/features/official_chemistry.json').read_text())['smiles_sha256']
    assert hashlib.sha256(json.dumps(structures).encode()).hexdigest() == expected
    RDLogger.DisableLog('rdApp.*')
    gs, exceptions = [], []
    for index, sm in enumerate(structures):
        mol, mode = chemistry.parse_molecule(sm)
        if mode != 'full_sanitization':
            exceptions.append(dict(index=index, mode=mode))
        aa = np.asarray([atom(a) for a in mol.GetAtoms()], dtype=np.float32)
        ee, ss, tt, reverse = [], [], [], []
        for edge in mol.GetBonds():
            j = len(ss)
            ss.extend([edge.GetBeginAtomIdx(), edge.GetEndAtomIdx()])
            tt.extend([edge.GetEndAtomIdx(), edge.GetBeginAtomIdx()])
            reverse.extend([j + 1, j]); ee.extend([bond(edge)] * 2)
        gs.append((aa, np.asarray(ee, np.float32).reshape(-1, 13), np.asarray(ss, np.int64),
                   np.asarray(tt, np.int64), np.asarray(reverse, np.int64)))
        if index % 10000 == 0:
            data.emit(stage='v79_graphs', done=index, total=len(structures))
    torch.save(dict(source_signature=signature(), graphs=gs), file)
    data.save(folder / 'audit.json', dict(rdkit_actual=rdBase.rdkitVersion, source_signature=signature(),
        graph_sha256=data.sha(file), molecules=len(gs), exceptions=exceptions,
        original_atoms_bonds_retained=True, external_data=False, pretrained_weights=False))
    return gs


def pack(gs, mols):
    aa, ee, ss, tt, rr, counts = [], [], [], [], [], []
    ao = eo = 0
    for index in mols:
        a, e, s, t, r = gs[int(index)]
        aa.append(a); ee.append(e); ss.append(s + ao); tt.append(t + ao); rr.append(r + eo)
        counts.append(len(a)); ao += len(a); eo += len(e)
    arrays = [np.concatenate(aa), np.concatenate(ee), np.concatenate(ss), np.concatenate(tt),
              np.concatenate(rr), np.repeat(np.arange(len(mols)), counts), np.asarray(counts, np.float32)]
    return [torch.from_numpy(x).cuda() for x in arrays]


class Head(nn.Module):
    def __init__(self, dim, tasks):
        super().__init__()
        self.w = nn.Parameter(torch.randn(tasks, dim, 32) * dim ** -.5)
        self.bias = nn.Parameter(torch.zeros(tasks, 32))
        self.out = nn.Parameter(torch.randn(tasks, 32) * 32 ** -.5)
        self.offset = nn.Parameter(torch.zeros(tasks))

    def forward(self, x):
        z = torch.einsum('bi,kih->bkh', x, self.w)
        z = nn.functional.dropout(nn.functional.silu(z + self.bias), .15, self.training)
        return (z * self.out).sum(-1) + self.offset


class Net(nn.Module):
    def __init__(self, desc_dim, tasks):
        super().__init__()
        h = HIDDEN
        def block(a, b):
            return nn.Sequential(nn.Linear(a, b), nn.LayerNorm(b), nn.SiLU(), nn.Dropout(.15))
        self.edge = nn.Linear(52 + 13, h)
        self.message = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(3)])
        self.atom = nn.Linear(52 + h, h)
        self.pool = block(2 * h + 1, h)
        self.desc = block(desc_dim, 96)
        self.fp = nn.Sequential(block(4096 + 96, 256), block(256, h))
        self.hybrid = nn.Sequential(block(2 * h + 96, 256), block(256, h))
        self.heads = nn.ModuleList([Head(h, tasks) for _ in range(3)])
        self.task_gate = nn.Parameter(torch.zeros(tasks, 3))
        self.molecule_gate = nn.Linear(2 * h, 3)

    def forward(self, graph, fp, desc):
        a, e, s, t, r, mi, count = graph
        h0 = nn.functional.silu(self.edge(torch.cat([a[s], e], 1)))
        h = h0
        for layer in self.message:
            incoming = h.new_zeros((len(a), HIDDEN)); incoming.index_add_(0, t, h)
            h = nn.functional.dropout(nn.functional.silu(h0 + layer(incoming[s] - h[r])), .12, self.training)
        incoming = h.new_zeros((len(a), HIDDEN)); incoming.index_add_(0, t, h)
        av = nn.functional.silu(self.atom(torch.cat([a, incoming], 1)))
        pool = av.new_zeros((len(count), HIDDEN)); pool.index_add_(0, mi, av)
        gh = self.pool(torch.cat([pool / count[:, None], pool / count.sqrt()[:, None], count.log1p()[:, None]], 1))
        dh = self.desc(desc)
        fh = self.fp(torch.cat([nn.functional.dropout(fp, .04, self.training), dh], 1))
        hh = self.hybrid(torch.cat([gh, fh, dh], 1))
        outputs = torch.stack([head(x) for head, x in zip(self.heads, (gh, fh, hh))])
        gate = (self.task_gate[None] + self.molecule_gate(torch.cat([gh, fh], 1))[:, None]).softmax(-1)
        merged = (outputs.permute(1, 2, 0) * gate).sum(-1)
        return torch.cat([outputs, merged[None]], 0)


def rows(d, part):
    if part == 'final':
        fit = np.flatnonzero(d['split'] < 2); q = np.flatnonzero(d['split'] == 2)
    elif part == 'valid':
        q = np.flatnonzero(d['split'] == 1)
        fit = np.flatnonzero((d['split'] == 0) & ~np.isin(d['groups'], d['groups'][q]))
    else:
        q = np.flatnonzero((d['split'] == 0) & (d['folds'] == int(part[-1])))
        fit = np.flatnonzero((d['split'] == 0) & ~np.isin(d['groups'], d['groups'][q]))
    assert not set(fit) & set(q) and np.all(d['split'][fit] < 2)
    if part != 'final':
        assert not set(d['groups'][fit]) & set(d['groups'][q])
        assert not set(d['canonical'][fit]) & set(d['canonical'][q])
    return fit, q


def targets(d, fit):
    n = len(d['views']['physchem']); tasks = len(d['tasks'])
    labels = np.zeros((n, tasks), np.float32)
    mask = np.zeros_like(labels, bool)
    states, members = [], []
    for k, metric in enumerate(d['metrics']):
        ix = fit[d['tid'][fit] == k]
        y = d['y'][ix]; mols = d['mi'][ix]
        state = dict(metric=metric, center=float(np.median(y)),
                     scale=max(float(np.subtract(*np.quantile(y, [.75, .25]))), .01))
        if metric == 'Spearman':
            knots, first = np.unique(y, return_index=True)
            vals = ndtri(np.clip((rankdata(y) - .5) / len(y), 1e-5, 1 - 1e-5))
            state.update(knots=knots, normal=vals[first])
            encoded = vals
        elif metric == 'MAE':
            encoded = (y - state['center']) / state['scale']
        else:
            encoded = y
        for m in np.unique(mols):
            value = encoded[mols == m]
            labels[m, k] = np.mean(value) if metric in ('AUROC', 'AUPRC') else np.median(value)
            mask[m, k] = True
        members.append(np.unique(mols)); states.append(state)
    return labels, mask, states, members


def design(d, fit):
    unique = np.unique(d['mi'][fit])
    raw = d['views']['physchem']
    median = np.nan_to_num(np.nanmedian(raw[unique], axis=0))
    x = np.where(np.isfinite(raw), raw, median)
    mean = x[unique].mean(0); scale = x[unique].std(0).clip(1e-4)
    desc = ((x - mean) / scale).clip(-8, 8).astype(np.float32)
    fp = d['views']['multiview'][:, :4096]
    return torch.from_numpy(fp).cuda(), torch.from_numpy(desc).cuda(), dict(median=median, mean=mean, scale=scale)


def infer(net, d, gs, fp, desc, q, states):
    net.eval(); mols = np.unique(d['mi'][q])
    results = np.empty((4, len(q)), float)
    lookup = {int(m): j for j, m in enumerate(mols)}
    values = []
    with torch.inference_mode():
        for start in range(0, len(mols), BATCH):
            mm = mols[start:start + BATCH]
            values.append(net(pack(gs, mm), fp[mm], desc[mm]).cpu().numpy())
    predictions = np.concatenate(values, axis=1)
    position = np.asarray([lookup[int(m)] for m in d['mi'][q]])
    for k, state in enumerate(states):
        ix = np.flatnonzero(d['tid'][q] == k)
        p = predictions[:, position[ix], k].astype(float)
        if state['metric'] in ('AUROC', 'AUPRC'):
            p = expit(p)
        elif state['metric'] == 'MAE':
            p = p * state['scale'] + state['center']
            if d['tasks'][k] == 'ppbr_az':
                p = p.clip(0, 100)
        results[:, ix] = p
    assert np.isfinite(results).all()
    return results


def train(d, gs, part, seed=SEED):
    fit, q = rows(d, part)
    dest = OUT / part / str(seed); dest.mkdir(parents=True, exist_ok=True)
    cache = dest / 'predictions.npz'
    if cache.exists():
        z = np.load(cache)
        assert str(z['signature']) == signature()
        np.testing.assert_array_equal(z['fit'], fit); np.testing.assert_array_equal(z['query'], q)
        return z['p'].copy()
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    label, mask, states, members = targets(d, fit)
    fp, desc, pre = design(d, fit)
    y = torch.from_numpy(label).cuda(); observed = torch.from_numpy(mask).cuda()
    cls = torch.tensor([m in ('AUROC', 'AUPRC') for m in d['metrics']], device='cuda')
    mae = torch.tensor([m == 'MAE' for m in d['metrics']], device='cuda')
    unique = np.unique(d['mi'][fit])
    net = Net(desc.shape[1], len(d['tasks'])).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=.0006, weight_decay=.015)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-5)
    snapshots = []; started = time.time()
    for step in range(1, STEPS + 1):
        net.train()
        k = int(rng.integers(len(d['tasks'])))
        mols = np.r_[rng.choice(members[k], BATCH // 2, replace=True),
                     rng.choice(unique, BATCH // 2, replace=True)]
        p = net(pack(gs, mols), fp[mols], desc[mols])
        truth, active = y[mols][None], observed[mols][None]
        err = p - truth
        binary = nn.functional.binary_cross_entropy_with_logits(p, truth.expand_as(p), reduction='none')
        smooth = nn.functional.smooth_l1_loss(p, truth.expand_as(p), beta=.3, reduction='none')
        element = torch.where(cls[None, None], binary,
                              torch.where(mae[None, None], .95 * err.abs() + .05 * smooth, smooth))
        count = active.sum(1).clamp_min(1)
        per_task = (element * active).sum(1) / count
        valid_tasks = (active.sum(1) > 0)
        loss_by_head = (per_task * valid_tasks).sum(1) / valid_tasks.sum(1).clamp_min(1)
        loss = .15 * loss_by_head[:3].sum() + .55 * loss_by_head[3]
        # Pairwise auxiliary on the sampled target, with bounded quadratic cost.
        if d['metrics'][k] == 'Spearman':
            chosen = np.flatnonzero(mask[mols, k])
            left = rng.choice(chosen, 256); right = rng.choice(chosen, 256)
            difference = label[mols[left], k] - label[mols[right], k]
            keep = abs(difference) > .2
            if keep.any():
                sign = torch.tensor(np.sign(difference[keep]), dtype=torch.float32, device='cuda')
                loss = loss + .05 * nn.functional.softplus(-sign * (p[3, left[keep], k] - p[3, right[keep], k])).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 3.)
        opt.step(); sched.step()
        if step % 250 == 0:
            data.emit(stage='v79_train', part=part, seed=seed, step=step,
                      seconds=round(time.time() - started), loss=float(loss.detach()))
        if step in SNAPSHOTS:
            pred = infer(net, d, gs, fp, desc, q, states)
            snapshots.append(pred)
            path = dest / f'checkpoint_{step}.pt'
            torch.save(dict(signature=signature(), state={key: value.detach().cpu().clone() for key, value in net.state_dict().items()},
                desc_dim=desc.shape[1], tasks=len(d['tasks']), target=states, pre=pre,
                fit=fit, query=q, seed=seed, step=step), path)
            replay = Net(desc.shape[1], len(d['tasks'])).cuda()
            saved = torch.load(path, weights_only=False); replay.load_state_dict(saved['state'])
            replay_fp = torch.from_numpy(d['views']['multiview'][:, :4096]).cuda()
            raw = d['views']['physchem']; pp = saved['pre']
            xp = ((np.where(np.isfinite(raw), raw, pp['median']) - pp['mean']) / pp['scale']).clip(-8, 8).astype(np.float32)
            replay_desc = torch.from_numpy(xp).cuda()
            np.testing.assert_allclose(pred, infer(replay, d, gs, replay_fp, replay_desc, q, saved['target']), atol=2e-5, rtol=2e-5)
            del replay, replay_fp, replay_desc
    p = np.mean(snapshots, axis=0)
    np.savez_compressed(cache, signature=signature(), fit=fit, query=q, p=p)
    data.save(dest / 'metadata.json', dict(signature=signature(), part=part, seed=seed,
        fit_rows=len(fit), query_rows=len(q), steps=STEPS, snapshots=SNAPSHOTS, seconds=time.time() - started,
        independent_reload_passed=True, test_labels_used=False, external_data=False, pretrained_weights=False,
        checkpoint_hashes={f'checkpoint_{s}.pt': data.sha(dest / f'checkpoint_{s}.pt') for s in SNAPSHOTS}))
    del net, opt, fp, desc, y, observed
    torch.cuda.empty_cache()
    data.emit(stage='v79_complete', part=part, seconds=round(time.time() - started))
    return p


def task_prediction(d, part, k):
    z = np.load(OUT / part / str(SEED) / 'predictions.npz')
    active = d['tid'][z['query']] == k
    return z['query'][active], z['p'][:, active]


def metric_gain(m, y, base, new):
    a = chemistry.metric_score(m, y, base); p = chemistry.metric_score(m, y, new)
    return (a - p) / max(a, 1e-8) if m == 'MAE' else p - a


def select(d):
    records = []
    for k, task in enumerate(d['tasks']):
        m = d['metrics'][k]
        for j, name in enumerate(VARS):
            parts = []
            for part in ('fold0', 'fold1'):
                q, ps = task_prediction(d, part, k)
                base = data.proxy(d, q, part); p = ps[j]
                mixed = chemistry.blend(m, base, p, .5)
                parts.append(dict(part=part, n=len(q), expert=chemistry.metric_score(m, d['y'][q], p),
                    proxy=chemistry.metric_score(m, d['y'][q], base),
                    blend=chemistry.metric_score(m, d['y'][q], mixed), gain=metric_gain(m, d['y'][q], base, mixed)))
            threshold = .02 if m == 'MAE' else .01
            eligible = all(r['gain'] >= threshold for r in parts)
            records.append(dict(task=task, metric=m, variant=j, name=name, parts=parts,
                eligible=eligible, objective=min(r['gain'] for r in parts)))
    eligible = [r for r in records if r['eligible']]
    # Keep at most three genuinely different tasks; method is frozen now.
    selected = []
    for r in sorted(eligible, key=lambda r: r['objective'], reverse=True):
        if r['task'] not in [x['task'] for x in selected] and len(selected) < 3:
            selected.append(r)
    result = dict(signature=signature(), status='FROZEN', selected=selected, candidates=records,
        limitations='Exact original folds but incomplete full v73 identity; proxy comparisons only. Historically reused labels.')
    path = OUT / 'frozen_selection.json'
    if path.exists():
        assert json.loads(path.read_text()) == result
    else:
        data.save(path, result)
    data.emit(stage='v79_selection', selected=selected)
    return selected


def confirm(d, gs):
    selected = json.loads((OUT / 'frozen_selection.json').read_text())['selected']
    if not selected:
        data.save(OUT / 'confirmation.json', dict(status='NO_CANDIDATE', accepted=[]))
        return
    for part in ('fold2', 'valid'):
        train(d, gs, part)
    records = []; accepted = []
    for choice in selected:
        k = d['tasks'].index(choice['task']); m = choice['metric']; parts = []
        for part in ('fold2', 'valid'):
            q, ps = task_prediction(d, part, k)
            base = data.proxy(d, q, part); p = ps[choice['variant']]
            mixed = chemistry.blend(m, base, p, .5)
            parts.append(dict(part=part, n=len(q), expert=chemistry.metric_score(m, d['y'][q], p),
                proxy=chemistry.metric_score(m, d['y'][q], base),
                blend=chemistry.metric_score(m, d['y'][q], mixed), gain=metric_gain(m, d['y'][q], base, mixed)))
        passed = all(p['gain'] > (.01 if m == 'MAE' else .005) for p in parts)
        record = dict(choice=choice, parts=parts, accepted=passed)
        records.append(record)
        if passed:
            accepted.append(choice)
    result = dict(signature=signature(), status='CONFIRMATION_COMPLETE', tasks=records, accepted=accepted,
        full_v73_oof_available=False, official_score=None)
    data.save(OUT / 'confirmation.json', result)
    data.emit(stage='v79_confirmation', **result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('run', 'features', 'develop', 'confirm'))
    args = parser.parse_args()
    assert torch.cuda.is_available()
    torch.set_num_threads(4); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    d = data.load_data()
    path = ROOT / 'recovered_history/legacy_v24_folds.npy'
    assert data.sha(path) == '3a24d8a1359df47c4197bff62a80809bbdc42f53db22c5dd1f478db2a95a4b98'
    d['folds'] = np.load(path)
    data.save(OUT / 'protocol.json', dict(signature=signature(), architecture='joint directed-message graph/fingerprint/hybrid with task and molecule gates',
        steps=STEPS, snapshots=SNAPSHOTS, batch=BATCH, seed=SEED,
        folds_sha256=data.sha(path), labels_historically_reused=True,
        selection='fold0/1, >=2 percent MAE or .01 rank gain both folds, at most three tasks',
        confirmation='fold2/valid, >1 percent MAE or .005 rank gain both parts, fixed half blend',
        all_query_scaffolds_excluded_across_all_tasks=True, test_labels_used=False,
        external_data=False, pretrained_weights=False, full_v73_oof_available=False))
    gs = graphs(d)
    if args.stage in ('run', 'develop'):
        for part in ('fold0', 'fold1'):
            train(d, gs, part)
        select(d)
    if args.stage in ('run', 'confirm'):
        confirm(d, gs)
    data.save(OUT / 'pipeline_status.json', dict(status='COMPLETE', stage=args.stage,
        signature=signature(), pid=os.getpid(), new_submission_created=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        data.save(OUT / 'pipeline_status.json', dict(status='FAILED', error=repr(exc), signature=signature()))
        raise
