"""Independent directed molecular graph experts against the frozen v94 anchor.

Each endpoint has its own weights, labels, and target-excluded measured inputs.
Graph-only/direct-hybrid/residual-hybrid heads are trained together. Fixed
checkpoints and blends are selected on fold0/1 before fold2/valid are opened.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '4'
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v94_isolated_assay as prev
import admet_v79_joint_graph as old_graph
b, r83 = prev.b, prev.r83
OUT = ROOT / 'admet_model_outputs_v95'
ANCHOR = ROOT / 'admet_submissions/v94_isolated_assay/我从山北来带着兰花草-小分子药物预测技术创新赛-20261025.csv'
ANCHOR_SHA = '78f228a4025d55149ce08fbd102b02c2b365e75627140bb4c300f9d7292d2583'
SEEDS = (954026, 954027)
STEPS = 1600
SNAPSHOTS = (400, 800, 1600)
BATCH = 96
HIDDEN = 96
MODES = ('graph', 'hybrid', 'residual')
WEIGHTS = (.25, .5, 1.)


def sig():
    return hashlib.sha256((b.sha(__file__) + prev.sig() + old_graph.signature()).encode()).hexdigest()


def emit(**obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def load():
    d = prev.load()
    assert b.sha(ANCHOR) == ANCHOR_SHA
    return d


def base(d, k, part):
    if part == 'final':
        return np.array([float(r['y_pred']) for r in b.table(ANCHOR) if r['task'] == d['tasks'][k]])
    old = prev.base(d, k, part)
    c = json.loads((prev.OUT / 'confirmation.json').read_text())
    assert c['signature'] == prev.sig() and c['status'] == 'COMPLETE'
    chosen = next((r for r in c['records'] if r['task'] == d['tasks'][k] and r['accepted']), None)
    if chosen:
        old = prev.mix(chosen['metric'], old, prev.expert(d, k, part, chosen['config']), chosen['weight'])
    return old


def threshold(m):
    return .01 if m == 'MAE' else .005 if m == 'Spearman' else .002


def mix(metric, old, new, weight):
    return prev.mix(metric, old, new, weight)


def graphs(d):
    if '_v95_graphs' not in d:
        gs = old_graph.graphs(d)
        assert len(gs) == len(d['views']['physchem'])
        d['_v95_graphs'] = gs
    return d['_v95_graphs']


def pack(gs, mols, device):
    aa, ee, ss, tt, rr, counts = [], [], [], [], [], []
    ao = eo = 0
    for index in mols:
        a, e, s, t, r = gs[int(index)]
        aa.append(a); ee.append(e); ss.append(s + ao); tt.append(t + ao); rr.append(r + eo)
        counts.append(len(a)); ao += len(a); eo += len(e)
    arrays = [np.concatenate(aa), np.concatenate(ee), np.concatenate(ss), np.concatenate(tt),
              np.concatenate(rr), np.repeat(np.arange(len(mols)), counts), np.asarray(counts, np.float32)]
    return tuple(torch.as_tensor(x, device=device) for x in arrays)


class Net(nn.Module):
    def __init__(self, desc_dim, assay_dim):
        super().__init__()
        h = HIDDEN
        def block(a, z, dropout=.2):
            return nn.Sequential(nn.Linear(a, z), nn.LayerNorm(z), nn.SiLU(), nn.Dropout(dropout))
        self.edge = nn.Linear(65, h)
        self.message = nn.ModuleList(nn.Linear(h, h, bias=False) for _ in range(3))
        self.atom = nn.Linear(52 + h, h)
        self.pool = block(2 * h + 1, h)
        self.desc = block(desc_dim, 32)
        self.fp = block(12288, 96, .4)
        self.assay = block(assay_dim, 32, .2)
        self.graph_head = nn.Sequential(block(h + 32, 64, .3), nn.Linear(64, 1))
        self.hybrid_head = nn.Sequential(block(h + 32 + 96 + 32, 96, .35), nn.Linear(96, 2))
        for head in (self.graph_head, self.hybrid_head):
            nn.init.normal_(head[-1].weight, std=.01)
            nn.init.zeros_(head[-1].bias)

    def forward(self, graph, fp, desc, assay):
        a, e, s, t, r, mi, count = graph
        h0 = nn.functional.silu(self.edge(torch.cat([a[s], e], 1)))
        h = h0
        for layer in self.message:
            incoming = h.new_zeros((len(a), HIDDEN))
            incoming.index_add_(0, t, h)
            h = nn.functional.dropout(nn.functional.silu(h0 + layer(incoming[s] - h[r])), .15, self.training)
        incoming = h.new_zeros((len(a), HIDDEN)); incoming.index_add_(0, t, h)
        av = nn.functional.silu(self.atom(torch.cat([a, incoming], 1)))
        pooled = av.new_zeros((len(count), HIDDEN)); pooled.index_add_(0, mi, av)
        gh = self.pool(torch.cat([pooled / count[:, None], pooled / count.sqrt()[:, None], count.log1p()[:, None]], 1))
        dh = self.desc(desc)
        if self.training:
            n = assay.shape[1] // 2
            keep = (torch.rand_like(assay[:, :n]) > .25).to(assay.dtype)
            assay = assay * torch.cat([keep, keep], 1)
        direct = self.graph_head(torch.cat([gh, dh], 1))
        hybrid = self.hybrid_head(torch.cat([gh, dh, self.fp(fp), self.assay(assay)], 1))
        return torch.cat([direct, hybrid], 1)


def prepare(d, k, part):
    return prev.prepare(d, k, part)


def tensors(a, device):
    return tuple(torch.as_tensor(a[key], device=device) for key in ('fp', 'desc', 'assay'))


def infer(net, d, a, ts, batch=128):
    net.eval(); gs = graphs(d); device = next(net.parameters()).device
    rows = np.r_[a['fit'], a['query']]; out = []
    with torch.inference_mode():
        for lo in range(len(a['fit']), len(rows), batch):
            ids = np.arange(lo, min(lo + batch, len(rows)))
            out.append(net(pack(gs, d['mi'][rows[ids]], device), *(x[ids] for x in ts)).cpu().numpy())
    return np.concatenate(out).astype(float)


def train(d, k, part, seed):
    folder = OUT / 'models' / d['tasks'][k] / part / str(seed)
    file = folder / 'predictions.npz'
    if file.exists():
        z = np.load(file); meta = json.loads((folder / 'metadata.json').read_text())
        assert str(z['signature']) == meta['signature'] == sig()
        assert b.sha(file) == meta['predictions_sha256']
        np.testing.assert_array_equal(z['query'], r83.split(d, k, part)[1])
        return z['values'].copy()
    folder.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    a = prepare(d, k, part); gs = graphs(d)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); rng = np.random.default_rng(seed)
    ts = tensors(a, 'cuda'); target = torch.as_tensor(a['target'][:, [0, 0, 1]], device='cuda')
    offset = torch.as_tensor(a['offset'], device='cuda'); net = Net(a['desc'].shape[1], a['assay'].shape[1]).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=.0007, weight_decay=.025)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=.000035)
    metric = d['metrics'][k]; snapshots = []; hashes = {}; losses = []
    for step in range(1, STEPS + 1):
        net.train(); ix = rng.choice(len(a['fit']), BATCH, replace=len(a['fit']) < BATCH)
        p = net(pack(gs, d['mi'][a['fit'][ix]], 'cuda'), *(x[ix] for x in ts)); yt = target[ix]
        if metric in ('AUROC', 'AUPRC'):
            off = torch.stack([torch.zeros_like(offset[ix]), torch.zeros_like(offset[ix]), offset[ix]], 1)
            loss = nn.functional.binary_cross_entropy_with_logits(p + off, yt)
        elif metric == 'MAE':
            loss = (p - yt).abs().mean()
        else:
            loss = (p - yt).square().mean()
        loss = loss + .005 * p.square().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 3.)
        opt.step(); sched.step()
        if step % 400 == 0:
            losses.append(dict(step=step, loss=float(loss.detach())))
            emit(stage='train', task=d['tasks'][k], part=part, seed=seed, step=step, seconds=round(time.monotonic()-started), loss=float(loss.detach()))
        if step in SNAPSHOTS:
            snapshots.append(infer(net, d, a, ts)); path = folder / f'checkpoint_{step}.pt'
            torch.save(dict(signature=sig(), weights={n: p.detach().cpu().clone() for n, p in net.state_dict().items()},
                            desc_dim=a['desc'].shape[1], assay_dim=a['assay'].shape[1],
                            fit=a['fit'], query=a['query'], seed=seed, step=step, state=a['state']), path)
            hashes[path.name] = b.sha(path)
    values = np.stack(snapshots)
    np.savez_compressed(file, signature=sig(), fit=a['fit'], query=a['query'], values=values, snapshots=SNAPSHOTS, qoffset=a['qoffset'])
    b.save(folder/'metadata.json', dict(signature=sig(), part=part, task=d['tasks'][k], seed=seed,
        checkpoint_hashes=hashes, predictions_sha256=b.sha(file), seconds=time.monotonic()-started, losses=losses))
    del net, opt, ts, target, offset; torch.cuda.empty_cache()
    return values


def expert(d, k, part, cfg):
    memo = d.setdefault('_v95_experts', {}); key = (k, part)
    if key not in memo:
        values = np.mean([train(d, k, part, s) for s in SEEDS], axis=0)
        a = prepare(d, k, part); memo[key] = (values, a['state'], a['qoffset'])
    values, state, qoff = memo[key]; mode = MODES.index(cfg['mode'])
    raw = values[SNAPSHOTS.index(cfg['step']), :, mode] * state['scale'] + (qoff if mode == 2 else state['center'])
    p = expit(raw) if d['metrics'][k] in ('AUROC', 'AUPRC') else raw
    return p.clip(0, 100) if d['tasks'][k] == 'ppbr_az' else p


def smoke(d):
    OUT.mkdir(exist_ok=True); k = d['tasks'].index('bioavailability_ma'); a = prepare(d, k, 'fold0')
    saved = d['y'][a['query']].copy(); d['y'][a['query']] = 1e6; d.pop('_v94_aux', None)
    perturbed = prepare(d, k, 'fold0'); d['y'][a['query']] = saved; d.pop('_v94_aux', None)
    for key in ('fp', 'desc', 'assay', 'target', 'offset', 'qoffset', 'fit', 'query', 'cols', 'source'):
        np.testing.assert_array_equal(a[key], perturbed[key])
    ts = tensors(a, 'cuda'); torch.manual_seed(SEEDS[0]); net = Net(a['desc'].shape[1], a['assay'].shape[1]).cuda()
    p = infer(net, d, a, ts); np.testing.assert_allclose(p, infer(net, d, a, ts, 31), atol=2e-6, rtol=2e-5)
    graph = pack(graphs(d), d['mi'][a['fit'][:32]], 'cuda')
    net.train(); started = time.monotonic(); opt = torch.optim.AdamW(net.parameters(), lr=.0007)
    for _ in range(50):
        out = net(graph, *(x[:32] for x in ts)); loss = out.square().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    elapsed = time.monotonic() - started
    path = OUT / 'smoke.pt'; torch.save(net.state_dict(), path)
    other = Net(a['desc'].shape[1], a['assay'].shape[1]); other.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    net.eval(); other.eval(); small = dict(a, query=a['query'][:7])
    # infer assumes fit+query ordering; tensors must follow the shortened rows.
    short = tuple(torch.cat([x[:len(a['fit'])], x[len(a['fit']):len(a['fit'])+7]]) for x in ts)
    cpu = tuple(x.cpu() for x in short)
    np.testing.assert_allclose(infer(net,d,small,short), infer(other,d,small,cpu), atol=2e-5, rtol=2e-5)
    b.save(OUT/'smoke.json', dict(status='PASS', signature=sig(), target_query_label_perturbation_invariance=True,
        target_assay_source_absent=True, alternate_batch=True, cpu_reload=True, seconds_for_50_updates_batch32=elapsed,
        graph_cache_sha256=b.sha(old_graph.OUT/'features/graphs.pt')))
    emit(stage='smoke', status='PASS', seconds_for_50_updates_batch32=elapsed)


def develop(d):
    OUT.mkdir(exist_ok=True)
    b.save(OUT/'protocol.json', dict(version=95, signature=sig(), baseline='v94', anchor_sha256=ANCHOR_SHA,
        baseline_official_score_user_reported=50.916, seeds=SEEDS, steps=STEPS, snapshots=SNAPSHOTS,
        modes=MODES, weights=WEIGHTS, batch=BATCH, selection_parts=['fold0','fold1'], confirmation_parts=['fold2','valid'],
        selection='One configuration per endpoint maximizing minimum development gain; threshold 1% MAE, .005 Spearman, .002 AUC/AP.',
        confirmation='All four gains positive; mean exceeds threshold. No fallback after confirmation.',
        architecture='Independent task-specific 3-layer directed bond message networks plus descriptor, fingerprint and target-excluded other official assay inputs. Graph-only/direct hybrid/residual hybrid heads.',
        label_isolation='Only target-task fit rows in loss. Entire target assay excluded from all auxiliary inputs. No shared trainable weights between tasks.',
        test_labels_used=False, external_data=False, new_pretrained_weights=False,
        limitations='Historical baseline proxies are not complete v94 OOF. Labels reused historically. Conditional official assays. No official score estimate.'))
    b.save(OUT/'data_audit.json', dict(input_hashes=d['sources'], test_labels_finite=0,
        graph_cache_sha256=b.sha(old_graph.OUT/'features/graphs.pt')))
    records=[]; selected=[]
    order=sorted(range(len(d['tasks'])), key=lambda k:int(((d['tid']==k)&(d['split']==0)).sum()))
    for k in order:
        task, metric=d['tasks'][k], d['metrics'][k]
        bases={p:base(d,k,p) for p in ('fold0','fold1')}
        for step in SNAPSHOTS:
            for mode in MODES:
                cfg=dict(step=step,mode=mode); ps={p:expert(d,k,p,cfg) for p in bases}
                for w in WEIGHTS:
                    parts=[]
                    for part,old in bases.items():
                        q=r83.split(d,k,part)[1]; p=mix(metric,old,ps[part],w)
                        parts.append(dict(part=part,rows=len(q),baseline=r83.score(metric,d['y'][q],old),
                            candidate=r83.score(metric,d['y'][q],p),gain=r83.gain(metric,d['y'][q],old,p)))
                    records.append(dict(task=task,metric=metric,config=cfg,weight=w,parts=parts,objective=min(x['gain'] for x in parts)))
        best=max((r for r in records if r['task']==task),key=lambda r:r['objective'])
        emit(stage='development', **best)
        if best['objective']>threshold(metric):selected.append(best)
        b.save(OUT/'development.json',dict(status='IN_PROGRESS',signature=sig(),records=records))
    b.save(OUT/'development.json',dict(status='COMPLETE',signature=sig(),records=records))
    b.save(OUT/'frozen_selection.json',dict(status='FROZEN_BEFORE_CONFIRMATION',signature=sig(),selected=selected,confirmation_inspected=False))
    emit(stage='frozen',selected=selected)


def confirm(d):
    frozen=json.loads((OUT/'frozen_selection.json').read_text()); assert frozen['signature']==sig(); records=[]
    for ch in frozen['selected']:
        k=d['tasks'].index(ch['task']); m=ch['metric']; parts=[]
        for part in ('fold0','fold1','fold2','valid'):
            q=r83.split(d,k,part)[1]; old=base(d,k,part); p=mix(m,old,expert(d,k,part,ch['config']),ch['weight'])
            parts.append(dict(part=part,rows=len(q),baseline=r83.score(m,d['y'][q],old),candidate=r83.score(m,d['y'][q],p),gain=r83.gain(m,d['y'][q],old,p)))
        r=dict(**ch,confirmation=parts,accepted=bool(all(x['gain']>0 for x in parts) and np.mean([x['gain'] for x in parts])>threshold(m)))
        records.append(r); emit(stage='confirmation',**r)
        b.save(OUT/'confirmation.json',dict(status='IN_PROGRESS',signature=sig(),records=records))
    b.save(OUT/'confirmation.json',dict(status='COMPLETE',signature=sig(),records=records,
        accepted=[r['task'] for r in records if r['accepted']],frozen_sha256=b.sha(OUT/'frozen_selection.json')))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['smoke','develop','confirm']); args=ap.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.set_float32_matmul_precision('highest')
    dict(smoke=smoke,develop=develop,confirm=confirm)[args.stage](load())
