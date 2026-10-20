"""Remote-only task-conditioned directed-bond graph learning.

Uses authorized task training rows and scaffold-excluded auxiliary assays only.
No labels enter graph features or query-time task routing.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.special import expit

from admet_v6_features import make_graph, graph_batch
from admet_v6_models import Preprocess, Target, train_loss
from admet_v8_models import load_task as load8, RobustTarget, allowed_auxiliary, source_names, source_data
from admet_transfer_upgrade import REL

GRAPH_CACHE = {}


def load_task(task):
    d = load8(task)
    schema = {'version': 10, 'graph': 'directed_bond_task_attention_v1', 'base': d['v8_feature_key']}
    d['v10_feature_key'] = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()
    return d


def graphs_for(d):
    key = hashlib.sha256(json.dumps(d['smiles']).encode()).hexdigest()
    if key not in GRAPH_CACHE:
        GRAPH_CACHE[key] = [make_graph(s) for s in d['smiles']]
    return GRAPH_CACHE[key]


def graph_sources(d, ix, y, metric, maximum):
    from admet_upgrade_v3 import get_metric
    protected = np.setdiff1d(np.arange(len(d['smiles'])), ix)
    forbidden_groups, forbidden_molecules = set(d['groups'][protected]), set(d['canonical'][protected])
    names = list(dict.fromkeys(REL.get(d['task'], []) + source_names(d['task'])))[:maximum]
    records = [(d['task'], d, np.asarray(ix), np.asarray(y), metric)]
    audits = []
    for name in names:
        assert name != d['task']
        sd = source_data(name); n = len(sd['frames']['train'])
        keep = allowed_auxiliary(sd['groups'][:n], sd['canonical'][:n], forbidden_groups, forbidden_molecules)
        si = np.flatnonzero(keep); sy = sd['frames']['train'].Y.to_numpy(float)[si]; sm = get_metric(name)
        enough = len(si) >= 30 and (sm not in ['AUROC', 'AUPRC'] or len(np.unique(sy)) == 2)
        audit = dict(source=name, origin='train.csv only', total=n, kept=len(si), excluded=int((~keep).sum()),
            used=bool(enough), heldout_scaffold_overlap=len(set(sd['groups'][si]) & forbidden_groups),
            heldout_molecule_overlap=len(set(sd['canonical'][si]) & forbidden_molecules))
        assert audit['heldout_scaffold_overlap'] == audit['heldout_molecule_overlap'] == 0
        audits.append(audit)
        if enough:
            records.append((name, sd, si, sy, sm))
    return records, audits


class TaskGraphNetwork(nn.Module):
    """Directed non-backtracking messages, task-specific atom attention and heads."""
    def __init__(self, atom_dim, bond_dim, desc_dim, emb_dim, fp_dim, tasks, hidden=128, depth=3):
        super().__init__()
        self.depth = depth; self.hidden = hidden
        self.initial = nn.Linear(atom_dim + bond_dim, hidden)
        self.message = nn.Linear(hidden, hidden, bias=False)
        self.atom = nn.Linear(atom_dim + hidden, hidden)
        self.atom_norm = nn.LayerNorm(hidden)
        self.task_embedding = nn.Embedding(tasks, 24)
        self.attention = nn.Sequential(nn.Linear(hidden + 24, 64), nn.Tanh(), nn.Linear(64, 1))
        self.graph_project = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.desc = nn.Sequential(nn.Linear(desc_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(.15))
        self.emb = nn.Sequential(nn.Linear(emb_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(.2))
        self.fp = nn.Sequential(nn.Linear(fp_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(.25))
        self.gate = nn.Sequential(nn.Linear(hidden + 24, 64), nn.SiLU(), nn.Linear(64, 4))
        self.shared = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(.2))
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden + 24, 64), nn.SiLU(), nn.Dropout(.15), nn.Linear(64, 1)) for _ in range(tasks)])
        self.dropout = nn.Dropout(.1)

    def forward(self, graph, views, task):
        atom, bond, src, dst, rev, group = graph
        h0 = torch.relu(self.initial(torch.cat([atom[src], bond], 1)))
        h = h0
        for _ in range(self.depth - 1):
            incoming = h.new_zeros((len(atom), self.hidden)).index_add_(0, dst, h)
            h = self.dropout(torch.relu(h0 + self.message(incoming[src] - h[rev])))
        incoming = h.new_zeros((len(atom), self.hidden)).index_add_(0, dst, h)
        ah = self.atom_norm(torch.relu(self.atom(torch.cat([atom, incoming], 1))))
        count = torch.bincount(group, minlength=len(task)).clamp_min(1).to(ah.dtype).unsqueeze(1)
        pooled = ah.new_zeros((len(task), self.hidden)).index_add_(0, group, ah) / count
        te = self.task_embedding(task)
        # Positive normalized attention avoids overflow, and is permutation invariant.
        weights = torch.sigmoid(self.attention(torch.cat([ah, te[group]], 1))) + 1e-6
        weighted = ah.new_zeros((len(task), self.hidden)).index_add_(0, group, ah * weights)
        mass = ah.new_zeros((len(task), 1)).index_add_(0, group, weights)
        gh = self.graph_project(torch.cat([pooled, weighted / mass], 1))
        encoded = torch.stack([gh, self.desc(views[0]), self.emb(views[1]), self.fp(views[2])], 1)
        mix = torch.softmax(self.gate(torch.cat([encoded.mean(1), te], 1)), 1)
        common = (encoded * mix.unsqueeze(-1)).sum(1)
        common = common + self.shared(common)
        out = common.new_zeros(len(task))
        for k, head in enumerate(self.heads):
            mask = task == k
            if mask.any():
                out[mask] = head(torch.cat([common[mask], te[mask]], 1)).squeeze(-1)
        return out


def raw_views(d, ix):
    return [d['desc'][ix], d['emb'][ix], np.log1p(d['count'][ix])]


def new_net(bundle):
    return TaskGraphNetwork(bundle['atom_dim'], bundle['bond_dim'], *bundle['view_dims'],
                            len(bundle['names']), bundle['hidden'], bundle['depth'])


def fit(family, d, ix, y, metric, protocol):
    from admet_v6_runner import remote_only
    remote_only(); assert family == 'graph_transfer'
    records, audits = graph_sources(d, ix, y, metric, protocol['max_aux_sources'])
    raw = [raw_views(sd, ri) for _, sd, ri, _, _ in records]
    pre = [Preprocess().fit(np.concatenate([v[k] for v in raw])) for k in range(3)]
    xx, yy, tf, metrics, pw, graphs = [], [], [], [], [], []
    for (_, sd, ri, labels, m), views in zip(records, raw):
        cls = m in ['AUROC', 'AUPRC']
        target = None if cls else (Target(labels, m) if m == 'Spearman' else RobustTarget(labels, m))
        tf.append(target); metrics.append(m)
        xx.append([torch.as_tensor(p.transform(a), device='cuda') for p, a in zip(pre, views)])
        yy.append(torch.as_tensor(labels if cls else target.encode(labels), dtype=torch.float32, device='cuda'))
        weight = min(3., max(1., np.sqrt((1-labels.mean()) / max(labels.mean(), .01)))) if cls else 1.
        pw.append(torch.tensor(weight, dtype=torch.float32, device='cuda'))
        all_graphs = graphs_for(sd); graphs.append([all_graphs[i] for i in ri])
    bundle = dict(family=family, task=d['task'], metric=metric, classification=metric in ['AUROC', 'AUPRC'],
        feature_key=d['v10_feature_key'], pre=pre, target=tf[0], view_dims=[x.shape[1] for x in xx[0]],
        atom_dim=graphs[0][0][0].shape[1], bond_dim=14, names=[r[0] for r in records],
        hidden=protocol['hidden'], depth=protocol['depth'], models=[], source_audit=audits,
        protocol=protocol, target_rows=len(ix), target_passes_per_seed=protocol['epochs'])
    for seed in protocol['seeds']:
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        net = new_net(bundle).cuda()
        optimizer = torch.optim.AdamW(net.parameters(), lr=protocol['learning_rate'], weight_decay=protocol['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, protocol['epochs'], eta_min=2e-5)
        for epoch in range(1, protocol['epochs'] + 1):
            net.train(); order = rng.permutation(len(ix))
            for start in range(0, len(ix), protocol['target_batch']):
                draws = [order[start:start + protocol['target_batch']]]
                draws += [rng.integers(len(v), size=protocol['aux_batch']) for v in yy[1:]]
                bg = graph_batch([g[int(i)] for g, indices in zip(graphs, draws) for i in indices])
                bx = [torch.cat([v[k][indices] for v, indices in zip(xx, draws)]) for k in range(3)]
                bt = torch.cat([torch.full((len(indices),), k, dtype=torch.long, device='cuda') for k, indices in enumerate(draws)])
                output = net(bg, bx, bt); losses = []
                for k, indices in enumerate(draws):
                    labels = yy[k][indices]
                    if metrics[k] in ['AUROC', 'AUPRC']:
                        labels = (1-protocol['label_smoothing'])*labels + protocol['label_smoothing']*.5
                    losses.append(train_loss(output[bt == k], labels, metrics[k], pw[k]))
                loss = losses[0] if len(losses) == 1 else .7*losses[0] + .3*torch.stack(losses[1:]).mean()
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite graph transfer loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 2.); optimizer.step()
            scheduler.step()
            if epoch in protocol['snapshots']:
                bundle['models'].append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
            if epoch % 15 == 0 or epoch == protocol['epochs']:
                print(json.dumps(dict(stage='v10_graph_epoch', task=d['task'], seed=seed, epoch=epoch,
                    target_rows=len(ix), source_count=len(records)-1, loss=float(loss.detach()))), flush=True)
        del net
    return bundle


def predict(bundle, d, ix):
    from admet_v6_runner import remote_only
    remote_only(); assert bundle['feature_key'] == d['v10_feature_key']
    xx = [torch.as_tensor(p.transform(a), device='cuda') for p, a in zip(bundle['pre'], raw_views(d, ix))]
    graphs = graphs_for(d); net = new_net(bundle).cuda(); values = []
    for state in bundle['models']:
        net.load_state_dict(state); net.eval(); parts = []
        with torch.inference_mode():
            for start in range(0, len(ix), 64):
                indices = ix[start:start+64]
                bg = graph_batch([graphs[int(i)] for i in indices])
                bx = [v[start:start+64] for v in xx]
                task = torch.zeros(len(indices), dtype=torch.long, device='cuda')
                parts.append(net(bg, bx, task).cpu().numpy())
        values.append(np.concatenate(parts))
    del net
    p = np.mean(values, axis=0)
    return expit(p) if bundle['classification'] else bundle['target'].decode(p)
