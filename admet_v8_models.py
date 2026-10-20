"""Remote-only multiscale kernels and scaffold-excluded gated assay transfer."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.special import expit
from scipy.stats import norm, rankdata
from sklearn.svm import SVC, SVR
from rdkit import Chem, rdBase
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from admet_v6_models import Preprocess, train_loss
from admet_v7_models import load_task as load7
from admet_transfer_upgrade import REL

SOURCE_CACHE = {}


def load_task(task):
    d = load7(task)
    schema = {'version': 8, 'rdkit': rdBase.rdkitVersion, 'morgan': 'radius3_count4096_chiral'}
    key = hashlib.sha256(json.dumps([schema, d['smiles']], sort_keys=True).encode()).hexdigest()
    path = Path('admet_model_outputs_v8/features') / (task + '.npz')
    if path.exists():
        with np.load(path) as z:
            assert str(z['key']) == key
            d['r3'] = z['r3']
    else:
        gen = GetMorganGenerator(radius=3, fpSize=4096, includeChirality=True)
        values = np.zeros((len(d['smiles']), 4096), np.float32)
        for i, s in enumerate(d['smiles']):
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                values[i] = gen.GetCountFingerprintAsNumPy(mol)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.stem + '.' + str(os.getpid()) + '.npz')
        np.savez_compressed(temp, r3=values, key=key)
        os.replace(temp, path)
        d['r3'] = values
    d['v8_feature_key'] = key
    d['task'] = task
    return d


class RobustTarget:
    """Rank transforms are fitted exclusively on the current fitting labels."""
    def __init__(self, y, metric):
        y = np.asarray(y, dtype=np.float64)
        self.metric = metric
        self.center = float(np.median(y))
        self.scale = max(float(np.quantile(y, .75) - np.quantile(y, .25)), 1e-6)
        if metric == 'Spearman':
            z = norm.ppf(np.clip((rankdata(y) - .5) / len(y), 1e-5, 1 - 1e-5))
            order = np.argsort(y, kind='stable')
            unique, ix = np.unique(y[order], return_index=True)
            self.y_knots = unique
            self.z_knots = z[order][ix]

    def encode(self, y):
        if self.metric == 'Spearman':
            # Keep knot interpolation in float64; neural tensors cast at the training boundary.
            return np.interp(y, self.y_knots, self.z_knots)
        return ((np.asarray(y) - self.center) / self.scale).astype(np.float32)

    def decode(self, p):
        if self.metric == 'Spearman':
            return np.interp(p, self.z_knots, self.y_knots)
        return np.asarray(p) * self.scale + self.center


def multiscale_kernel(d, query, reference, pre):
    """Fixed positive mixture of PSD Tanimoto and descriptor RBF kernels."""
    pieces = []
    specs = [('r3', .30, False), ('count', .15, True), ('fcfp', .15, False),
             ('rdkit', .10, False), ('ap', .15, False)]
    refs = []
    for name, weight, binary in specs:
        a = d[name][reference]
        a = (a > 0).astype(np.float32) if binary else a
        a = torch.as_tensor(a, device='cuda')
        refs.append((name, weight, binary, a, a.square().sum(1)))
    desc_ref = torch.as_tensor(pre.transform(d['desc'][reference]), device='cuda')
    for start in range(0, len(query), 256):
        ix = query[start:start + 256]
        out = torch.zeros((len(ix), len(reference)), device='cuda')
        for name, weight, binary, b, bn in refs:
            a = d[name][ix]
            a = (a > 0).astype(np.float32) if binary else a
            a = torch.as_tensor(a, device='cuda')
            dot = a @ b.T
            out += weight * dot / (a.square().sum(1)[:, None] + bn[None] - dot).clamp_min(1e-8)
        a = torch.as_tensor(pre.transform(d['desc'][ix]), device='cuda')
        dist = (a.square().sum(1)[:, None] + desc_ref.square().sum(1)[None] - 2*a@desc_ref.T).clamp_min(0) / a.shape[1]
        out += .075 * torch.exp(-dist) + .075 * torch.exp(-dist/4)
        pieces.append(out.cpu().numpy())
    return np.concatenate(pieces).astype(np.float64)


def views(d, ix):
    return [np.log1p(d['count'][ix]), np.log1p(d['fcfp'][ix]),
            np.log1p(d['ap'][ix]), d['desc'][ix], d['emb'][ix]]


class GatedAssayNetwork(nn.Module):
    def __init__(self, dimensions, tasks, hidden=128):
        super().__init__()
        self.views = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(.2))
            for dim in dimensions])
        self.task_embedding = nn.Embedding(tasks, 24)
        self.gate = nn.Sequential(nn.Linear(hidden + 24, 64), nn.SiLU(), nn.Linear(64, len(dimensions)))
        self.shared = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(.2))
            for _ in range(2)])
        self.expert_gate = nn.Linear(hidden + 24, 2)
        self.private = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, 64), nn.SiLU(), nn.Dropout(.15)) for _ in range(tasks)])
        self.heads = nn.ModuleList([nn.Linear(hidden + 64, 1) for _ in range(tasks)])

    def forward(self, x, task):
        encoded = torch.stack([layer(v) for layer, v in zip(self.views, x)], dim=1)
        embedding = self.task_embedding(task)
        weights = torch.softmax(self.gate(torch.cat([encoded.mean(1), embedding], dim=1)), dim=1)
        h = (encoded * weights.unsqueeze(-1)).sum(1)
        expert_weights = torch.softmax(self.expert_gate(torch.cat([h, embedding], dim=1)), dim=1)
        shared = h + (torch.stack([layer(h) for layer in self.shared], dim=1) * expert_weights.unsqueeze(-1)).sum(1)
        output = h.new_zeros(len(h))
        for k in range(len(self.heads)):
            mask = task == k
            if mask.any():
                output[mask] = self.heads[k](torch.cat([shared[mask], self.private[k](h[mask])], dim=1)).squeeze(-1)
        return output


def allowed_auxiliary(groups, canonical, forbidden_groups, forbidden_canonical):
    return np.asarray([g not in forbidden_groups and c not in forbidden_canonical
                       for g, c in zip(groups, canonical)], dtype=bool)


def source_names(task):
    names = set(REL.get(task, []))
    if 'substrate_carbonmangels' in task:
        names.update(['cyp2c9_substrate_carbonmangels', 'cyp2d6_substrate_carbonmangels',
                      'cyp3a4_substrate_carbonmangels'])
    if task in {'clearance_hepatocyte_az', 'clearance_microsome_az', 'half_life_obach', 'vdss_lombardo', 'ppbr_az'}:
        names.update(['clearance_hepatocyte_az', 'clearance_microsome_az', 'half_life_obach',
                      'vdss_lombardo', 'ppbr_az'])
    return sorted(names - {task})


def source_data(task):
    if task not in SOURCE_CACHE:
        SOURCE_CACHE[task] = load7(task)
    return SOURCE_CACHE[task]


def transfer_training(d, ix, y, metric):
    from admet_upgrade_v3 import get_metric
    task = d['task']
    # Everything outside target fit is protected, including official valid/test structures.
    protected = np.setdiff1d(np.arange(len(d['smiles'])), ix)
    forbidden_groups = set(d['groups'][protected])
    forbidden_canonical = set(d['canonical'][protected])
    records = [(task, views(d, ix), np.asarray(y), metric)]
    audits = []
    for source in source_names(task):
        sd = source_data(source)
        n = len(sd['frames']['train'])
        keep = allowed_auxiliary(sd['groups'][:n], sd['canonical'][:n], forbidden_groups, forbidden_canonical)
        si = np.flatnonzero(keep)
        sy = sd['frames']['train'].Y.to_numpy(float)[si]
        sm = get_metric(source)
        enough = len(si) >= 30 and (sm not in ['AUROC', 'AUPRC'] or len(np.unique(sy)) == 2)
        audits.append(dict(source=source, origin='train.csv only', total=n, kept=len(si), excluded=int((~keep).sum()),
                           used=bool(enough), heldout_scaffold_overlap=len(set(sd['groups'][si]) & forbidden_groups),
                           heldout_molecule_overlap=len(set(sd['canonical'][si]) & forbidden_canonical)))
        assert audits[-1]['heldout_scaffold_overlap'] == audits[-1]['heldout_molecule_overlap'] == 0
        if enough:
            records.append((source, views(sd, si), sy, sm))
    return records, audits


def fit(family, d, ix, y, metric, protocol):
    assert torch.cuda.is_available(), 'Remote CUDA required'
    cls = metric in ['AUROC', 'AUPRC']
    target = None if cls else RobustTarget(y, metric)
    bundle = dict(family=family, metric=metric, classification=cls, target=target,
                  feature_key=d['v8_feature_key'], task=d['task'], models=[])
    if family == 'multiscale_kernel':
        pre = Preprocess().fit(d['desc'][ix])
        x = multiscale_kernel(d, ix, ix, pre)
        yy = np.asarray(y) if cls else target.encode(y)
        for c in protocol['kernel_cs']:
            model = (SVC(C=c, kernel='precomputed', class_weight='balanced', cache_size=1536) if cls
                     else SVR(C=c, kernel='precomputed', epsilon=.02, cache_size=1536))
            model.fit(x, yy)
            bundle['models'].append(model)
        bundle.update(pre=pre, reference=np.asarray(ix))
        return bundle
    assert family == 'gated_transfer'
    records, audits = transfer_training(d, ix, y, metric)
    pre = [Preprocess().fit(np.concatenate([r[1][k] for r in records])) for k in range(5)]
    xx = []; yy = []; transforms = []; weights = []; metrics = []
    for name, vx, labels, m in records:
        classification = m in ['AUROC', 'AUPRC']
        tf = None if classification else RobustTarget(labels, m)
        transforms.append(tf); metrics.append(m)
        xx.append([torch.as_tensor(p.transform(a), device='cuda') for p, a in zip(pre, vx)])
        yy.append(torch.as_tensor(labels if classification else tf.encode(labels), dtype=torch.float32, device='cuda'))
        weight = min(3., max(1., np.sqrt((1-labels.mean())/max(labels.mean(), .01)))) if classification else 1.
        weights.append(torch.tensor(weight, dtype=torch.float32, device='cuda'))
    dimensions = [x.shape[1] for x in xx[0]]
    bundle.update(pre=pre, dimensions=dimensions, names=[r[0] for r in records], hidden=protocol['mtl_hidden'],
                  source_audit=audits, target=transforms[0], protocol=protocol)
    for seed in protocol['seeds']:
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        net = GatedAssayNetwork(dimensions, len(records), protocol['mtl_hidden']).cuda()
        optimizer = torch.optim.AdamW(net.parameters(), lr=protocol['mtl_lr'], weight_decay=.015)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, protocol['mtl_epochs'], eta_min=1e-5)
        for epoch in range(1, protocol['mtl_epochs'] + 1):
            net.train()
            for step in range(protocol['mtl_steps_per_epoch']):
                draws = [rng.integers(len(v), size=protocol['mtl_target_batch'] if k == 0 else protocol['mtl_aux_batch']) for k, v in enumerate(yy)]
                bx = [torch.cat([v[k][idx] for v, idx in zip(xx, draws)]) for k in range(5)]
                bt = torch.cat([torch.full((len(idx),), k, device='cuda', dtype=torch.long) for k, idx in enumerate(draws)])
                pred = net(bx, bt)
                losses = []
                for k, idx in enumerate(draws):
                    loss = train_loss(pred[bt == k], yy[k][idx], metrics[k], weights[k])
                    losses.append((2. if k == 0 else 1.) * loss)
                loss = torch.stack(losses).sum() / (len(records) + 1)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite gated transfer loss')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 2.)
                optimizer.step()
            schedule.step()
            if epoch in protocol['mtl_snapshots']:
                bundle['models'].append({k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
            if epoch % 15 == 0 or epoch == protocol['mtl_epochs']:
                print(json.dumps(dict(stage='v8_mtl_epoch', task=d['task'], seed=seed, epoch=epoch,
                                      loss=float(loss.detach()), source_count=len(records)-1)), flush=True)
        del net
    return bundle


def predict(bundle, d, ix):
    assert bundle['feature_key'] == d['v8_feature_key']
    family = bundle['family']; preds = []
    if family == 'multiscale_kernel':
        x = multiscale_kernel(d, ix, bundle['reference'], bundle['pre'])
        for model in bundle['models']:
            p = model.decision_function(x) if bundle['classification'] else model.predict(x)
            preds.append(p)
        p = np.mean(preds, axis=0)
        return expit(p) if bundle['classification'] else bundle['target'].decode(p)
    raw = views(d, ix)
    x = [torch.as_tensor(pre.transform(a), device='cuda') for pre, a in zip(bundle['pre'], raw)]
    net = GatedAssayNetwork(bundle['dimensions'], len(bundle['names']), bundle['hidden']).cuda()
    for state in bundle['models']:
        net.load_state_dict(state); net.eval(); parts = []
        with torch.inference_mode():
            for start in range(0, len(ix), 256):
                batch = [v[start:start+256] for v in x]
                task = torch.zeros(len(batch[0]), device='cuda', dtype=torch.long)
                parts.append(net(batch, task).cpu().numpy())
        preds.append(np.concatenate(parts))
    del net
    p = np.mean(preds, axis=0)
    return expit(p) if bundle['classification'] else bundle['target'].decode(p)
