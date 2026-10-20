"""Masked, task-balanced mixture-of-experts neural network for all 22 tasks.

Canonical molecules share parameters, not fabricated labels. All molecules in
the evaluated split are excluded from every task's training records. No TDC
download or hidden-test target lookup is performed.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import GroupShuffleSplit
from admet_features_v2 import features, DescriptorPreprocessor, SCHEMA


class TaskGatedExperts(nn.Module):
    def __init__(self, input_dim, tasks, width=128, experts=3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.LayerNorm(256),
                                     nn.SiLU(), nn.Dropout(.15), nn.Linear(256, width), nn.SiLU())
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.SiLU(),
                                                  nn.Dropout(.15)) for _ in range(experts)])
        self.gates = nn.Parameter(torch.randn(tasks, width, experts)*.03)
        self.gate_bias = nn.Parameter(torch.zeros(tasks, experts))
        self.head = nn.Parameter(torch.randn(tasks, width)*.03)
        self.bias = nn.Parameter(torch.zeros(tasks))

    def forward(self, x, task):
        h = self.encoder(x)
        e = torch.stack([expert(h) for expert in self.experts], dim=1)
        attention = torch.softmax(torch.einsum('bh,bhe->be', h, self.gates[task])+self.gate_bias[task], dim=-1)
        mixed = (e*attention[:, :, None]).sum(1)
        return (mixed*self.head[task]).sum(1)+self.bias[task]


def collect(root, cache):
    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    frames = {n: {s: pd.read_csv(root/n/f'{s}.csv') for s in ['train', 'valid', 'test']} for n in names}
    # SMILES-only caching involves no fitted population statistics.
    smi = sorted(set(s for n in names for df in frames[n].values() for s in df.Drug))
    x, canon, scaffold = features(smi, cache)
    smi_ix = dict(zip(smi, range(len(smi))))
    records, split_idx = {}, {}
    for n in names:
        split_idx[n] = {s: np.asarray([smi_ix[v] for v in df.Drug]) for s, df in frames[n].items()}
        df = frames[n]['train']
        ix = split_idx[n]['train']
        fit, hold = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=20260924).split(ix, groups=np.asarray(scaffold)[ix]))
        records[n] = {'fit': fit, 'hold': hold}
    return names, frames, x, np.asarray(canon), split_idx, records


def fit_stage(names, frames, x, canon, indices, splits, stage, out, epochs=32):
    torch.set_num_threads(6)
    torch.manual_seed(1729)
    rng = np.random.default_rng(1729)
    # Exclude evaluation molecules across tasks so auxiliary heads cannot see
    # labelled versions of validation/test structures.
    forbidden = set()
    for task in names:
        forbidden.update(canon[indices[task]['test']])
        if stage != 'full':
            forbidden.update(canon[indices[task]['valid']])
        if stage == 'scaffold':
            forbidden.update(canon[indices[task]['train'][splits[task]['hold']]])
    by_task, transforms, audit = [], [], []
    official = pd.read_csv('admet_official/submission_template.csv').groupby('task').metric.first().to_dict()
    for task in names:
        permitted_splits = ['train', 'valid'] if stage == 'full' else ['train']
        ix = np.concatenate([indices[task][s] for s in permitted_splits])
        y = np.concatenate([frames[task][s].Y.to_numpy(float) for s in permitted_splits])
        mask = np.asarray([s not in forbidden for s in canon[ix]])
        ix, y = ix[mask], y[mask]
        assert not set(canon[ix]) & forbidden
        # Collapse repeated assays of identical canonical molecules. Classification
        # disagreement remains a soft target; regression uses measured mean.
        agg = pd.DataFrame({'canonical': canon[ix], 'ix': ix, 'y': y}).groupby('canonical').agg(ix=('ix','first'),y=('y','mean'))
        ix, y = agg.ix.to_numpy(int), agg.y.to_numpy(float)
        classification = official[task] in ['AUROC', 'AUPRC']
        log = not classification and official[task] == 'Spearman' and y.min() >= 0
        yy = np.log1p(y) if log else y.copy()
        mean, scale = (0., 1.) if classification else (float(yy.mean()), max(float(yy.std()), 1e-6))
        yy = (yy-mean)/scale
        by_task.append((ix, yy.astype(np.float32)))
        transforms.append(dict(classification=classification, log=log, mean=mean, scale=scale))
        audit.append(dict(task=task, stage=stage, usable_molecules=len(ix),
                          excluded_records=int((~mask).sum()), forbidden_intersection=0))
        if not len(ix):
            raise ValueError(f'{task}: empty train after leakage exclusions')
    training_ix = np.unique(np.concatenate([r[0] for r in by_task]))
    pre = DescriptorPreprocessor().fit(x[training_ix])
    xx = torch.from_numpy(pre.transform(x))
    net = TaskGatedExperts(x.shape[1], len(names))
    opt = torch.optim.AdamW(net.parameters(), lr=8e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    class_tasks = torch.tensor([t['classification'] for t in transforms])
    logrows = []
    # Fixed epochs selected before seeing this experiment's valid results.
    # Equal examples/task prevent the largest CYP endpoint dominating gradients.
    snapshots = []
    for ep in range(epochs):
        net.train()
        losses = []
        for _ in range(55):
            bix, by, bt = [], [], []
            for t, (ix, yy) in enumerate(by_task):
                draw = rng.integers(len(ix), size=24)
                bix.extend(ix[draw]); by.extend(yy[draw]); bt.extend([t]*len(draw))
            target = torch.tensor(np.asarray(by), dtype=torch.float32)
            task = torch.tensor(bt, dtype=torch.long)
            p = net(xx[np.asarray(bix)], task)
            binary = nn.functional.binary_cross_entropy_with_logits(p, target, reduction='none')
            regression = nn.functional.smooth_l1_loss(p, target, reduction='none')
            loss = torch.where(class_tasks[task], binary, regression).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 3.)
            opt.step(); losses.append(float(loss.detach()))
        scheduler.step()
        logrows.append(dict(epoch=ep+1, loss=float(np.mean(losses))))
        if ep+1 in [epochs-4, epochs]:
            snapshots.append(copy.deepcopy(net.state_dict()))
        if (ep+1)%4 == 0:
            print(f'MTL {stage} epoch {ep+1}/{epochs} loss={np.mean(losses):.4f}', flush=True)
    stageout = out/stage; stageout.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit).to_csv(stageout/'leakage_audit.csv', index=False)
    pd.DataFrame(logrows).to_csv(stageout/'training_loss.csv', index=False)
    torch.save({'states': snapshots, 'input_dim': x.shape[1], 'names': names, 'transforms': transforms,
                'schema': SCHEMA, 'architecture': 'TaskGatedExperts', 'epochs': epochs}, stageout/'model.pt')
    joblib.dump(pre, stageout/'preprocessor.joblib')
    for t, task in enumerate(names):
        if stage == 'scaffold':
            idx = indices[task]['train'][splits[task]['hold']]
        else:
            idx = indices[task]['test' if stage == 'full' else 'valid']
        preds = []
        for state in snapshots:
            net.load_state_dict(state); net.eval()
            p = []
            with torch.no_grad():
                for s in range(0,len(idx),512):
                    batch = idx[s:s+512]
                    p.extend(net(xx[batch], torch.full((len(batch),), t, dtype=torch.long)).numpy())
            p = np.asarray(p)
            trans = transforms[t]
            if trans['classification']:
                p = 1/(1+np.exp(-np.clip(p,-30,30)))
            else:
                p = p*trans['scale']+trans['mean']
                if trans['log']:
                    p = np.expm1(np.clip(p,-30,30))
            preds.append(p)
        pd.DataFrame({'Y_pred': np.mean(preds,axis=0)}).to_csv(stageout/f'{task}.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', nargs='+', default=['scaffold','valid','full'])
    ap.add_argument('--epochs', type=int, default=32)
    a = ap.parse_args()
    out = Path('admet_mtl_outputs'); out.mkdir(exist_ok=True)
    print('Building shared molecular features', flush=True)
    data = collect(Path('admet_competition_data/data_benchmark'), out/'features')
    print(f'Unique raw SMILES {len(data[2])}; {len(data[0])} tasks', flush=True)
    for stage in a.stages:
        fit_stage(*data, stage, out, epochs=a.epochs)


if __name__ == '__main__':
    main()
