"""Read-only remote audit for the next model experiment."""
import json
import os
from pathlib import Path
import numpy as np
import torch
import admet_v24_models as b

assert os.name != 'nt'
torch.set_num_threads(2)
d = b.build_data()
print('RESOURCE', torch.cuda.get_device_name(), torch.cuda.mem_get_info(), flush=True)
print('BASELINE_PATHS', [str(p) for p in Path('admet_submissions').glob('v4*/*.csv')], flush=True)
for k, task in enumerate(d['tasks']):
    parts = []
    for split in range(3):
        rows = np.flatnonzero((d['tid'] == k) & (d['split'] == split))
        train = np.flatnonzero((d['tid'] == k) & (d['split'] < (2 if split == 2 else 1)))
        groups = d['rg'][rows]
        overlap = np.isin(groups, d['rg'][train]).mean()
        ring = np.mean([not x.startswith(('ACYCLIC:', 'INVALID:')) for x in groups])
        parts.append(dict(n=len(rows),groups=len(np.unique(groups)),ring=float(ring),scaffold_seen=float(overlap)))
    print(json.dumps(dict(task=task,metric=d['metrics'][k],parts=parts)), flush=True)
for v in [24, 27, 33, 37, 41, 43, 44, 46]:
    root = Path(f'admet_model_outputs_v{v}')
    for name in ['valid_predictions.npy', 'cv_predictions.npy', 'baseline_valid.npy', 'baseline_cv_proxy.npy']:
        p = root/name
        if p.exists():
            a = np.load(p)
            print('CACHE', str(p), a.shape, int(np.isfinite(a).sum()), flush=True)
for p in [Path('admet_model_outputs_v43/features/valid/latent.npz'),Path('admet_model_outputs_v41/features/parents.npz')]:
    print('FEATURE', str(p), p.exists(), flush=True)
