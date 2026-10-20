import importlib.util
import json
import os
from pathlib import Path
import torch

root = Path('/home/czp123/admet_v23_20260928')
os.chdir(root)
print(json.dumps(dict(cuda=torch.cuda.is_available(), device=torch.cuda.get_device_name(0),
    memory=torch.cuda.mem_get_info(), packages={x: bool(importlib.util.find_spec(x))
    for x in ['catboost','lightgbm','xgboost','sklearn']})))
for version in [24,27,33,37,40,43,44,45]:
    out = root / f'admet_model_outputs_v{version}'
    print(version, [p.name for p in out.glob('*') if p.is_file()])
    if version == 45:
        print('v45 export', (out/'export.json').read_text())
print('v45 files', [(str(p.relative_to(root)),p.stat().st_size) for p in (root/'admet_submissions/v45').glob('*')])
print('available disk', os.statvfs(root).f_bavail*os.statvfs(root).f_frsize)
