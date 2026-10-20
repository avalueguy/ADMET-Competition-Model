import json
import os
from pathlib import Path
import numpy as np

root=Path('/home/czp123/admet_v23_20260928')
os.chdir(root)
versions=['24','27','29','32','33','36','37','39','40','42','43','44','45','45b','45c','46']
for v in versions:
    out=root/f'admet_model_outputs_v{v}'
    print('\nV',v,'exists',out.exists())
    for name in ['analysis.json','valid_predictions.npy','cv_predictions.npy','test_predictions_compact.json','baseline_valid.npy','baseline_cv_proxy.npy','fusion_replay.json']:
        p=out/name
        if not p.exists():continue
        try:
            if p.suffix=='.npy':print(name,np.load(p).shape)
            else:print(name,p.stat().st_size)
        except Exception as exc:print(name,exc)
    p=out/'analysis.json'
    if p.exists():
        try:
            a=json.loads(p.read_text())
            print('status',a.get('status'),'selected',a.get('selected_tasks',[])[:20])
        except Exception as exc:print('analysis error',exc)
for v in [24,27,33,37,40,43,44,45,46]:
    p=root/f'admet_model_outputs_v{v}/test_predictions_compact.json'
    if p.exists():
        a=json.loads(p.read_text()); print('compact',v,len(a),list(a)[:8])
