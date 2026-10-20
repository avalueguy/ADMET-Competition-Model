"""Compact progress and diagnostic-only CV comparison (does not select)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from admet_upgrade_v3 import metrics, utility, get_metric

root=Path('admet_model_outputs_v6')
for dest in sorted(root.iterdir()):
    if not dest.is_dir() or dest.name in ['features','legacy_features','legacy_mtl']:continue
    candidates=sorted(p.name[:-8] for p in dest.glob('*_oof.csv') if p.name!='baseline_oof.csv')
    line=dict(task=dest.name,candidates=candidates,baseline=(dest/'baseline_oof.csv').exists())
    if (dest/'decision.json').exists():
        d=json.loads((dest/'decision.json').read_text())
        line.update(decision=d.get('reason'),accepted=d.get('accepted'))
        if 'before' in d:line.update(before=d['before'],after=d['after'])
        if 'selection' in d:line.update(family=d['selection']['family'],alpha=d['selection']['alpha'])
    elif (dest/'baseline_oof.csv').exists():
        metric=get_metric(dest.name);kind='classification' if metric in ['AUROC','AUPRC'] else 'regression'
        b=pd.read_csv(dest/'baseline_oof.csv');y=b.Y_true.to_numpy();ix=b.fold.to_numpy();base=b.Y_pred.to_numpy()
        bm=[metrics(kind,y[ix==f],base[ix==f]) for f in range(3)]
        line['base_mean']=float(np.mean([utility(metric,m) for m in bm]))
        scores={}
        for name in candidates:
            q=pd.read_csv(dest/(name+'_oof.csv')).Y_pred.to_numpy()
            scores[name]=round(float(np.mean([utility(metric,metrics(kind,y[ix==f],q[ix==f])) for f in range(3)])),6)
        line['candidate_mean']=scores
    print(json.dumps(line))
