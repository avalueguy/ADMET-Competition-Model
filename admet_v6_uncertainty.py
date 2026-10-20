"""Conditional scaffold bootstrap; does not adjust for model selection bias."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from admet_v6_runner import load_task, metrics, kind, utility

rows=[]
for path in sorted(Path('admet_model_outputs_v6').glob('*/decision.json')):
    d=json.loads(path.read_text())
    if not d['accepted']:continue
    task=d['task'];data=load_task(task);n=len(data['frames']['train']);nv=len(data['frames']['valid'])
    groups=data['groups'][n:n+nv];unique=np.unique(groups)
    frame=pd.read_csv(path.parent/'validation_comparison.csv')
    y=frame.Y_true.to_numpy();old=frame.baseline.to_numpy();new=frame.blend.to_numpy()
    metric=d['selection']['metric'];kk=kind(metric)
    mapping=[np.flatnonzero(groups==g) for g in unique];rng=np.random.default_rng(602509);gains=[]
    if len(unique)>1:
        for _ in range(400):
            ix=np.concatenate([mapping[i] for i in rng.integers(len(unique),size=len(unique))])
            if kk=='classification' and len(np.unique(y[ix]))<2:continue
            before=metrics(kk,y[ix],old[ix]);after=metrics(kk,y[ix],new[ix])
            gains.append(utility(metric,after)-utility(metric,before))
    row=dict(task=task,metric=metric,valid_rows=len(y),valid_scaffolds=len(unique),
             conditional_ci_low=float(np.quantile(gains,.025)) if gains else None,
             conditional_ci_high=float(np.quantile(gains,.975)) if gains else None,
             bootstrap_samples=len(gains),note='Conditional on reused selected predictions; not selection-adjusted or blind score')
    fp=data['raw'][:n,:2048];query=data['raw'][n:n+nv,:2048];nearest=[]
    for start in range(0,nv,256):
        q=query[start:start+256];dot=q@fp.T
        nearest.extend((dot/np.maximum(q.sum(1)[:,None]+fp.sum(1)[None,:]-dot,1)).max(1))
    nearest=np.array(nearest)
    for name,mask in [('distant',nearest<.3),('closer',nearest>=.3)]:
        row[name+'_n']=int(mask.sum())
        if mask.sum()<20 or (kk=='classification' and len(np.unique(y[mask]))<2):continue
        row[name+'_primary_before']=utility(metric,metrics(kk,y[mask],old[mask]))
        row[name+'_primary_after']=utility(metric,metrics(kk,y[mask],new[mask]))
    rows.append(row)
pd.DataFrame(rows).to_csv('admet_diagnostics/v6_uncertainty.csv',index=False)
print(json.dumps(rows))
