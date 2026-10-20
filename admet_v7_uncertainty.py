"""Conditional scaffold bootstrap, descriptive only: never used for reselection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from admet_v6_runner import metrics,kind,utility
from admet_v7_models import load_task


def main():
    rows=[]
    for path in sorted(Path('admet_model_outputs_v7').glob('*/decision.json')):
        d=json.loads(path.read_text())
        if not d['accepted']:continue
        task=d['task'];data=load_task(task);n=len(data['frames']['train']);nv=len(data['frames']['valid']);groups=data['groups'][n:n+nv]
        unique=np.unique(groups);indices=[np.flatnonzero(groups==g) for g in unique]
        frame=pd.read_csv(path.parent/'validation_comparison.csv');y=frame.Y_true.to_numpy();b=frame.baseline.to_numpy();p=frame.blend.to_numpy()
        metric=d['selection']['metric'];kk=kind(metric);rng=np.random.default_rng(792509);gains=[]
        if len(unique)>1:
            for _ in range(500):
                ix=np.concatenate([indices[i] for i in rng.integers(len(indices),size=len(indices))])
                if kk=='classification' and len(np.unique(y[ix]))<2:continue
                gains.append(utility(metric,metrics(kk,y[ix],p[ix]))-utility(metric,metrics(kk,y[ix],b[ix])))
        rows.append(dict(task=task,metric=metric,valid_rows=nv,scaffold_count=len(unique),conditional_ci_low=float(np.quantile(gains,.025)) if gains else None,
                         conditional_ci_high=float(np.quantile(gains,.975)) if gains else None,bootstrap_draws=len(gains),
                         limitation='Conditional on selected reused validation; not selection-adjusted; not platform score'))
    pd.DataFrame(rows,columns=['task','metric','valid_rows','scaffold_count','conditional_ci_low','conditional_ci_high','bootstrap_draws','limitation']).to_csv('admet_diagnostics/v7_uncertainty.csv',index=False)
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
