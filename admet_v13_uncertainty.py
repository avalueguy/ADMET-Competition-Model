"""Conditional uncertainty of adopted changes; not an additional selection gate."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from admet_v6_runner import metrics, kind, utility
from admet_v13_models import load_task


def main():
    rows = []
    for path in sorted(Path('admet_model_outputs_v13').glob('*/decision.json')):
        decision = json.loads(path.read_text())
        if not decision['accepted']:
            continue
        task = decision['task']; d = load_task(task)
        n = len(d['frames']['train']); nv = len(d['frames']['valid'])
        groups = d['groups'][n:n+nv]; unique = np.unique(groups)
        indices = [np.flatnonzero(groups==group) for group in unique]
        frame = pd.read_csv(path.parent/'validation_comparison.csv')
        y = frame.Y_true.to_numpy(); before = frame.baseline.to_numpy(); after = frame.blend.to_numpy()
        metric = decision['selection']['metric']; kk = kind(metric)
        rng = np.random.default_rng(992509); gains = []
        if len(unique)>1:
            for _ in range(500):
                ix = np.concatenate([indices[i] for i in rng.integers(len(indices), size=len(indices))])
                if kk=='classification' and len(np.unique(y[ix]))<2:
                    continue
                gains.append(utility(metric,metrics(kk,y[ix],after[ix]))-utility(metric,metrics(kk,y[ix],before[ix])))
        rows.append(dict(task=task, metric=metric, valid_rows=nv, scaffold_count=len(unique),
            conditional_ci_low=float(np.quantile(gains,.025)) if gains else None,
            conditional_ci_high=float(np.quantile(gains,.975)) if gains else None,
            bootstrap_draws=len(gains), limitation='Conditional on reused selected validation; not selection-adjusted or a platform score interval'))
    columns=['task','metric','valid_rows','scaffold_count','conditional_ci_low','conditional_ci_high','bootstrap_draws','limitation']
    pd.DataFrame(rows,columns=columns).to_csv('admet_diagnostics/v13_uncertainty.csv',index=False)
    print(json.dumps(rows),flush=True)


if __name__=='__main__':
    main()
