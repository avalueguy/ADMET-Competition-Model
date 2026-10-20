"""Inspect source errors and prediction scale, without reading any test target values."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import admet_v24_models as b
assert os.name!='nt'
d=b.build_data()
valid=np.load('admet_model_outputs_v46/baseline_valid.npy')
cv=np.load('admet_model_outputs_v46/baseline_cv_proxy.npy')
folds=np.load(b.OUT/'folds.npy')
source=pd.read_csv(next(Path('admet_submissions/v46').glob('*.csv')),dtype=str)
for k,t in enumerate(d['tasks']):
    if d['metrics'][k]!='MAE':continue
    tr=np.flatnonzero((d['tid']==k)&(d['split']==0));vi=np.flatnonzero((d['tid']==k)&(d['split']==1))
    te=np.flatnonzero((d['tid']==k)&(d['split']==2));p=source.loc[source.task.eq(t),'y_pred'].to_numpy(float)
    q=lambda x:np.quantile(x,[0,.01,.1,.5,.9,.99,1]).round(4).tolist()
    print(json.dumps(dict(task=t,train_target=q(d['y'][tr]),valid_target=q(d['y'][vi]),test_prediction=q(p),
        cv_mae=[b.metric('MAE',d['y'][ix],cv[ix]) for f in range(3) for ix in [tr[folds[tr]==f]]],
        valid_mae=b.metric('MAE',d['y'][vi],valid[vi]),
        cv_median_residual=[float(np.median(d['y'][ix]-cv[ix])) for f in range(3) for ix in [tr[folds[tr]==f]]])),flush=True)
