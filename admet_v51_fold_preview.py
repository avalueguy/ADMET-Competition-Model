"""Report completed fold metrics only; no selection from partial folds."""
import json
import os
import numpy as np
import admet_v24_models as b
assert os.name!='nt'
d=b.build_data();folds=np.load(b.OUT/'folds.npy')
base=np.load('admet_model_outputs_v46/baseline_cv_proxy.npy')
from admet_v51_run import OUT,variants
for task in ['solubility_aqsoldb','lipophilicity_astrazeneca','ld50_zhu','ppbr_az','caco2_wang']:
    k=d['tasks'].index(task)
    for f in range(3):
        path=OUT/task/f'fold{f}'/'predictions.npz'
        if not path.exists():continue
        z=np.load(path);q=z['query'];pred=variants(z['p'])
        print(json.dumps(dict(task=task,fold=f,n=len(q),baseline_mae=b.metric('MAE',d['y'][q],base[q]),
            candidate_mae=[b.metric('MAE',d['y'][q],p) for p in pred],
            relative_gains=[b.delta('MAE',d['y'][q],base[q],p) for p in pred])),flush=True)
