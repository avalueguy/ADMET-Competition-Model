"""Report selected-expert gains by label-free chemistry domains; no extra tuning."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import json
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import admet_v24_models as b
import admet_v61_run as v
b.remote_only();d=b.build_data();f=np.load(b.OUT/'folds.npy');bv,bc=v.baseline(d);names=[n for n,fn in Descriptors.descList if n not in {'Ipc','AvgIpc','BalabanJ'}];cyclic=d['desc'][:,names.index('RingCount')]>0;report=json.loads((v.OUT/'release.json').read_text());records=[]
for task,r in report['selected'].items():
    k=d['tasks'].index(task);m=d['metrics'][k];s=r['selection'];cp=np.load(v.OUT/task/'cv_predictions.npy')[s['variant']];tr=np.flatnonzero((d['tid']==k)&(d['split']==0));te=np.flatnonzero((d['tid']==k)&(d['split']==2));parts=[]
    for fold in range(3):
        ix=tr[f[tr]==fold];merged=b.blend(m,bc[ix],cp[ix],s['weight'])
        for ring in [0,1]:
            active=cyclic[d['mi'][ix]]==ring
            if active.sum()<20 or (m in ['AUROC','AUPRC'] and len(np.unique(d['y'][ix[active]]))<2):continue
            parts.append(dict(fold=fold,cyclic=ring,rows=int(active.sum()),gain=b.delta(m,d['y'][ix[active]],bc[ix[active]],merged[active])))
    records.append(dict(task=task,test_cyclic=float(np.mean(cyclic[d['mi'][te]])),diagnostics=parts))
b.save(v.OUT/'domain_check.json',dict(status='COMPLETE',tasks=records,selection_changed=False,test_labels_used=False))
print(json.dumps(records),flush=True)
