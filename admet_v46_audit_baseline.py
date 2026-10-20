"""Reproduce the exact locally delivered v45 from its remote source predictions."""
import csv
import io
import json
import os
import sys
from pathlib import Path
import numpy as np

root=Path('/home/czp123/admet_v23_20260928')
os.chdir(root)
sys.path.insert(0,str(root))
import admet_v24_models as b
import admet_v27_models as graph

out=root/'admet_model_outputs_v46'
def read(path):return list(csv.reader(io.StringIO(path.read_bytes().decode('utf-8-sig'),newline='')))
old=read(next((root/'admet_submissions/v44').glob('*.csv')))
new=read(out/'baseline_v45.csv')
spec=json.loads((out/'v45_source_export.json').read_text())
compact=json.loads((graph.OUT/'test_predictions_compact.json').read_text())
report={}
for task,s in spec['changed_tasks'].items():
    pos=[i for i,a in enumerate(new) if i>0 and a[0]==task]
    a=np.array([float(old[i][4]) for i in pos])
    z=np.array(compact[task])[s['branch']]
    expected=b.blend(new[pos[0]][5],a,z,s['weight'])
    actual=np.array([float(new[i][4]) for i in pos])
    np.testing.assert_array_equal(actual,expected)
    report[task]=dict(rows=len(pos),replay_exact=True)
for a,c in zip(old,new):
    if a[0] not in spec['changed_tasks']:assert a==c
assert b.sha(out/'baseline_v45.csv')==spec['sha256']
b.save(out/'baseline_file_audit.json',dict(status='PASS',sha256=spec['sha256'],
    exact_export_replayed=True,changed_tasks=report))
print(json.dumps(dict(status='PASS',changed_tasks=list(report)),ensure_ascii=False))
