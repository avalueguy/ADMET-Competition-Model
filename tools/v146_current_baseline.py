"""Frozen v145 model controls and current official feedback, no new tuning."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='3'
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v131_current_controls as old
import v135_broad_boosting as boost
import v139_classification_kernels as kernel
b,r83=old.b,old.r83
OUT=ROOT/'admet_model_outputs_v146'
ANCHOR=next((ROOT/'admet_submissions/v145_feedback_repair').glob('*.csv'))
ANCHOR_SHA='3ffd68785b3fc321c8e81a32b23a09cf2493ff26240555a3472a315c572706e7'
PARTS=old.PARTS

def sig():return hashlib.sha256((b.sha(__file__)+old.sig()+boost.sig()+kernel.sig()+ANCHOR_SHA).encode()).hexdigest()
def load():
    d=old.load();assert b.sha(ANCHOR)==ANCHOR_SHA;d['v145_rows']=b.table(ANCHOR)
    d['v144_choices']=json.loads((ROOT/'admet_model_outputs_v144/evidence.json').read_text())['selected'];return d
def split(d,k,part):return old.split(d,k,part)
def baseline(d,k,part):
    path=OUT/'controls'/d['tasks'][k]/(part+'.npz');fit,q=split(d,k,part)
    if path.exists():
        z=np.load(path);assert str(z['signature'])==sig();np.testing.assert_array_equal(z['query'],q);return z['p'].copy()
    if part=='final':p=np.array([float(r['y_pred']) for r in d['v145_rows'] if r['task']==d['tasks'][k]])
    else:
        p=old.baseline(d,k,part);task=d['tasks'][k]
        if task in ('ames','cyp2c9_veith','cyp2d6_veith'):
            ch=d['v144_choices'][task];m=boost if ch['version']=='v135' else kernel
            p=old.mix(d['metrics'][k],p,m.predict(d,k,part,ch['record']['config']),ch['record']['weight'])
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,signature=sig(),fit=fit,query=q,p=p);return p
def mix(m,a,z,w):return old.mix(m,a,z,w)
def threshold(m):return .012 if m=='MAE' else .008 if m=='Spearman' else .003
def tasks(d):
    f=json.loads((ROOT/'docs/platform_scorecard_v145_20261027.json').read_text());return [r['task'] for r in sorted(f['tasks'],key=lambda r:r['score']) if r['task']!='hia_hou']
def audit():
    d=load();records=[]
    for t in tasks(d):
        k=d['tasks'].index(t);parts=[]
        for part in PARTS:
            fit,q=split(d,k,part);p=baseline(d,k,part)
            assert np.isfinite(p).all() and not set(fit)&set(q)
            parts.append(dict(part=part,fit=len(fit),query=len(q),groups=len(np.unique(d['groups'][q])),score=r83.score(d['metrics'][k],d['y'][q],p) if part!='final' else None))
        records.append(dict(task=t,parts=parts));print(json.dumps(records[-1]),flush=True)
    b.save(OUT/'baseline_audit.json',dict(status='PASS',signature=sig(),anchor_sha256=ANCHOR_SHA,records=records,
        limitations='v145 carried predictions reconstructed on original folds. Inherits approximate historical upstream OOF; not a full nested refit. New structural controls and perturbation checks are additional safeguards, not new independent data.'))
if __name__=='__main__':audit()
