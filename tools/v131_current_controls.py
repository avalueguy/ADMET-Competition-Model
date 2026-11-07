"""Frozen v130 controls and target-excluded context for broad task experiments."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='4'
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v125_low_task_baseline as old
import v127_low_classification as tree
import v119_matched_half_life as half
import v102_count_margin as history
import v109_pharmacophore as pharma
b,r83=old.b,old.r83
OUT=ROOT/'admet_model_outputs_v131'
ANCHOR=next((ROOT/'admet_submissions/v130_low_task_experts').glob('*.csv'))
ANCHOR_SHA='4d1d24761f737b0ffe79e707022fabd1db01cf14799abdab3b23d4b2f249e5cb'
PARTS=('fold0','fold1','fold2','valid','final')

def sig():return hashlib.sha256((b.sha(__file__)+old.sig()+tree.sig()+half.sig()+history.sig()+ANCHOR_SHA).encode()).hexdigest()
def load():
    d=old.load();assert b.sha(ANCHOR)==ANCHOR_SHA
    d['v131_anchor']=b.table(ANCHOR)
    d['v130_selected']=json.loads((ROOT/'admet_model_outputs_v130/evidence.json').read_text())['selected']
    meta=json.loads((pharma.OUT/'features/audit.json').read_text())
    fp=pharma.OUT/'features/pharmacophore.npz';assert meta['signature']==pharma.sig() and b.sha(fp)==meta['sha256']
    d['pharmacophore']=np.load(fp)['x'];return d
def split(d,k,part):return half.split(d,k,part) if d['tasks'][k]=='half_life_obach' else old.split(d,k,part)
def anchor(d,k):return np.array([float(r['y_pred']) for r in d['v131_anchor'] if r['task']==d['tasks'][k]])
def baseline(d,k,part):
    path=OUT/'controls'/d['tasks'][k]/(part+'.npz')
    if path.exists():
        z=np.load(path);assert str(z['signature'])==sig();np.testing.assert_array_equal(z['query'],split(d,k,part)[1]);return z['p'].copy()
    task=d['tasks'][k]
    if part=='final':p=anchor(d,k)
    elif task in d['v130_selected']:
        r=d['v130_selected'][task];p=old.mix(d['metrics'][k],old.baseline(d,k,part),tree.predict(d,k,part,r['record']['view']),r['weight'])
    elif task=='half_life_obach':p=old.mix('Spearman',half.base(d,k,part),half.predict(d,k,part,'full'),.5)
    elif task in old.TASKS:p=old.baseline(d,k,part)
    elif task in ('cyp2c9_veith','cyp2c9_substrate_carbonmangels'):p=r83.baseline(d,k,part)
    elif task in ('bioavailability_ma','cyp2d6_veith'):p=history.base(d,k,part)
    elif task=='ppbr_az':p=old.smooth.r84.baseline(d,k,part)
    else:p=r83.raw_proxy(d,split(d,k,part)[1])
    fit,q=split(d,k,part);assert len(p)==len(q) and np.isfinite(p).all();path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,p=p,fit=fit,query=q,signature=sig());return p
def features(d,k,part,view):
    fit,q=split(d,k,part);aux,cols,source,_,_=old.old.assays.assays(d,k,part)
    assert k not in cols and np.all(d['tid'][source]!=k) and np.all(d['split'][source]<(2 if part=='final' else 1))
    if view=='full':raw=d['views']['multiview']
    else:raw=np.c_[d['views']['physchem'],d['conformer_features'],d['pharmacophore']]
    x=np.c_[aux[fit],raw[d['mi'][fit]]].astype(np.float32);qx=np.c_[aux[q],raw[d['mi'][q]]].astype(np.float32)
    x[~np.isfinite(x)]=np.nan;qx[~np.isfinite(qx)]=np.nan
    return fit,q,x,qx,source
def mix(metric,a,p,w):return old.mix(metric,a,p,w)
def threshold(metric):return .008 if metric=='MAE' else .006 if metric=='Spearman' else .002
def tasks(d):
    score=json.loads((ROOT/'docs/platform_scorecard_v130_20261027.json').read_text())
    return [r['task'] for r in sorted(score['tasks'],key=lambda r:r['score']) if r['task']!='hia_hou']
def audit():
    d=load();records=[]
    for task in tasks(d):
        k=d['tasks'].index(task);parts=[]
        for part in PARTS:
            fit,q=split(d,k,part);p=baseline(d,k,part)
            parts.append(dict(part=part,fit=len(fit),query=len(q),score=r83.score(d['metrics'][k],d['y'][q],p) if part!='final' else None))
        if task in ('cyp3a4_substrate_carbonmangels','half_life_obach'):quality='Matched archived fitting/query rows; reconstructed current branch'
        elif task in ('herg','clearance_hepatocyte_az','cyp2d6_substrate_carbonmangels','pgp_broccatelli','cyp2c9_veith','cyp2c9_substrate_carbonmangels','bioavailability_ma','cyp2d6_veith','ppbr_az'):quality='Reconstructed carried branch; upstream proxy limitations'
        else:quality='Approximate history proxy; full historical OOF unavailable'
        records.append(dict(task=task,metric=d['metrics'][k],control_quality=quality,parts=parts));print(json.dumps(records[-1]),flush=True)
    b.save(OUT/'baseline_audit.json',dict(status='PASS',signature=sig(),anchor_sha256=ANCHOR_SHA,records=records,test_labels_used=False,limitations='Development labels repeatedly reused. Some controls approximate; official total improvement unconfirmed.'))
if __name__=='__main__':audit()
