"""Expand frozen inputs to nine continuous outcomes; local packaging only."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import sys
import tarfile
import numpy as np
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/regression9_v377'
TASKS=('caco2_wang','ld50_zhu','lipophilicity_astrazeneca','ppbr_az','solubility_aqsoldb',
       'clearance_hepatocyte_az','clearance_microsome_az','half_life_obach','vdss_lombardo')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
 assert not (EXP/'data_manifest.json').exists(),'Frozen inputs already prepared'
 sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
 import v146_current_baseline as legacy
 d=legacy.load();old=ROOT/'experiments/mae_experts_v374'
 manifest=json.loads((old/'data_manifest.json').read_text(encoding='utf-8'))
 raw=dict(np.load(old/'official_rows.npz'))
 for k in ('mi','tid','split','y'):np.testing.assert_array_equal(d[k],raw[k])
 best=json.loads((ROOT/'docs/current_best_after_v374_20261105.json').read_text(encoding='utf-8'))
 assert sha(ROOT/best['csv'])==best['csv_sha256']
 with (ROOT/best['csv']).open(encoding='utf-8-sig',newline='') as f:table=list(csv.DictReader(f))
 EXP.mkdir(parents=True,exist_ok=True)
 # Reused large official arrays are inherited remotely and verified by hash.
 for name in ('official_rows.npz','expert_features.npz','stereo_keys.npz','submission_template.csv'):
  shutil.copyfile(old/name,EXP/name)
 shutil.copyfile(ROOT/best['csv'],EXP/'v374_baseline.csv')
 records=[]
 for task in TASKS:
  for part in ('fold0','fold1','fold2','valid','final'):
   if task in TASKS[:5]:
    a=dict(np.load(old/'controls'/task/(part+'.npz')));z=dict(np.load(old/'design'/task/(part+'.npz')))
    if task=='solubility_aqsoldb':
     extra=np.load(old/'outputs/tasks'/task/part/'honest_local_k16/predictions.npz')
     for k in ('fit','query'):np.testing.assert_array_equal(a[k],extra[k])
     a['p']=.8*a['p']+.2*extra['p']
   else:
    path=ROOT/'admet_model_outputs_v146/controls'/task/(part+'.npz');a=dict(np.load(path))
    fit,query,x,qx,source=legacy.old.features(d,d['tasks'].index(task),part,'augmented')
    for k,v in (('fit',fit),('query',query)):np.testing.assert_array_equal(a[k],v)
    z=dict(fit=fit,query=query,x=x,qx=qx,source=source)
   if part=='final':
    expected=np.array([float(r['y_pred']) for r in table if r['task']==task]);np.testing.assert_array_equal(a['p'],expected)
   raw_score=a['p'].copy()
   if task in TASKS[5:]:
    # Historical score output has arbitrary units. Fit-label empirical quantiles
    # supply a stated reference in native units; no query labels are consulted.
    yy=np.sort(raw['y'][a['fit']]);u=(rankdata(raw_score,method='average')-.5)/len(raw_score)
    a['p']=np.quantile(yy,u,method='linear')
   a['score_p']=raw_score
   for k in ('fit','query'):np.testing.assert_array_equal(a[k],z[k])
   assert not np.intersect1d(a['fit'],a['query']).size
   if part!='final':assert not np.intersect1d(raw['groups'][raw['mi'][a['fit']]],raw['groups'][raw['mi'][a['query']]]).size
   assert np.isfinite(a['p']).all()
   cp=EXP/'controls'/task/(part+'.npz');cp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cp,**a)
   dp=EXP/'design'/task/(part+'.npz');dp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dp,**z)
   records.append(dict(task=task,part=part,fit=len(a['fit']),query=len(a['query']),control=cp.relative_to(EXP).as_posix(),control_sha256=sha(cp),design=dp.relative_to(EXP).as_posix(),design_sha256=sha(dp)))
   print(json.dumps(dict(prepared=task,part=part)),flush=True)
 value=dict(tasks=manifest['tasks'],metrics=manifest['metrics'],pilot_tasks=TASKS,baseline=best,rows=manifest['rows'],
   official_rows_sha256=sha(EXP/'official_rows.npz'),expert_features_sha256=sha(EXP/'expert_features.npz'),stereo_keys_sha256=sha(EXP/'stereo_keys.npz'),controls=records,
   units='Native official Y values. Four rank-task controls calibrated by fit-label empirical quantiles of query score ranks; not historical native-unit model errors.',
   historical_control_limitation='Partial historical proxy OOF, exact recent layers. Four rank MAEs are calibrated reference only. Public labels reused; no new blind set.',
   all_fitting_remote=True,training_host='connect.nmb2.seetacloud.com:25870',external_data=False,test_labels_parsed=False)
 save(EXP/'data_manifest.json',value)
 for name in ('v371_mae_experts.py','v377_regression_experts.py','v377_run_experts.py','v377_verify_experts.py'):
  shutil.copyfile(ROOT/'tools'/name,EXP/name)
 names=[p for p in EXP.rglob('*') if p.is_file() and p.name not in ('official_rows.npz','expert_features.npz','stereo_keys.npz')]
 archive=ROOT/'releases/v377_regression9_delta.tar.gz'
 with tarfile.open(archive,'w:gz') as tf:
  for p in names:tf.add(p,arcname=p.relative_to(EXP).as_posix())
 save(ROOT/'docs/v377_input_package.json',dict(status='PASS',archive=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,large_arrays_inherited_from='mae_experts_v374',partitions=len(records)))
 print(json.dumps(dict(status='PREPARED',bytes=archive.stat().st_size,partitions=len(records))))


if __name__=='__main__':main()
