"""Prepare immutable inputs with full v382 control predictions; no fitting."""
from pathlib import Path
import hashlib,json,os,shutil,tarfile,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/regression9_v384'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 assert not (EXP/'source_freeze.json').exists(),'Never overwrite frozen experiment'
 EXP.mkdir(parents=True,exist_ok=True);prev=ROOT/'experiments/regression9_v377'
 manifest=json.loads((prev/'data_manifest.json').read_text());best=json.loads((ROOT/'docs/current_best_after_v382_20261105.json').read_text())
 assert sha(ROOT/best['csv'])==best['csv_sha256'];manifest['baseline']=best
 names=['official_rows.npz','expert_features.npz','stereo_keys.npz','submission_template.csv']+[r['design'] for r in manifest['controls']]
 for name in names:
  src,dest=prev/name,EXP/name;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert sha(src)==sha(dest)
  else:os.link(src,dest)
 shutil.copyfile(ROOT/best['csv'],EXP/'v382_baseline.csv')
 updated=[]
 for r in manifest['controls']:
  src,dest=prev/r['control'],EXP/r['control'];dest.parent.mkdir(parents=True,exist_ok=True)
  a=dict(np.load(src));t,p=r['task'],r['part']
  if t in ('lipophilicity_astrazeneca','ppbr_az'):
   version,name,w=('v380','sqrt_polynomial_C30_native',.1) if t=='lipophilicity_astrazeneca' else ('v381','simplex_reg01',.25)
   expert=ROOT/f'experiments/regression9_{version}/outputs/tasks/{t}/{p}/{name}/predictions.npz';z=np.load(expert)
   for k in ('fit','query'):np.testing.assert_array_equal(a[k],z[k])
   a['p']=(1-w)*a['p']+w*z['p'];a['score_p']=a['p'].copy();np.savez_compressed(dest,**a)
   updated.append(dict(task=t,part=p,old_sha256=sha(src),expert=expert.relative_to(ROOT).as_posix(),expert_sha256=sha(expert),weight=w,new_sha256=sha(dest)))
  elif not dest.exists():os.link(src,dest)
  else:assert sha(src)==sha(dest)
  r['control_sha256']=sha(dest)
 rows=list(csv.DictReader((EXP/'v382_baseline.csv').open(encoding='utf-8-sig',newline='')))
 for t in manifest['pilot_tasks']:
  a=np.load(EXP/'controls'/t/'final.npz');expected=np.array([float(r['y_pred'])for r in rows if r['task']==t])
  key='p' if t in manifest['pilot_tasks'][:5] else 'score_p'
  np.testing.assert_allclose(a[key],expected,rtol=0,atol=1e-10)
 manifest['v382_baseline_sha256']=sha(EXP/'v382_baseline.csv');manifest['control_updates']=updated
 save(EXP/'data_manifest.json',manifest)
 # Keep familiar runner semantics, but train one complete pair of curves per job.
 runner=(ROOT/'tools/v381_run_experts.py').read_text().replace('v381_oof_pareto','v384_recursive_median').replace('v381_verify_experts','v384_verify_experts').replace('381000','384000')
 runner=runner.replace("[m.predict(d,%r,%r,c) for c in m.configs(%r)]","[m.train_curve(d,%r,%r,b) for b in m.BASES]").replace("%(t,p,t)","%(t,p)")
 (ROOT/'tools/v384_run_experts.py').write_text(runner,encoding='utf-8')
 for n in ('completion','monitor'):
  text=(ROOT/f'tools/v381_{n}.py').read_text().replace('v381','v384').replace('216','324')
  if n=='monitor':
   text=text.replace("52","144").replace("3*sum(p.parts[-3] in ('fold0','fold1') for p in (ROOT/'outputs/base').glob('*/*/*/state.joblib'))","9*sum(p.parts[-3] in ('fold0','fold1') for p in (ROOT/'outputs/curves').glob('*/*/*/state.joblib'))")
  (ROOT/f'tools/v384_{n}.py').write_text(text,encoding='utf-8')
 sources=('v371_mae_experts.py','v377_regression_experts.py','v384_recursive_median.py','v384_run_experts.py','v384_verify_experts.py','v384_smoke.py','v384_monitor.py','v384_completion.py')
 entries=[]
 for name in sources:shutil.copyfile(ROOT/'tools'/name,EXP/name);entries.append(dict(name=name,sha256=sha(EXP/name)))
 freeze=dict(files=entries,all_fitting_remote=True,baseline=best,workers=2,control_final_matches_v382=True)
 save(EXP/'source_freeze.json',freeze)
 archive=ROOT/'releases/v384_input_delta.tar.gz'
 with tarfile.open(archive,'w:gz')as tf:
  for name in sources+('source_freeze.json','data_manifest.json','v382_baseline.csv')+tuple(r['control']for r in manifest['controls'] if r['task']in('lipophilicity_astrazeneca','ppbr_az')):tf.add(EXP/name,arcname=name)
 receipt=dict(archive=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,updated_controls=10,baseline_csv_matches=True)
 save(ROOT/'docs/v384_input_package.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
