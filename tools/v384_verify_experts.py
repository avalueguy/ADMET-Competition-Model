"""Independently rebuild prediction kernels, replay models and selection."""
from pathlib import Path
import json,tarfile,joblib,numpy as np
import v384_recursive_median as m
import v384_run_experts as run
ROOT=Path(__file__).resolve().parent
def main():
 m.setup();d=m.load();sig=m.signature();manifest=json.loads((ROOT/'data_manifest.json').read_text())
 for n,k in (('official_rows.npz','official_rows_sha256'),('expert_features.npz','expert_features_sha256'),('stereo_keys.npz','stereo_keys_sha256'),('v382_baseline.csv','v382_baseline_sha256')):assert m.sha(ROOT/n)==manifest[k]
 for r in manifest['controls']:
  for key in ('control','design'):assert m.sha(ROOT/r[key])==r[key+'_sha256']
 for r in json.loads((ROOT/'source_freeze.json').read_text())['files']:assert m.sha(ROOT/r['name'])==r['sha256']
 assert np.isnan(d['y'][d['split']==2]).all()
 curves=[];records=[];models=0
 for path in sorted((m.OUT/'curves').glob('*/*/*/state.joblib')):
  t,p,b=path.parts[-4:-1]
  if p=='smoke':continue
  a=m.design(d,t,p);s=joblib.load(path);assert s['signature']==sig
  for k in ('fit','query'):np.testing.assert_array_equal(a[k],s[k])
  assert len(s['history'])==9 and [r['round']for r in s['snapshots']]==list(m.HORIZONS)
  meta=json.loads(path.with_name('metadata.json').read_text());assert meta['model_sha256']==m.sha(path)
  for r in s['snapshots']:assert r['model'].fit_status_==0 and np.isfinite(r['metric']).all() and r['metric'].min()>0
  x,qx,_=m.make_arrays(d,a,s['transform']);changed={**d,'y':d['y'].copy()};changed['y'][a['query']]=987654.
  xx,qq,_=m.make_arrays(changed,a,s['transform']);np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(qx,qq)
  for r in s['snapshots']:
   block=x[:40];k=m.kernel(block,block,r['metric'],s['base']['bandwidth']);assert np.linalg.eigvalsh(k.astype(float)).min()>-1e-4
  curves.append(dict(task=t,part=p,base=b,converged_solves=9,history=meta['history'],seconds=meta['seconds'],query_labels_excluded=True));models+=9
 for path in sorted((m.OUT/'tasks').glob('*/*/*/predictions.npz')):
  t,p,name=path.parts[-4:-1]
  if p=='smoke':continue
  c=next(c for c in m.CONFIGS if c['name']==name);z=np.load(path);assert str(z['signature'])==sig
  rp=m.replay(d,t,p,c);np.testing.assert_array_equal(rp,z['p'])
  a=m.design(d,t,p);changed={**d,'y':d['y'].copy()};changed['y'][a['query']]=987654.
  np.testing.assert_array_equal(rp,m.replay(changed,t,p,c))
  meta=json.loads(path.with_name('metadata.json').read_text());assert m.sha(path)==meta['prediction_sha256'] and m.sha(ROOT/meta['model_path'])==meta['model_sha256']
  records.append(dict(task=t,part=p,config=name,replay_error=0.,query_label_perturbation_pass=True))
 dev=json.loads((m.OUT/'development.json').read_text());chosen=[]
 for r in dev['records']:
  choices=[];t=r['task']
  for c in m.configs(t):
   for w in m.WEIGHTS:
    parts=[run.score(d,t,p,c,w)for p in ('fold0','fold1')];objective=min(x['gain']for x in parts)
    if t in m.TASKS[5:] and min(x['rank_gain']for x in parts)<=0:objective=min(objective,-1.)
    choices.append(dict(config=c,weight=w,objective=objective))
  b=max(choices,key=lambda x:(x['objective'],-x['weight'],-list(m.configs(t)).index(x['config'])))
  for k in ('config','weight','objective'):assert b[k]==r['best'][k]
  if b['objective']>0:chosen.append((t,b['config'],b['weight']))
 frozen=json.loads((m.OUT/'frozen_selection.json').read_text());assert chosen==[(r['task'],r['config'],r['weight'])for r in frozen['selected']]
 conf=json.loads((m.OUT/'confirmation.json').read_text())
 for r in conf['records']:
  assert [run.score(d,r['task'],p,r['config'],r['weight'])for p in ('fold0','fold1','fold2','valid')]==r['confirmation']
  for b in r['bootstrap']:assert run.boot(d,r['task'],b['part'],r['config'],r['weight'])==b
 value=dict(status='PASS',signature=sig,models=models,curves=len(curves),predictions=len(records),full_replay_max_error=0.,no_fitting_performed=True,curve_records=curves,records=records)
 m.save(m.OUT/'independent_verification.json',value)
 files=[p for p in m.OUT.rglob('*')if p.is_file() and p.suffix!='.joblib' and 'smoke'not in p.parts]
 files += [ROOT/r['name']for r in json.loads((ROOT/'source_freeze.json').read_text())['files']]
 for r in conf['records']:
  if r['experimental_eligible']:files.append(m.OUT/'curves'/r['task']/'final'/r['config']['base']/'state.joblib')
 m.save(ROOT/'result_manifest.json',dict(signature=sig,files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=m.sha(p),bytes=p.stat().st_size)for p in files],all_other_curves_retained_remote=True))
 archive=ROOT/'results.tar.gz'
 with tarfile.open(archive,'w:gz')as tf:
  for p in files+[ROOT/'result_manifest.json']:tf.add(p,arcname=p.relative_to(ROOT).as_posix())
 m.save(ROOT/'result_archive.json',dict(status='PASS',sha256=m.sha(archive),bytes=archive.stat().st_size,signature=sig))
 print(json.dumps({k:v for k,v in value.items()if k not in ('records','curve_records')}))
if __name__=='__main__':main()
