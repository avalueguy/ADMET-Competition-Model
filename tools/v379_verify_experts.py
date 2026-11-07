"""Independent model reload, selection, input hashes, query-label perturbation."""
from pathlib import Path
import json
import tarfile
import numpy as np
import v379_distribution_transformer as m
import v379_run_experts as run
ROOT=Path(__file__).resolve().parent


def main():
 d=m.load();m.setup();sig=m.signature();manifest=json.loads((ROOT/'data_manifest.json').read_text())
 for name,key in (('official_rows.npz','official_rows_sha256'),('expert_features.npz','expert_features_sha256'),('stereo_keys.npz','stereo_keys_sha256')):assert m.sha(ROOT/name)==manifest[key]
 for r in manifest['controls']:assert m.sha(ROOT/r['control'])==r['control_sha256'] and m.sha(ROOT/r['design'])==r['design_sha256']
 assert np.isnan(d['y'][d['split']==2]).all()
 records=[];maximum=0.
 for path in sorted((m.OUT/'tasks').glob('*/*/*/predictions.npz')):
  t,p,name=path.parts[-4:-1];c=next(c for c in m.CONFIGS if c['name']==name);z=np.load(path);a=m.design(d,t,p)
  assert str(z['signature'])==sig
  for k in ('fit','query'):np.testing.assert_array_equal(z[k],a[k])
  out=m.replay(d,t,p,c);err=float(np.max(np.abs(out-z['p'])));assert err<2e-5;maximum=max(maximum,err)
  changed={**d,'y':d['y'].copy()};changed['y'][a['query']]=987654.
  np.testing.assert_array_equal(out,m.replay(changed,t,p,c))
  meta=json.loads(path.with_name('metadata.json').read_text());assert m.sha(ROOT/meta['model_path'])==meta['model_sha256'] and m.sha(path)==meta['prediction_sha256']
  records.append(dict(task=t,part=p,config=name,error=err));m.emit('verified',task=t,part=p,config=name)
 # Recompute development choice from saved predictions, without new fitting.
 dev=json.loads((m.OUT/'development.json').read_text());frozen=json.loads((m.OUT/'frozen_selection.json').read_text())
 chosen=[]
 for r in dev['records']:
  values=[]
  for c in m.CONFIGS:
   for w in m.WEIGHTS:
    parts=[run.score(d,r['task'],p,c,w) for p in ('fold0','fold1')];objective=min(x['gain'] for x in parts)
    if r['task'] in m.TASKS[5:] and min(x['rank_gain'] for x in parts)<=0:objective=min(objective,-1.)
    values.append(dict(config=c,weight=w,objective=objective))
  best=max(values,key=lambda x:(x['objective'],-x['weight'],-x['config']['rounds'],-list(m.CONFIGS).index(x['config'])))
  for k in ('config','weight','objective'):assert best[k]==r['best'][k]
  if best['objective']>0:chosen.append((r['task'],best['config'],best['weight']))
 assert chosen==[(r['task'],r['config'],r['weight']) for r in frozen['selected']]
 conf=json.loads((m.OUT/'confirmation.json').read_text())
 for r in conf['records']:
  parts=[run.score(d,r['task'],p,r['config'],r['weight']) for p in ('fold0','fold1','fold2','valid')]
  assert parts==r['confirmation']
  for b in r['bootstrap']:assert run.boot(d,r['task'],b['part'],r['config'],r['weight'])==b
 result=dict(status='PASS',signature=sig,models=len(records),full_replay_max_error=maximum,records=records,no_fitting_performed=True)
 m.save(m.OUT/'independent_verification.json',result)
 files=[p for p in m.OUT.rglob('*') if p.is_file() and p.suffix!='.joblib']+[ROOT/n for n in ('v371_mae_experts.py','v377_regression_experts.py','v379_distribution_transformer.py','v379_run_experts.py','v379_verify_experts.py')]
 final_models=[]
 for r in conf['records']:
  if r['experimental_eligible']:
   p=m.OUT/'curves'/r['task']/'final'/r['config']['base']/'state.joblib';final_models.append(dict(remote_path=str(p),sha256=m.sha(p),bytes=p.stat().st_size))
 m.save(ROOT/'result_manifest.json',dict(signature=sig,files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=m.sha(p),bytes=p.stat().st_size) for p in files],final_models=final_models))
 archive=ROOT/'results.tar.gz'
 with tarfile.open(archive,'w:gz') as tf:
  for p in files+[ROOT/'result_manifest.json']:tf.add(p,arcname=p.relative_to(ROOT).as_posix())
 m.save(ROOT/'result_archive.json',dict(status='PASS',signature=sig,sha256=m.sha(archive),bytes=archive.stat().st_size))
 print(json.dumps({k:v for k,v in result.items() if k!='records'}))


if __name__=='__main__':main()
