"""No-fit replay of kernel specialists, hash checks and frozen selection."""
from pathlib import Path
import json,tarfile,joblib,numpy as np
import v385_warped_median as m
import v385_run_experts as run
ROOT=Path(__file__).resolve().parent
def main():
 m.setup();d=m.load();sig=m.signature();manifest=json.loads((ROOT/'data_manifest.json').read_text())
 for n,k in (('official_rows.npz','official_rows_sha256'),('expert_features.npz','expert_features_sha256'),('stereo_keys.npz','stereo_keys_sha256'),('v382_baseline.csv','v382_baseline_sha256')):assert m.sha(ROOT/n)==manifest[k]
 for r in manifest['controls']:
  for key in('control','design'):assert m.sha(ROOT/r[key])==r[key+'_sha256']
 for r in json.loads((ROOT/'source_freeze.json').read_text())['files']:assert m.sha(ROOT/r['name'])==r['sha256']
 assert np.isnan(d['y'][d['split']==2]).all();records=[]
 for path in sorted((m.OUT/'tasks').glob('*/*/*/predictions.npz')):
  t,p,name=path.parts[-4:-1]
  if p=='smoke':continue
  c=next(c for c in m.CONFIGS if c['name']==name);z=np.load(path);assert str(z['signature'])==sig
  rp=m.replay(d,t,p,c);np.testing.assert_array_equal(rp,z['p']);s=joblib.load(path.with_name('state.joblib'));assert s['model'].fit_status_==0
  a=m.design(d,t,p);changed={**d,'y':d['y'].copy()};changed['y'][a['query']]=987654.;np.testing.assert_array_equal(rp,m.replay(changed,t,p,c))
  meta=json.loads(path.with_name('metadata.json').read_text());assert m.sha(path)==meta['prediction_sha256']and m.sha(path.with_name('state.joblib'))==meta['model_sha256']
  yy,w,ss=m.target(d['y'][a['fit']],c);np.testing.assert_array_equal(w,s['weights'])
  for key in('mode','center','scale'):assert ss[key]==s[key]
  records.append(dict(task=t,part=p,config=name,replay_error=0.,iterations=s['iterations']))
 assert sum(r['part']in('fold0','fold1')for r in records)==296
 # Rebuild tiny kernels and check query-label invariance separately from caches.
 for t in m.TASKS:
  a=m.design(d,t,'fold0');small={**a,'fit':a['fit'][:32],'query':a['query'][:6],'x':a['x'][:32],'qx':a['qx'][:6]}
  k,q=m.geometry.matrices(d,small);changed={**d,'y':d['y'].copy()};changed['y'][a['query']]=987654.;kk,qq=m.geometry.matrices(changed,small)
  for key in k:np.testing.assert_array_equal(k[key],kk[key]);np.testing.assert_array_equal(q[key],qq[key]);assert np.linalg.eigvalsh(k[key].astype(float)).min()>-1e-4
 dev=json.loads((m.OUT/'development.json').read_text());chosen=[]
 for r in dev['records']:
  choices=[];t=r['task']
  for c in m.configs(t):
   for w in m.WEIGHTS:
    parts=[run.score(d,t,p,c,w)for p in('fold0','fold1')];objective=min(x['gain']for x in parts)
    if t in m.TASKS[5:]and min(x['rank_gain']for x in parts)<=0:objective=min(objective,-1.)
    choices.append(dict(config=c,weight=w,objective=objective))
  b=max(choices,key=lambda x:(x['objective'],-x['weight'],-list(m.configs(t)).index(x['config'])))
  for key in('config','weight','objective'):assert b[key]==r['best'][key]
  if b['objective']>0:chosen.append((t,b['config'],b['weight']))
 frozen=json.loads((m.OUT/'frozen_selection.json').read_text());assert chosen==[(r['task'],r['config'],r['weight'])for r in frozen['selected']]
 conf=json.loads((m.OUT/'confirmation.json').read_text())
 for r in conf['records']:
  assert [run.score(d,r['task'],p,r['config'],r['weight'])for p in('fold0','fold1','fold2','valid')]==r['confirmation']
  for b in r['bootstrap']:assert run.boot(d,r['task'],b['part'],r['config'],r['weight'])==b
 value=dict(status='PASS',signature=sig,models=len(records),development_fits=296,full_replay_max_error=0.,no_fitting_performed=True,records=records);m.save(m.OUT/'independent_verification.json',value)
 files=[p for p in m.OUT.rglob('*')if p.is_file()and p.suffix!='.joblib'and 'smoke'not in p.parts and 'kernels'not in p.parts]
 files += [ROOT/r['name']for r in json.loads((ROOT/'source_freeze.json').read_text())['files']]
 for r in conf['records']:
  if r['experimental_eligible']:files.append(m.OUT/'tasks'/r['task']/'final'/r['config']['name']/'state.joblib')
 m.save(ROOT/'result_manifest.json',dict(signature=sig,files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=m.sha(p),bytes=p.stat().st_size)for p in files],unselected_models_and_kernels_retained_remote=True))
 archive=ROOT/'results.tar.gz'
 with tarfile.open(archive,'w:gz')as tf:
  for p in files+[ROOT/'result_manifest.json']:tf.add(p,arcname=p.relative_to(ROOT).as_posix())
 m.save(ROOT/'result_archive.json',dict(status='PASS',sha256=m.sha(archive),bytes=archive.stat().st_size,signature=sig));print(json.dumps({k:v for k,v in value.items()if k!='records'}))
if __name__=='__main__':main()
