"""Replay all fitted bases, calibration certificates and frozen selection."""
from pathlib import Path
import json,joblib,tarfile,numpy as np
import v386_oof_calibrated as m
import v386_run_experts as run
ROOT=Path(__file__).resolve().parent
def main():
 m.setup();d=m.load();sig=m.signature();manifest=json.loads((ROOT/'data_manifest.json').read_text())
 for n,k in(('official_rows.npz','official_rows_sha256'),('expert_features.npz','expert_features_sha256'),('stereo_keys.npz','stereo_keys_sha256'),('v385_baseline.csv','baseline_csv_sha256')):assert m.sha(ROOT/n)==manifest[k]
 for r in manifest['controls']:
  for k in('control','design'):assert m.sha(ROOT/r[k])==r[k+'_sha256']
 for r in json.loads((ROOT/'source_freeze.json').read_text())['files']:assert m.sha(ROOT/r['name'])==r['sha256']
 assert np.isnan(d['y'][d['split']==2]).all();base_records=[];oof_records=[]
 for path in sorted((m.OUT/'oof').glob('*/*/state.joblib')):
  t,p=path.parts[-3:-1];a=m.design(d,t,p);o=joblib.load(path);assert o['signature']==sig
  for k in('fit','query'):np.testing.assert_array_equal(o[k],a[k])
  reconstructed=np.full_like(o['oof'],np.nan)
  for i,split in enumerate(o['splits']):
   tr,ho=split['train'],split['held'];sub=m.subdesign(a,tr,ho);assert not np.intersect1d(d['rg'][sub['fit']],d['rg'][sub['query']]).size
   reconstructed[ho]=m.replay_base(d,sub,t,p,'inner'+str(i))
  np.testing.assert_array_equal(reconstructed,o['oof']);np.testing.assert_array_equal(m.replay_base(d,a,t,p,'full'),o['full'])
  small={**a,'fit':a['fit'][:32],'query':a['query'][:6],'x':a['x'][:32],'qx':a['qx'][:6]}
  changed={**d,'y':d['y'].copy()};changed['y'][a['query']]=987654.
  k,q=m.matrices(d,small);kk,qq=m.matrices(changed,small)
  for key in k:
   np.testing.assert_array_equal(k[key],kk[key]);np.testing.assert_array_equal(q[key],qq[key]);assert np.linalg.eigvalsh(k[key].astype(float)).min()>-1e-4
  oof_records.append(dict(task=t,part=p,group_disjoint=True,query_labels_excluded=True,replay_error=0.))
  for tag in('inner0','inner1','inner2','full'):
   bp=m.OUT/'base'/t/p/tag/'state.joblib';meta=json.loads(bp.with_name('metadata.json').read_text());assert m.sha(bp)==meta['model_sha256']and meta['converged']
   base_records.append(dict(task=t,part=p,tag=tag,models=3,iterations=meta['iterations']))
 assert sum(r['models']for r in base_records if r['part']in('fold0','fold1'))==216
 records=[];calibrators=0
 for path in sorted((m.OUT/'tasks').glob('*/*/*/predictions.npz')):
  t,p,name=path.parts[-4:-1];c=next(c for c in m.CONFIGS if c['name']==name);a=m.design(d,t,p);z=np.load(path);assert str(z['signature'])==sig
  for k in('fit','query'):np.testing.assert_array_equal(z[k],a[k])
  rp=m.replay(d,t,p,c);np.testing.assert_array_equal(rp,z['p'])
  changed={**d,'y':d['y'].copy()};changed['y'][a['query']]=987654.;np.testing.assert_array_equal(rp,m.replay(changed,t,p,c))
  state=joblib.load(path.with_name('state.joblib'));meta=json.loads(path.with_name('metadata.json').read_text())
  assert m.sha(path.with_name('state.joblib'))==meta['model_sha256']and m.sha(path)==meta['prediction_sha256']
  cert=None
  if state['calibration']is not None:
   o=joblib.load(m.OUT/'oof'/t/p/'state.joblib');j=m.BASES.index(c['base']);s=state['calibration'];cert=m.calibration.certificate(o['oof'][:,j],d['y'][a['fit']],s);calibrators+=1
   q=np.linspace(s['knots'].min()-3,s['knots'].max()+3,301)*s['scale']+s['center'];assert np.diff(m.calibration.apply(q,s)).min()>0
  records.append(dict(task=t,part=p,config=name,replay_error=0.,calibration_certificate=cert))
 assert sum(r['part']in('fold0','fold1')for r in records)==162
 dev=json.loads((m.OUT/'development.json').read_text());chosen=[]
 for r in dev['records']:
  choices=[];t=r['task']
  for c in m.configs(t):
   for w in m.WEIGHTS:
    parts=[run.score(d,t,p,c,w)for p in('fold0','fold1')];objective=min(x['gain']for x in parts)
    if t in m.TASKS[5:]and min(x['rank_gain']for x in parts)<=0:objective=min(objective,-1.)
    choices.append(dict(config=c,weight=w,objective=objective))
  b=max(choices,key=lambda x:(x['objective'],-x['weight'],-list(m.configs(t)).index(x['config'])))
  for k in('config','weight','objective'):assert b[k]==r['best'][k]
  if b['objective']>0:chosen.append((t,b['config'],b['weight']))
 frozen=json.loads((m.OUT/'frozen_selection.json').read_text());assert chosen==[(r['task'],r['config'],r['weight'])for r in frozen['selected']]
 conf=json.loads((m.OUT/'confirmation.json').read_text())
 for r in conf['records']:
  assert [run.score(d,r['task'],p,r['config'],r['weight'])for p in('fold0','fold1','fold2','valid')]==r['confirmation']
  for b in r['bootstrap']:assert run.boot(d,r['task'],b['part'],r['config'],r['weight'])==b
 value=dict(status='PASS',signature=sig,models=len(records),base_models=sum(r['models']for r in base_records),development_base_models=216,
  calibration_models=calibrators,full_replay_max_error=0.,base_records=base_records,records=records,oof_records=oof_records,no_fitting_performed=True)
 m.save(m.OUT/'independent_verification.json',value)
 files=[p for p in m.OUT.rglob('*')if p.is_file()and p.suffix!='.joblib'and 'kernels'not in p.parts]
 files += [ROOT/r['name']for r in json.loads((ROOT/'source_freeze.json').read_text())['files']]
 for r in conf['records']:
  if r['experimental_eligible']:
   t=r['task'];files += [m.OUT/'tasks'/t/'final'/r['config']['name']/'state.joblib',m.OUT/'oof'/t/'final'/'state.joblib']
   files += list((m.OUT/'base'/t/'final').glob('*/state.joblib'))
 files=sorted(set(files));m.save(ROOT/'result_manifest.json',dict(signature=sig,files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=m.sha(p),bytes=p.stat().st_size)for p in files],unselected_models_and_kernels_retained_remote=True))
 archive=ROOT/'results.tar.gz'
 with tarfile.open(archive,'w:gz')as tf:
  for p in files+[ROOT/'result_manifest.json']:tf.add(p,arcname=p.relative_to(ROOT).as_posix())
 m.save(ROOT/'result_archive.json',dict(status='PASS',sha256=m.sha(archive),bytes=archive.stat().st_size,signature=sig));print(json.dumps({k:v for k,v in value.items()if k not in('records','base_records','oof_records')}))
if __name__=='__main__':main()
