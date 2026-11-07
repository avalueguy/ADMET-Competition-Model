"""Independent replay, candidate selection, feature and arithmetic audit."""
from pathlib import Path
import hashlib,json,tarfile
import numpy as np
import v375_stereo_context as m

ROOT=Path(__file__).resolve().parent


def mae(y,p):return float(np.mean(np.abs(y-p)))
def gain(y,a,p):return (mae(y,a)-mae(y,p))/mae(y,a)
def main():
 m.setup();d=m.load();sig=m.signature();f=json.loads((m.OUT/'frozen_selection.json').read_text());confirmation=json.loads((m.OUT/'confirmation.json').read_text());assert f['signature']==confirmation['signature']==sig
 manifest=json.loads((ROOT/'data_manifest.json').read_text())
 for name,key in (('official_rows.npz','official_rows_sha256'),('expert_features.npz','expert_features_sha256'),('stereo_keys.npz','stereo_keys_sha256')):assert m.sha(ROOT/name)==manifest[key]
 for r in manifest['controls']:assert m.sha(ROOT/r['control'])==r['control_sha256'] and m.sha(ROOT/r['design'])==r['design_sha256']
 assert np.isnan(d['y'][d['split']==2]).all()
 records=[];max_error=0.
 for path in sorted((m.OUT/'tasks').glob('*/*/*/predictions.npz')):
  t,p,cname=path.parts[-4:-1];c=next(c for c in m.CONFIGS if c['name']==cname);z=dict(np.load(path));a=m.design(d,t,p)
  assert str(z['signature'])==sig
  for name in ('fit','query'):np.testing.assert_array_equal(z[name],a[name])
  replay=m.replay(d,t,p,c);err=float(np.max(np.abs(replay-z['p'])));assert err<1e-6,(t,p,cname,err);max_error=max(max_error,err)
  changed={**d,'y':d['y'].copy()};changed['y'][a['query']]=np.arange(len(a['query']))+1234567.
  changed.pop('_stereo_context',None)
  np.testing.assert_array_equal(m.replay(changed,t,p,c),replay)
  state=m.joblib.load(path.parent/'model.joblib')
  ctx=m.context(d,t,p);assert d['tasks'].index(t) not in ctx['cols'] and np.all(d['tid'][ctx['source']]!=d['tasks'].index(t))
  meta=json.loads(path.with_name('metadata.json').read_text());assert m.sha(path.parent/'model.joblib')==meta['model_sha256'] and m.sha(path)==meta['predictions_sha256']
  records.append(dict(task=t,part=p,config=cname,full_replay_error=err,query_label_perturbation_invariant=True));m.emit('verified',task=t,part=p,config=cname)
 selected=[]
 for t in m.TASKS:
  values=[]
  for ci,c in enumerate(m.CONFIGS):
   for w in m.WEIGHTS:
    g=[]
    for part in ('fold0','fold1'):
     a=m.ctrl(t,part);z=np.load(m.OUT/'tasks'/t/part/c['name']/'predictions.npz');y=d['y'][a['query']];g.append(gain(y,a['p'],(1-w)*a['p']+w*z['p']))
    values.append((min(g),-w,-ci,c['name'],w))
  b=max(values)
  if b[0]>0:selected.append((t,b[3],b[4]))
 assert selected==[(r['task'],r['config']['name'],r['weight']) for r in f['selected']]
 for r in confirmation['records']:
  for part in ('fold0','fold1','fold2','valid'):
   a=m.ctrl(r['task'],part);q=a['query'];z=np.load(m.OUT/'tasks'/r['task']/part/r['config']['name']/'predictions.npz');pred=(1-r['weight'])*a['p']+r['weight']*z['p'];g=gain(d['y'][q],a['p'],pred);reported=next(x for x in r['confirmation'] if x['part']==part);assert abs(g-reported['gain'])<1e-12
   if part in ('fold2','valid'):
    groups=d['rg'][q];ug=np.unique(groups);chunks=[np.flatnonzero(groups==g) for g in ug];rng=np.random.default_rng(375000+d['tasks'].index(r['task']));deltas=[]
    if len(ug)>1:
     for _ in range(500):
      ix=np.concatenate([chunks[i] for i in rng.integers(len(chunks),size=len(chunks))]);deltas.append(gain(d['y'][q][ix],a['p'][ix],pred[ix]))
    boot=next(x for x in r['bootstrap'] if x['part']==part)
    if deltas:np.testing.assert_allclose(np.quantile(deltas,[.05,.95]),boot['ci90'],atol=1e-12,rtol=0)
    else:assert boot['ci90'] is None
  positive=all(x['gain']>0 for x in r['confirmation']);strict=positive and r['objective']>.012 and np.mean([x['gain'] for x in r['confirmation']])>.012 and all(x['lower_positive'] for x in r['bootstrap'])
  assert bool(positive)==r['experimental_eligible'] and bool(strict)==r['strict_accepted']
 result=dict(status='PASS',signature=sig,models=len(records),records=records,full_replay_max_error=max_error,independent_selection_recomputed=True,confirmation_arithmetic_recomputed=True,no_fitting_performed=True)
 m.save(m.OUT/'independent_verification.json',result)
 files=[p for p in m.OUT.rglob('*') if p.is_file() and p.suffix not in ('.joblib',) and '/kernels/' not in p.as_posix()]
 files+=[ROOT/n for n in ('v371_mae_experts.py','v375_stereo_context.py','v375_run_experts.py','v375_verify_experts.py')]
 weights=[]
 for r in confirmation['records']:
  if r['experimental_eligible']:
   path=m.OUT/'tasks'/r['task']/'final'/r['config']['name']/'model.joblib';weights.append(dict(remote_path=str(path),sha256=m.sha(path),bytes=path.stat().st_size))
 manifest=dict(signature=sig,files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=m.sha(p),bytes=p.stat().st_size) for p in sorted(files)],final_models=weights)
 m.save(ROOT/'result_manifest.json',manifest);archive=ROOT/'results.tar.gz'
 with tarfile.open(archive,'w:gz') as tf:
  for p in files+[ROOT/'result_manifest.json']:tf.add(p,arcname=p.relative_to(ROOT).as_posix())
 m.save(ROOT/'result_archive.json',dict(status='PASS',signature=sig,sha256=m.sha(archive),bytes=archive.stat().st_size,files=len(files)))
 print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
