import os,sys,json,pathlib,hashlib,time
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='4'
sys.path[:0]=['.','tools']
import v87_metric_ranking as v
import xgboost as x,numpy as np
from scipy.special import expit
out=v.OUT;records=[];started=time.monotonic()
for cachepath in sorted((out/'models').glob('*/*/*/predictions.npz')):
 f=cachepath.parent;z=np.load(cachepath);meta=json.loads((f/'metadata.json').read_text());assert str(z['signature'])==meta['signature']==v.signature();assert v.b.sha(cachepath)==meta['predictions_sha256'];members=[];splits=[]
 for seed in v.SEEDS:
  p=f/f'model_{seed}.ubj';assert v.b.sha(p)==meta['model_hashes'][p.name]
  model=x.Booster();model.load_model(p);model.set_param(dict(device='cuda',nthread=4));assert model.num_boosted_rounds()==meta['config']['trees']
  margin=model.predict(x.DMatrix(z['qx'],base_margin=z['qoff']),output_margin=True).astype(float);members.append(margin);splits.append(sum(t.count('split') for t in model.get_dump(dump_format='json')))
 np.testing.assert_allclose(z['members'],members,atol=1e-8,rtol=1e-8)
 np.testing.assert_allclose(z['p'],expit(np.mean(members,axis=0)),atol=1e-10,rtol=1e-10)
 records.append(dict(path=f.relative_to(v.ROOT).as_posix(),models=len(members),splits=splits,status='PASS'))
report=dict(status='PASS',scope='All rejected-branch saved booster hashes and GPU prediction replay; not an independent full feature audit',cpu_attempt='A CPU replay exceeded probability tolerance by 3.68e-7; no selection or model changes. GPU replay uses original predictor and strict tolerance.',models=sum(r['models'] for r in records),ensembles=len(records),zero_split_models=sum(sum(s==0 for s in r['splits']) for r in records),seconds=time.monotonic()-started,records=records)
v.b.save(out/'model_prediction_replay.json',report);print(json.dumps({k:val for k,val in report.items() if k!='records'}))
