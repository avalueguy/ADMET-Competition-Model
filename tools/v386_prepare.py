"""Freeze the next experiment, with v385 controls and fit-only OOF calibration."""
from pathlib import Path
import hashlib,json,os,shutil,tarfile,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/regression9_v386'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 assert not(EXP/'source_freeze.json').exists(),'Do not overwrite frozen experiment';EXP.mkdir(exist_ok=True);prev=ROOT/'experiments/regression9_v385'
 manifest=json.loads((prev/'data_manifest.json').read_text());best=json.loads((ROOT/'docs/current_best_after_v385_20261106.json').read_text());manifest['baseline']=best
 assert sha(ROOT/best['csv'])==best['csv_sha256'];shutil.copyfile(ROOT/best['csv'],EXP/'v385_baseline.csv')
 names=['official_rows.npz','expert_features.npz','stereo_keys.npz','submission_template.csv']+[r['design']for r in manifest['controls']]
 for n in names:
  src,dest=prev/n,EXP/n;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert sha(src)==sha(dest)
  else:os.link(src,dest)
 updates=[]
 for r in manifest['controls']:
  src,dest=prev/r['control'],EXP/r['control'];dest.parent.mkdir(parents=True,exist_ok=True)
  if r['task']=='ppbr_az':
   z=np.load(prev/f"outputs/tasks/ppbr_az/{r['part']}/sqrt_gaussian_rank_C30/predictions.npz");a=dict(np.load(src))
   for k in('fit','query'):np.testing.assert_array_equal(a[k],z[k])
   a['p']=.9*a['p']+.1*z['p'];a['score_p']=a['p'].copy();np.savez_compressed(dest,**a)
   updates.append(dict(task=r['task'],part=r['part'],prior_sha256=sha(src),new_sha256=sha(dest),expert_sha256=sha(prev/f"outputs/tasks/ppbr_az/{r['part']}/sqrt_gaussian_rank_C30/predictions.npz"),weight=.1))
  elif dest.exists():assert sha(src)==sha(dest)
  else:os.link(src,dest)
  r['control_sha256']=sha(dest)
 manifest['baseline_csv']='v385_baseline.csv';manifest['baseline_csv_sha256']=sha(EXP/'v385_baseline.csv');manifest['v385_control_updates']=updates
 manifest.pop('v382_baseline_sha256',None)
 save(EXP/'data_manifest.json',manifest)
 tab=list(csv.DictReader((EXP/'v385_baseline.csv').open(encoding='utf-8-sig',newline='')))
 for t in manifest['pilot_tasks']:
  a=np.load(EXP/f'controls/{t}/final.npz');expected=np.array([float(r['y_pred'])for r in tab if r['task']==t]);np.testing.assert_allclose(a['score_p'],expected,rtol=0,atol=1e-10)
 # Retain audited grouped inner/full base fitting, replace only its stack layer.
 s=(ROOT/'tools/v381_oof_pareto.py').read_text(encoding='utf-8');s=s[:s.index('def pair_sample(')]
 start=s.index('CONFIGS =');end=s.index('sha, save, emit',start)
 s=s[:start]+'''CONFIGS = tuple(dict(name=b+'_'+mode, base=b, calibration=mode, penalty=penalty)
                for b in BASES for mode,penalty in (('raw',0.),('mono02',.02),('mono10',.1)))
WEIGHTS = (.1,.25,.5,.75,1.)
SETTINGS = dict(inner_group_folds=3,C=30.,epsilon=0.,tolerance=1e-5,maximum_iterations=1000000)
import v386_calibration_core as calibration
'''+s[end:]
 s=s.replace("histogram.base.shared.__file__, ROOT/'data_manifest.json'","histogram.base.shared.__file__, calibration.__file__, ROOT/'data_manifest.json'")
 s=s.replace("    histogram.setup()","    histogram.ROOT=ROOT; histogram.base.ROOT=ROOT; histogram.base.shared.ROOT=ROOT; histogram.setup()")
 s=s.replace("    return histogram.load()","    histogram.ROOT=ROOT; return histogram.load()")
 start=s.index('def configs(');end=s.index('def matrices(',start);s=s[:start]+"def configs(t):\n    return CONFIGS\n\n\n"+s[end:]
 s+='''def clip(t,p):
    return np.clip(p,0,100) if t=='ppbr_az' else np.maximum(p,0) if t in TASKS[5:] else p

def predict(d,t,p,c):
    setup();path=OUT/'tasks'/t/p/c['name']/'predictions.npz'
    if path.exists():
        z=np.load(path);assert str(z['signature'])==signature();return z['p']
    a=design(d,t,p);o=grouped_oof(d,t,p);j=BASES.index(c['base']);state=None
    if c['calibration']!='raw':state=calibration.fit(o['oof'][:,j],d['y'][a['fit']],c['penalty'])
    pred=clip(t,o['full'][:,j] if state is None else calibration.apply(o['full'][:,j],state))
    s=dict(signature=signature(),config=c,fit=a['fit'],query=a['query'],calibration=state)
    path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(s,path.with_name('state.joblib'),compress=3)
    np.savez_compressed(path,signature=signature(),fit=a['fit'],query=a['query'],p=pred)
    save(path.with_name('metadata.json'),dict(signature=signature(),model_sha256=sha(path.with_name('state.joblib')),prediction_sha256=sha(path)))
    emit('calibration_done',task=t,part=p,config=c['name']);return pred

def replay(d,t,p,c):
    a=design(d,t,p);s=joblib.load(OUT/'tasks'/t/p/c['name']/'state.joblib');assert s['signature']==signature()
    for k in ('fit','query'):np.testing.assert_array_equal(s[k],a[k])
    raw=replay_base(d,a,t,p,'full')[:,BASES.index(c['base'])]
    return clip(t,raw if s['calibration'] is None else calibration.apply(raw,s['calibration']))

def freeze():
    save(OUT/'protocol.json',dict(signature=signature(),tasks=TASKS,configs=CONFIGS,weights=WEIGHTS,settings=SETTINGS,
        baseline='complete v385 associated official51.308',all_fitting_remote=True,external_data=False,external_weights=False,
        base='Three existing converged SVRs: sqrt native, Tanimoto native, sqrt Gaussian rank. Refit inner3/full with per-inner feature scaling; no historical outputs as calibration features.',
        method='Nine fit-OOF-prediction quantile knots; monotone piecewise linear MAE fit with slope bounds0.05..5, knot shift bounds0.5IQR, L1 identity shrink0.02/0.1. Outside knots fixed slope1. Raw expert controls.',
        budget='18 development outer partitions x4 inner/full x3 SVRs=216 base fits, plus18x3x2=108 calibration LPs; all before selection.',
        novelty='Extends prior affine/empirical marginal calibration and inner OOF stacking; calibrated single-expert monotone shape is the tested difference, not invention of calibration.',
        selection='Unique maxmin fold0/1 config and blend; freeze before fold2/valid; no confirmation fallback. Four MAE gains positive, and four Spearman gains positive for rank tasks.',
        limitations='Legacy proxy controls, calibrated rank MAE references, repeated public validation. Monotonicity preserves a single unbounded expert ranking, not ranking of a blend or improvement vs historical ensemble.'))
'''
 (ROOT/'tools/v386_oof_calibrated.py').write_text(s,encoding='utf-8')
 for n in('run_experts','completion'):
  s=(ROOT/f'tools/v381_{n}.py').read_text(encoding='utf-8').replace('v381_oof_pareto','v386_oof_calibrated').replace('v381','v386').replace('381000','386000')
  (ROOT/f'tools/v386_{n}.py').write_text(s,encoding='utf-8')
 sources=('v371_mae_experts.py','v377_regression_experts.py','v380_histogram_median.py','v386_calibration_core.py','v386_oof_calibrated.py','v386_run_experts.py','v386_verify_experts.py','v386_smoke.py','v386_completion.py')
 entries=[]
 for n in sources:shutil.copyfile(ROOT/'tools'/n,EXP/n);entries.append(dict(name=n,sha256=sha(EXP/n)))
 save(EXP/'source_freeze.json',dict(files=entries,all_fitting_remote=True,baseline=best,controls_final_match_csv=True,workers=2))
 archive=ROOT/'releases/v386_input_delta.tar.gz'
 with tarfile.open(archive,'w:gz',dereference=True)as tf:
  for n in sources+('source_freeze.json','data_manifest.json','v385_baseline.csv')+tuple(r['control']for r in manifest['controls']if r['task']=='ppbr_az'):tf.add(EXP/n,arcname=n)
 receipt=dict(archive=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,baseline=best,updated_controls=5)
 save(ROOT/'docs/v386_input_package.json',receipt);print(json.dumps({k:v for k,v in receipt.items()if k!='baseline'}))
if __name__=='__main__':main()
