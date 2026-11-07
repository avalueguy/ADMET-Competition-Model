"""Prepare immutable delta inputs; no local model fitting."""
from pathlib import Path
import json,hashlib,shutil,tarfile,numpy as np
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/mae_experts_v374'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    assert not (EXP/'data_manifest.json').exists(),'Preserve frozen experiment inputs'
    EXP.mkdir(parents=True,exist_ok=True);old=ROOT/'experiments/mae_experts_v371';tools=ROOT/'tools'
    for n in ('official_rows.npz','expert_features.npz','submission_template.csv'):
        shutil.copyfile(old/n,EXP/n)
    shutil.copytree(old/'design',EXP/'design',dirs_exist_ok=True)
    best=json.loads((ROOT/'docs/current_best_after_v371_20261105.json').read_text(encoding='utf-8'));baseline=ROOT/best['csv'];assert sha(baseline)==best['csv_sha256'];shutil.copyfile(baseline,EXP/'v371_baseline.csv')
    d=np.load(EXP/'official_rows.npz');manifest=json.loads((old/'data_manifest.json').read_text(encoding='utf-8'));records=[]
    for r in manifest['controls']:
        a=dict(np.load(old/r['control']));t,part=r['task'],r['part']
        if t=='ld50_zhu':
            z=np.load(old/'outputs/tasks'/t/part/'crossfit_local_median/predictions.npz')
            for k in ('fit','query'):np.testing.assert_array_equal(a[k],z[k])
            a['p']=.8*a['p']+.2*z['p']
        path=EXP/r['control'];path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**a)
        records.append({**r,'control_sha256':sha(path)})
    from rdkit import Chem,RDLogger
    RDLogger.DisableLog('rdApp.*');keys=[];failed=0
    for sm in d['smiles']:
        m=Chem.MolFromSmiles(str(sm))
        if m is None:keys.append('unparsed:'+str(sm));failed+=1
        else:keys.append(Chem.MolToSmiles(m,isomericSmiles=True))
    np.savez_compressed(EXP/'stereo_keys.npz',isomeric=np.array(keys))
    manifest.update(baseline=best,controls=records,stereo_keys_sha256=sha(EXP/'stereo_keys.npz'),stereo_parse_fallback=failed,
        baseline_control_limitation='Historicalv364 partlyproxy; v371LD50 incremental layer exact replay; solubility provenance discrepancy retained, localCSV preserved.',
        seed=3741106,source_manifest_sha256=sha(old/'data_manifest.json'))
    save(EXP/'data_manifest.json',manifest)
    runner=(tools/'v371_run_experts.py').read_text(encoding='utf-8').replace('import v371_mae_experts as m','import v374_specialists as m').replace('371000','374000').replace('total=70','total=120').replace('v371_verify_experts.py','v374_verify_experts.py')
    begin=runner.index('def smoke(d):');end=runner.index('\ndef main():',begin)
    smoke='''def smoke(d):
 m.setup();a=m.design(d,m.TASKS[0],'fold0');sub={**a,'fit':a['fit'][:70],'query':a['query'][:9],'x':a['x'][:70],'qx':a['qx'][:9]}
 mat=m.matrices(d,sub,'SMOKE','fold0');stats={}
 for name,(k,kq) in mat.items():
  assert np.linalg.eigvalsh(k.astype(float)).min()>-1e-4
  model=m.SVR(kernel='precomputed',C=3.,epsilon=.01).fit(k,d['y'][sub['fit']]);assert np.isfinite(model.predict(kq)).all();stats[name]=float(np.diag(k).mean())
 w,audit=m.replicate_weights(d,sub['fit']);assert np.isfinite(w).all() and (w>0).all()
 m.save(m.OUT/'smoke.json',dict(status='PASS',signature=m.signature(),kernels=stats,cuda_device=m.torch.cuda.get_device_name(),replicate_weight_fit_only=True))
'''
    runner=runner[:begin]+smoke+runner[end:];(tools/'v374_run_experts.py').write_text(runner,encoding='utf-8')
    verifier=(tools/'v371_verify_experts.py').read_text(encoding='utf-8').replace('import v371_mae_experts as m','import v374_specialists as m').replace('371000','374000').replace("'v371_mae_experts.py','v371_run_experts.py','v371_verify_experts.py'","'v371_mae_experts.py','v374_specialists.py','v374_run_experts.py','v374_verify_experts.py'")
    verifier=verifier.replace("  if c['family']=='local':\n   for s in state['inner_splits']:\n    assert not np.intersect1d(d['rg'][s['train']],d['rg'][s['held']]).size", """  if c['family']=='residual':
   rs=m.joblib.load(m.OUT/'residuals'/t/p/'state.joblib')
   for inner in rs['splits']:
    assert not np.intersect1d(d['rg'][a['fit'][inner['train']]],d['rg'][a['fit'][inner['held']]]).size""")
    verifier=verifier.replace(' selected=[]\n', ''' # Independently replay inner referenceOOF used to learn residuals.
 for path in (m.OUT/'residuals').glob('*/*/state.joblib'):
  task,part=path.parts[-3:-1];a=m.design(d,task,part);s=m.joblib.load(path);assert s['signature']==sig
  rebuilt=np.full(len(a['fit']),np.nan)
  for i,inner in enumerate(s['splits']):
   tr,ho=inner['train'],inner['held'];sub=dict(fit=a['fit'][tr],query=a['fit'][ho],x=a['x'][tr],qx=a['x'][ho],source=a['source'])
   _,kq=m.matrices(d,sub,task,part,'inner'+str(i))['raw_count'];st=inner['state'];rebuilt[ho]=kq@st['coef']*st['scale']+st['center']
  np.testing.assert_allclose(rebuilt,s['oof'],atol=1e-9,rtol=0)
  np.testing.assert_allclose(d['y'][a['fit']]-rebuilt,s['residual'],atol=1e-9,rtol=0)
  weights,audit=m.replicate_weights(d,a['fit']);np.testing.assert_array_equal(weights,s['weights'])
 selected=[]
''',1)
    verifier=verifier.replace(' records=[];max_error=0.', ''' manifest=json.loads((ROOT/'data_manifest.json').read_text())
 for name,key in (('official_rows.npz','official_rows_sha256'),('expert_features.npz','expert_features_sha256'),('stereo_keys.npz','stereo_keys_sha256')):assert m.sha(ROOT/name)==manifest[key]
 for r in manifest['controls']:
  assert m.sha(ROOT/r['control'])==r['control_sha256'] and m.sha(ROOT/r['design'])==r['design_sha256']
 assert np.isnan(d['y'][d['split']==2]).all()
 records=[];max_error=0.''')
    verifier=verifier.replace("  if c['family']=='residual':", """  weights,audit=m.replicate_weights(d,a['fit'])
  if c['family']=='kernel':np.testing.assert_array_equal(weights,state['weights'])
  if c['family']=='residual':""")
    verifier=verifier.replace(' weights=[]\n', ''' weights=[]
 residual_files=[p for p in (m.OUT/'residuals').glob('*/*/state.joblib')]
 residual_manifest=[dict(remote_path=str(p),sha256=m.sha(p),bytes=p.stat().st_size) for p in residual_files]
''')
    verifier=verifier.replace('final_models=weights)', 'final_models=weights,residual_models=residual_manifest)')
    (tools/'v374_verify_experts.py').write_text(verifier,encoding='utf-8')
    monitor=(tools/'v371_monitor.py').read_text(encoding='utf-8');(tools/'v374_monitor.py').write_text(monitor,encoding='utf-8')
    deliver=(tools/'v371_deliver_experts.py').read_text(encoding='utf-8').replace('v371','v374').replace('V371','V374').replace('v364','v371').replace('51.269','51.270')
    deliver=deliver.replace("('v374_mae_experts.py','v374_run_experts.py','v374_verify_experts.py')","('v371_mae_experts.py','v374_specialists.py','v374_run_experts.py','v374_verify_experts.py')")
    deliver=deliver.replace('精确中位数核、GPU分位数树、理化先验残差树及内部骨架交叉验证局部校正均已完成有界比较。失败记录保留。','原count、圆形count、圆形二值核、重复测量降权、内部OOF局部残差和GPU残差树已完成有界比较。失败记录保留。官方v371卡溶解度与本地文件的来源差异尚未消除。')
    (tools/'v374_deliver_experts.py').write_text(deliver,encoding='utf-8')
    sources=('v371_mae_experts.py','v374_specialists.py','v374_run_experts.py','v374_verify_experts.py','v374_monitor.py')
    for n in sources:shutil.copyfile(tools/n,EXP/n)
    save(EXP/'source_freeze.json',dict(files=[dict(name=n,sha256=sha(EXP/n)) for n in sources]))
    archive=ROOT/'releases/v374_delta_inputs.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        files=[EXP/n for n in sources+('data_manifest.json','stereo_keys.npz','v371_baseline.csv','source_freeze.json')]+list((EXP/'controls').glob('*/*.npz'))
        for p in files:tf.add(p,arcname=p.relative_to(EXP).as_posix())
    save(ROOT/'docs/v374_input_package.json',dict(status='PASS',archive=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,reused_immutable_input='v371_expert_inputs.tar.gz',stereo_parse_fallback=failed))
    print(json.dumps(dict(status='PREPARED',bytes=archive.stat().st_size,sha256=sha(archive),configs=12,tasks=5)))
if __name__=='__main__':main()
