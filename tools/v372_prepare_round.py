"""Freeze bounded next-round source files; local file assembly, no training."""
from pathlib import Path
import json,shutil,hashlib

ROOT=Path(__file__).resolve().parents[1]

def main():
    tools=ROOT/'tools'
    runner=(tools/'v371_run_experts.py').read_text(encoding='utf-8').replace('import v371_mae_experts as m','import v372_crossfit_experts as m').replace('371000','372000').replace('total=70','total=20').replace('v371_verify_experts.py','v372_verify_experts.py')
    start=runner.index('def smoke(d):');end=runner.index('\ndef main():',start)
    smoke='''def smoke(d):
 m.setup();a=m.design(d,m.TASKS[0],'fold0');sub={**a,'fit':a['fit'][:70],'query':a['query'][:9],'x':a['x'][:70],'qx':a['qx'][:9]}
 k,kq,_=m.shared.kernels(d,sub);assert np.linalg.eigvalsh(k.astype(float)).min()>-1e-4
 x,qx,cols,keep=m.tree_inputs(d,sub);net=m.CatBoostRegressor(loss_function='MAE',iterations=8,depth=3,task_type='GPU',devices='0',verbose=False,allow_writing_files=False)
 net.fit(x[:,keep],d['y'][sub['fit']]);assert np.isfinite(net.predict(qx[:,keep])).all()
 yy=np.arange(30,dtype=float);p=np.c_[yy,yy+3,yy-2,yy*.5];s=m.fit_simplex(p,yy,np.ones(4)/4,.01)
 assert np.mean(np.abs(p@s['weights']+s['bias']-yy))<.1
 m.save(m.OUT/'smoke.json',dict(status='PASS',signature=m.signature(),cuda_device=m.torch.cuda.get_device_name(),catboost_mae_gpu=True,kernel_psd=True,convex_L1_synthetic_check=True))
'''
    runner=runner[:start]+smoke+runner[end:]
    (tools/'v372_run_experts.py').write_text(runner,encoding='utf-8')
    verifier=(tools/'v371_verify_experts.py').read_text(encoding='utf-8').replace('import v371_mae_experts as m','import v372_crossfit_experts as m').replace('371000','372000').replace("'v371_mae_experts.py','v371_run_experts.py','v371_verify_experts.py'","'v371_mae_experts.py','v372_crossfit_experts.py','v372_run_experts.py','v372_verify_experts.py'")
    verifier=verifier.replace("  if c['family']=='local':\n   for s in state['inner_splits']:\n    assert not np.intersect1d(d['rg'][s['train']],d['rg'][s['held']]).size", "  for inner in state['splits']:\n   assert not np.intersect1d(d['rg'][a['fit'][inner['train']]],d['rg'][a['fit'][inner['held']]]).size\n  assert abs(state['global_stack']['weights'].sum()-1)<1e-7 and state['global_stack']['weights'].min()>=-1e-7")
    marker=' selected=[]\n'
    extra=''' # Audit every base model including its inner held predictions, not just the final stack.
 base_records=[]
 for path in sorted((m.OUT/'base').glob('*/*/*/state.joblib')):
  t,part,tag=path.parts[-4:-1];a=m.design(d,t,part);st=m.joblib.load(path)
  if tag!='full':
   tr=np.flatnonzero(np.isin(a['fit'],st['fit']));ho=np.flatnonzero(np.isin(a['fit'],st['query']));a=m.subdesign(a,tr,ho)
  pred=m.replay_base(d,a,t,part,tag);error=float(np.max(np.abs(pred-st['p'])));assert error<1e-6
  altered={**d,'y':d['y'].copy()};altered['y'][a['query']]=1234567.
  np.testing.assert_array_equal(pred,m.replay_base(altered,a,t,part,tag))
  meta=json.loads(path.with_name('metadata.json').read_text());assert m.sha(path)==meta['model_sha256']
  max_error=max(max_error,error);base_records.append(dict(task=t,part=part,tag=tag,error=error))
  for cp in (m.OUT/'tasks'/t/part).glob('*/model.joblib'):
   stack=m.joblib.load(cp)
   if tag=='full':np.testing.assert_array_equal(stack['full'],pred)
   else:
    held=stack['splits'][int(tag[5:])]['held'];np.testing.assert_allclose(stack['oof'][held],pred,atol=1e-6,rtol=0)
 selected=[]
'''
    verifier=verifier.replace(marker,extra,1).replace('models=len(records),records=records','models=len(records),base_model_fits=len(base_records)*4,base_records=base_records,records=records')
    verifier=verifier.replace("np.testing.assert_array_equal(stack['full'],pred)", "np.testing.assert_allclose(stack['full'],pred,atol=1e-6,rtol=0)")
    verifier=verifier.replace("  assert abs(state['global_stack']['weights'].sum()-1)<1e-7 and state['global_stack']['weights'].min()>=-1e-7", """  assert abs(state['global_stack']['weights'].sum()-1)<1e-7 and state['global_stack']['weights'].min()>=-1e-7
  rebuilt,weights=m.fit_stack(d,a,t,state['oof'],state['full'],c)
  np.testing.assert_allclose(rebuilt,z['p'],atol=1e-10,rtol=0)
  np.testing.assert_allclose(weights['global_stack']['weights'],state['global_stack']['weights'],atol=1e-10,rtol=0)
  for new,old in zip(weights['locals'],state['locals']):
   np.testing.assert_allclose(new['weights'],old['weights'],atol=1e-10,rtol=0)
   assert abs(new['bias']-old['bias'])<1e-10""")
    verifier=verifier.replace('no_fitting_performed=True','independent_stack_weights_recomputed=True,base_models_not_refitted=True')
    verifier=verifier.replace(" records=[];max_error=0.", """ manifest=json.loads((ROOT/'data_manifest.json').read_text())
 assert m.sha(ROOT/'official_rows.npz')==manifest['official_rows_sha256']
 assert m.sha(ROOT/'expert_features.npz')==manifest['expert_features_sha256']
 for row in manifest['controls']:
  assert m.sha(ROOT/row['design'])==row['design_sha256']
  assert m.sha(ROOT/row['control'])==row['control_sha256']
 assert np.isnan(d['y'][d['split']==2]).all()
 records=[];max_error=0.""")
    (tools/'v372_verify_experts.py').write_text(verifier,encoding='utf-8')
    monitor=(tools/'v371_monitor.py').read_text(encoding='utf-8')
    (tools/'v372_monitor.py').write_text(monitor,encoding='utf-8')
    deliver=(tools/'v371_deliver_experts.py').read_text(encoding='utf-8').replace('v371','v372').replace('V371','V372').replace("('v372_mae_experts.py','v372_run_experts.py','v372_verify_experts.py')", "('v371_mae_experts.py','v372_crossfit_experts.py','v372_run_experts.py','v372_verify_experts.py')")
    deliver=deliver.replace('精确中位数核、GPU分位数树、理化先验残差树及内部骨架交叉验证局部校正均已完成有界比较。失败记录保留。','中位数SVR、核岭、GPU CatBoost MAE和XGBoost中位数专家，用拟合集内部骨架OOF学习单纯形权重与理化分区门控；内折独立归一化/列选择。失败记录保留。')
    (tools/'v372_deliver_experts.py').write_text(deliver,encoding='utf-8')
    exp=ROOT/'experiments/mae_experts_v372';exp.mkdir(exist_ok=True)
    previous=ROOT/'experiments/mae_experts_v371'
    for name in ('official_rows.npz','expert_features.npz','data_manifest.json','v364_baseline.csv','submission_template.csv'):
        if not (exp/name).exists():shutil.copyfile(previous/name,exp/name)
    for name in ('controls','design'):
        shutil.copytree(previous/name,exp/name,dirs_exist_ok=True)
    files=('v371_mae_experts.py','v372_crossfit_experts.py','v372_run_experts.py','v372_verify_experts.py','v372_monitor.py')
    for n in files:shutil.copyfile(tools/n,exp/n)
    receipt=dict(status='FROZEN',source_v371_data_manifest_sha256=hashlib.sha256((previous/'data_manifest.json').read_bytes()).hexdigest(),
        all_fitting_remote=True,files=[dict(name=n,sha256=hashlib.sha256((exp/n).read_bytes()).hexdigest()) for n in files])
    (exp/'source_freeze.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))

if __name__=='__main__':main()
