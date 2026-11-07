"""Create stereo keys and source delta; no local model fitting."""
from pathlib import Path
import json,hashlib,shutil,tarfile,numpy as np
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/mae_experts_v375'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    assert not (EXP/'data_manifest.json').exists(),'Preserve frozen inputs'
    old=ROOT/'experiments/mae_experts_v374';tools=ROOT/'tools';EXP.mkdir(exist_ok=True,parents=True)
    for n in ('official_rows.npz','expert_features.npz','submission_template.csv','v371_baseline.csv'):shutil.copyfile(old/n,EXP/n)
    for n in ('design','controls'):shutil.copytree(old/n,EXP/n,dirs_exist_ok=True)
    from rdkit import Chem,RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize as ms
    RDLogger.DisableLog('rdApp.*');d=np.load(EXP/'official_rows.npz');keys=[];neutral=[];fallback=0
    chooser=ms.LargestFragmentChooser();uncharger=ms.Uncharger()
    for sm in d['smiles']:
        mol=Chem.MolFromSmiles(str(sm))
        if mol is None:keys.append('unparsed:'+str(sm));neutral.append('unparsed:'+str(sm));fallback+=1
        else:
            keys.append(Chem.MolToSmiles(mol,isomericSmiles=True));p=uncharger.uncharge(chooser.choose(mol));neutral.append(Chem.MolToSmiles(p,isomericSmiles=True))
    np.savez_compressed(EXP/'stereo_keys.npz',isomeric=np.array(keys),neutral_stereo=np.array(neutral))
    manifest=json.loads((old/'data_manifest.json').read_text(encoding='utf-8'));manifest.update(stereo_keys_sha256=sha(EXP/'stereo_keys.npz'),stereo_parse_fallback=fallback,previous_inputs='v374 same exact localv371 baseline',seed=3751106)
    save(EXP/'data_manifest.json',manifest)
    runner=(tools/'v374_run_experts.py').read_text(encoding='utf-8').replace('v374_specialists','v375_stereo_context').replace('374000','375000').replace('total=120','total=40').replace('v374_verify_experts.py','v375_verify_experts.py')
    begin=runner.index('def smoke(d):');end=runner.index('\ndef main():',begin)
    smoke='''def smoke(d):
 m.setup();a=m.design(d,m.TASKS[0],'fold0');sub={**a,'fit':a['fit'][:70],'query':a['query'][:9],'x':a['x'][:70],'qx':a['qx'][:9]}
 c=m.CONFIGS[0];xs,qs,cols=m.features(d,sub,m.TASKS[0],'fold0',c)
 net=m.CatBoostRegressor(loss_function='MAE',iterations=8,depth=3,task_type='GPU',devices='0',verbose=False,allow_writing_files=False)
 net.fit(xs[0],d['y'][sub['fit']]);assert np.isfinite(net.predict(qs[0])).all()
 ctx=m.context(d,m.TASKS[0],'fold0');assert np.all(d['tid'][ctx['source']]!=d['tasks'].index(m.TASKS[0]))
 integrated=m.samples(d,sub['query'],m.TASKS[0],'fold0','integrated');assert len(integrated)==4
 m.save(m.OUT/'smoke.json',dict(status='PASS',signature=m.signature(),stereo_gpu_fit=True,target_context_excluded=True,empirical_samples=4))
'''
    runner=runner[:begin]+smoke+runner[end:];(tools/'v375_run_experts.py').write_text(runner,encoding='utf-8')
    verifier=(tools/'v371_verify_experts.py').read_text(encoding='utf-8').replace('v371_mae_experts as m','v375_stereo_context as m').replace('371000','375000')
    verifier=verifier.replace("  if c['family']=='local':\n   for s in state['inner_splits']:\n    assert not np.intersect1d(d['rg'][s['train']],d['rg'][s['held']]).size", "  ctx=m.context(d,t,p);assert d['tasks'].index(t) not in ctx['cols'] and np.all(d['tid'][ctx['source']]!=d['tasks'].index(t))")
    verifier=verifier.replace("'v371_mae_experts.py','v371_run_experts.py','v371_verify_experts.py'","'v371_mae_experts.py','v375_stereo_context.py','v375_run_experts.py','v375_verify_experts.py'")
    verifier=verifier.replace(' records=[];max_error=0.', ''' manifest=json.loads((ROOT/'data_manifest.json').read_text())
 for name,key in (('official_rows.npz','official_rows_sha256'),('expert_features.npz','expert_features_sha256'),('stereo_keys.npz','stereo_keys_sha256')):assert m.sha(ROOT/name)==manifest[key]
 for r in manifest['controls']:assert m.sha(ROOT/r['control'])==r['control_sha256'] and m.sha(ROOT/r['design'])==r['design_sha256']
 assert np.isnan(d['y'][d['split']==2]).all()
 records=[];max_error=0.''')
    verifier=verifier.replace("  np.testing.assert_array_equal(m.replay(changed,t,p,c),replay)","  changed.pop('_stereo_context',None)\n  np.testing.assert_array_equal(m.replay(changed,t,p,c),replay)")
    (tools/'v375_verify_experts.py').write_text(verifier,encoding='utf-8')
    (tools/'v375_monitor.py').write_text((tools/'v374_monitor.py').read_text(encoding='utf-8'),encoding='utf-8')
    deliver=(tools/'v374_deliver_experts.py').read_text(encoding='utf-8').replace('v374','v375').replace('V374','V375').replace('v374_specialists','v375_stereo_context')
    deliver=deliver.replace('v375_specialists.py','v375_stereo_context.py').replace('原count、圆形count、圆形二值核、重复测量降权、内部OOF局部残差和GPU残差树已完成有界比较。','立体结构保留的其他测量上下文、4点经验重复测量积分和两种深度GPU MAE树已完成有界比较。')
    (tools/'v375_deliver_experts.py').write_text(deliver,encoding='utf-8')
    names=('v371_mae_experts.py','v375_stereo_context.py','v375_run_experts.py','v375_verify_experts.py','v375_monitor.py')
    for n in names:shutil.copyfile(tools/n,EXP/n)
    save(EXP/'source_freeze.json',dict(files=[dict(name=n,sha256=sha(EXP/n)) for n in names]))
    archive=ROOT/'releases/v375_delta_inputs.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for n in names+('data_manifest.json','stereo_keys.npz','source_freeze.json'):tf.add(EXP/n,arcname=n)
    save(ROOT/'docs/v375_input_package.json',dict(status='PASS',archive=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,reused_inputs='v374 inputs without stereo_keys/data_manifest',stereo_parse_fallback=fallback))
    print(json.dumps(dict(status='PREPARED',bytes=archive.stat().st_size,sha256=sha(archive),candidates=4,tasks=5)))
if __name__=='__main__':main()
