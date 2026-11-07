"""Prepare task-specific MAE experts; local CPU packaging only, no fitting."""
from pathlib import Path
import hashlib,json,sys,shutil,tarfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
OUT=ROOT/'experiments/mae_experts_v371'
TASKS=('caco2_wang','ld50_zhu','lipophilicity_astrazeneca','ppbr_az','solubility_aqsoldb')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    import v146_current_baseline as v
    from rdkit.Chem import Descriptors
    assert not (OUT/'data_manifest.json').exists(),'Preserve frozen inputs'
    previous=ROOT/'experiments/transformer_stage5_v370'
    old=json.loads((previous/'data_manifest.json').read_text(encoding='utf-8'))
    OUT.mkdir(exist_ok=True,parents=True)
    d=v.load();raw=dict(np.load(previous/'official_rows.npz'))
    for k in ('mi','tid','split','y'):np.testing.assert_array_equal(d[k],raw[k])
    for k in ('canonical','groups'):np.testing.assert_array_equal(d[k],raw[k][raw['mi']])
    names=[name for name,_ in Descriptors.descList if name not in {'Ipc','AvgIpc','BalabanJ'}]
    assert len(names)==raw['descriptor'].shape[1]
    physical_names=('MolWt','MolLogP','TPSA','NumHDonors','NumHAcceptors','NumRotatableBonds','FractionCSP3','RingCount','NumAromaticRings','HeavyAtomCount')
    physical=raw['descriptor'][:,[names.index(n) for n in physical_names]]
    np.savez_compressed(OUT/'expert_features.npz',fingerprints=d['views']['multiview'][:,:12288],physical=physical)
    shutil.copyfile(previous/'official_rows.npz',OUT/'official_rows.npz')
    shutil.copyfile(previous/'v364_baseline.csv',OUT/'v364_baseline.csv')
    shutil.copyfile(previous/'submission_template.csv',OUT/'submission_template.csv')
    records=[]
    for task in TASKS:
        for part in ('fold0','fold1','fold2','valid','final'):
            a=dict(np.load(previous/'controls'/task/(part+'.npz')))
            fit,q,x,qx,source=v.old.features(d,d['tasks'].index(task),part,'augmented')
            np.testing.assert_array_equal(fit,a['fit']);np.testing.assert_array_equal(q,a['query'])
            assert np.all(d['tid'][source]!=d['tasks'].index(task))
            assert np.all(d['split'][source]<(2 if part=='final' else 1))
            assert np.isfinite(d['y'][fit]).all() and not np.intersect1d(fit,q).size
            if part!='final':assert not np.intersect1d(raw['groups'][raw['mi'][fit]],raw['groups'][raw['mi'][q]]).size
            p=OUT/'controls'/task/(part+'.npz');p.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(p,**a)
            dest=OUT/'design'/task/(part+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,fit=fit,query=q,x=x,qx=qx,source=source)
            records.append(dict(task=task,part=part,fit=len(fit),query=len(q),design=dest.relative_to(OUT).as_posix(),design_sha256=sha(dest),control=p.relative_to(OUT).as_posix(),control_sha256=sha(p)))
            print(json.dumps(dict(stage='prepared',task=task,part=part,fit=len(fit),query=len(q))),flush=True)
    sources={str(Path(v.__file__).relative_to(ROOT)):sha(v.__file__),str(Path(v.old.__file__).relative_to(ROOT)):sha(v.old.__file__)}
    manifest=dict(tasks=old['tasks'],metrics=old['metrics'],pilot_tasks=TASKS,baseline=old['baseline'],
        official_rows_sha256=sha(OUT/'official_rows.npz'),expert_features_sha256=sha(OUT/'expert_features.npz'),
        physical_names=physical_names,rows=old['rows'],unique_exact_smiles=old['unique_exact_smiles'],
        controls=records,feature_source_hashes=sources,context_target_and_mask_excluded=True,
        context_sources='Other official train labels; final additionally valid. Existing context implementation unchanged.',
        historical_control_limitation='Partial historical proxy OOF; every adopted v364 layer replayed.',
        source_inputs=old['source_inputs'],training_host='connect.nmb2.seetacloud.com:25870',all_fitting_remote=True,test_labels_parsed=False)
    save(OUT/'data_manifest.json',manifest)
    for n in ('v371_mae_experts.py','v371_run_experts.py','v371_verify_experts.py'):
        shutil.copyfile(ROOT/'tools'/n,OUT/n)
    archive=ROOT/'releases/v371_expert_inputs.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():tf.add(p,arcname=p.relative_to(OUT).as_posix())
    report=dict(status='PASS',archive=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,task_count=5,partitions=25)
    save(ROOT/'docs/v371_input_package.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
