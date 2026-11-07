"""Package source, data, specialist weights and independent verification."""
from pathlib import Path
import sys, json, hashlib, tarfile, platform
from importlib.metadata import version

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import admet_v67_data as b

def main():
    out=ROOT/'admet_model_outputs_v72_v56'
    verification=json.loads((out/'model_verification.json').read_text())
    assert verification['status']=='PASS'
    release=json.loads((out/'release.json').read_text())
    assert {r['task'] for r in verification['final_records']}=={r['task'] for r in release['selected']}
    release.update(status='CANDIDATE_READY_AWAITING_PLATFORM',
        independent_reload_status='PASS',independent_reload_sha256=b.sha(out/'model_verification.json'))
    b.save(out/'release.json',release)
    csv=ROOT/release['file'];assert b.sha(csv)==release['sha256']
    audit_path=csv.with_suffix('.verification.json');audit=json.loads(audit_path.read_text())
    audit.update(model_reload_status='PASS',independent_reload_sha256=b.sha(out/'model_verification.json'))
    b.save(audit_path,audit)
    import torch,rdkit
    b.save(out/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),
        gpu=torch.cuda.get_device_name(0),packages={p:version(p) for p in ('numpy','scipy','scikit-learn','pandas','catboost','torch','rdkit')},
        training_root=str(ROOT),credentials_included=False))
    include=['admet_v71_v56_catboost.py','admet_v67_data.py','admet_v67_train.py','admet_v67_protocol.json',
        'admet_v68_kernel.py','admet_v68_protocol.json','admet_competition_data/data_benchmark',
        'admet_official/submission_template.csv','docs/official_data_restore_20261020.json','docs/ITERATION_20261021.md',
        'admet_model_outputs_v67/features','admet_model_outputs_v67/folds.npy','admet_model_outputs_v67/split_audit.json',
        'admet_model_outputs_v67/selection.json','admet_model_outputs_v67/valid',
        'admet_model_outputs_v68','admet_model_outputs_v71_v56','admet_model_outputs_v72_v56',
        'admet_model_outputs_v49/fusion.json','admet_submissions/v56','admet_submissions/v57','admet_submissions/v72_v56',
        'recovered_history/baseline_valid.npy','recovered_history/baseline_cv_proxy.npy','recovered_history/baseline_replay.json',
        'tools/analyze_v71_v56.py','tools/release_v72_v56.py','tools/verify_v71_models.py',
        'tools/verify_submission.py','tools/repack_upload_template.py','tools/package_v72_release.py','tools/finalize_v72_delivery.py',
        'v71.log','v72_final.log','v72_verify.log','v72_verify_fixed.log']
    paths=[]
    for name in include:
        p=ROOT/name
        assert p.exists(),name
        paths.extend([p] if p.is_file() else [x for x in p.rglob('*') if x.is_file()])
    paths=sorted(set(paths));records=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=b.sha(p)) for p in paths]
    dest=ROOT/'releases';dest.mkdir(exist_ok=True)
    manifest=dest/'v72_v56_20261021.manifest.json'
    b.save(manifest,dict(format='sha256 per file',files=records,complete_historical_v56_weights=False,
        historical_v56_original_submission_included=True,baseline_contains_historical_pretrained_sources=True,
        new_specialist_weights_and_fit_only_state_included=True))
    archive=dest/'v72_v56_20261021.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=3) as tf:
        for p in paths+[manifest]:tf.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    # Read every archived member and compare its actual bytes with the manifest.
    with tarfile.open(archive,'r:gz') as tf:
        for rec in records:
            data=tf.extractfile(rec['path']).read()
            assert len(data)==rec['bytes'] and hashlib.sha256(data).hexdigest()==rec['sha256']
    result=dict(status='PASS',path=archive.relative_to(ROOT).as_posix(),bytes=archive.stat().st_size,
        sha256=b.sha(archive),members=len(records)+1,all_member_hashes_verified=True)
    b.save(dest/'v72_v56_20261021.archive.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
