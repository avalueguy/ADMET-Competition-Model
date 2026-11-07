"""Archive all active v74 experiment weights and verify every member hash."""
from pathlib import Path
import sys,json,tarfile,hashlib,platform
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import admet_v67_data as b

def main():
    out=ROOT/'admet_model_outputs_v74';release=json.loads((out/'release.json').read_text(encoding='utf-8'));audit=json.loads((out/'model_verification.json').read_text(encoding='utf-8'))
    assert release['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM' and audit['status']=='PASS'
    assert b.sha(ROOT/release['file'])==release['sha256']
    b.save(out/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),packages={p:version(p) for p in ('numpy','scipy','scikit-learn','catboost','torch','rdkit')},gpu='NVIDIA A100-PCIE-40GB',credentials_included=False))
    names=['admet_v74_specialists.py','admet_v74_protocol.json','admet_v67_data.py','admet_v67_train.py','admet_v67_protocol.json',
        'tools/select_v74.py','tools/run_v74_confirmation.py','tools/run_v74_pipeline.py','tools/finalize_v74.py','tools/verify_v74_models.py','tools/package_v74_release.py','tools/verify_submission.py','tools/repack_upload_template.py',
        'admet_model_outputs_v74','admet_competition_data/data_benchmark','admet_official/submission_template.csv','docs/artifact-manifests/runtime-inputs.manifest.json',
        'admet_model_outputs_v67/features','admet_model_outputs_v67/folds.npy','admet_model_outputs_v67/split_audit.json',
        'admet_submissions/v73_v72','admet_submissions/v74_v73','recovered_history/baseline_valid.npy','recovered_history/baseline_cv_proxy.npy','recovered_history/baseline_replay.json',
        'v74_selection_training_fixed.log','v74_confirmation.log','v74_finalize.log','v74_verify.log']
    paths=[]
    for name in names:
        p=ROOT/name;assert p.exists(),name;paths.extend([p] if p.is_file() else [x for x in p.rglob('*') if x.is_file()])
    paths=sorted(set(paths));records=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=b.sha(p)) for p in paths]
    dest=ROOT/'releases';dest.mkdir(exist_ok=True);prefix='v74_v73_20261021';manifest=dest/(prefix+'.manifest.json')
    b.save(manifest,dict(files=records,complete_new_model_weights=True,full_historical_anchor_weights=False,official_score=None,credentials_included=False))
    archive=dest/(prefix+'.tar.gz')
    with tarfile.open(archive,'w:gz',compresslevel=3) as tf:
        for p in paths+[manifest]:tf.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    with tarfile.open(archive,'r:gz') as tf:
        for r in records:
            content=tf.extractfile(r['path']).read();assert len(content)==r['bytes'] and hashlib.sha256(content).hexdigest()==r['sha256']
    result=dict(status='PASS',path=archive.relative_to(ROOT).as_posix(),bytes=archive.stat().st_size,sha256=b.sha(archive),members=len(records)+1,all_member_hashes_verified=True)
    b.save(dest/(prefix+'.archive.json'),result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
