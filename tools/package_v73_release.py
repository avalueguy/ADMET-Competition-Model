"""Archive all v73 weights and exact inputs after independent verification."""
from pathlib import Path
import sys,json,tarfile,hashlib,platform
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import admet_v67_data as b

def main():
    out=ROOT/'admet_model_outputs_v73';release=json.loads((out/'release.json').read_text())
    audit=json.loads((out/'model_verification.json').read_text())
    assert release['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM'
    assert audit['status']=='PASS' and audit['models']==194 and audit['final_models']==10
    assert b.sha(ROOT/release['file'])==release['sha256']
    b.save(out/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),
        packages={p:version(p) for p in ('numpy','scipy','scikit-learn','catboost','torch','rdkit')},
        training_gpu='NVIDIA A100-PCIE-40GB',training_root=str(ROOT),credentials_included=False))
    include=['admet_v73_experts.py','admet_v73_conditional.py','admet_v67_data.py','admet_v67_train.py','admet_v67_protocol.json',
        'admet_competition_data/data_benchmark','admet_official/submission_template.csv',
        'docs/artifact-manifests/runtime-inputs.manifest.json','docs/official_rules_20261020.json',
        'docs/platform_feedback_v72_20261021.json','docs/ITERATION_V73_20261021.md',
        'admet_model_outputs_v73','admet_model_outputs_v73_conditional',
        'admet_model_outputs_v67/features','admet_model_outputs_v67/folds.npy','admet_model_outputs_v67/split_audit.json',
        'admet_submissions/v72_v56','admet_submissions/v73_v72','admet_model_outputs_v72_v56/diagnostic.json','recovered_history/baseline_valid.npy',
        'recovered_history/baseline_cv_proxy.npy','recovered_history/baseline_replay.json',
        'tools/analyze_v73.py','tools/audit_v73_controls.py','tools/finalize_v73.py','tools/verify_v73_models.py',
        'tools/package_v73_release.py','tools/verify_submission.py','tools/repack_upload_template.py',
        'v73_development.log','v73_conditional.log','v73_analysis.log','v73_finalize.log','v73_finalize_fixed.log','v73_verify.log']
    paths=[]
    for name in include:
        p=ROOT/name;assert p.exists(),name
        paths.extend([p] if p.is_file() else [x for x in p.rglob('*') if x.is_file()])
    paths=sorted(set(paths));records=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=b.sha(p)) for p in paths]
    dest=ROOT/'releases';dest.mkdir(exist_ok=True);manifest=dest/'v73_v72_20261021.manifest.json'
    b.save(manifest,dict(files=records,official_score=None,baseline='v72',
        complete_historical_baseline_weights=False,baseline_submission_included=True,
        all_new_expert_weights_included=True,credentials_included=False))
    archive=dest/'v73_v72_20261021.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=3) as tf:
        for p in paths+[manifest]:tf.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    with tarfile.open(archive,'r:gz') as tf:
        for r in records:
            data=tf.extractfile(r['path']).read();assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
    result=dict(status='PASS',path=archive.relative_to(ROOT).as_posix(),sha256=b.sha(archive),
        bytes=archive.stat().st_size,members=len(records)+1,all_member_hashes_verified=True)
    b.save(dest/'v73_v72_20261021.archive.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
