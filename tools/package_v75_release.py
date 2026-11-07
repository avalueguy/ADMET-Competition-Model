"""Archive v75 models, exact sources and official inputs with member hashes."""
from pathlib import Path
import sys,json,tarfile,hashlib,platform
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import admet_v67_data as b


def main():
    out=ROOT/'admet_model_outputs_v75';release=json.loads((out/'release.json').read_text());audit=json.loads((out/'model_verification.json').read_text())
    assert release['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM' and audit['status']=='PASS';assert b.sha(ROOT/release['file'])==release['sha256']
    b.save(out/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),gpu='RTX 4090 48GB',
        packages={p:version(p) for p in ['numpy','scipy','scikit-learn','catboost','torch','rdkit']},credentials_included=False))
    names=['admet_v75_experiments.py','admet_v75_protocol.json','admet_v67_data.py','admet_v67_train.py','admet_v67_protocol.json',
        'tools/select_v75.py','tools/run_v75_pipeline.py','tools/finalize_v75.py','tools/verify_v75_models.py','tools/package_v75_release.py',
        'tools/verify_submission.py','tools/repack_upload_template.py','admet_model_outputs_v75','admet_competition_data/data_benchmark',
        'admet_official/submission_template.csv','admet_model_outputs_v67/features','admet_model_outputs_v67/folds.npy','admet_model_outputs_v67/split_audit.json',
        'docs/artifact-manifests/runtime-inputs.manifest.json','admet_submissions/v73_v72','admet_submissions/v75_v73','recovered_history',
        'v75_pipeline.log','v75_kernels.log','finalize_v75.log','verify_v75_models.log']
    paths=[]
    for name in names:
        p=ROOT/name;assert p.exists(),name;paths.extend([p] if p.is_file() else [x for x in p.rglob('*') if x.is_file()])
    # Logs of running parent processes are not sealed artifacts; archive a snapshot.
    snapshot=out/'log_snapshots';snapshot.mkdir(exist_ok=True)
    for name in ['v75_pipeline.log','v75_kernels.log','finalize_v75.log','verify_v75_models.log']:
        p=ROOT/name;target=snapshot/name;target.write_bytes(p.read_bytes());paths=[q for q in paths if q!=p];paths.append(target)
    paths=sorted(set(paths));records=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=b.sha(p)) for p in paths]
    dest=ROOT/'releases';dest.mkdir(exist_ok=True);prefix='v75_v73_20261022';manifest=dest/(prefix+'.manifest.json')
    b.save(manifest,dict(files=records,complete_new_model_weights=True,full_historical_anchor_weights=False,credentials_included=False,official_score=None))
    archive=dest/(prefix+'.tar.gz')
    with tarfile.open(archive,'w:gz',compresslevel=3) as tf:
        for p in paths+[manifest]:tf.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    with tarfile.open(archive,'r|gz') as tf:
        expected={r['path']:r for r in records}
        for member in tf:
            if member.name not in expected:continue
            raw=tf.extractfile(member).read();r=expected[member.name];assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
    result=dict(status='PASS',path=archive.relative_to(ROOT).as_posix(),bytes=archive.stat().st_size,sha256=b.sha(archive),members=len(records)+1,all_member_hashes_verified=True)
    b.save(dest/(prefix+'.archive.json'),result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
