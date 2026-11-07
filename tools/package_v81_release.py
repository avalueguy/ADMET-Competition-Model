"""Package v81 exact inputs, models and verification records."""
from pathlib import Path
import sys,json,tarfile,hashlib,platform,subprocess,shutil
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import admet_v77_ranking as b
def main():
    out=ROOT/'admet_model_outputs_v81_final';release=json.loads((out/'release.json').read_text());audit=json.loads((out/'model_verification.json').read_text())
    assert audit['status']=='PASS' and release['status']=='CANDIDATE_PENDING_REPLAY';assert b.sha(ROOT/release['file'])==release['sha256']
    release['status']='VERIFIED_CANDIDATE_AWAITING_PLATFORM';release['verification_sha256']=b.sha(out/'model_verification.json');b.save(out/'release.json',release)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip()
    b.save(out/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),gpu=gpu,packages={p:version(p) for p in ['numpy','scipy','scikit-learn','catboost','rdkit']},credentials_included=False))
    logs=out/'log_snapshots';logs.mkdir(exist_ok=True)
    for p in [Path('/tmp/v81_class_probe.out'),Path('/tmp/v81_confirm.out'),Path('/tmp/v81_export.out'),Path('/tmp/v81_verify.out')]:
        if p.exists():shutil.copyfile(p,logs/p.name)
    names=['tools/v81_export.py','tools/verify_v81_models.py','tools/package_v81_release.py','tools/finalize_v81_delivery.py','tools/verify_submission.py','tools/repack_upload_template.py','admet_v77_ranking.py','admet_v67_data.py','admet_v67_protocol.json','docs/ITERATION_V81_20261024.md','admet_model_outputs_v81_class','admet_model_outputs_v81_final','admet_submissions/v81_multitask_residual','admet_submissions/v80_cyp2c9_residual','admet_official/submission_template.csv','recovered_history/baseline_cv_proxy.npy','recovered_history/baseline_valid.npy','recovered_history/legacy_v24_folds.npy','admet_model_outputs_v67/features/official_chemistry.npz','admet_model_outputs_v73/features/views.npz']
    paths=[]
    for name in names:
        p=ROOT/name;assert p.exists(),name;paths.extend([p] if p.is_file() else [x for x in p.rglob('*') if x.is_file() and '__pycache__' not in x.parts])
    paths=sorted(set(paths));records=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=b.sha(p)) for p in paths]
    dest=ROOT/'releases';dest.mkdir(exist_ok=True);prefix='v81_multitask_residual_20261024';manifest=dest/(prefix+'.manifest.json');b.save(manifest,dict(files=records,complete_new_model_weights=True,credentials_included=False,official_score=None))
    archive=dest/(prefix+'.tar.gz')
    with tarfile.open(archive,'w:gz',compresslevel=3) as tf:
        for p in paths+[manifest]:tf.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    seen=set();expected={r['path']:r for r in records}
    with tarfile.open(archive,'r|gz') as tf:
        for m in tf:
            assert m.isfile() and m.name not in seen;seen.add(m.name);raw=tf.extractfile(m).read()
            if m.name==manifest.relative_to(ROOT).as_posix():assert raw==manifest.read_bytes();continue
            r=expected[m.name];assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
    result=dict(status='PASS',path=archive.relative_to(ROOT).as_posix(),bytes=archive.stat().st_size,sha256=b.sha(archive),members=len(seen),all_member_hashes_verified=True);b.save(dest/(prefix+'.archive.json'),result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
