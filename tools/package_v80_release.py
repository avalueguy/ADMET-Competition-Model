"""Package exact new-branch inputs and weights; never includes credentials."""
from pathlib import Path
import sys
import json
import tarfile
import hashlib
import platform
import shutil
import subprocess
from importlib.metadata import version

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import admet_v77_ranking as data

def main():
    out=ROOT/'admet_model_outputs_v80_final'
    release=json.loads((out/'release.json').read_text())
    audit=json.loads((out/'model_verification.json').read_text())
    assert release['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM' and audit['status']=='PASS'
    assert data.sha(ROOT/release['file'])==release['sha256']
    release.update(external_data_new_branch=False,pretrained_weights_new_branch=False,
        anchor_contains_historical_pretraining=True,full_historical_anchor_weights_included=False,
        verification_sha256=data.sha(out/'model_verification.json'))
    release.pop('external_data',None)
    data.save(out/'release.json',release)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip()
    data.save(out/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),gpu=gpu,
        packages={p:version(p) for p in ['numpy','scipy','scikit-learn','catboost','rdkit']},credentials_included=False))
    logs=out/'log_snapshots';logs.mkdir(exist_ok=True)
    for path in Path('/tmp').glob('v80*.out'):
        shutil.copyfile(path,logs/path.name)
    for name in ['v80_residual_probe.json']:
        p=Path('/tmp')/name
        if p.exists():shutil.copyfile(p,logs/name)
    names=['admet_v80_offset.py','admet_v80_nested.py','admet_v80_protocol.json',
        'admet_v77_ranking.py','admet_v67_data.py','admet_v67_protocol.json',
        'tools/verify_v80_models.py','tools/package_v80_release.py','tools/finalize_v80_delivery.py',
        'tools/verify_submission.py','tools/repack_upload_template.py','tools/experiments_v80',
        'docs/ITERATION_V80_20261023.md','docs/v80_nested_confirmation.json',
        'docs/artifact-manifests/runtime-inputs.manifest.json',
        'admet_model_outputs_v80_final','admet_model_outputs_v80_nested','admet_model_outputs_v80_meta',
        'admet_model_outputs_v80_offset','admet_submissions/v80_cyp2c9_residual',
        'admet_official/submission_template.csv',
        'recovered_history/baseline_cv_proxy.npy','recovered_history/baseline_valid.npy',
        'recovered_history/legacy_v24_folds.npy','admet_model_outputs_v67/folds.npy',
        'admet_model_outputs_v67/features/official_chemistry.npz',
        'admet_model_outputs_v67/features/official_chemistry.json','admet_model_outputs_v73/features/views.npz']
    paths=[next((ROOT/'admet_submissions/v73_v72').glob('*.csv'))]
    for name in names:
        p=ROOT/name;assert p.exists(),name
        paths.extend([p] if p.is_file() else [x for x in p.rglob('*') if x.is_file() and '__pycache__' not in x.parts])
    paths.extend((ROOT/'admet_competition_data/data_benchmark').glob('*/train.csv'))
    paths.extend((ROOT/'admet_competition_data/data_benchmark').glob('*/valid.csv'))
    for name in ['probe.json','selection.json']:
        paths.append(ROOT/'admet_model_outputs_v80_probe'/name)
    paths=sorted(set(paths))
    records=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=data.sha(p)) for p in paths]
    dest=ROOT/'releases';dest.mkdir(exist_ok=True);prefix='v80_cyp2c9_residual_20261023'
    manifest=dest/(prefix+'.manifest.json')
    data.save(manifest,dict(files=records,complete_new_model_weights=True,
        full_historical_anchor_weights=False,credentials_included=False,official_score=None))
    archive=dest/(prefix+'.tar.gz')
    with tarfile.open(archive,'w:gz',compresslevel=3) as tf:
        for p in paths+[manifest]:tf.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    expected={r['path']:r for r in records};seen=set()
    with tarfile.open(archive,'r|gz') as tf:
        for member in tf:
            assert member.isfile() and member.name not in seen;seen.add(member.name)
            raw=tf.extractfile(member).read()
            if member.name==manifest.relative_to(ROOT).as_posix():assert raw==manifest.read_bytes();continue
            rec=expected[member.name]
            assert len(raw)==rec['bytes'] and hashlib.sha256(raw).hexdigest()==rec['sha256']
    assert seen==set(expected)|{manifest.relative_to(ROOT).as_posix()}
    result=dict(status='PASS',path=archive.relative_to(ROOT).as_posix(),bytes=archive.stat().st_size,
        sha256=data.sha(archive),members=len(seen),all_member_hashes_verified=True)
    data.save(dest/(prefix+'.archive.json'),result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
