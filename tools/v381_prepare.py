"""Freeze remote-only OOF comparison and immutable inherited inputs."""
from pathlib import Path
import hashlib,json,os,shutil,tarfile
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/regression9_v381'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (EXP/'source_freeze.json').exists(),'Never overwrite frozen sources'
    EXP.mkdir(parents=True,exist_ok=True)
    runner=(ROOT/'tools/v380_run_experts.py').read_text(encoding='utf-8')
    runner=runner.replace('v380_histogram_median','v381_oof_pareto').replace('v380_verify_experts','v381_verify_experts').replace('380000','381000')
    runner=runner.replace('max_workers=3','max_workers=2')
    (ROOT/'tools/v381_run_experts.py').write_text(runner,encoding='utf-8')
    prev=ROOT/'experiments/regression9_v377'
    manifest=json.loads((prev/'data_manifest.json').read_text(encoding='utf-8'))
    names=['official_rows.npz','expert_features.npz','stereo_keys.npz','submission_template.csv','v374_baseline.csv','data_manifest.json']
    names += [row[k] for row in manifest['controls'] for k in ('control','design')]
    for name in names:
        src,dest=prev/name,EXP/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha(src)==sha(dest)
        else:os.link(src,dest)
    sources=('v371_mae_experts.py','v377_regression_experts.py','v380_histogram_median.py',
             'v381_oof_pareto.py','v381_run_experts.py','v381_verify_experts.py','v381_smoke.py')
    entries=[]
    for name in sources:
        shutil.copyfile(ROOT/'tools'/name,EXP/name);entries.append(dict(name=name,sha256=sha(EXP/name)))
    freeze=dict(files=entries,official_best=51.271,all_fitting_remote=True,workers=2,
                inherited_inputs='regression9_v377',no_external_data=True,comparison_baseline='v374 scored association; v380 unscored candidate kept separate')
    (EXP/'source_freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    archive=ROOT/'releases/v381_oof_delta.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for name in sources+('source_freeze.json',):tf.add(EXP/name,arcname=name)
    receipt=dict(status='PASS',archive=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,
                 source_files=entries,local_input_files_verified=len(names))
    (ROOT/'docs/v381_input_package.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
