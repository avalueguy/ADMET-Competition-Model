"""Package small review artifacts; keep remote training caches/checkpoints remote."""
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    verified=json.loads(Path('admet_release_v5/verification.json').read_text())
    assert verified['status']=='VERIFIED'
    names=set()
    for root in ['admet_model_outputs_v5','admet_v5_confirmation','admet_release_v5']:
        for p in Path(root).rglob('*'):
            if p.is_file() and p.suffix in {'.csv','.json','.txt'}:
                names.add(p)
    for p in Path('admet_diagnostics').glob('v5_*'):
        if p.is_file():names.add(p)
    for p in Path('admet_submissions').glob('admet-v5*'):
        if p.is_file():names.add(p)
    for p in Path('logs').glob('*'):
        if p.is_file() and p.suffix in {'.txt','.log'}:names.add(p)
    names.update(Path('.').glob('admet_v5_*.py'))
    names.update(Path('.').glob('test_admet_v5*.py'))
    names.update(Path(p) for p in ['admet_v5_protocol.json','admet_legacy_recipes.json','admet_remote_requirements.txt'])
    dest=Path('admet_v5_review.tar.gz')
    with tarfile.open(dest,'w:gz') as tf:
        for p in sorted(names):tf.add(p,arcname=str(p),recursive=False)
    report=dict(archive=str(dest),bytes=dest.stat().st_size,
                sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),files=len(names),
                model_location='remote /root/autodl-tmp/admet_20260925',
                checkpoint_policy='Heavy checkpoints/caches excluded from local review package')
    Path('admet_v5_review_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
