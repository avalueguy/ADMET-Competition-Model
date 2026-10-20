"""Package compact experiment evidence without copying GPU model files locally."""
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    paths = []
    for name in ['admet_model_outputs_v20', 'admet_model_outputs_v20_rare',
                 'admet_model_outputs_v20_map', 'admet_release_v20_rare', 'admet_submissions/v20']:
        paths.extend(p for p in (ROOT/name).rglob('*') if p.is_file() and p.suffix in {'.csv','.json'})
    paths.extend(p for p in ROOT.glob('admet_v20*') if p.suffix in {'.py','.json'})
    paths.append(ROOT/'test_admet_v20_rare.py')
    archive = ROOT/'admet_v20_review.tar.gz'
    with tarfile.open(archive, 'w:gz') as out:
        for path in sorted(set(paths)):
            if path.exists():
                out.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size, files=len(paths),
                          excludes_model_checkpoints=True)), flush=True)


if __name__ == '__main__':
    main()
