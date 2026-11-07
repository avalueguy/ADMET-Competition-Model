"""Copy immutable v377 data into v378 for local result arithmetic only."""
from pathlib import Path
import hashlib
import json
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'experiments/regression9_v377'
    dest = root / 'experiments/regression9_v378'
    manifest = json.loads((source / 'data_manifest.json').read_text(encoding='utf-8'))
    names = ['official_rows.npz', 'expert_features.npz', 'stereo_keys.npz',
             'submission_template.csv', 'v374_baseline.csv', 'data_manifest.json']
    names += [r[key] for r in manifest['controls'] for key in ('control', 'design')]
    for name in names:
        src, dst = source / name, dest / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            assert sha(src) == sha(dst), name
        else:
            shutil.copyfile(src, dst)
    for row in manifest['controls']:
        for key in ('control', 'design'):
            assert sha(dest / row[key]) == row[key + '_sha256']
    result = dict(status='PASS', files=len(names), all_fitting_remote=True,
                  note='Local byte copy and hash checks only; no fitting.')
    (root / 'docs/v378_local_input_verification.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
