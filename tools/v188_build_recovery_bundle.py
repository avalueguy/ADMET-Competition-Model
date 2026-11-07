"""Rebuild the unfinished v181 workspace from immutable local inputs only."""
import ast
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'v188_v181_recovery_20261028'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    files = set()

    def add(path):
        path = ROOT / path
        assert path.is_file(), path
        files.add(path)

    def folder(path, pattern='*'):
        for p in (ROOT / path).rglob(pattern):
            if p.is_file():
                files.add(p)

    folder('tools', '*.py')
    files.update(ROOT.glob('*.py'))
    files.update(ROOT.glob('*protocol.json'))
    folder('docs', '*.json')
    add('admet_official/submission_template.csv')
    for part in ('train.csv', 'valid.csv'):
        folder('admet_competition_data/data_benchmark', part)
    for name in ('admet_model_outputs_v67/features',
                 'admet_model_outputs_v73/features',
                 'admet_model_outputs_v95/identity_audit',
                 'admet_model_outputs_v98/features',
                 'admet_model_outputs_v109/features',
                 'admet_model_outputs_v146/controls'):
        folder(name)
    for name in ('admet_model_outputs_v67/folds.npy',
                 'recovered_history/legacy_v24_folds.npy',
                 'recovered_history/baseline_cv_proxy.npy',
                 'recovered_history/baseline_valid.npy'):
        add(name)
    # Read-only metadata for the historical baseline loader. No failed model
    # training or old model weights are needed when v146 controls are cached.
    for p in ROOT.glob('admet_model_outputs_v*/*.json'):
        if p.parent.name not in ('admet_model_outputs_v181', 'admet_model_outputs_v182'):
            files.add(p)
    versions = (60, 73, 82, 83, 84, 85, 86, 88, 92, 94, 101, 114, 124, 130, 145)
    for version in versions:
        folders = list((ROOT / 'admet_submissions').glob('v' + str(version) + '_*'))
        folders += list((ROOT / 'admet_submissions').glob('v' + str(version)))
        for path in folders:
            for p in path.glob('*.csv'):
                files.add(p)
    training = ROOT / 'tools/v181_quantile_forest.py'
    assert sha(training) == 'b7d1828a0e5802f61edea3e6e23eeb4c8f8cf2bbf3bfc105383a5eba59e4218d'
    manifest = dict(version='v188', purpose='Restore unfinished v181 on user-provided replacement instance',
                    files=[dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
                           for p in sorted(files)], credentials_included=False,
                    test_label_files_included=False, trained_rejected_routes_restarted=False)
    mp = ROOT / 'releases' / (PREFIX + '.manifest.json')
    ap = ROOT / 'releases' / (PREFIX + '.tar.gz')
    assert not ap.exists() and not mp.exists()
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with tarfile.open(ap, 'w:gz', compresslevel=2) as tf:
        for p in sorted(files):
            tf.add(p, arcname=p.relative_to(ROOT).as_posix())
        tf.add(mp, arcname=mp.relative_to(ROOT).as_posix())
    info = dict(path=ap.relative_to(ROOT).as_posix(), sha256=sha(ap), bytes=ap.stat().st_size,
                files=len(files), manifest_sha256=sha(mp))
    (ROOT / 'releases' / (PREFIX + '.archive.json')).write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(info))


if __name__ == '__main__':
    main()
