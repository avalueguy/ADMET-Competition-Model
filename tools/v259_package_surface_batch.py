"""Verify and preserve the distinct v256 exposed polarity route."""
import ast
import json
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path
from v161_package import sha, save

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'v257_exposed_polarity_20261029'


def package():
    out = ROOT / 'admet_model_outputs_v257'
    ep = out / 'evidence.json'
    assert ep.exists() or (out / 'no_candidate.json').exists()
    assert json.loads((ROOT / 'admet_model_outputs_v256/confirmation.json').read_text())['status'] == 'COMPLETE'
    if ep.exists():
        report = json.loads(ep.read_text())
        assert sha(ROOT / report['file']) == report['sha256']
        assert json.loads((out / 'independent_verification.json').read_text())['status'] == 'PASS'
    paths = []
    for folder in ('admet_model_outputs_v256', 'admet_model_outputs_v257',
                   'admet_submissions/v257_exposed_polarity'):
        paths.extend(p for p in (ROOT / folder).rglob('*')
                     if p.is_file() and p.suffix not in ('.tmp', '.pyc') and p.name != 'pipeline_status.json')
    paths.extend([ROOT / 'tools/v256_exposed_polarity.py',
                  ROOT / 'tools/v257_verify_exposed_polarity.py', Path(__file__),
                  ROOT / 'docs/v256_training_plan.json', ROOT / 'v256_exposed_polarity.log',
                  ROOT / 'v256_smoke.log', ROOT / 'v257_verify.log',
                  ROOT / 'docs/v256_source_review.json',
                  ROOT / 'tools/finalize_v257_local.py', ROOT / 'tools/v260_surface_driver.py',
                  ROOT / 'docs/v256_frozen_sources.json',
                  ROOT / 'docs/v243_v145_error_review.json', ROOT / 'tools/v243_v145_error_review.py',
                  ROOT / 'docs/v249_review_decision.json',
                  ROOT / 'docs/v255_surface_feasibility.json',
                  ROOT / 'tools/v258_build_surface.py',
                  ROOT / 'v256_feature_build.log',
                  ROOT / 'tools/v261_start_surface.py', *ROOT.glob('v260_worker_*.log')])
    queue = [p for p in paths if p.suffix == '.py']
    seen = set()
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for name in names:
                stem = name.split('.')[0]
                for p in (ROOT / (stem + '.py'), ROOT / 'tools' / (stem + '.py')):
                    if p.is_file() and p not in seen:
                        queue.append(p)
    files = {p.relative_to(ROOT).as_posix(): p for p in paths + list(seen) if p.is_file()}
    prefix = 'v176_shared_graph_batch_20261027'
    dep = ROOT / 'releases' / (prefix + '.tar.gz')
    info = json.loads((ROOT / 'releases' / (prefix + '.archive.json')).read_text())
    assert sha(dep) == info['sha256']
    manifest = ROOT / 'releases' / (PREFIX + '.manifest.json')
    save(manifest, dict(version='v257', files=[dict(path=n, bytes=p.stat().st_size, sha256=sha(p)) for n, p in sorted(files.items())],
                        dependencies=[dict(path=dep.relative_to(ROOT).as_posix(), sha256=info['sha256'])],
                        credentials_included=False, test_label_files_included=False,
                        submission_created=ep.exists(), official_score=None))
    archive = ROOT / 'releases' / (PREFIX + '.tar.gz')
    assert not archive.exists()
    with tarfile.open(archive, 'w:gz', compresslevel=4) as tf:
        for name, path in sorted(files.items()):
            tf.add(path, arcname=name)
        tf.add(manifest, arcname=manifest.relative_to(ROOT).as_posix())
    info = dict(path=archive.relative_to(ROOT).as_posix(), sha256=sha(archive),
                bytes=archive.stat().st_size, files=len(files), manifest_sha256=sha(manifest),
                submission_created=ep.exists(), official_score=None)
    save(ROOT / 'releases' / (PREFIX + '.archive.json'), info)
    print(json.dumps(info), flush=True)


def main():
    out = ROOT / 'admet_model_outputs_v257'
    out.mkdir(exist_ok=True)
    status = out / 'pipeline_status.json'
    for _ in range(720):
        failure = ROOT / 'admet_model_outputs_v256/failure.json'
        if failure.exists():
            raise RuntimeError('v256 failed; preserve failed exposed polarity batch without retry')
        cp = ROOT / 'admet_model_outputs_v256/confirmation.json'
        if cp.exists() and json.loads(cp.read_text())['status'] == 'COMPLETE':
            break
        save(status, dict(status='WAITING_TRAINING', route='v256'))
        time.sleep(30)
    else:
        raise TimeoutError('v256 incomplete; preserve partial results')
    save(status, dict(status='INDEPENDENT_VERIFICATION'))
    with (ROOT / 'v257_verify.log').open('a', encoding='utf-8') as log:
        subprocess.run([sys.executable, str(ROOT / 'tools/v257_verify_exposed_polarity.py')],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    save(status, dict(status='PACKAGING'))
    package()
    save(status, dict(status='COMPLETE', submission_created=(out / 'evidence.json').exists(),
                      official_score=None, automatically_uploaded=False))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        save(ROOT / 'admet_model_outputs_v257/pipeline_status.json', dict(status='FAILED', traceback=traceback.format_exc()))
        raise
