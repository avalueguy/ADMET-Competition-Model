"""Verify and preserve the distinct v289 stereochemical context repair route."""
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
PREFIX = 'v290_stereo_schema_fix_20261031'


def package():
    out = ROOT / 'admet_model_outputs_v290'
    ep = out / 'evidence.json'
    assert ep.exists() or (out / 'no_candidate.json').exists()
    assert json.loads((ROOT / 'admet_model_outputs_v289/confirmation.json').read_text())['status'] == 'COMPLETE'
    if ep.exists():
        report = json.loads(ep.read_text())
        assert sha(ROOT / report['file']) == report['sha256']
        assert json.loads((out / 'independent_verification.json').read_text())['status'] == 'PASS'
    paths = []
    for task in ('herg', 'cyp3a4_substrate_carbonmangels'):
        for part in ('fold0', 'fold1', 'fold2', 'valid', 'final'):
            prior = ROOT / 'admet_model_outputs_v127/models' / task / part / 'physchem'
            paths.extend(p for p in prior.iterdir() if p.is_file())
    paths.extend([ROOT / 'tools/v283_stereochemical_context_audit.py',
                  ROOT / 'docs/v283_stereochemical_context_audit.json',
                  ROOT / 'docs/v283_review_decision.json',
                  ROOT / 'docs/v289_original_models.json', ROOT / 'docs/v289_schema_defect_review.json',
                  ROOT / 'docs/v285_final_status.json', ROOT / 'tools/prepare_v289_schema_fix.py'])
    for folder in ('admet_model_outputs_v289', 'admet_model_outputs_v290',
                   'admet_submissions/v290_stereo_schema_fix'):
        paths.extend(p for p in (ROOT / folder).rglob('*')
                     if p.is_file() and p.suffix not in ('.tmp', '.pyc') and p.name != 'pipeline_status.json')
    paths.extend([ROOT / 'tools/v289_stereo_schema_fix.py',
                  ROOT / 'tools/v290_verify_stereo_schema_fix.py', Path(__file__),
                  ROOT / 'docs/v289_training_plan.json', ROOT / 'v289_stereo_schema_fix.log',
                  ROOT / 'v289_smoke.log', ROOT / 'v290_verify.log',
                  ROOT / 'docs/v289_source_review.json',
                  ROOT / 'tools/finalize_v290_local.py', ROOT / 'tools/v292_stereo_schema_driver.py',
                  ROOT / 'docs/v289_frozen_sources.json', ROOT / 'docs/v289_remote_start.json',
                  ROOT / 'docs/v243_v145_error_review.json', ROOT / 'tools/v243_v145_error_review.py',
                  ROOT / 'docs/v249_review_decision.json',
                  ROOT / 'tools/v293_start_stereo_schema_fix.py', ROOT / 'tools/v270_assay_dependence_audit.py',
                  ROOT / 'docs/v270_assay_dependence_audit.json', ROOT / 'docs/v270_review_decision.json', *ROOT.glob('v292_worker_*.log')])
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
    save(manifest, dict(version='v290', files=[dict(path=n, bytes=p.stat().st_size, sha256=sha(p)) for n, p in sorted(files.items())],
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
    out = ROOT / 'admet_model_outputs_v290'
    out.mkdir(exist_ok=True)
    status = out / 'pipeline_status.json'
    for _ in range(720):
        failure = ROOT / 'admet_model_outputs_v289/failure.json'
        if failure.exists():
            raise RuntimeError('v289 failed; preserve failed stereochemical context repair batch without retry')
        cp = ROOT / 'admet_model_outputs_v289/confirmation.json'
        if cp.exists() and json.loads(cp.read_text())['status'] == 'COMPLETE':
            break
        save(status, dict(status='WAITING_TRAINING', route='v289'))
        time.sleep(30)
    else:
        raise TimeoutError('v289 incomplete; preserve partial results')
    save(status, dict(status='INDEPENDENT_VERIFICATION'))
    with (ROOT / 'v290_verify.log').open('a', encoding='utf-8') as log:
        subprocess.run([sys.executable, str(ROOT / 'tools/v290_verify_stereo_schema_fix.py')],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    save(status, dict(status='PACKAGING'))
    package()
    save(status, dict(status='COMPLETE', submission_created=(out / 'evidence.json').exists(),
                      official_score=None, automatically_uploaded=False))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        save(ROOT / 'admet_model_outputs_v290/pipeline_status.json', dict(status='FAILED', traceback=traceback.format_exc()))
        raise
