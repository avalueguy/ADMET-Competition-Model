"""Build a credential-free handoff with exact model source bytes and resumable artifacts."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import ast
import hashlib
import json
import re
import shutil
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'work/github_handoff_v366_20261104'
DATE = '20261104'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    best = json.loads((ROOT/'docs/current_best_after_v364_20261104.json').read_text(encoding='utf-8'))
    candidate = json.loads((ROOT/'docs/v366_stage2_delivery.json').read_text(encoding='utf-8'))
    assert sha(ROOT/best['csv']) == best['csv_sha256']
    assert sha(ROOT/candidate['csv']) == candidate['csv_sha256']
    weight_archive = ROOT/'releases/v366_final_weights.tar.gz'
    assert sha(weight_archive) == '36a821c5ff55f2f86d10176f4729f8f3eff7702e03920635e6a9e6793990ceb5'
    with tarfile.open(weight_archive) as tf:
        remote_weights = json.loads(tf.extractfile('docs/v367_remote_weight_manifest.json').read())
        for item in remote_weights['files']:
            path = ROOT/item['path']
            assert path.resolve().is_relative_to(ROOT.resolve())
            data = tf.extractfile(item['path']).read()
            assert hashlib.sha256(data).hexdigest() == item['sha256']
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                assert sha(path) == item['sha256']
            else:
                path.write_bytes(data)
    save(ROOT/'docs/v367_remote_weight_manifest.json', remote_weights)

    current = json.loads((ROOT/'CURRENT_STATUS.json').read_text(encoding='utf-8'))
    current.update(handoff_date='2026-11-04', timezone='Asia/Hong_Kong',
                   latest_completed_model_branch='v366', latest_completed_model_branch_is_proven_best=False,
                   rollback_baseline='v364', latest_aggregate_plan='v366',
                   aggregate_status='V366_EXPERIMENTAL_DELIVERED_AWAITING_OFFICIAL_FEEDBACK',
                   official_score=51.269, known_best_version='v364', target_achieved=False,
                   latest_candidate_official_score=None, local_gpu_training=False,
                   automatic_platform_submission=False, active_gpu_training_at_snapshot=False,
                   current_handoff='docs/HANDOFF_CURRENT.md',
                   v364_official_feedback=best, v366_experimental_delivery=candidate,
                   source_of_official_score='user-reported full scorecard; platform upload hash unavailable')
    save(ROOT/'CURRENT_STATUS.json', current)
    save(ROOT/'docs/v367_github_handoff_status.json', dict(prepared_at_hong_kong=datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
         repository='avalueguy/ADMET-Competition-Model', account='avalueguy', repository_visibility='public',
         known_best=best, latest_candidate=candidate, passwords_included=False, final_weights=6,
         historical_oof_proxy_limitations_retained=True, new_external_data_or_weights=False))
    guide = ROOT/'docs/HANDOFF_CURRENT.md'
    assert guide.is_file()
    shutil.copyfile(guide, ROOT/'HANDOFF_CURRENT.md')

    # All current source and written evidence, plus exact inputs for the two new stages.
    sources = {p for p in ROOT.iterdir() if p.is_file() and p.suffix.lower() in ('.py','.json','.md','.txt','.toml','.yml','.yaml')}
    sources |= {ROOT/'.gitignore', ROOT/'.gitattributes'}
    for folder in ('tools', 'docs'):
        sources |= {p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix.lower() in ('.py','.json','.md','.txt','.toml','.yml','.yaml','.mjs')
                    and not any(x in p.parts for x in ('browser_proof','__pycache__','archive_conflicts'))}
    # Local transport wrappers are safe to include, but no browser telemetry or UI task files.
    sources = {p for p in sources if not p.name.startswith(('v367_upload_', 'github-publish-'))}
    forbidden = re.compile(rb'ghp_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{20,}|BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY')
    for p in sources:
        content = p.read_bytes()
        assert not forbidden.search(content), 'Potential credential: '+p.relative_to(ROOT).as_posix()
        if p.suffix == '.py':
            ast.parse(content.decode('utf-8-sig'), filename=p.name)
    evidence = set()
    for stage in ('transformer_stage1_v364','transformer_stage2_v366'):
        base = ROOT/'experiments'/stage
        evidence |= {p for p in base.rglob('*') if p.is_file() and p.name not in ('lock','graphs.pt','training.log')
                     and '__pycache__' not in p.parts and not p.name.endswith('.tmp')}
    # Fixed old outputs are retained as inputs, not advertised as newly reproduced models.
    for version in ('v361','v364','v366'):
        patterns = {'v361':'v361_herg_feedback_repair','v364':'v364_graph_transformer_experimental','v366':'v366_graph_transformer_experimental'}
        base = ROOT/'admet_submissions'/patterns[version]
        assert base.is_dir(), base
        evidence |= {p for p in base.iterdir() if p.is_file()}
    evidence.add(ROOT/'admet_official/submission_template.csv')
    for name in ('v364_graph_transformer_stage1_20261104.tar.gz','v364_remote_results.tar.gz',
                 'v366_stage2_20261104.tar.gz','v366_remote_results.tar.gz'):
        evidence.add(ROOT/'releases'/name)
    # Preserve the original v361 baseline consumed by stage 1 and final prediction NPZs.
    entries = []
    for p in sorted(sources | evidence):
        assert p.resolve().is_relative_to(ROOT.resolve())
        entries.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    runtime = DEST/'admet-v366-runtime.tar.gz'
    with tarfile.open(runtime,'w:gz') as tf:
        for p in sorted(evidence):
            tf.add(p,arcname=p.relative_to(ROOT).as_posix())
    with tarfile.open(runtime) as tf:
        for p in sorted(evidence):
            assert hashlib.sha256(tf.extractfile(p.relative_to(ROOT).as_posix()).read()).hexdigest()==sha(p)
    parts=[]
    with runtime.open('rb') as f:
        number=1
        while block:=f.read(20*1024*1024):
            path=DEST/f'admet-v366-runtime.tar.gz.part{number:03d}'
            path.write_bytes(block)
            parts.append(dict(name=path.name,sha256=sha(path),bytes=len(block)))
            number+=1
    source_zip=DEST/'admet-v366-source.zip'
    with zipfile.ZipFile(source_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as zf:
        for p in sorted(sources):
            zf.write(p,arcname=p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(source_zip) as zf:
        assert zf.testzip() is None
        for p in sources:
            assert hashlib.sha256(zf.read(p.relative_to(ROOT).as_posix())).hexdigest()==sha(p)
    manifest=dict(format_version=1,version='v366',created_at_hong_kong=datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
        known_official_best=dict(version='v364',score=51.269,csv=best['csv'],sha256=best['csv_sha256']),
        candidate=dict(version='v366',official_score=None,csv=candidate['csv'],sha256=candidate['csv_sha256'],experimental_only=True),
        source_archive=dict(name=source_zip.name,sha256=sha(source_zip),bytes=source_zip.stat().st_size,files=len(sources)),
        runtime_archive=dict(name=runtime.name,sha256=sha(runtime),bytes=runtime.stat().st_size,files=len(evidence),parts=parts),
        files=entries,remote_environment=remote_weights['environment'],full_historical_ensemble_reconstructed=False,
        limitations=['Historical ensemble OOF is partially proxy','Earlier full ensemble weights are incomplete','No new official result for v366',
                     'All development checkpoints remain remote; handoff includes the six final checkpoints',
                     'Graph cache excluded: rebuild deterministic graphs with original source and RDKit2026.09.1'],
        passwords_included=False)
    save(DEST/'handoff-manifest-v366.json',manifest)
    shutil.copyfile(ROOT/'tools/restore_v366_handoff.py',DEST/'restore_v366_handoff.py')
    for name in ('README.md','CURRENT_STATUS.json','HANDOFF_CURRENT.md'):
        shutil.copyfile(ROOT/name,DEST/name)
    save(ROOT/'docs/v367_handoff_package_verification.json',dict(status='PASS',files=len(entries),source_files=len(sources),runtime_files=len(evidence),
         source_zip_sha256=sha(source_zip),runtime_sha256=sha(runtime),parts=parts,final_checkpoints=6,all_member_hashes_verified=True))
    print(json.dumps(dict(status='PASS',destination=str(DEST),source_files=len(sources),runtime_files=len(evidence),
                         bytes=runtime.stat().st_size,parts=parts),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
