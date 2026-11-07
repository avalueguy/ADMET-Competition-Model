"""Freeze exact launched sources and record the actual v210 process evidence."""
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    now = datetime.now(timezone.utc).isoformat()
    start = json.loads((ROOT / 'docs/v210_remote_start.json').read_text())
    plan = json.loads((ROOT / 'docs/v210_training_plan.json').read_text())
    assert plan['source_sha256'] == sha(ROOT / 'tools/v210_standardized_chemistry.py')
    assert plan['runner_sha256'] == sha(ROOT / 'tools/v148_experiment_runner.py')
    assert start['smoke']['status'] == 'PASS'
    paths = ['tools/v210_standardized_chemistry.py', 'tools/v211_verify_standardized_chemistry.py',
             'tools/v212_package_standardized_batch.py', 'tools/finalize_v211_local.py',
             'docs/v210_training_plan.json', 'docs/v210_source_review.json']
    archive = ROOT / 'releases/v210_frozen_sources_20261028.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive, 'w:gz') as tf:
        for path in paths:
            tf.add(ROOT / path, arcname=path)
    info = dict(path=archive.relative_to(ROOT).as_posix(), sha256=sha(archive),
                files=[dict(path=path, sha256=sha(ROOT / path)) for path in paths], credentials_included=False)
    save(ROOT / 'docs/v210_frozen_sources.json', info)
    follow = dict(status='TRAINING_NEW_STANDARDIZED_CHEMISTRY_EXPERTS', checked_at_utc=now,
        training_version='v210', delivery_version='v211', baseline='v145', official_best=51.166,
        target_score=52, target_achieved=False, host='connect.nmb2.seetacloud.com', port=25870,
        remote_workspace='/root/autodl-tmp/v76', python='/root/miniconda3/bin/python',
        training_pid=start['processes']['training']['pid'], pipeline_pid=start['processes']['pipeline']['pid'],
        process_evidence='docs/v210_remote_start.json', log='v210_standardized_chemistry.log',
        pipeline_log='v212_pipeline.log', pipeline_status='admet_model_outputs_v211/pipeline_status.json',
        archive_prefix='v211_standardized_chemistry_20261028', source_sha256=plan['source_sha256'],
        signature=start['smoke']['signature'], smoke='PASS', tasks='All21 improvable endpoints; HIA protected',
        training_plan='docs/v210_training_plan.json', frozen_sources_archive=info['path'],
        verification_source='tools/v211_verify_standardized_chemistry.py',
        local_finalizer='tools/finalize_v211_local.py', credentials_stored=False,
        source_immutable_after_start=True, thresholds_unchanged=True, new_csv_created=False,
        new_upload_performed=False, new_official_score=None, running_training_processes=1,
        last_observed='Source feature generation reached45000/46202, actual training13106/pipeline13163 identities verified. Complete feature audit now saved,6576 new columns.',
        obsolete_automation_prompt='Earlier v181/v189 routes completed. Defer stale heartbeat PIDs to CURRENT_STATUS and this followup. App prompt refresh previously blocked by Transport closed.',
        next_steps=['Check exact process identity and actual output before acting; never blindly restart.',
            'After completion fetch all3release files, run tools/restore_iteration_archive.py v211_standardized_chemistry_20261028.',
            'No accepted tasks: record noCSV/no upload and do not rerun rejected route.',
            'Accepted tasks: run tools/finalize_v211_local.py; all gates/model hashes/template nonprediction bytes/Drug_ID/backup hashes required.',
            'Only eligible changed predictions may be uploaded through visible logged-in Edge/cua_repl, then read new timestamp row complete22metrics.',
            'Official baseline staysv14551.166 unless actual official score higher. No official gain guarantee.'])
    save(ROOT / 'docs/v210_followup.json', follow)
    state = json.loads((ROOT / 'CURRENT_STATUS.json').read_text(encoding='utf-8-sig'))
    state.update(active_experiment='v210', active_experiment_status=follow['status'],
        current_followup='docs/v210_followup.json', last_updated_hong_kong='2026-10-28',
        current_state_checked_at_utc=now, recommended_submission_version=None,
        new_submission_created=False, running_new_training_processes=1,
        last_known_running_new_training_processes=1,
        running_new_training_processes_note='v210training13106 and v212verification/archive13163 live; original frozen gates preserved.')
    state['v210_followup'] = follow
    save(ROOT / 'CURRENT_STATUS.json', state)
    with (ROOT / 'docs/EDGE_SUBMISSION_20261027.md').open('a', encoding='utf-8') as file:
        file.write('\n2026-10-28 newv210/v211: neutral organic-parent and bounded canonical-tautomer countfingerprints/descriptors, plus original chemistry/context. Two fixed CatBoost seeds,1000treesdepth6,all21 improvable endpoints/HIA protected. Salt and keto/enol invariance, originalSMILES mapping, query-label isolation and actual classification/MAE/Spearman independent reload smoke PASS. Actualtraining13106 and verifier/archive13163 verified by bootID/startticks. Gates unchanged, noCSV/upload. Followdocs/v210_followup.json; earlier heartbeatv181/v189 is stale.\n')
    print(json.dumps(follow, ensure_ascii=False))


if __name__ == '__main__':
    main()
