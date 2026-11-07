"""Close the fully verified v189 batch without manufacturing a submission."""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))


def save(path, value):
    (ROOT / path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    now = datetime.now(timezone.utc).isoformat()
    archive = load('docs/v190_local_archive_verification.json')
    confirmation = load('admet_model_outputs_v189/confirmation.json')
    dev = load('admet_model_outputs_v189/development.json')
    no = load('admet_model_outputs_v190/no_candidate.json')
    assert archive['status'] == 'PASS'
    assert dev['status'] == confirmation['status'] == 'COMPLETE'
    assert archive['all_member_hashes_verified'] and archive['dependency_hashes_verified']
    assert not confirmation['accepted'] and not no['submission_created']
    assert not (ROOT / 'admet_model_outputs_v190/evidence.json').exists()
    assert not list((ROOT / 'admet_submissions/v190_leafwise_boosting').glob('*.csv'))
    reasons = []
    for record in confirmation['records']:
        reasons.append(dict(task=record['task'], accepted=False,
                            gains={r['part']: r['gain'] for r in record['confirmation']},
                            bootstrap90={r['part']: r['gain_ci90'] for r in record['bootstrap']}))
    result = dict(status='COMPLETE_NO_CONFIRMED_NEW_MODEL', checked_at_utc=now,
                  training_version='v189', delivery_version='v190',
                  baseline='v145', official_best=51.166, target_score=52, target_achieved=False,
                  development_tasks=len({r['task'] for r in dev['records']}),
                  nominated_tasks=[r['task'] for r in confirmation['records']], accepted_tasks=[],
                  confirmation_results=reasons, archive=archive, new_csv_created=False,
                  new_upload_performed=False, new_official_score=None,
                  frozen_gates_unchanged=True, rejected_route_restarted=False,
                  no_confirmation_fallback=True)
    save('docs/v190_final_status.json', result)
    follow = load('docs/v189_followup.json')
    follow.update(status=result['status'], checked_at_utc=now, accepted_tasks=[],
                  completed_development_tasks=list(dict.fromkeys(r['task'] for r in dev['records'])),
                  training_pid=None, pipeline_pid=None, new_csv_created=False,
                  last_observed='All21 development endpoints and five confirmations complete; accepted []; archive331 files and dependency verified locally.',
                  final_status='docs/v190_final_status.json',
                  next_steps=['Do not restart rejected v189, generate a CSV, or upload the old baseline.',
                              'A genuinely different future method must preserve the frozen v148 gates.'])
    save('docs/v189_followup.json', follow)
    recovery = load('docs/v188_recovery_status.json')
    recovery.update(status='RECOVERY_COMPLETE_V181_AND_V189_ARCHIVED', checked_at_utc=now,
                    running_model_training_pids=[], running_validation_pipeline_pids=[],
                    training_pid=None, pipeline_pid=None,
                    scheduling_note='Recovered frozen v181 completed and archived; original and replacement training PIDs are historical.',
                    v189_result='docs/v190_final_status.json')
    save('docs/v188_recovery_status.json', recovery)
    old_follow = load('docs/v181_followup.json')
    old_follow.update(training_pid=None, pipeline_pid=None,
                      last_observed='Frozen v181 completed, all five tasks below development threshold; archive locally verified.',
                      scheduling_note='Recovery scheduling complete; all previously listed PIDs are historical.')
    save('docs/v181_followup.json', old_follow)
    heartbeat = load('docs/admet_heartbeat_latest.json')
    heartbeat.update(checked_at_utc=now, remote_training_status='V181_AND_V189_COMPLETE_NO_CANDIDATE',
                     running_model_training_pids=[], running_validation_pipeline_pids=[],
                     pipeline_status=dict(v182='COMPLETE_NO_SUBMISSION', v190='COMPLETE_NO_SUBMISSION'),
                     v189_completed_development_tasks=follow['completed_development_tasks'],
                     v189_confirmation_complete=True, v189_archive_verification=archive,
                     new_verified_candidate=False, new_upload_performed=False, pending_platform_upload=False,
                     decision='DONT_NOTIFY',
                     reason='Both model routes complete, neither passed frozen validation; no new file, upload or score.')
    save('docs/admet_heartbeat_latest.json', heartbeat)
    state = load('CURRENT_STATUS.json')
    state.update(active_experiment='v189', active_experiment_status=result['status'],
                 v189_training=follow, v190_result=result, recommended_submission_version=None,
                 new_submission_created=False, running_new_training_processes=0,
                 last_known_running_new_training_processes=0,
                 running_new_training_processes_note='v181/v189 completed and archived; inspect new route state before starting anything.',
                 priority_next_step='Retain official v145=51.166. Completed v181/v189 rejected; preserve gates and use a genuinely new method only.',
                 platform_followup_last_check=heartbeat)
    save('CURRENT_STATUS.json', state)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
