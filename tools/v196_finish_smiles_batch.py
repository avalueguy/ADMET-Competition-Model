"""Record a fully archived rejected sequence experiment without exporting CSV."""
import hashlib
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
    archive = load('docs/v194_local_archive_verification.json')
    conf = load('admet_model_outputs_v193/confirmation.json')
    dev = load('admet_model_outputs_v193/development.json')
    no = load('admet_model_outputs_v194/no_candidate.json')
    frozen = load('admet_model_outputs_v193/frozen_selection.json')
    plan = load('docs/v193_training_plan.json')
    assert archive['status'] == 'PASS' and archive['all_member_hashes_verified'] and archive['dependency_hashes_verified']
    assert dev['status'] == conf['status'] == 'COMPLETE' and not conf['accepted']
    assert conf['signature'] == frozen['signature'] == dev['signature']
    assert not no['submission_created']
    assert no['confirmation_sha256'] == hashlib.sha256((ROOT / 'admet_model_outputs_v193/confirmation.json').read_bytes()).hexdigest()
    assert plan['source_sha256'] == hashlib.sha256((ROOT / 'tools/v193_smiles_convolution.py').read_bytes()).hexdigest()
    assert not (ROOT / 'admet_model_outputs_v194/evidence.json').exists()
    assert not list((ROOT / 'admet_submissions/v194_smiles_convolution').glob('*.csv'))
    result = dict(status='COMPLETE_NO_CONFIRMED_NEW_MODEL', checked_at_utc=now,
                  training_version='v193', delivery_version='v194', baseline='v145', official_best=51.166,
                  target_score=52, target_achieved=False, development_tasks=21,
                  nominated_tasks=[r['task'] for r in conf['records']], accepted_tasks=[],
                  confirmation_results=[dict(task=r['task'], metric=r['metric'],
                      gains={p['part']: p['gain'] for p in r['confirmation']},
                      bootstrap90={p['part']: p['gain_ci90'] for p in r['bootstrap']}) for r in conf['records']],
                  archive=archive, new_csv_created=False, new_upload_performed=False, new_official_score=None,
                  frozen_gates_unchanged=True, no_confirmation_fallback=True, rejected_route_restarted=False)
    save('docs/v194_final_status.json', result)
    f = load('docs/v193_followup.json')
    f.update(status=result['status'], checked_at_utc=now, training_pid=None, pipeline_pid=None,
             accepted_tasks=[], completed_development_tasks=list(dict.fromkeys(r['task'] for r in dev['records'])),
             final_status='docs/v194_final_status.json',
             last_observed='All21 development tasks and two frozen confirmations complete; both rejected. Archive251 members and dependencies verified locally; no CSV/upload.',
             next_steps=['Do not restart rejected v193 or upload old predictions.',
                         'Continue only genuinely different model methods under unchanged frozen gates.'])
    save('docs/v193_followup.json', f)
    h = load('docs/admet_heartbeat_latest.json')
    h.update(checked_at_utc=now, remote_training_status='V193_COMPLETE_NO_CANDIDATE',
             running_model_training_pids=[], running_validation_pipeline_pids=[],
             v193_completed_development_tasks=f['completed_development_tasks'], v193_confirmation_complete=True,
             v193_archive_verification=archive, new_verified_candidate=False, new_upload_performed=False,
             pending_platform_upload=False, decision='DONT_NOTIFY',
             reason='Sequence model completed21 endpoints; both frozen nominees regressed on validation. Archive verified, no CSV/upload or new official score.')
    h['pipeline_status']['v194'] = 'COMPLETE_NO_SUBMISSION'
    save('docs/admet_heartbeat_latest.json', h)
    s = load('CURRENT_STATUS.json')
    s.update(v193_training=f, v194_result=result, active_experiment='v193', active_experiment_status=result['status'],
             running_new_training_processes=0, last_known_running_new_training_processes=0,
             recommended_submission_version=None, new_submission_created=False, platform_followup_last_check=h,
             running_new_training_processes_note='v1935754/v1955755 exited normally; no training active.',
             priority_next_step='Retain v14551.166; v193 completed and rejected, do not rerun. Preserve frozen gates for a genuinely new method.')
    save('CURRENT_STATUS.json', s)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
