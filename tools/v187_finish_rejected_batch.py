"""Finalize only an archived batch that produced no accepted new task."""
import json
from pathlib import Path
from finalize_v124_local import ROOT, load, save


def main():
    archive = load(ROOT / 'docs/v182_local_archive_verification.json')
    assert archive['status'] == 'PASS'
    assert archive['all_member_hashes_verified'] and archive['dependency_hashes_verified']
    cp = ROOT / 'admet_model_outputs_v181/confirmation.json'
    confirmation = load(cp)
    assert confirmation['status'] == 'COMPLETE' and confirmation['accepted'] == []
    no = load(ROOT / 'admet_model_outputs_v182/no_candidate.json')
    assert not no['submission_created']
    assert not (ROOT / 'admet_model_outputs_v182/evidence.json').exists()
    assert not list((ROOT / 'admet_submissions/v182_quantile_forest').glob('*.csv'))
    scheduling = load(ROOT / 'admet_model_outputs_v181/scheduling.json')
    assert scheduling['status'] == 'PRECOMPUTE_COMPLETE'
    assert not any(scheduling['worker_exit_codes'].values())
    status = dict(status='COMPLETE_NO_NEW_SUBMISSION', baseline_version='v145',
                  baseline_official_score=51.166, target_score=52, target_achieved=False,
                  completed_new_routes=['v175', 'v178', 'v181'], accepted_tasks=[],
                  new_csv_created=False, new_upload_performed=False, new_official_score=None,
                  archive_verifications=[load(ROOT / 'docs' / (v + '_local_archive_verification.json'))
                                         for v in ('v176', 'v179', 'v182')],
                  objective_audit='docs/v184_official_objective_audit.json',
                  no_failed_fallback=True, frozen_gates_unchanged=True,
                  source_mapping_correction='v174 rejected; previous51.188 estimate withdrawn',
                  running_new_training_processes=0)
    save(ROOT / 'docs/v182_final_status.json', status)
    state = load(ROOT / 'CURRENT_STATUS.json')
    other = state.get('v189_training', {})
    other_running = other.get('status') in (
        'TRAINING_AND_PIPELINE_STARTED', 'TRAINING_RUNNING', 'DEVELOPMENT_RUNNING',
        'CONFIRMATION_RUNNING', 'WAITING_TRAINING')
    if other_running:
        state.update(active_experiment='v189', active_experiment_status=other['status'])
    else:
        state.update(active_experiment='v181', active_experiment_status=status['status'],
                     recommended_submission_version=None, new_submission_created=False,
                     running_new_training_processes=0)
    state['v181_training'].update(status='COMPLETE_NO_CONFIRMED_NEW_MODEL', accepted_tasks=[])
    state['v182_result'] = status
    state['v175_v181_batch_result'] = status
    save(ROOT / 'CURRENT_STATUS.json', state)
    followup = load(ROOT / 'docs/v181_followup.json')
    followup.update(status=status['status'], accepted_tasks=[], submission_created=False,
                    running_training_processes=0, final_status='docs/v182_final_status.json',
                    next_steps=['Only inspect genuinely new verified candidates or pending platform scores',
                                'Do not rerun completed rejected routes or upload the old baseline as a new version'])
    save(ROOT / 'docs/v181_followup.json', followup)
    print(json.dumps({k: v for k, v in status.items() if k != 'archive_verifications'}))


if __name__ == '__main__':
    main()
