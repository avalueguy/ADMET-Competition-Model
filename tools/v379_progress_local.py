"""Persist a verified remote progress snapshot without fitting or selection."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    remote=json.loads((ROOT/'experiments/regression9_v379/progress.json').read_text())
    source=json.loads((ROOT/'experiments/regression9_v379/source_freeze.json').read_text())
    assert remote['run_status']['signature']=='f8c8b90e4a93b1032b13e66c604daeece93892199a28c3761fb9269c119888dd'
    value=dict(status=remote['run_status']['stage'],official_best=51.271,remote_snapshot=remote,
               all_fitting_remote=True,source_files=source['files'],goal_achieved=False,
               current_plan='docs/V379_REGULARIZED_DISTRIBUTION_EXPERIMENT.md',
               no_new_official_evaluation=True)
    (ROOT/'docs/v379_progress.json').write_text(json.dumps(value,indent=2)+'\n')
    state=json.loads((ROOT/'CURRENT_STATUS.json').read_text(encoding='utf-8'))
    state.update(latest_aggregate_plan='v377_v378_v379',aggregate_status='REGRESSION9_V379_'+value['status'],
                 current_experiment_progress='docs/v379_progress.json',official_score=51.271,
                 remote_training_running=value['status'] not in ('EVIDENCE_READY','FAILED'))
    (ROOT/'CURRENT_STATUS.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=value['status'],completed=remote['completed_development_curves'],total=36)))


if __name__=='__main__':main()
