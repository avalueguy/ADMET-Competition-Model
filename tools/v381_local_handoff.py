"""Synchronize observed remote status without fitting or touching frozen code."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]


def main():
    exp=ROOT/'experiments/regression9_v381'
    remote=json.loads((exp/'progress.json').read_text())
    smoke=json.loads((exp/'smoke_receipt.json').read_text())
    assert smoke['signature']=='6d69d42a1795b6086f426b887d088ad06df48f959c835a9fd21d731bd667cc40'
    status=remote['run_status']['stage']
    value=dict(status=status,official_best=51.271,remote_snapshot=remote,all_fitting_remote=True,
               goal_achieved=False,current_plan='docs/V381_GROUPED_OOF_PARETO_PLAN.md',no_new_official_evaluation=True)
    (ROOT/'docs/v381_progress.json').write_text(json.dumps(value,indent=2)+'\n')
    path=ROOT/'CURRENT_STATUS.json';state=json.loads(path.read_text(encoding='utf-8'))
    state.update(latest_aggregate_plan='v377_v378_v379_v380_v381',aggregate_status='REGRESSION9_V381_'+status,
                 current_experiment_progress='docs/v381_progress.json',official_score=51.271,
                 remote_training_running=status not in ('EVIDENCE_READY','FAILED'))
    path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=status,completed_base_models=remote['completed_development_base_models'],total=216)))


if __name__=='__main__':main()
