"""Local status arithmetic, no model fitting or scoring."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    remote=json.loads((ROOT/'experiments/regression9_v380/progress.json').read_text())
    smoke=json.loads((ROOT/'experiments/regression9_v380/smoke_receipt.json').read_text())
    assert smoke['signature']=='1fad079687840ca420ee8663000c3bffedd29e2c82857acbee9c662deaab8452'
    status=remote['run_status']['stage']
    value=dict(status=status,official_best=51.271,remote_snapshot=remote,all_fitting_remote=True,
               goal_achieved=False,current_plan='docs/V380_HISTOGRAM_MEDIAN_PLAN.md',no_new_official_evaluation=True)
    (ROOT/'docs/v380_progress.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    path=ROOT/'CURRENT_STATUS.json'
    state=json.loads(path.read_text(encoding='utf-8'))
    state.update(latest_aggregate_plan='v377_v378_v379_v380',aggregate_status='REGRESSION9_V380_'+status,
                 current_experiment_progress='docs/v380_progress.json',official_score=51.271,
                 remote_training_running=status not in ('EVIDENCE_READY','FAILED'))
    path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=status,completed=remote['completed_development_fits'],total=104)))


if __name__=='__main__':main()
