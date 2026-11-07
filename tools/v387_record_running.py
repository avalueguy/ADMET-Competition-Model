"""Update a compact running-state pointer; no fitting or frozen-source edits."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def main():
    state=json.loads((ROOT/'CURRENT_STATUS.json').read_text(encoding='utf-8'))
    launch=json.loads((ROOT/'experiments/regression9_v387/launch.json').read_text())
    smoke=json.loads((ROOT/'experiments/regression9_v387/smoke_receipt.json').read_text())
    assert launch['status']=='STARTED'and smoke['status']=='PASS'
    state.update(aggregate_status='V386_COMPLETE_V387_RUNNING',current_experiment_progress='docs/V387_CORRELATED_METRIC_PLAN.md',
        remote_training_running=True,latest_completed_model_branch='v386',latest_completed_model_branch_is_proven_best=False)
    state['releases']['386']=json.loads((ROOT/'docs/v386_delivery.json').read_text(encoding='utf-8'))
    (ROOT/'CURRENT_STATUS.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(stage=state['aggregate_status'],remote_pid=launch['pid'])))
if __name__=='__main__':main()
