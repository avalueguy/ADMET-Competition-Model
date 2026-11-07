"""Persist collected scheduling state without interpreting partial scores."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    remote=json.loads((ROOT/'docs/v377_v378_remote_progress.json').read_text(encoding='utf-8'))
    for version in (377,378):
        snapshot=remote['experiments'][str(version)]
        path=ROOT/f'docs/v{version}_progress.json'
        value=json.loads(path.read_text(encoding='utf-8'))
        value.update(remote_snapshot=snapshot,remote_snapshot_at_unix=remote['at_unix'])
        value['status']=snapshot.get('run_status',{}).get('stage',value['status'])
        if 'launch' in snapshot:value['remote_pid']=snapshot['launch']['pid']
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    path=ROOT/'CURRENT_STATUS.json'
    value=json.loads(path.read_text(encoding='utf-8'))
    value.update(latest_aggregate_plan='v377_v378',aggregate_status='REGRESSION9_TRAINING',
                 current_experiment_progress='docs/v378_progress.json',
                 remote_training_running=True,official_score=51.271,
                 latest_regression9_remote_progress=remote)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='RECORDED',remote_at_unix=remote['at_unix'])))


if __name__=='__main__':main()
