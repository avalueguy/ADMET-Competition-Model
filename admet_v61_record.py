"""Update score uncertainty and remote experiment state without altering old releases."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
file=ROOT/'admet_current_candidate.json';state=json.loads(file.read_text(encoding='utf-8'))
backup=ROOT/'admet_diagnostics/candidate_before_v61_20261017.json'
if not backup.exists():backup.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
state.update(current_official_score=None,current_score_status='Unknown; user confirms below tenth-place threshold 50.804792',current_score_upper_bound=50.804792,current_rank='outside_top_10',latest_user_goal='Exceed 50.804792 and return to top ten',target_achieved=False,experimental_release='v61',experimental_status='REMOTE_TRAINING',training_status='RUNNING',active_training=True,remote_training_host='czp123@100.109.143.42',local_gpu_training=False)
file.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
score=dict(recorded_date='2026-10-17',timezone='Asia/Hong_Kong',current_official_score=None,current_rank='outside_top_10',current_score_upper_bound=50.804792,source='User reports score below current tenth place 50.804792.',release_prepared='v60',experimental_release='v61',release_official_score=None,training_status='RUNNING',automatically_uploaded=False,local_gpu_training=False,remote_host='czp123@100.109.143.42')
(ROOT/'admet_diagnostics/current_score_status_20261017.json').write_text(json.dumps(score,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status='RECORDED',threshold=50.804792,exact_score=None,remote_experiment='v61')))
