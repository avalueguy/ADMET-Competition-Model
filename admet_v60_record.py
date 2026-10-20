"""Record verified artifact identity and the user's latest score bound, without guessing score."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
meta=json.loads((ROOT/'admet_submissions/v60/local_verification.json').read_text(encoding='utf-8'))
remote=json.loads((ROOT/'admet_model_outputs_v60/model_replay.json').read_text(encoding='utf-8'))
assert meta['status']==remote['status']=='PASS'
file=next((ROOT/'admet_submissions/v60').glob('*.csv'))
statefile=ROOT/'admet_current_candidate.json';previous=json.loads(statefile.read_text(encoding='utf-8'))
backup=ROOT/'admet_diagnostics/candidate_before_v60_20261017.json'
if not backup.exists():backup.write_text(json.dumps(previous,ensure_ascii=False,indent=2),encoding='utf-8')
state=dict(recommended_release='v60',submission=str(file.resolve()),sha256=meta['sha256'],
    experimental_release='v60',experimental_status='READY_FOR_PLATFORM_EVALUATION',
    ready_for_platform_evaluation=True,automatically_uploaded=False,
    current_official_score=None,current_score_status='Unknown; user confirms current score below 50.755035',
    current_score_upper_bound=50.755035,current_rank='outside_top_10',
    latest_user_goal='Exceed 50.755035 and return to top ten',target_achieved=False,
    base_release='v57',base_sha256='e5aef127f9d093860e2e77aad67e559bca320a34e82e8272ea066aa86354d31b',
    remote_training_host='czp123@100.109.143.42',remote_gpu='RTX 4090',local_gpu_training=False,
    training_status='COMPLETE',active_training=False,rows=meta['rows'],columns=meta['columns'],
    encoding=meta['encoding'],line_endings=meta['line_endings'],changed_tasks=meta['changed_tasks'],
    changed_rows=meta['changed_rows'],validation=dict(local_format_and_fusion='PASS',remote_final_model_replay='PASS',
    official_score_available=False,full_ensemble_independent_oof=False),
    report=str((ROOT/'admet_diagnostics/v60_report.md').resolve()),
    fallback=str(next((ROOT/'admet_submissions/v57').glob('*.csv')).resolve()))
statefile.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
score=dict(recorded_date='2026-10-17',timezone='Asia/Hong_Kong',current_official_score=None,
    current_rank='outside_top_10',current_score_upper_bound=50.755035,
    source='User reports score remains unknown and below the current tenth-place score 50.755035.',
    release_prepared='v60',release_official_score=None,training_status='COMPLETE',automatically_uploaded=False,
    local_gpu_training=False,remote_host='czp123@100.109.143.42')
(ROOT/'admet_diagnostics/current_score_status_20261017.json').write_text(json.dumps(score,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status='RECORDED',release='v60',sha256=meta['sha256'],official_score=None,threshold=50.755035)))
