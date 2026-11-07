"""Record the scorecard associated with the immediately preceding v385 delivery."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 delivery=read(ROOT/'docs/v385_delivery.json');path=ROOT/delivery['csv'];assert hashlib.sha256(path.read_bytes()).hexdigest()==delivery['csv_sha256']
 old=read(ROOT/'docs/platform_scorecard_v382_20261105.json');new=json.loads(json.dumps(old))
 for r in new['tasks']:
  if r['task']=='ppbr_az':r.update(raw=6.442906324588329,score=53.222526)
 new.update(version='v385_associated',date='2026-11-06',submitted_at='2026/11/06 12:15',score=51.308,
  task_score_mean=sum(r['score']for r in new['tasks'])/22,recorded_at_utc=datetime.now(timezone.utc).isoformat(),
  source='User full22 scorecard immediately following v385 delivery; only PPBR changed exactly as expected. Version associated, not explicitly named in this feedback.',
  csv=delivery['csv'],submission_sha256=delivery['csv_sha256'],uploaded_byte_hash_confirmed=False,
  previous_score=51.299,changed_tasks=['ppbr_az'],unchanged_tasks=21,
  provenance_note='Local v385 artifact and expected changed-task set verified. Version inferred from conversation; platform file bytes unavailable.')
 save(ROOT/'docs/platform_scorecard_v385_20261106.json',new)
 best=dict(version='v385_associated',official_user_reported=51.308,exact_task_mean=new['task_score_mean'],csv=delivery['csv'],csv_sha256=delivery['csv_sha256'],
  source='docs/platform_scorecard_v385_20261106.json',platform_uploaded_byte_hash_confirmed=False,association=new['source'])
 save(ROOT/'docs/current_best_after_v385_20261106.json',best)
 state=read(ROOT/'CURRENT_STATUS.json');state.update(official_score=51.308,current_training_baseline='v385_associated:51.308',
  current_best='docs/current_best_after_v385_20261106.json',current_official_feedback='docs/platform_scorecard_v385_20261106.json',target_achieved=False)
 state['releases']['385'].update(official_score_user_reported=51.308,official_score=51.308,status='USER_REPORTED_OFFICIAL_BEST_ASSOCIATED')
 save(ROOT/'CURRENT_STATUS.json',state)
 print(json.dumps(dict(score=51.308,exact_mean=new['task_score_mean'],ppbr_relative_MAE_gain=(6.495122547350538-6.442906324588329)/6.495122547350538)))
if __name__=='__main__':main()
