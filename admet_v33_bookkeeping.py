"""Record the human-reported platform result without inferring task scores."""
import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
dest=ROOT/'admet_submissions/v32';csv=next(dest.glob('*.csv'))
digest=hashlib.sha256(csv.read_bytes()).hexdigest()
assert digest=='179f39d79eabe7ebc7d4fda8d3abe45fa56a2a0e1a1e45bc58b07c12945f28ac'
record=dict(version='v32',reported_score=50.399027,previous_score=50.386293,
    observed_display_gain=0.012734,submitted_sha256=digest,reported_on='2026-10-06',
    source='User report after v32 delivery; platform file hash and display semantics not independently exposed',
    leader_score_last_reported=52.723518,aggregation_formula='unknown',official_per_task_scores=None)
write(dest/'platform_score.json',record)
p=ROOT/'admet_diagnostics/platform_score_history.json';history=json.loads(p.read_text(encoding='utf-8'))
if not any(r.get('version')=='v32' for r in history):history.append(record)
write(p,history);write(ROOT/'admet_diagnostics/platform_score.json',record)
p=ROOT/'admet_current_candidate.json';state=json.loads(p.read_text(encoding='utf-8'))
state.update(recommended_release='v32',submission=str(csv),sha256=digest,platform_anchor_release='v32',
    platform_anchor_score=50.399027,latest_displayed_score=50.399027,observed_display_gain=.012734,
    candidate_platform_score=50.399027,awaiting='Remote v33 neighborhood-transfer experiments',
    experimental_release='v33',experimental_status='TRAINING_NOT_YET_STARTED',ready_for_platform_evaluation=False,
    target_achieved=False)
for key in ['experimental_submission','experimental_sha256']:state.pop(key,None)
write(p,state)
print(json.dumps(record,ensure_ascii=False))
