"""Record user-reported score without altering historical model evidence."""
import json
from pathlib import Path
from decimal import Decimal

score=Path('admet_diagnostics/platform_score.json')
old=json.loads(score.read_text(encoding='utf8'))
history=Path('admet_diagnostics/platform_score_history.json')
entries=json.loads(history.read_text(encoding='utf8')) if history.exists() else [old]
sha='8a677b09bbc8f263ef9ae27055cd2cc88de56a0ed7c7991de67a5256cdb628a4'
new=dict(old,reported_score=49.247501,previous_score=48.98128,latest_improvement=float(Decimal('49.247501')-Decimal('48.98128')),
         gap_to_target=float(Decimal('51.584929')-Decimal('49.247501')),submitted_sha256=sha,version=6,
         source='user reported platform result; linked to v6 submission')
if not any(e.get('submitted_sha256')==sha and e.get('reported_score')==49.247501 for e in entries):entries.append(new)
for path,obj in [(score,new),(history,entries)]:path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
run=Path('admet_v6_remote_run.json');state=json.loads(run.read_text(encoding='utf8'))
state.update(status='submitted_platform_score_user_reported',latest_platform_score=49.247501)
files=list(Path('admet_submissions/v6').glob('*.csv'))
if len(files)==1:
    import hashlib
    assert hashlib.sha256(files[0].read_bytes()).hexdigest()==sha
    state['submission']=str(files[0])
run.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(new,ensure_ascii=False))
