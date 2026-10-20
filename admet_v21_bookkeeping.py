"""Record the user's reported v20 platform score before the next experiment."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
p=next((ROOT/'admet_submissions/v20').glob('*.csv'))
h=hashlib.sha256(p.read_bytes()).hexdigest()
assert h=='7c4001cf13f7cd98f714575a1dce12a827ac7c9800cb65ec163c4f5826587789'
record=ROOT/'admet_current_candidate.json'
d=json.loads(record.read_text(encoding='utf-8'))
d.update(recommended_release='v20',submission=str(p.resolve()),sha256=h,
         platform_anchor_release='v20',platform_anchor_score=50.019727,
         candidate_platform_score=50.019727,score_source='user report',
         awaiting='v21 representation fine-tuning experiments',target_achieved=False,
         next_version_minimum_target=50.619727)
record.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
result=dict(version='v20',score=50.019727,previous_version='v19',previous_score=49.993751,
            gain=round(50.019727-49.993751,6),submission_sha256=h,source='user report')
(ROOT/'admet_submissions/v20/platform_score.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
