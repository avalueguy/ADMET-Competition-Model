"""Record the user-reported platform anchor; no ML imports or local training."""
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def main():
    path=next((ROOT/'admet_submissions/v24').glob('*.csv'))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest=='d8d6a5f776aeea0f71149702f267894974ade314b87ccfe48925f9719e02aa86'
    score=dict(version='v24',reported_score=50.248759,previous_score=50.077413,
               latest_improvement=0.171346,submitted_sha256=digest,
               source='User report; file association inferred from delivered v24, platform hash unavailable',
               aggregation_formula='unknown',leader_score_last_reported=52.723518)
    (path.parent/'platform_score.json').write_text(json.dumps(score,indent=2),encoding='utf-8')
    hp=ROOT/'admet_diagnostics/platform_score_history.json'
    history=json.loads(hp.read_text(encoding='utf-8'))
    if not any(str(x.get('version'))=='v24' for x in history):history.append(score)
    hp.write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding='utf-8')
    cp=ROOT/'admet_current_candidate.json';cur=json.loads(cp.read_text(encoding='utf-8'))
    cur.update(recommended_release='v24',submission=str(path),sha256=digest,
               platform_anchor_release='v24',platform_anchor_score=50.248759,
               candidate_platform_score=50.248759,observed_display_gain=0.171346,
               target_score=53.723518,latest_leader_score=52.723518,target_achieved=False,
               experimental_release='v25',experimental_status='PROTOCOL_FIXED_NOT_RELEASED',
               awaiting='Remote paired v25 training and verification',
               report='admet_diagnostics/v25_report.md',
               verification='admet_submissions/v24/local_verification.json')
    for key in ['experimental_submission','experimental_sha256','next_version_minimum_target','requested_gain']:
        cur.pop(key,None)
    cp.write_text(json.dumps(cur,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(score,ensure_ascii=False))
if __name__=='__main__':main()
