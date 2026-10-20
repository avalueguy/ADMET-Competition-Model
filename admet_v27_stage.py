"""Lightweight platform feedback bookkeeping without local tensor imports."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    old=next((ROOT/'admet_submissions/v26').glob('*.csv'))
    rec=dict(version='v26',reported_score=50.248759,previous_score=50.248759,observed_display_gain=0.,
       source='User reports score unchanged and rank sixth on 2026-10-06; version linkage inferred from last v26 delivery',
       submitted_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),leader_score_user_reported=52.723518,
       ranking_user_reported=6,platform_display_semantics='Unverified whether latest score or historical best',aggregation_formula='unknown')
    (old.parent/'platform_score.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2),encoding='utf-8')
    hp=ROOT/'admet_diagnostics/platform_score_history.json';h=json.loads(hp.read_text(encoding='utf-8'))
    if not any(r.get('version')=='v26' for r in h):h.append(rec)
    hp.write_text(json.dumps(h,ensure_ascii=False,indent=2),encoding='utf-8')
    cp=ROOT/'admet_current_candidate.json';c=json.loads(cp.read_text(encoding='utf-8'))
    c.update(experimental_release='v27',experimental_status='REMOTE_GRAPH_TRAINING',latest_leader_score=52.723518,
       ranking_user_reported=6,latest_displayed_score=50.248759,awaiting='Remote v27 directed graph evaluation',
       report='admet_diagnostics/v27_report.md',target_score=53.723518,target_achieved=False)
    c.pop('experimental_submission',None);c.pop('experimental_sha256',None)
    cp.write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(rec,ensure_ascii=False))
if __name__=='__main__':main()
