"""Record new human scorecard, associating it with the preceding v374 delivery."""
from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib

ROOT = Path(__file__).resolve().parents[1]


def main():
    previous = json.loads((ROOT/'docs/platform_scorecard_v371_20261105.json').read_text(encoding='utf-8'))
    delivery = json.loads((ROOT/'docs/v374_delivery.json').read_text(encoding='utf-8'))
    tasks = [dict(r) for r in previous['tasks']]
    for row in tasks:
        if row['task'] == 'solubility_aqsoldb':
            row.update(raw=0.7006168876998713, score=51.314265)
    score = sum(r['score'] for r in tasks)/22
    assert abs(score-51.271)<.000501
    path = ROOT/delivery['csv']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==delivery['csv_sha256']
    card = dict(version='v374_associated', date='2026-11-05', submitted_at='2026/11/05 17:59',
        timezone='Asia/Hong_Kong', recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        score=51.271, task_score_mean=score, tasks=tasks, status='user_reported_official',
        source='Full human scorecard immediately following v374 delivery; version associated, not explicitly named in this feedback.',
        csv=delivery['csv'], submission_sha256=delivery['csv_sha256'], uploaded_byte_hash_confirmed=False,
        previous_score=51.270, changed_tasks=['solubility_aqsoldb'], unchanged_tasks=21,
        provenance_note='Version associated with preceding v374 delivery. Platform byte hash unavailable. Earlier v371 solubility discrepancy preserved in earlier record.')
    dest=ROOT/'docs/platform_scorecard_v374_20261105.json'
    if dest.exists():
        old=json.loads(dest.read_text(encoding='utf-8'));assert old['tasks']==tasks;card=old
    else:dest.write_text(json.dumps(card,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    best=dict(version='v374_associated', official_user_reported=51.271, exact_task_mean=score,
        csv=delivery['csv'], csv_sha256=delivery['csv_sha256'], source=dest.relative_to(ROOT).as_posix(),
        platform_uploaded_byte_hash_confirmed=False, association=card['source'])
    (ROOT/'docs/current_best_after_v374_20261105.json').write_text(json.dumps(best,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    p=ROOT/'CURRENT_STATUS.json';s=json.loads(p.read_text(encoding='utf-8'))
    s.update(official_score=51.271,official_best_version_user_reported='v374_associated',official_best_record='docs/current_best_after_v374_20261105.json',rollback_baseline='v374_local_csv',official_scorecard_provenance_note=card['provenance_note'])
    p.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='RECORDED',official=51.271,exact_task_mean=score,solubility_relative_gain=1-.7006168876998713/.7008642743228115)))


if __name__=='__main__':main()
