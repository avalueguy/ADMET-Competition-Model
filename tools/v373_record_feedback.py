"""Record human v371 scorecard without inventing uploaded-byte identity."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,copy
ROOT=Path(__file__).resolve().parents[1]
def main():
    previous=json.loads((ROOT/'docs/platform_scorecard_v364_20261104.json').read_text(encoding='utf-8'))
    delivery=json.loads((ROOT/'docs/v371_delivery.json').read_text(encoding='utf-8'))
    card=copy.deepcopy(previous);card.update(version='v371',date='2026-11-05',submitted_at='2026/11/05 16:11',score=51.270,
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),source='Full human-provided scorecard, human confirms v371',platform_uploaded_byte_hash_confirmed=False)
    changes={'ld50_zhu':(.5550199723529466,52.074579),'solubility_aqsoldb':(.7008642743228115,51.305445)}
    for row in card['tasks']:
        if row['task'] in changes:row['raw'],row['score']=changes[row['task']]
        if row['metric']=='MAE':assert abs(100*row['reference_score']/(row['raw']+row['reference_score'])-row['score'])<5.1e-7
    card['task_score_mean']=sum(r['score'] for r in card['tasks'])/22
    assert abs(card['task_score_mean']-card['score'])<.000501
    card['provenance_note']='Human confirms v371. Local delivered v371 changes LD50 only; scorecard also changes solubility. Uploaded-byte identity cannot be established; retain discrepancy, do not manufacture a matching CSV.'
    dest=ROOT/'docs/platform_scorecard_v371_20261105.json'
    if dest.exists():
        old=json.loads(dest.read_text(encoding='utf-8'));assert old['tasks']==card['tasks'];card=old
    else:dest.write_text(json.dumps(card,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    p=ROOT/delivery['csv'];assert hashlib.sha256(p.read_bytes()).hexdigest()==delivery['csv_sha256']
    best=dict(version='v371',official_user_reported=51.270,exact_task_mean=card['task_score_mean'],csv=delivery['csv'],csv_sha256=delivery['csv_sha256'],
        source=dest.relative_to(ROOT).as_posix(),previous_best='v364',previous_best_official=51.269,platform_uploaded_byte_hash_confirmed=False,
        solubility_discrepancy_unresolved=True,reason=card['provenance_note'])
    (ROOT/'docs/current_best_after_v371_20261105.json').write_text(json.dumps(best,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    status=ROOT/'CURRENT_STATUS.json';s=json.loads(status.read_text(encoding='utf-8'));s.update(official_score=51.270,official_best_version_user_reported='v371',official_best_record='docs/current_best_after_v371_20261105.json',rollback_baseline='v371_local_csv',official_scorecard_provenance_note=card['provenance_note']);status.write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='RECORDED',official=51.270,task_mean=card['task_score_mean'],ld50_relative_reduction=1-changes['ld50_zhu'][0]/.55547536869963,solubility_relative_reduction=1-changes['solubility_aqsoldb'][0]/.7010230434713527,provenance_discrepancy=True)))
if __name__=='__main__':main()
