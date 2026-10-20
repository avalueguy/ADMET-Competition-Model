"""Local bookkeeping and lightweight evidence transfer; no local training."""
from pathlib import Path
import hashlib,json,tarfile

ROOT=Path(__file__).resolve().parent
def main():
    path=ROOT/'admet_submissions/v23/v23_upload.csv'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest=='b2dc3eacc5c621f007fbacde47b55f751d87509e4721e76d47b686dde36004e1'
    score=dict(version='v23',reported_score=50.077413,previous_score=50.019727,
               latest_improvement=0.057686,submitted_sha256=digest,
               source='User report; association inferred from prior delivery',
               aggregation_formula='unknown',leader_score_last_reported=52.723518)
    (path.parent/'platform_score.json').write_text(json.dumps(score,indent=2),encoding='utf-8')
    hp=ROOT/'admet_diagnostics/platform_score_history.json'
    history=json.loads(hp.read_text(encoding='utf-8'))
    if not any(x.get('version')=='v23' for x in history):history.append(score)
    hp.write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding='utf-8')
    cp=ROOT/'admet_current_candidate.json'; cur=json.loads(cp.read_text(encoding='utf-8'))
    cur.update(recommended_release='v23',submission=str(path),sha256=digest,
               platform_anchor_release='v23',platform_anchor_score=50.077413,
               candidate_platform_score=50.077413,observed_display_gain=0.057686,
               target_score=53.723518,latest_leader_score=52.723518,
               target_achieved=False,experimental_release='v24',experimental_status='TRAINING_NOT_RELEASED',
               awaiting='Remote v24 audit and group-disjoint model comparison')
    for k in ['experimental_submission','experimental_sha256']:cur.pop(k,None)
    cp.write_text(json.dumps(cur,ensure_ascii=False,indent=2),encoding='utf-8')
    files=[]
    for version in [16,19]:
        for name in ['validation_comparison.csv','selection.json']:
            files+=list((ROOT/f'admet_model_outputs_v{version}').glob('*/'+name))
    special=ROOT/'admet_model_outputs_v20_rare/cyp2d6_substrate_carbonmangels'
    files += [special/'decision.json',special/'rare_valid.csv']
    dest=ROOT/'admet_v24_evidence.tar.gz'
    with tarfile.open(dest,'w:gz') as tar:
        for p in files:tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    print(json.dumps(dict(score=score,evidence_files=len(files),archive_bytes=dest.stat().st_size)))
if __name__=='__main__':main()
