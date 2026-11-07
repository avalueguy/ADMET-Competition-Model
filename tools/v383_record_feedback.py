"""Record the user's v382 scorecard and make a separate, reversible task rollback."""
from pathlib import Path
import csv, hashlib, io, json
from datetime import datetime, timezone
from repack_upload_template import field_spans
from verify_submission import verify
ROOT=Path(__file__).resolve().parents[1]
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(raw): return list(csv.reader(io.StringIO(raw.decode('utf-8-sig'),newline='')))
def main():
    delivery=read(ROOT/'docs/v382_delivery.json'); base=ROOT/delivery['csv']
    assert sha(base)==delivery['csv_sha256']
    old=read(ROOT/'docs/platform_scorecard_v374_20261105.json'); new=json.loads(json.dumps(old))
    changes={'lipophilicity_astrazeneca':(.4173184854676313,53.121527),'ppbr_az':(6.495122547350538,53.021518)}
    for r in new['tasks']:
        if r['task'] in changes: r['raw'],r['score']=changes[r['task']]
    new.update(version='v382',submitted_at='2026/11/05 22:47',score=51.299,
        recorded_at_utc=datetime.now(timezone.utc).isoformat(),task_score_mean=sum(r['score'] for r in new['tasks'])/22,
        source='User explicitly identifies v382 and provides complete 22-task official scorecard.',
        csv=delivery['csv'],submission_sha256=sha(base),uploaded_byte_hash_confirmed=False,
        previous_score=51.271,changed_tasks=list(changes),unchanged_tasks=20,
        provenance_note='Version explicitly named by user. Local artifact hash checked; platform upload bytes unavailable.')
    save(ROOT/'docs/platform_scorecard_v382_20261105.json',new)
    best=dict(version='v382',official_user_reported=51.299,exact_task_mean=new['task_score_mean'],
        csv=delivery['csv'],csv_sha256=sha(base),source='docs/platform_scorecard_v382_20261105.json',
        platform_uploaded_byte_hash_confirmed=False,association=new['source'])
    save(ROOT/'docs/current_best_after_v382_20261105.json',best)
    raw=base.read_bytes(); tab=rows(raw); spans=field_spans(raw)
    prior=ROOT/old['csv']; priorraw=prior.read_bytes(); priorrows=rows(priorraw); priorspans=field_spans(priorraw)
    chunks=[];cursor=0; changed=0
    for i,(a,b) in enumerate(zip(tab,priorrows)):
        assert a[:4]+a[5:]==b[:4]+b[5:]
        if i and a[0]=='lipophilicity_astrazeneca':
            lo,hi=spans[i][4];pl,ph=priorspans[i][4]
            chunks.extend([raw[cursor:lo],priorraw[pl:ph]]); cursor=hi;changed+=a!=b
    chunks.append(raw[cursor:]);payload=b''.join(chunks)
    folder=ROOT/'admet_submissions/v383_v382_lipophilicity_rollback';folder.mkdir(exist_ok=True)
    path=folder/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261106.csv'
    if path.exists(): assert path.read_bytes()==payload
    else: path.write_bytes(payload)
    # This is exactly the existing PPBR-only candidate, apart from its name.
    source=ROOT/'admet_submissions/v381_regression9_experimental/我从山北来带着兰花草-小分子药物预测技术创新赛-20261105.csv'
    assert payload==source.read_bytes()
    check=verify(path,ROOT/'admet_official/submission_template.csv')
    receipt=dict(status='VERIFIED_EXPERIMENTAL_DELIVERY',version='v383',baseline=best,csv=path.relative_to(ROOT).as_posix(),
        csv_sha256=sha(path),changed_rows={'lipophilicity_astrazeneca':changed},unchanged_tasks=21,new_training=False,
        official_score=None,conditional_task_independent_arithmetic=new['task_score_mean']+(53.148026-53.121527)/22,
        conditional_arithmetic_is_official_score=False,format=check,
        note='Restores the previously scored lipophilicity task. No evidence of new training gain. Subsequent experts compare to full v382, not this rollback.')
    save(ROOT/'docs/v383_delivery.json',receipt);save(path.with_suffix('.verification.json'),receipt)
    state=read(ROOT/'CURRENT_STATUS.json');state.update(official_score=51.299,rollback_baseline='v382_local_csv',
        current_best='docs/current_best_after_v382_20261105.json',current_official_feedback='docs/platform_scorecard_v382_20261105.json',
        latest_feedback_version='v382',target_achieved=False)
    state['releases'].setdefault('382',delivery.copy()).update(official_score=51.299,official_score_user_reported=51.299,status='USER_REPORTED_OFFICIAL_BEST')
    state['releases']['383']=receipt
    save(ROOT/'CURRENT_STATUS.json',state)
    print(json.dumps(dict(score=51.299,exact_mean=new['task_score_mean'],rollback_csv=receipt['csv'],conditional_mean=receipt['conditional_task_independent_arithmetic'])))
if __name__=='__main__':main()
