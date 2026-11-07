"""Parse the two user scorecards and bind them to explicitly named versions."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,re,csv,io,shutil
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('C:/Users/lenovo/.codex/attachments/5481bac4-f936-4f9f-b203-378d2f388a95/已粘贴的文本.txt')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def table(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))
def main():
    text=SOURCE.read_text(encoding='utf-8-sig');objects=[json.loads(s)for s in re.findall(r'\{[^{}]*"metric"[^{}]*\}',text)]
    assert len(objects)==44
    prior=read(ROOT/'docs/platform_scorecard_v385_20261106.json');records={}
    for index,(version,total)in enumerate((('v386',51.308),('v387',51.314))):
        delivery=read(ROOT/f'docs/{version}_delivery.json');assert sha(ROOT/delivery['csv'])==delivery['csv_sha256']
        tasks=[]
        for old,row in zip(prior['tasks'],objects[index*22:(index+1)*22]):
            assert row['metric']==old['metric'] and row['reference_score']==old['reference_score']
            tasks.append(dict(task=old['task'],raw=row.pop('raw_metric'),**row))
        mean=sum(r['score']for r in tasks)/22;assert abs(mean-total)<.00051
        value=dict(version=version,date='2026-11-06',score=total,task_score_mean=mean,tasks=tasks,
            source='User explicitly names v386=51.308 and v387=51.314 with ordered full22 scorecards in attachment.',
            attachment_sha256=sha(SOURCE),csv=delivery['csv'],submission_sha256=delivery['csv_sha256'],
            uploaded_byte_hash_confirmed=False,recorded_at_utc=datetime.now(timezone.utc).isoformat(),
            note='First supplied scorecard timestamp is12:15 and exactly matches prior v385 card; accepted as user attribution to v386, no platform byte-hash verification.')
        save(ROOT/f'docs/platform_scorecard_{version}_20261106.json',value);records[version]=value
    a,b=records['v386'],records['v387'];diff=[]
    for old,new in zip(a['tasks'],b['tasks']):
        if old!=new:
            diff.append(dict(task=old['task'],metric=old['metric'],old_raw=old['raw'],new_raw=new['raw'],
                relative_MAE_reduction=(old['raw']-new['raw'])/old['raw'],task_score_gain=new['score']-old['score']))
    assert [r['task']for r in diff]==['lipophilicity_astrazeneca']
    ta,tb=table(ROOT/a['csv']),table(ROOT/b['csv']);changed={}
    for x,y in zip(ta,tb):
        assert x[:4]+x[5:]==y[:4]+y[5:]
        if x!=y:changed[x[0]]=changed.get(x[0],0)+1
    assert changed=={'lipophilicity_astrazeneca':840}
    comparison=dict(v386_score=51.308,v387_score=51.314,task_differences=diff,csv_changed_rows=changed,
        exact_mean_gain=b['task_score_mean']-a['task_score_mean'],
        whole_task_best_merge_equals='v387',cannot_add_gains_on_same_task=True,
        v386_card_identical_to_previous_v385=all(x==y for x,y in zip(a['tasks'],prior['tasks'])),
        no_test_labels_available=True)
    save(ROOT/'docs/v389_scorecard_comparison.json',comparison)
    best=dict(version='v387',official_user_reported=51.314,exact_task_mean=b['task_score_mean'],csv=b['csv'],csv_sha256=b['submission_sha256'],
        source='docs/platform_scorecard_v387_20261106.json',platform_uploaded_byte_hash_confirmed=False,
        association='User explicitly confirms v387 version; full task scorecard supplied.')
    save(ROOT/'docs/current_best_after_v387_20261106.json',best)
    state=read(ROOT/'CURRENT_STATUS.json');state.update(official_score=51.314,current_best='docs/current_best_after_v387_20261106.json',
        current_official_feedback='docs/platform_scorecard_v387_20261106.json',latest_feedback_version='v387',current_training_baseline='v387:51.314')
    for v in('v386','v387'):state['releases'][v[1:]].update(official_score_user_reported=records[v]['score'],official_score=records[v]['score'],status='USER_REPORTED_OFFICIAL_BEST'if v=='v387'else'USER_REPORTED_OFFICIAL')
    save(ROOT/'CURRENT_STATUS.json',state)
    shutil.copyfile(SOURCE,ROOT/'docs/v386_v387_user_scorecards_20261106.txt')
    print(json.dumps(comparison,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
