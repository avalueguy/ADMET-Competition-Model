"""Merge disjoint v380/v381 task outputs without fitting, search or rescaling."""
from pathlib import Path
import csv,hashlib,io,json
from datetime import datetime
from zoneinfo import ZoneInfo
from repack_upload_template import field_spans
from verify_submission import verify
ROOT=Path(__file__).resolve().parents[1]
PARTS=('fold0','fold1','fold2','valid')
TASKS=('caco2_wang','ld50_zhu','lipophilicity_astrazeneca','ppbr_az','solubility_aqsoldb',
       'clearance_hepatocyte_az','clearance_microsome_az','half_life_obach','vdss_lombardo')


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(raw):return list(csv.reader(io.StringIO(raw.decode('utf-8-sig'),newline='')))


def main():
    best=read(ROOT/'docs/current_best_after_v374_20261105.json');base=ROOT/best['csv']
    assert sha(base)==best['csv_sha256']
    raw=base.read_bytes();old=table(raw);spans=field_spans(raw);replacements={};sources=[];evidence=[];seen=set()
    for version in ('v380','v381'):
        delivery=read(ROOT/f'docs/{version}_delivery.json')
        assert delivery['status']=='VERIFIED_EXPERIMENTAL_DELIVERY'
        assert delivery['baseline']=='v374_associated' and delivery['all_fitting_remote']
        path=ROOT/delivery['csv'];assert sha(path)==delivery['csv_sha256']
        newraw=path.read_bytes();new=table(newraw);news=field_spans(newraw)
        assert len(old)==len(new)
        chosen=set(delivery['accepted_tasks']);assert not chosen&seen;seen|=chosen
        for i,(a,b) in enumerate(zip(old,new)):
            assert a[:4]+a[5:]==b[:4]+b[5:]
            if i and a[0] in chosen:
                lo,hi=news[i][4];replacements[i]=newraw[lo:hi]
            else:assert a==b
        evidence.extend([r for r in delivery['task_evidence'] if r['accepted']])
        sources.append(dict(version=version,csv=delivery['csv'],sha256=delivery['csv_sha256'],tasks=delivery['accepted_tasks'],fusion=delivery['fusion']))
    assert seen=={'lipophilicity_astrazeneca','ppbr_az'}
    cursor=0;chunks=[]
    for i,value in sorted(replacements.items()):
        lo,hi=spans[i][4];chunks.extend([raw[cursor:lo],value]);cursor=hi
    chunks.append(raw[cursor:]);payload=b''.join(chunks)
    date=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d')
    folder=ROOT/'admet_submissions/v382_regression9_combined_experimental';folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'我从山北来带着兰花草-小分子药物预测技术创新赛-{date}.csv'
    if path.exists():assert path.read_bytes()==payload
    else:path.write_bytes(payload)
    checked=verify(path,ROOT/'admet_official/submission_template.csv')
    final=table(payload);changed={t:0 for t in seen}
    for i,(a,b) in enumerate(zip(old,final)):
        assert a[:4]+a[5:]==b[:4]+b[5:]
        if i not in replacements:assert a==b
        if a!=b:changed[a[0]]+=1
    for source in sources:
        src=table((ROOT/source['csv']).read_bytes())
        for task in source['tasks']:assert [r for r in final if r[0]==task]==[r for r in src if r[0]==task]
    nine={p:sum(next(x['gain'] for x in r['parts'] if x['part']==p) for r in evidence)/9 for p in PARTS}
    five={p:sum(next(x['gain'] for x in r['parts'] if x['part']==p) for r in evidence if r['task'] in TASKS[:5])/5 for p in PARTS}
    result=dict(status='VERIFIED_EXPERIMENTAL_DELIVERY',version='v382',assembly_only=True,new_training=False,
       csv=path.relative_to(ROOT).as_posix(),csv_sha256=sha(path),official_best=51.271,new_official_score=None,
       experimental_only=True,strict_stability_confirmed=False,baseline=best,sources=sources,changed_rows=changed,
       changed_tasks=sorted(seen),unchanged_tasks=20,format=checked,nine_task_mean_relative_MAE_gain_by_part=nine,
       five_scored_task_mean_relative_MAE_gain_by_part=five,goal_mean_5percent_achieved_local=all(g>=.05 for g in nine.values()),
       five_scored_goal_achieved_local=all(g>=.05 for g in five.values()),all_fitting_remote=True,automatic_platform_submission=False,
       limitation='Disjoint experimental updates from repeatedly reused public partitions. Local gains are not official score estimates. Rank task calibrated reference MAEs not original physical-scale errors.',task_evidence=evidence)
    for dest in (path.with_suffix('.verification.json'),ROOT/'docs/v382_delivery.json'):
        dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (folder/'README.md').write_text('# v382 实验提交文件\n\n只合并v380脂溶性及v381 PPBR整任务输出，无新增训练、融合调参或逐样本选择。两个来源均未确认严格稳定性，官方新分未知，9项平均5%目标未达到。其他20项逐行保留51.271关联基线。\n\n请手动提交本目录CSV；verification.json记录来源哈希和格式检查。\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('baseline','sources','format','task_evidence')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
