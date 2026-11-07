"""Archive v144 official feedback and route whole task outputs to v145."""
import csv,hashlib,io,json,shutil,sys,tarfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from repack_upload_template import field_spans
from verify_submission import verify
PREFIX='v145_feedback_repair_20261027'
CHANGES={
 'ames':(0.8769096712291213,50.141941),
 'clearance_hepatocyte_az':(0.5560507844295146,50.064522),
 'cyp2c9_veith':(0.8349408577894758,51.364323),
 'cyp2d6_veith':(0.7439447168745902,50.846574),
 'cyp3a4_substrate_carbonmangels':(0.7075045207956601,49.380756)}
KEEP=('ames','cyp2c9_veith','cyp2d6_veith')

def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return list(csv.reader(io.StringIO(p.read_bytes().decode('utf-8-sig'),newline='')))

def main():
    base=load(ROOT/'docs/platform_scorecard_v130_20261027.json');feedback=json.loads(json.dumps(base));deltas=[]
    for r in feedback['tasks']:
        if r['task'] in CHANGES:
            prior=next(z for z in base['tasks'] if z['task']==r['task']);r['raw'],r['score']=CHANGES[r['task']]
            deltas.append(dict(task=r['task'],baseline_raw=prior['raw'],candidate_raw=r['raw'],baseline_score=prior['score'],candidate_score=r['score'],score_delta=r['score']-prior['score'],contribution_to_total=(r['score']-prior['score'])/22))
    avg=sum(r['score'] for r in feedback['tasks'])/22
    feedback={k:feedback[k] for k in ('tasks',)}|dict(version='v144',date='2026-10-27',score=51.099,task_score_mean=avg,status='user_reported_official',
        submitted_at='2026-10-27 17:09',source='User-pasted 22-task official scorecard; association to most recently delivered v144 inferred; platform CSV hash not exposed.',
        submission_sha256=load(ROOT/'docs/v144_local_delivery.json')['sha256'],uploaded_byte_hash_confirmed=False,baseline='v130',baseline_score=51.146,
        delta_task_mean=avg-base['task_score_mean'],task_deltas=deltas,unchanged_tasks=17)
    assert round(avg,3)==51.099
    save(ROOT/'docs/platform_scorecard_v144_20261027.json',feedback)
    oldfile=next((ROOT/'admet_submissions/v130_low_task_experts').glob('*.csv'));newfile=next((ROOT/'admet_submissions/v144_broad_task_models').glob('*.csv'))
    assert sha(oldfile)==base['submission_sha256'] and sha(newfile)==feedback['submission_sha256']
    a,z=rows(oldfile),rows(newfile);processed=load(ROOT/'work/v145_export/v145_rows.json');assert len(a)==len(z)==len(processed)==16380
    expected=[a[0]]+[r if r[0] not in KEEP else z[i] for i,r in enumerate(a[1:],1)]
    assert processed[0][0].lstrip('\ufeff')==a[0][0]
    assert processed[0][1:]==a[0][1:] and processed[1:]==expected[1:]
    template=ROOT/'admet_official/submission_template.csv';raw=template.read_bytes();sp=field_spans(raw);pieces=[];cursor=0
    for i,r in enumerate(expected[1:],1):
        lo,hi=sp[i][4];pieces.extend((raw[cursor:lo],r[4].encode('ascii')));cursor=hi
    pieces.append(raw[cursor:]);payload=b''.join(pieces)
    folder=ROOT/'admet_submissions/v145_feedback_repair';folder.mkdir(parents=True,exist_ok=True);file=folder/oldfile.name
    if file.exists():assert file.read_bytes()==payload
    else:file.write_bytes(payload)
    assert rows(file)==expected
    fmt=verify(file,template);assert fmt['utf8_bom'] and fmt['crlf_only']
    rsp=field_spans(payload);pieces=[];cursor=0
    for i in range(1,len(expected)):
        lo,hi=rsp[i][4];tlo,thi=sp[i][4];pieces.extend((payload[cursor:lo],raw[tlo:thi]));cursor=hi
    pieces.append(payload[cursor:]);assert b''.join(pieces)==raw
    counts=Counter(r[0] for r,s in zip(a[1:],expected[1:]) if r[4]!=s[4]);assert set(counts)==set(KEEP)
    retrospective=sum(CHANGES[r['task']][1] if r['task'] in KEEP else r['score'] for r in base['tasks'])/22
    fmt.update(whole_task_predictions_exact=True,nonprediction_bytes_exact=True,changed_tasks=dict(counts),unchanged_tasks=19)
    save(file.with_suffix('.verification.json'),fmt)
    sources={t:dict(version='v144' if t in KEEP else 'v130',csv_sha256=sha(newfile if t in KEEP else oldfile)) for t in [r['task'] for r in base['tasks']]}
    report=dict(version='v145',status='VERIFIED_TASK_ROUTING_AWAITING_PLATFORM',baseline='v130',baseline_official_score=51.146,csv=file.relative_to(ROOT).as_posix(),sha256=sha(file),
        kept_v144_tasks=list(KEEP),reverted_v144_tasks=[t for t in CHANGES if t not in KEEP],changed_tasks=dict(counts),changed_predictions=sum(counts.values()),sources=sources,
        official_score=None,retrospective_recombined_task_mean=retrospective,retrospective_recombined_task_mean_is_official=False,
        new_model_trained=False,automatically_uploaded=False,test_labels_used=False,external_data_added=False,
        method='Deterministic whole-task routing of immutable model outputs. No sample-specific editing, retraining, new blend tuning, or label inference.',
        limitations='Selected using public leaderboard feedback; retrospective 51.1661275 is arithmetic from prior measured task scores, not a new official result or unbiased validation.')
    save(folder/'model_evidence.json',report);save(ROOT/'docs/v145_local_delivery.json',report)
    lines=['# v144 官方回退分析与 v145 修复','',
        f'用户提供 v144 官方总分 51.099，22 项均值 {avg:.9f}。v130 均值 {base["task_score_mean"]:.9f}，下降 {avg-base["task_score_mean"]:.9f}。17 项完全不变，5 项模型改动与成绩变化对应。平台未提供所提交 CSV 的哈希，版本对应按最近交付文件推定。','',
        '| 任务 | v130 分项 | v144 分项 | 分项变化 | 总分贡献 |','|---|---:|---:|---:|---:|']
    for r in deltas:lines.append(f"| {r['task']} | {r['baseline_score']:.6f} | {r['candidate_score']:.6f} | {r['score_delta']:+.6f} | {r['contribution_to_total']:+.6f} |")
    lines+=['',
        'CYP3A4 底物 raw AUROC 从 0.7460443 降至 0.7075045，是主要损失；肝细胞清除率 Spearman 从 0.5652090 降至 0.5560508。AMES、CYP2C9 抑制及 CYP2D6 抑制改善。','',
        '本地四分区点估计均提升并不能保证隐藏测试集提升。两项失败候选的骨架 bootstrap 区间跨零，75% 的 CYP3A4 新模型融合比例使错序影响较大。重复使用公开标签、部分历史 OOF 基线为代理，也降低了筛选的可信度。无法仅凭总分证明唯一原因是过拟合或数据分布变化；需要更可靠的同条件基线与全新结构验证。','',
        'v145 保留上述三项 v144 模型输出，其余十九项直接沿用 v130 模型输出，包括恢复失败的两项。全部为任务级自动路由，未手工修改任何单样本预测，未重新调整融合比例。','',
        f'按两次已测分项重组的算术均值为 {retrospective:.7f}（显示约 51.166），这是回溯计算，不是 v145 官方成绩，也未达到 52。此修复只保住已经观察到的收益，不作为新模型突破。','',
        '后续新训练：优先补齐当前完整模型的同条件 OOF；CYP3A4 从头核查跨骨架稳定性，低样本任务采用更小的新专家权重并要求实际增益区间支持。保留旧模型旁路，区别完整新模型与基于成绩反馈的路由修复。不得将格式校验或模型重载 PASS 当作泛化保证。','',
        f'CSV SHA256：`{sha(file)}`。16,379 行、22 项，BOM/CRLF、Drug_ID 原字符串和全部非预测字节均与官方模板一致。','']
    doc=ROOT/'docs/ITERATION_V145_20261027.md';doc.write_text('\n'.join(lines),encoding='utf-8')
    dest=Path('D:/bili_download/v145_feedback_repair');dest.mkdir(parents=True,exist_ok=True)
    for p in (file,file.with_suffix('.verification.json'),doc):
        target=dest/(p.name if p!=doc else 'v145_说明.md')
        if target.exists():assert target.read_bytes()==p.read_bytes()
        else:shutil.copy2(p,target)
    report['upload_copy']=str(dest/file.name);save(ROOT/'docs/v145_local_delivery.json',report);save(dest/'v145_delivery.json',report)
    state=load(ROOT/'CURRENT_STATUS.json');state.update(latest_completed_model_branch='v145',latest_completed_model_branch_is_proven_best=False,latest_aggregate_plan='v145',
        aggregate_status=report['status'],active_experiment='v145',active_experiment_status=report['status'],recommended_submission_version='v145',
        best_official_version='v130',best_official_version_user_reported='v130',best_official_score_user_reported=51.146,last_user_reported_official_version='v144',
        last_user_reported_official_score=51.099,official_score=51.099,new_submission_created=True)
    state['latest_user_reported_official_score']=dict(version='v144',score=51.099,previous_version='v130',previous_score=51.146,observed_gain=-.047,source='User-pasted task scorecard',reported_date='2026-10-27',uploaded_byte_hash_confirmed=False)
    state['active_iteration']=dict(version='v145',baseline='v130 + confirmed v144 tasks',status=report['status'],official_score=None,running_training_processes=0)
    state['releases']['144'].update(official_score=51.099,status='OFFICIAL_REGRESSION_RECORDED');state['v144_result'].update(official_score=51.099,status='OFFICIAL_REGRESSION_RECORDED')
    state['releases']['145']=report;state['v145_result']=report
    state['priority_next_step']='Evaluate task-routing repair v145; rebuild matched OOF controls before another broad model candidate.';save(ROOT/'CURRENT_STATUS.json',state)
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8-sig');head='# v144 官方 51.099；v145 保留三项收益并恢复两项退步\n\n官方最佳仍为 v130（51.146）。v145 为完整任务输出路由修复，尚无官方分数。已测分项重组均值约 51.166，仅为回溯计算。\n\n[分析与修复](docs/ITERATION_V145_20261027.md) · [提交文件记录](docs/v145_local_delivery.json)\n\n以下为历史快照，部分待评测状态已由上述反馈更新。\n\n'
    if not text.startswith(head):readme.write_text(head+text,encoding='utf-8')
    paths=[Path(__file__),ROOT/'work/v145_export/build.mjs',ROOT/'docs/platform_scorecard_v144_20261027.json',doc,ROOT/'docs/v145_local_delivery.json',ROOT/'tools/repack_upload_template.py',ROOT/'tools/verify_submission.py']
    paths+=list(folder.iterdir())
    manifest=dict(version='v145',files=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in paths],
        model_dependencies=[dict(path='releases/v144_broad_task_models_20261027.tar.gz',sha256=load(ROOT/'releases/v144_broad_task_models_20261027.archive.json')['sha256']),dict(path='releases/v130_low_task_experts_20261027.tar.gz',sha256=load(ROOT/'releases/v130_low_task_experts_20261027.archive.json')['sha256'])])
    mp=ROOT/'releases'/(PREFIX+'.manifest.json');save(mp,manifest);archive=ROOT/'releases'/(PREFIX+'.tar.gz');assert not archive.exists()
    with tarfile.open(archive,'w:gz') as tf:
        for p in paths+[mp]:tf.add(p,arcname=p.relative_to(ROOT).as_posix())
    with tarfile.open(archive,'r:gz') as tf:
        for r in manifest['files']:assert hashlib.sha256(tf.extractfile(r['path']).read()).hexdigest()==r['sha256']
    save(ROOT/'releases'/(PREFIX+'.archive.json'),dict(path=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,all_member_hashes_verified=True))
    print(json.dumps(dict(official_v144=51.099,regression=feedback['delta_task_mean'],csv=str(dest/file.name),sha256=sha(file),retrospective_task_mean=retrospective,official_v145=None,changed_tasks=dict(counts)),ensure_ascii=True,indent=2))
if __name__=='__main__':main()
