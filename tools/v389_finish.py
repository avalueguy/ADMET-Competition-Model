"""Persist user feedback, experimental fusion results and reproducible delta."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile,shutil
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/regression9_v389'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    delivery=read(ROOT/'docs/v389_delivery.json');receipt=read(EXP/'completion_receipt.json');assert receipt['status']=='PASS'
    models=read(EXP/'selected_models_manifest.json');archive=ROOT/'releases/v389_selected_models.tar.gz';assert sha(archive)==receipt['selected_models_archive_sha256']
    # Preserve legacy directories and place newly generated source models under a separate root.
    recovered=EXP/'source_model_snapshot'
    with tarfile.open(archive)as tf:
        expected={r['path']:r['sha256']for r in models['files']}
        for member in tf.getmembers():
            assert member.isfile()and member.name in expected;dest=(recovered/member.name).resolve();assert dest.is_relative_to(recovered.resolve())
            payload=tf.extractfile(member).read();assert hashlib.sha256(payload).hexdigest()==expected[member.name]
            if dest.exists():assert dest.read_bytes()==payload
            else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
    state=read(ROOT/'CURRENT_STATUS.json');state.update(aggregate_status='V389_COMPLETE',remote_training_running=False,
        official_score=51.314,latest_completed_model_branch='v389',latest_completed_model_branch_is_proven_best=False,
        current_training_baseline='v387:51.314',current_experiment_progress='docs/V389_RESULTS_AND_HANDOFF.md',
        latest_aggregate_plan='v389_restricted_expert_combination',latest_completed_experiments=['v389'],
        latest_MAE_experimental_submission=delivery if delivery['csv']else state.get('latest_MAE_experimental_submission'),
        target_achieved=delivery['goal_mean_5percent_achieved_local'])
    state['releases']['389']=delivery;save(ROOT/'CURRENT_STATUS.json',state)
    comparison=read(ROOT/'docs/v389_scorecard_comparison.json');lines=['# 最新进展：v387官方51.314与v389组合实验','',
        '用户明确反馈v386=51.308、v387=51.314。两份完整成绩只有脂溶性不同：MAE由0.4173184854676313降至0.41505260605543154，下降0.54296%；分项增加0.135557，总分精确均值增加0.00616168。其余21项相同。',
        '', '附件第一张表时间为12:15且逐项与上一份v385成绩一致，当前按用户明确版本归属登记v386，平台字节未确认。',
        '', '直接按任务取优的组合等于原v387，因此不把同一任务收益重复相加。v387仍为官方最好文件。',
        '', 'v389以完整v387为基线，固定两轮各任务此前开发选择的唯一专家，只搜索受限非负两系数，总新增权重≤75%；由fold0/1固定唯一配方，再确认fold2/valid，失败不换后备。全部选择、模型拟合和重放在指定25870远端。',
        '', '|任务|v386专家权重|v387专家权重|fold0 MAE降幅|fold1|fold2|valid|结果|',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in delivery['task_evidence']:
        gains={x['part']:f"{x['gain']:+.3%}"for x in r['parts']};label='实验采用'if r['eligible']else'未采用'
        lines.append('|'+ '|'.join([r['task'],str(r['a']),str(r['b']),*[gains.get(p,'未确认')for p in('fold0','fold1','fold2','valid')],label])+'|')
    confirmation=read(EXP/'confirmation.json')
    ppbr=next(r for r in confirmation['records']if r['task']=='ppbr_az')
    lines += ['', '本轮没有任何两个专家均为正权重的配方成为开发最优，因此未证实v386与v387专家之间存在额外融合收益。',
        '', '最终文件保留v387脂溶性，仅将PPBR改为95%完整v387基线加5%相关度量专家gaussian_full_C3_r8；共修改559行，其余21项保持不变。',
        '', 'PPBR确认分区的90%分组bootstrap相对MAE降幅区间：'+ '；'.join(r['part']+' ['+f"{r['ci90'][0]:+.4%}, {r['ci90'][1]:+.4%}"+']'for r in ppbr['bootstrap'])+'。两个区间均跨零。',
        '', '新实验提交：`'+str(delivery['csv'])+'`。新官方分数未知。',
        '', '九项平均相对MAE降幅：'+str(delivery['nine_task_mean_relative_MAE_gain_by_part'])+'，未采用任务计零。平均5%目标达到：'+str(delivery['goal_mean_5percent_achieved_local'])+'。',
        '', '严格稳定性确认：'+str(delivery['strict_stability_confirmed'])+'。方向一致门槛通过但置信区间跨零的文件仍为实验候选，不能保证涨分。',
        '', '局限：旧公开分区和组件配置已被反复研究，确认不等于全新盲测；四个Spearman任务的MAE基线是校准参考；MAE改善同时须有排序改善才采用。',
        '', '冻结脚本verification中的new_training_fits=False仅指系数组合层，不代表组件生成无新增拟合；完整原模型包盘点和更正见completion_receipt.json。',
        '', '复现：本轮增量依赖v388交接包及其递归依赖。运行原模型重放需要指定远端Linux CUDA。source_model_snapshot保留本轮采用组件最终完整状态，不覆盖旧冻结实验。',
        '', '下一步应扩展已出现官方收益的相关度量表示与骨架外泛化，并补齐强历史集成的真实OOF；继续以完整预算和多分区验收，不能由这一轮小幅收益推出几分跃升。','']
    summary='\n'.join(lines);(ROOT/'docs/V389_RESULTS_AND_HANDOFF.md').write_text(summary,encoding='utf-8')
    for path in(ROOT/'README.md',ROOT/'HANDOFF_CURRENT.md',ROOT/'docs/HANDOFF_CURRENT.md'):
        old=path.read_text(encoding='utf-8')
        if old.startswith(lines[0]):old=old.split('\n---\n\n',1)[1]
        path.write_text(summary+'\n---\n\n'+old,encoding='utf-8')
    prior=read(ROOT/'docs/v388_release.json');assert sha(ROOT/prior['archive'])==prior['sha256']
    deps=dict(required_archive={k:prior[k]for k in('archive','sha256','bytes')},recursive_dependencies=prior['dependencies'],delta_alone_is_not_complete=True,credentials_included=False)
    save(ROOT/'docs/v389_handoff_dependencies.json',deps)
    files=[ROOT/'CURRENT_STATUS.json',ROOT/'README.md',ROOT/'HANDOFF_CURRENT.md',ROOT/'docs/HANDOFF_CURRENT.md',
        ROOT/'docs/current_best_after_v387_20261106.json',ROOT/'docs/platform_scorecard_v386_20261106.json',ROOT/'docs/platform_scorecard_v387_20261106.json',ROOT/'docs/v386_v387_user_scorecards_20261106.txt',
        ROOT/'releases/v389_remote_results.tar.gz',ROOT/'releases/v389_selected_models.tar.gz']
    files+=list((ROOT/'tools').glob('v389_*'))+list((ROOT/'docs').glob('v389_*'))+list((ROOT/'docs').glob('V389_*'))
    files +=[p for p in EXP.rglob('*')if p.is_file()and'__pycache__'not in p.parts]
    if delivery['csv']:files+=list((ROOT/delivery['csv']).parent.glob('*'))
    files=sorted(set(p for p in files if p.is_file()and p.name not in('v389_release.json','v389_release_manifest.json')))
    entries=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size)for p in files]
    manifest=ROOT/'docs/v389_release_manifest.json';save(manifest,dict(files=entries,dependencies=deps,credentials_included=False))
    dest=ROOT/'releases/v389_handoff_delta.tar.gz'
    with tarfile.open(dest,'w:gz',dereference=True)as tf:
        for p in files+[manifest]:tf.add(p,arcname=p.relative_to(ROOT).as_posix())
    expected={r['path']:r['sha256']for r in entries};expected[manifest.relative_to(ROOT).as_posix()]=sha(manifest)
    with tarfile.open(dest)as tf:
        members=tf.getmembers();assert len(members)==len(expected)
        for m in members:assert m.isfile()and hashlib.sha256(tf.extractfile(m).read()).hexdigest()==expected[m.name]
    release=dict(status='PASS',archive=dest.relative_to(ROOT).as_posix(),sha256=sha(dest),bytes=dest.stat().st_size,members=len(expected),all_member_hashes_verified=True,dependencies=deps,completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(ROOT/'docs/v389_release.json',release);print(json.dumps({k:v for k,v in release.items()if k!='dependencies'}))
if __name__=='__main__':main()
