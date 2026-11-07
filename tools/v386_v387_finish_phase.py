"""Finalize verified remote experiments and a reproducible dependency delta."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, tarfile

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    deliveries={v:read(ROOT/f'docs/{v}_delivery.json')for v in('v386','v387','v388')}
    for v in('v386','v387'):
        exp=ROOT/f'experiments/regression9_{v}'
        checked=read(exp/'outputs/independent_verification.json');receipt=read(exp/'completion_receipt.json')
        assert checked['status']==receipt['status']=='PASS' and checked['signature']==receipt['signature']==deliveries[v]['signature']
    best=read(ROOT/'docs/current_best_after_v385_20261106.json')
    official=read(ROOT/'docs/platform_scorecard_v385_20261106.json')
    assert sha(ROOT/best['csv'])==best['csv_sha256']
    union=deliveries['v388'];chosen=union if union['csv'] else next((deliveries[v]for v in('v386','v387')if deliveries[v]['csv']),None)
    goal=bool(chosen and chosen['goal_mean_5percent_achieved_local']);fivegoal=bool(chosen and chosen['five_scored_goal_achieved_local'])
    state=read(ROOT/'CURRENT_STATUS.json')
    state.update(aggregate_status='V386_V387_COMPLETE_GOAL_MET_LOCAL'if goal else'V386_V387_COMPLETE_GOAL_NOT_MET',
        handoff_date='2026-11-06',timezone='Asia/Hong_Kong',latest_aggregate_plan='v386_v387_independent_experts',
        latest_completed_model_branch='v388'if union['csv']else'v387',latest_completed_model_branch_is_proven_best=False,
        current_experiment_progress='docs/V386_V387_PHASE_HANDOFF.md',remote_training_running=False,official_score=51.308,
        latest_feedback_version='v385_associated',rollback_baseline='v385_local_csv',
        current_training_baseline='v385_associated:51.308',target_achieved=goal,
        goal_nine_task_mean_MAE_five_percent_achieved=goal,goal_five_scored_mean_MAE_five_percent_achieved=fivegoal,
        current_best='docs/current_best_after_v385_20261106.json',current_official_feedback='docs/platform_scorecard_v385_20261106.json',
        latest_MAE_experimental_submission=chosen,latest_completed_experiments=['v386','v387'])
    for v,d in deliveries.items():state['releases'][v[1:]]=d
    save(ROOT/'CURRENT_STATUS.json',state)
    scored=[r for r in official['tasks']if r['metric']=='MAE'];fixed=sum(r['score']for r in official['tasks']if r['metric']!='MAE')
    def conditional(f):return(fixed+sum(100*r['reference_score']/(r['reference_score']+r['raw']*(1-f))for r in scored))/22
    lo,hi=0.,.5
    for _ in range(70):
        mid=(lo+hi)/2
        if conditional(mid)<52:lo=mid
        else:hi=mid
    save(ROOT/'docs/v388_score_scenarios.json',dict(formula='100*reference/(reference+MAE)',other17_fixed=True,
        uniform_five_task_MAE_reduction5percent=conditional(.05),uniform_five_task_MAE_reduction10percent=conditional(.1),
        uniform_five_task_MAE_reduction_for52=hi,scenario_is_model_prediction=False))
    v386,v387=deliveries['v386'],deliveries['v387']
    lines=['# 最新进展：51.308反馈与v386/v387完整实验','',
        '最新用户成绩为51.308，关联v385提交；只有PPBR变化，MAE由6.495122547350538降至6.442906324588329，下降0.80393%。平台上传字节未确认。',
        '',f"v386：216次开发基础SVR、108次开发单调校准LP全部完成；包括确认/最终，基础拟合{v386['all_base_fits_verified']}次，校准{v386['all_calibration_fits_verified']}次。采用任务：{v386['accepted_tasks']}。",
        '',f"v387：324次开发求解全部完成，每条曲线0..8轮；包括确认/最终共{v387['all_solves_verified']}次。采用任务：{v387['accepted_tasks']}。",
        '', '两轮所有拟合仅指定25870远端。独立重放误差0，源码/输入/模型哈希、查询标签扰动、OOF隔离/LP证书或度量正定检查通过。完整学习曲线和拒绝记录保留。',
        '', '两轮都比较完整v385。任务合成规则在v387开发结果产生前记录：不相交任务直接合成，重叠固定优先v386，不按确认收益二次挑选。',
        '', '最终候选：`'+str(chosen['csv']if chosen else None)+'`。新官方成绩未知。',
        '', '独立备选：`'+str(v387['csv'])+'`。若与v386采用同一任务，两者不合成；主文件依预先固定优先顺序确定，并非根据确认收益选出。',
        '', '9项平均相对MAE降幅：'+str(chosen['nine_task_mean_relative_MAE_gain_by_part']if chosen else{})+'；拒绝任务计零。',
        '', '5项官方MAE平均相对降幅：'+str(chosen['five_scored_task_mean_relative_MAE_gain_by_part']if chosen else{})+'。',
        '', '9项平均5%目标达到：'+str(goal)+'；5项平均5%目标达到：'+str(fivegoal)+'。',
        '', '局部方向一致但未通过严格幅度或分组置信区间门槛的候选，只标为实验文件。公开分区已反复使用，历史控制部分proxy，四个Spearman任务的MAE使用校准参考，不能称新盲测。',
        '',f'条件算术：其余17项固定，5项MAE共同降低5%约{conditional(.05):.3f}，降低10%约{conditional(.1):.3f}，共同降低{hi:.2%}约52。这不是新模型分数预测。',
        '', '逐任务证据：`docs/V386_RESULTS_AND_HANDOFF.md`、`docs/V387_RESULTS_AND_HANDOFF.md`、`docs/v388_delivery.json`。',
        '', '复现交接：按依赖顺序解压v380完整包、v382增量、v385增量、v388增量，再运行`python tools/v388_restore_inputs.py`重建未重复打包的相同输入路径并核对哈希。随后可运行两份deliver脚本及v388_combine；这些接收器不拟合模型。模型重放需Linux CUDA远端，全部选中最终状态已入包，未采用状态仍保留远端。',
        '', '依赖输入检查已核对188个目标输入哈希与两轮选中最终模型：`docs/v388_dependency_audit.json`。恢复器会展开历史嵌套输入包，再按哈希恢复本轮路径；不会再次运行prepare或改动冻结源码。',
        '', 'v387冻结源码开头仍有继承的diagonal说明文字；实际实现、协议和签名对应完整相关度量。冻结后未为改说明文字重签缓存。',
        '', '后续优先从更强的结构表示、重建历史集成的真实嵌套OOF和任务专属残差入手；沿开发区完整学习曲线决定表示和正则化，不在确认集连续重选。',
        '']
    summary='\n'.join(lines);(ROOT/'docs/V386_V387_PHASE_HANDOFF.md').write_text(summary,encoding='utf-8')
    for p in(ROOT/'README.md',ROOT/'HANDOFF_CURRENT.md',ROOT/'docs/HANDOFF_CURRENT.md'):
        old=p.read_text(encoding='utf-8')if p.exists()else''
        if old.startswith(lines[0]):old=old.split('\n---\n\n',1)[1]
        p.write_text(summary+'\n---\n\n'+old,encoding='utf-8')
    prior=[read(ROOT/f'docs/{v}_release.json')for v in('v380','v382','v385')]
    for r in prior:assert sha(ROOT/r['archive'])==r['sha256']
    dependencies=dict(required_archives=[{k:r[k]for k in('archive','sha256','bytes')}for r in prior],
        delta_alone_is_not_complete=True,all_fitting_remote=True,local_assembly_no_fitting=True,credentials_included=False,
        remote_roots=['/root/autodl-tmp/regression9_v386','/root/autodl-tmp/regression9_v387'])
    save(ROOT/'docs/v388_handoff_dependencies.json',dependencies)
    files=[ROOT/'CURRENT_STATUS.json',ROOT/'README.md',ROOT/'HANDOFF_CURRENT.md',ROOT/'docs/HANDOFF_CURRENT.md',
        ROOT/'docs/platform_scorecard_v385_20261106.json',ROOT/'docs/current_best_after_v385_20261106.json']
    for v in('v386','v387','v388'):
        files+=list((ROOT/'tools').glob(v+'_*'))+list((ROOT/'docs').glob(v+'_*'))+list((ROOT/'docs').glob(v.upper()+'_*'))
    for v in('v386','v387'):
        files.append(ROOT/f'releases/{v}_input_delta.tar.gz')
        files.append(ROOT/f'releases/{v}_remote_results.tar.gz')
        exp=ROOT/f'experiments/regression9_{v}'
        files += [p for p in exp.rglob('*')if p.is_file()and('outputs'in p.parts or p.suffix in('.json','.py','.log')or p.name=='v385_baseline.csv')]
        files += list((exp/'controls/ppbr_az').glob('*.npz'))
    for d in deliveries.values():
        if d['csv']:files+=list((ROOT/d['csv']).parent.glob('*'))
    files=sorted(set(p for p in files if p.is_file()and p.name not in('v388_release.json','v388_release_manifest.json')and'__pycache__'not in p.parts))
    entries=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size)for p in files]
    manifest=ROOT/'docs/v388_release_manifest.json';save(manifest,dict(files=entries,dependencies='docs/v388_handoff_dependencies.json',credentials_included=False))
    archive=ROOT/'releases/v388_handoff_delta.tar.gz'
    with tarfile.open(archive,'w:gz',dereference=True)as tf:
        for p in files+[manifest]:tf.add(p,arcname=p.relative_to(ROOT).as_posix())
    expected={r['path']:r['sha256']for r in entries};expected[manifest.relative_to(ROOT).as_posix()]=sha(manifest)
    with tarfile.open(archive)as tf:
        members=tf.getmembers();assert len(members)==len(expected)
        for m in members:assert m.isfile()and hashlib.sha256(tf.extractfile(m).read()).hexdigest()==expected[m.name]
    release=dict(status='PASS',archive=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,
        members=len(expected),all_member_hashes_verified=True,dependencies=dependencies,completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(ROOT/'docs/v388_release.json',release);print(json.dumps({k:v for k,v in release.items()if k!='dependencies'}))


if __name__=='__main__':main()
