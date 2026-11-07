"""Finalize completed experiments, current status and verified dependency delta."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,tarfile
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 deliveries={v:read(ROOT/f'docs/{v}_delivery.json')for v in('v383','v384','v385')}
 for v in('v384','v385'):
  exp=ROOT/f'experiments/regression9_{v}';checked=read(exp/'outputs/independent_verification.json');receipt=read(exp/'completion_receipt.json')
  assert checked['status']==receipt['status']=='PASS'and checked['signature']==receipt['signature']==deliveries[v]['signature']
 best=read(ROOT/'docs/current_best_after_v382_20261105.json');official=read(ROOT/'docs/platform_scorecard_v382_20261105.json')
 assert sha(ROOT/best['csv'])==best['csv_sha256']
 result=deliveries['v385'];completed=datetime.now(timezone.utc).isoformat()
 smoke=ROOT/'experiments/regression9_v385/smoke_amendment.json';amendment=read(smoke)
 assert amendment['frozen_training_sources_unchanged']and sha(ROOT/'tools/v385_smoke_amended.py')==amendment['amended_smoke_sha256']
 state=read(ROOT/'CURRENT_STATUS.json');state.update(aggregate_status='V383_FEEDBACK_V384_V385_COMPLETE_GOAL_NOT_MET',
  latest_completed_model_branch='v385',latest_completed_model_branch_is_proven_best=False,current_experiment_progress='docs/V385_RESULTS_AND_HANDOFF.md',
  remote_training_running=False,official_score=51.299,target_achieved=False,latest_MAE_experimental_submission=result if result['csv']else deliveries['v383'],
  current_training_baseline='v382:51.299',goal_nine_task_mean_MAE_five_percent_achieved=result['goal_mean_5percent_achieved_local'],
  goal_five_scored_mean_MAE_five_percent_achieved=result['five_scored_goal_achieved_local'])
 for v,d in deliveries.items():state['releases'][v[1:]]=d
 save(ROOT/'CURRENT_STATUS.json',state)
 scored=[r for r in official['tasks']if r['metric']=='MAE'];fixed=sum(r['score']for r in official['tasks']if r['metric']!='MAE')
 def conditional(f):return(fixed+sum(100*r['reference_score']/(r['reference_score']+r['raw']*(1-f))for r in scored))/22
 lo,hi=0.,.5
 for _ in range(70):
  mid=(lo+hi)/2
  if conditional(mid)<52:lo=mid
  else:hi=mid
 arithmetic=dict(formula='MAE task score=100*reference/(reference+MAE), confirmed against supplied task scorecard',
  scenario_other17_fixed=True,uniform_five_task_MAE_reduction5percent=conditional(.05),uniform_five_task_MAE_reduction10percent=conditional(.1),
  uniform_five_task_MAE_reduction_for52=hi,scenario_is_model_prediction=False)
 save(ROOT/'docs/v385_score_scenarios.json',arithmetic)
 lines=['# 最新进展：v382官方反馈与v384/v385完整实验','',
  '最新用户确认官方成绩为v382：51.299。PPBR MAE下降约2.56%，脂溶性小退。完整v382作为新实验基线。',
  '','v384完成324次开发、342次总求解；任务自适应对角度量没有通过采用门槛。全部训练轮次已完成，未因早期表现中止。',
  '',f"v385完成296次开发拟合；结果：{result['status']}；采用任务：{result['accepted_tasks']}。",
  '', '两轮全部远端训练，独立重放误差0，源码/输入/模型哈希和查询标签扰动通过。公开分区反复使用，历史控制部分proxy，四个排序MAE是校准参考。',
  '', '实验提交文件：`'+str(result['csv'])+'`。新官方分数未知。',
  '', 'PPBR采用10%高斯秩中位数核专家融合。四区MAE方向为正，但fold2和valid的90%分组bootstrap区间跨零，未确认严格稳定性。其余21项预测保持完整v382。',
  '', '单独回退文件：`'+deliveries['v383']['csv']+'`，仅恢复旧脂溶性，保留v382 PPBR；不是新增模型收益。按任务独立可合成的条件算术51.300112，未官方测试。',
  '', '完整9项平均MAE降低5%目标尚未达到。五项按MAE直接计分；其他四项连续任务按Spearman计分，MAE下降不保证其官方收益。',
  '',f"条件算术：其他17项固定，五项MAE都降5%约{conditional(.05):.3f}；都降10%约{conditional(.1):.3f}；均降{hi:.2%}才约52。这不是实验预期。",
  '', '结果入口：`docs/V384_RESULTS_AND_HANDOFF.md`、`docs/V385_RESULTS_AND_HANDOFF.md`；方案、开发选型、确认失败及置信区间保留。',
  '', '下一步优先验证能否重建更强的结构表示和完整嵌套OOF，或针对边界分布的模型。对角度量有几项后期缓解退步但仍未赢过基线，不能据此永久否定特征学习；下一轮需改变表示/相关方向，不能原样重扫确认集。',
  '', 'v385 smoke曾有一条错误的有限经验秩逆映射断言，修订仅限测试，冻结训练源码未改；原脚本、修订和失败记录均保留。','']
 summary='\n'.join(lines);(ROOT/'docs/V385_PHASE_HANDOFF.md').write_text(summary,encoding='utf-8')
 for p in(ROOT/'HANDOFF_CURRENT.md',ROOT/'docs/HANDOFF_CURRENT.md'):
  old=p.read_text(encoding='utf-8')
  if not old.startswith('# 最新进展：v382官方反馈与v384/v385完整实验'):p.write_text(summary+'\n---\n\n'+old,encoding='utf-8')
 p=ROOT/'README.md';old=p.read_text(encoding='utf-8')
 if not old.startswith('# 最新进展：v382官方反馈与v384/v385完整实验'):p.write_text(summary+'\n---\n\n'+old,encoding='utf-8')
 prior=[read(ROOT/'docs/v380_release.json'),read(ROOT/'docs/v382_release.json')]
 for r in prior:assert sha(ROOT/r['archive'])==r['sha256']
 dependencies=dict(required_archives=prior,delta_alone_is_not_complete=True,
  all_fitting_remote=True,local_assembly_no_fitting=True,credentials_included=False,
  remote_roots=['/root/autodl-tmp/regression9_v384','/root/autodl-tmp/regression9_v385'])
 save(ROOT/'docs/v385_handoff_dependencies.json',dependencies)
 files=[ROOT/'CURRENT_STATUS.json',ROOT/'README.md',ROOT/'HANDOFF_CURRENT.md',ROOT/'docs/HANDOFF_CURRENT.md',
  ROOT/'releases/v384_input_delta.tar.gz',ROOT/'releases/v385_input_delta.tar.gz']
 for pattern in('v383_*','v384_*','v385_*','V384_*','V385_*'):
  files+=list((ROOT/'tools').glob(pattern))+list((ROOT/'docs').glob(pattern))
 files += [ROOT/'docs/platform_scorecard_v382_20261105.json',ROOT/'docs/current_best_after_v382_20261105.json']
 for v in('v384','v385'):
  exp=ROOT/f'experiments/regression9_{v}'
  files += [p for p in exp.rglob('*')if p.is_file() and ('outputs'in p.parts or p.suffix in('.json','.py','.log')or p.name=='v382_baseline.csv')]
  files += [p for t in('lipophilicity_astrazeneca','ppbr_az')for p in(exp/'controls'/t).glob('*.npz')]
 for d in deliveries.values():
  if d['csv']:files+=list((ROOT/d['csv']).parent.glob('*'))
 files=sorted(set(p for p in files if p.is_file()and p.name not in('v385_release.json','v385_release_manifest.json')and '__pycache__'not in p.parts))
 entries=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size)for p in files]
 manifest=ROOT/'docs/v385_release_manifest.json';save(manifest,dict(files=entries,dependencies='docs/v385_handoff_dependencies.json',credentials_included=False))
 archive=ROOT/'releases/v385_handoff_delta.tar.gz'
 with tarfile.open(archive,'w:gz',dereference=True)as tf:
  for p in files+[manifest]:tf.add(p,arcname=p.relative_to(ROOT).as_posix())
 expected={r['path']:r['sha256']for r in entries};expected[manifest.relative_to(ROOT).as_posix()]=sha(manifest)
 with tarfile.open(archive)as tf:
  members=tf.getmembers();assert len(members)==len(expected)
  for m in members:assert m.isfile()and hashlib.sha256(tf.extractfile(m).read()).hexdigest()==expected[m.name]
 release=dict(status='PASS',archive=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,members=len(expected),all_member_hashes_verified=True,dependencies=dependencies,completed_at_utc=completed)
 save(ROOT/'docs/v385_release.json',release);print(json.dumps({k:v for k,v in release.items()if k!='dependencies'}))
if __name__=='__main__':main()
