"""Record the full user-reported 51.269 scorecard; preserve frozen stage-one evidence."""
from __future__ import annotations
import copy, csv, hashlib, json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from v184_score_objective_audit import normalized

ROOT=Path(__file__).resolve().parents[1]
# Full ordered raw metrics and normalized scores, transcribed from the user's card.
REPORTED_VALUES=[
 (.8769096712291213,50.141941),(.9289790494058787,50.032316),(.7948121050881277,52.391603),
 (.27592992051587734,50.689546),(.565208996561239,50.211228),(.6756845436053647,50.495262),
 (.5862085536546348,56.936749),(.8349408577894758,51.364323),(.7583244699983189,50.358383),
 (.7439447168745902,50.846574),(.7460443037974683,50.706702),(.8991915344045904,50.422543),
 (.935,50.715743),(.6509691357439866,52.539161),(.861119293078056,49.93509),(1.,50.14341),
 (.55547536869963,52.05411),(.4168746341957797,53.148026),(.9504798720341242,50.320749),
 (6.665484533215087,52.376135),(.7010230434713527,51.299786),(.7145137427047057,50.792751),
]
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):
 p=ROOT/name;assert not p.exists(),f'Preserve existing record: {p}'
 p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 old=read('docs/platform_scorecard_v361_20261104.json');delivery=read('docs/v364_stage1_delivery.json');quality=read('docs/v364_final_quality_check.json')
 control=ROOT/'experiments/transformer_stage1_v364/outputs/confirmation.json'
 preserved=[ROOT/'docs/v364_stage1_delivery.json',control,ROOT/'docs/V364_STAGE1_RESULTS.md',ROOT/'docs/v364_release_archive.json',ROOT/'docs/current_best_after_v361_20261104.json']
 hashes={str(p.relative_to(ROOT)):sha(p) for p in preserved}
 submission=ROOT/delivery['csv'];assert sha(submission)==delivery['csv_sha256']==quality['csv_sha256'];assert quality['status']=='PASS'
 tasks=copy.deepcopy(old['tasks']);assert len(tasks)==len(REPORTED_VALUES)==22
 for row,(raw,points) in zip(tasks,REPORTED_VALUES):
  row.update(raw=raw,score=points)
  assert abs(normalized(row['metric'],raw,row['reference_score'])-points)<.00000051
 changes=[]
 for row,before in zip(tasks,old['tasks']):
  if row==before:continue
  assert row['metric']=='MAE' and row['raw']<before['raw']
  changes.append(dict(task=row['task'],metric=row['metric'],before_raw=before['raw'],after_raw=row['raw'],relative_mae_reduction=(before['raw']-row['raw'])/before['raw'],before_score=before['score'],after_score=row['score'],task_point_gain=row['score']-before['score'],total_point_contribution=(row['score']-before['score'])/22))
 assert [r['task'] for r in changes]==delivery['experimental_tasks']
 with submission.open(encoding='utf-8-sig',newline='') as f:current=list(csv.DictReader(f))
 with (ROOT/old['csv']).open(encoding='utf-8-sig',newline='') as f:previous=list(csv.DictReader(f))
 assert len(current)==len(previous)==16379
 changed_rows={}
 for new,base in zip(current,previous):
  assert all(new[k]==base[k] for k in ('task','sample_idx','drug_id','smiles','metric'))
  if new['y_pred']!=base['y_pred']:changed_rows[new['task']]=changed_rows.get(new['task'],0)+1
 assert changed_rows==delivery['changed_rows']
 mean=sum(r['score'] for r in tasks)/22;delta=mean-old['task_score_mean'];assert round(mean,3)==51.269
 assert abs(delta-sum(r['total_point_contribution'] for r in changes))<1e-12
 at=datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat();source='docs/platform_scorecard_v364_20261104.json'
 platform_time='2026/11/04 15:08'
 card=dict(version='v364',date='2026-11-04',submitted_at=platform_time,recorded_at_hong_kong=at,timezone='Asia/Hong_Kong',score=51.269,task_score_mean=mean,tasks=tasks,
  status='user_reported_official',platform_displayed_certification='未认证',source='Full ordered22-task scorecard pasted by user after v364 delivery.',
  csv=delivery['csv'],submission_sha256=delivery['csv_sha256'],uploaded_byte_hash_confirmed=False,
  association='Inferred from immediate feedback, both changed endpoints and20 unchanged metrics. Platform upload hash not available.',
  platform_timestamp_preserved_verbatim=True,platform_timestamp_precedes_local_delivery_clock=True,
  timestamp_note='The pasted platform timestamp is earlier than the local delivery timestamp. Clock alignment and uploaded bytes have not been independently verified.',
  baseline='v361',baseline_score=old['score'],changed_tasks=[r['task'] for r in changes],unchanged_tasks=20,delta_task_mean=delta,original_local_strict_rejections_preserved=True)
 best=dict(version='v364',official_user_reported=51.269,exact_task_mean=mean,recorded_at_hong_kong=at,csv=delivery['csv'],csv_sha256=delivery['csv_sha256'],source=source,previous_best='v361',previous_best_official=old['score'],delta_task_mean=delta,platform_uploaded_byte_hash_confirmed=False,
  reason='User-reported full scorecard improves both changed endpoints, other20 match. Original local strict-validation failures are preserved.')
 report=dict(status='PASS_USER_REPORTED_OFFICIAL_IMPROVEMENT',recorded_at_hong_kong=at,best=best,changes=changes,unchanged_tasks=20,changed_prediction_rows=changed_rows,
  rounded_display_delta=round(51.269-old['score'],3),exact_task_mean_delta=delta,gap_to52=52-mean,required_sum_task_points_to52=22*(52-mean),
  strict_accepted_tasks=[],historical_strict_rejections_unchanged=True,official_feedback_is_not_independent_private_test_validation=True,
  no_new_training=True,no_new_submission=True,platform_submission=False,no_platform_access=True,preserved_source_hashes=hashes,
  recommendation='Adopt v364 as the best user-reported official baseline. Next bounded experiment should extend the frozen graph representation to PPBR and LD50 with original-scale L1 target heads, then inspect task-specific residual models. Do not copy unsuccessful hERG/CYP2D6/Caco2 recipes or promise a52 score.',
  forecast='Current evidence supports a small incremental gain, not a multi-point jump; no guaranteed score forecast.',
  version_association_limit=card['association'],timestamp_note=card['timestamp_note'])
 save(source,card);save('docs/current_best_after_v364_20261104.json',best);save('docs/v365_feedback_review.json',report)
 lines=['# v364 官方反馈复核','',f'用户成绩单：51.269；22项分数均值 {mean:.12f}。上版 v361 为51.240，精确均值增量为 {delta:.12f}。','',
  '|任务|上版 MAE|本次 MAE|误差下降|分项增加|对总分的贡献|','|---|---:|---:|---:|---:|---:|']
 names={'lipophilicity_astrazeneca':'脂溶性','solubility_aqsoldb':'溶解度'}
 for r in changes:lines.append(f"|{names[r['task']]}|{r['before_raw']:.9f}|{r['after_raw']:.9f}|{r['relative_mae_reduction']:.3%}|+{r['task_point_gain']:.6f}|+{r['total_point_contribution']:.6f}|")
 lines += ['', '其余20项原始指标和分项分数均与v361一致；本地文件也仅这两项的2837条预测变化。',
  '这次反馈支持Graph Transformer迁移专家与现有基线的互补性。新专家单独的开发成绩仍弱于基线，不能据此直接替换原模型。',
  '原实验未通过完整严格稳定性验收，本记录不改写冻结选择、确认结果或交付证据。官方公开成绩改善也不等于证明独立私榜稳定性。',
  f'距离52分仍差约 {52-mean:.3f}；此次是小幅增益，没有几分飞跃的证据。',
  '下一阶段建议优先在PPBR和LD50上验证相同表征的任务专用L1头，继续逐任务筛选。所有后续GPU训练仍使用已指定的远端服务器。',
  '成绩单与v364的关联由紧接交付的反馈及分项匹配推断，未取得平台上传哈希。平台时间按原文记录，其与本地交付时间的先后关系未独立核实。','']
 path=ROOT/'docs/V364_OFFICIAL_FEEDBACK.md';assert not path.exists();path.write_text('\n'.join(lines),encoding='utf-8')
 assert all(sha(ROOT/p)==value for p,value in hashes.items())
 print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
