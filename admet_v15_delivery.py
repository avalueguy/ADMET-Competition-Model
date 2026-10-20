"""Record verified unscored v15 and explain gains and regressions without forecasting points."""
import csv,json,hashlib
from datetime import datetime,timezone
from pathlib import Path


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    v=read('admet_release_v15/local_verification.json');assert v['status']=='VERIFIED_LOCAL_AND_REMOTE'
    assert read('admet_model_outputs_v15/status.json')['status']=='COMPLETE_VERIFIED'
    upload=Path(v['local_submission']);assert sha(upload)==v['sha256']
    agg=read('admet_diagnostics/v15_aggregate.json');routes=read('admet_release_v15/routing.json')
    audit=read('admet_diagnostics/v15_input_audit.json');assert audit['status']=='PASSED' and len(audit['tasks'])==22
    regression=[r for r in routes if r['selection']['valid_gain']<0]
    repeat=[r for r in routes if r['repeat'].get('primary_gain',0)<0]
    fit_records=[read(p) for p in Path('admet_model_outputs_v15').glob('*/*/*.json')]
    fit_records=[r for r in fit_records if 'training_seconds' in r]
    caveat='Historical OOF, official validation and repeated scaffold diagnostics are reused and selection-biased. Their gains are not an estimate of official platform points. Hidden-score improvement and +0.6 are not guaranteed.'
    run=dict(status='complete_verified_awaiting_platform_score',completed_at_utc=datetime.now(timezone.utc).isoformat(),
        candidate_version='v15',candidate_platform_score=None,baseline_release='v14',baseline_score=49.419624,
        requested_gain=.6,next_target=50.019624,ultimate_target=51.584929,target_achieved=False,
        remote_only=True,remote_directory='/root/autodl-tmp/admet_20260925',local_training=False,
        automatic_submission=False,per_metric_nonregression_required=False,trained_bundles=len(fit_records),
        adopted=v['changed'],retained=v['unchanged'],valid_primary_regressions=[r['task'] for r in regression],
        repeat_primary_regressions=[r['task'] for r in repeat],submission=str(upload),sha256=v['sha256'],bytes=v['bytes'],limitation=caveat)
    save('admet_v15_remote_run.json',run)
    save('admet_current_candidate.json',dict(recommended_release='v15',submission=str(upload),sha256=v['sha256'],
        platform_anchor_release='v14',platform_anchor_score=49.419624,candidate_platform_score=None,
        next_version_minimum_target=50.019624,requested_gain=.6,target_score=51.584929,target_achieved=False,
        awaiting='User-reported official score for this exact verified v15 file',
        report='admet_diagnostics/v15_delivery_report.md',verification='admet_release_v15/local_verification.json',
        constraint_change='Per-task and per-metric non-regression vetoes removed as requested',limitation=caveat))
    save('admet_diagnostics/v15_unscored_candidate.json',dict(release='v15',sha256=v['sha256'],platform_score=None,
        baseline_release='v14',baseline_score=49.419624,submission=str(upload)))
    lines=['# v15 远程训练与提交记录','',
        '已评分基线 v14：49.419624；下一目标为 50.019624（+0.6），最终目标为 51.584929。v15 尚无平台分数，不能宣称达到任一目标。','',
        '本轮按用户新要求移除了逐任务、逐折、辅助指标和低相似度子组的不退步门槛，允许为整体目标接受局部退步。保留旧文件供对照。','',
        '模型从 v6–v13 的原始候选家族中，按匹配的骨架折外预测为每个任务选两个候选；随后在远程重新训练，比较原值融合和排名融合，采用 0、25%、50%、75%、100% 的凸组合权重。模型包括预训练分子表示、神经网络、多任务迁移、分子核和树模型。所有短名单在新验证结果产生前固定。','',
        '选择目标为 65% 折外主指标改善 + 35% 官方验证集主指标改善，再减去折间波动和新增模型权重的小额惩罚。MAE 使用相对改善，AUROC/AUPRC/Spearman 使用绝对差值；任务间等权，仅是替代目标，不能换算成平台总分。','',
        f"最终修改 {len(v['changed'])}/22 个任务，保留 {len(v['unchanged'])} 个任务的原始预测字符串；本轮完成 {len(fit_records)} 个模型包训练（一个模型包可能含多个随机种子或参数模型），再独立重载复算验证、重复切分和最终预测。训练及重推理全部在远程服务器进行。",'',
        f"22 任务平均主指标替代改善：折外 {agg['mean_oof_primary_surrogate']:.6f}，官方验证 {agg['mean_valid_primary_surrogate']:.6f}。这些数字不是平台分数或预计涨分。",'',
        '|任务|主指标|v14 验证|v15 验证|验证改善¹|重复切分改善¹|融合|v14 / 候选1 / 候选2 权重|候选|',
        '|---|---|---:|---:|---:|---:|---|---|---|']
    for r in routes:
        s=r['selection'];key=s['metric'].lower();rg=r['repeat'].get('primary_gain');rs='未重训（保留基线）' if rg is None else f'{rg:+.6f}'
        lines.append(f"|{r['task']}|{s['metric']}|{r['before'][key]:.6f}|{r['after'][key]:.6f}|{s['valid_gain']:+.6f}|{rs}|{s['mode']}|{' / '.join(str(w) for w in s['weights'])}|{', '.join(s['names'])}|")
    lines+=['','¹ MAE 为相对改善比例，排名指标为绝对差；负值代表退步。','',
        f"官方验证集主指标下降：{', '.join(r['task'] for r in regression) or '无'}。", 
        f"重复骨架切分主指标下降：{', '.join(r['task'] for r in repeat) or '无'}。",'',
        '全体与低相似度子组的主/辅助指标见 v15_all_diagnostics.csv。上述退步没有触发旧版本的否决条件。','',
        '历史折外预测、官方验证集与重复切分被多轮用于选型，存在选择偏差。验证集本轮也参与权重选择；元模型留一折诊断不是独立嵌套交叉验证。这些结果不能保证隐藏测试集上涨或达到 +0.6。没有使用外部 ADMET 实验标签、盲测标签或答案检索。','',
        '提交已在远程及本地核验：16,379 条数据、22 个任务、6 列，UTF-8 BOM、CRLF、唯一表头；所有非预测字段与官方模板逐行一致，分类值范围和全部有限性检查通过。继续使用之前成功的整文件上传方式；网页刷新会重置网站分片配置。未自动上传。','',
        f'提交文件：{upload}',f'SHA256：{v["sha256"]}',f'文件字节数：{v["bytes"]}','']
    Path('admet_diagnostics/v15_delivery_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(run,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
