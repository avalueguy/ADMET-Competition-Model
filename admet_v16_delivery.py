import json
from pathlib import Path

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

proof=read('admet_release_v16/local_verification.json');assert proof['status']=='VERIFIED_LOCAL_AND_REMOTE'
agg=read('admet_diagnostics/v16_aggregate.json')
run=dict(status='complete_verified_awaiting_platform_score',candidate_version='v16',candidate_platform_score=None,
    baseline_release='v15',baseline_score=49.750495,next_target=50.019624,ultimate_target=51.584929,
    requested_gain=.6,target_achieved=False,remote_only=True,remote_directory='/root/autodl-tmp/admet_20260925',
    local_training=False,automatic_submission=False,changed=proof['changed'],unchanged=proof['unchanged'],
    mean_oof_primary_surrogate=agg['mean_oof_primary_surrogate'],mean_valid_primary_surrogate=agg['mean_valid_primary_surrogate'],
    valid_primary_regression_tasks=agg['valid_primary_regression_tasks'],submission=proof['submission'],sha256=proof['sha256'],bytes=proof['bytes'],
    limitation='OOF, validation and repeat splits were reused for selection; surrogate gains are not an official score forecast.')
save('admet_v16_remote_run.json',run)
save('admet_current_candidate.json',dict(recommended_release='v16',submission=proof['submission'],sha256=proof['sha256'],
    platform_anchor_release='v15',platform_anchor_score=49.750495,candidate_platform_score=None,
    next_version_minimum_target=50.019624,requested_gain=.6,target_score=51.584929,target_achieved=False,
    awaiting='User-reported platform score for exact v16 CSV',report='admet_diagnostics/v16_delivery_report.md',
    verification='admet_release_v16/local_verification.json',limitation=run['limitation']))
lines=['# v16 远程模型迭代记录','',
    'v15 平台分数为 49.750495。本轮 v16 已完成全部 22 个任务的远程训练、融合选择、最终重训和独立模型重放；尚未获得平台实测分数。','',
    f"v16 修改 {len(proof['changed'])}/22 个任务，保留 {len(proof['unchanged'])} 个任务的 v15 预测。折外替代目标平均改善 {agg['mean_oof_primary_surrogate']:.6f}，官方验证替代目标平均改善 {agg['mean_valid_primary_surrogate']:.6f}。这些数值不能直接换算为平台分数。",'',
    '明显验证收益包括：生物利用度 AUROC +0.023256；CYP2C9 底物 AUPRC +0.100291；CYP2C9 Veith AUPRC +0.007935；hERG AUROC +0.010913；LD50 MAE 相对改善约 0.012093。半衰期 Spearman 下降约 0.001517，已按当前激进优化策略保留。','',
    f"验证主指标回落任务：{', '.join(agg['valid_primary_regression_tasks']) or '无'}。",'',
    '所有训练和模型推理在远程 RTX 4090 完成。本地未进行 GPU 训练，也未复制模型权重。提交文件已通过远程独立重放和本地逐行检查：16,379 行、22 任务、6 列、UTF-8 BOM、CRLF，非预测字段与官方模板完全一致。','',
    f"提交文件：{proof['submission']}",f"SHA256：{proof['sha256']}",f"字节数：{proof['bytes']}",'',
    '官方分数仍需用户上传后确认；没有自动上传。']
Path('admet_diagnostics/v16_delivery_report.md').write_text('\n'.join(lines),encoding='utf-8')
print(json.dumps(run,ensure_ascii=False,indent=2))
