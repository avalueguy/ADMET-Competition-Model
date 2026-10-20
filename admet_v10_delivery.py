"""Combine cumulative v7-to-v10 evidence without training or altering predictions."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write_json(p,x):
    Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')


def read_csv(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def main():
    local=read_json('admet_release_v10/local_verification.json'); assert local['status']=='VERIFIED_LOCAL_AND_REMOTE'
    final=Path(local['submission']); original=Path('admet_submissions/v7/我从山北来带着兰花草-小分子药物预测技术创新赛-20260925.csv')
    assert hashlib.sha256(original.read_bytes()).hexdigest()=='88e81b9d16e48cb941f18c5fd6c4a49e067c73659de808e896e51dea8de5acf0'
    assert hashlib.sha256(final.read_bytes()).hexdigest()==local['sha256']
    original_rows=read_csv(original); new_rows=read_csv(final); assert len(original_rows)==len(new_rows)==16379
    changed=Counter(); tasks=set()
    for a,b in zip(original_rows,new_rows):
        tasks.add(a['task'])
        assert {k:v for k,v in a.items() if k!='y_pred'}=={k:v for k,v in b.items() if k!='y_pred'}
        if a['y_pred']!=b['y_pred']:changed[a['task']]+=1
    adopted=set()
    for v in [8,9,10]:adopted.update(read_json(f'admet_release_v{v}/independent_verification.json')['accepted'])
    assert set(changed)==adopted
    comparisons={v:{r['task']:r for r in read_csv(f'admet_diagnostics/v{v}_comparison.csv')} for v in [8,9,10]}
    rows=[]
    for task in sorted(tasks):
        first=comparisons[8][task]; last=comparisons[10][task]; metric=first['metric']; k=metric.lower()
        for v in [9,10]:
            prior=comparisons[v-1][task]; now=comparisons[v][task]
            for m in ['mae','spearman','auroc','auprc']:
                if prior['after_'+m]:assert abs(float(prior['after_'+m])-float(now['before_'+m]))<1e-12
        before=float(first['before_'+k]); after=float(last['after_'+k]); gain=before-after if metric=='MAE' else after-before
        assert gain>=-1e-12
        if task not in changed:assert gain==0
        rows.append(dict(task=task,metric=metric,v7_valid=before,v10_valid=after,favorable_change=gain,changed_prediction_rows=changed[task]))
    with Path('admet_diagnostics/v10_vs_scored_v7.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    decisions=[read_json(p) for p in Path('admet_model_outputs_v10').glob('*/decision.json')]
    rejection=Counter(d['reason'] for d in decisions if not d['accepted'] and d['task'] in read_json('admet_v10_protocol.json')['tasks'])
    audit_n=0
    for p in Path('admet_model_outputs_v10').glob('*/*source_audit.json'):
        for record in read_json(p):
            assert record['heldout_scaffold_overlap']==record['heldout_molecule_overlap']==0;audit_n+=1
    evidence=dict(status='VERIFIED_CUMULATIVE_V7_TO_V10',submission=str(final.resolve()),sha256=local['sha256'],
        generated_at=datetime.now(timezone.utc).isoformat(),baseline_version=7,baseline_platform_score=49.343821,
        target_score=51.584929,platform_score=None,target_achieved=False,automatic_submission=False,local_gpu_training=False,
        bytes=final.stat().st_size,rows=16379,columns=6,tasks=22,changed_tasks=sorted(changed),changed_prediction_rows=dict(changed),
        unchanged_tasks=sorted(tasks-set(changed)),adopted_v10=local['changed'],valid_primary_declines=0,
        audited_source_records=audit_n,source_overlap_count=0,rejection_reasons=dict(rejection),
        caveat='Reused validation; selected results are not unbiased hidden-score estimates. Some fold metrics decline within fixed tolerances.')
    write_json('admet_release_v10/cumulative_verification.json',evidence)
    run=read_json('admet_v10_remote_run.json');run.update(status='complete_verified_not_submitted',sha256=local['sha256'],
        submission=str(final.resolve()),accepted=local['changed'],cumulative_changed_vs_v7=sorted(changed),
        remote_independent_replay=True,local_template_verification=True,cumulative_v7_verification=True,
        training_processes_finished=True,awaiting_platform_feedback=True,target_achieved=False)
    write_json('admet_v10_remote_run.json',run)
    lines=['# v10 累计交付与证据', '',
        'v10 包含 v8、v9 已采用的改进，并加入本轮通过全部门槛的图网络预测。当前只需选择一个最终版本提交。v7 的平台实测仍是 49.343821；目标 51.584929，后续版本尚无用户反馈的分数。', '',
        '## 本轮实验', '',
        '18 个任务、54 次骨架折拟合、两个固定种子，另对通过 OOF 的候选进行验证与额外切分。模型学习定向化学键消息，按任务进行原子注意力与多视图融合，使用独立任务输出头。辅助任务仅使用已授权的 train 数据，并排除目标留出分子和骨架。', '',
        '全部 GPU 训练在远程完成。五个契约测试通过，随后独立重载发布模型复算预测，并重算逐折、验证与重复切分的低相似度主/辅助指标。未通过的任务保留 v9 原始预测。', '',
        '## 相对已实测 v7 的验证集结果', '',
        '|任务|指标|v7|v10|', '|---|---|---:|---:|']
    for r in rows:
        if r['task'] in changed:lines.append(f"|{r['task']}|{r['metric']}|{r['v7_valid']:.6f}|{r['v10_valid']:.6f}|")
    lines+=['',f'累计改变 {len(changed)} 项任务，其余 {22-len(changed)} 项的预测字符串与 v7 完全一致。', '',
        '## 局部回退与统计限制', '',
        '所有这些结果来自反复使用的验证数据，存在选型偏差。官方 valid 主指标不下降，但骨架折和低相似度子组允许预先规定的小幅波动；不能保证每个指标都上升，也不能保证盲测总分提升。低覆盖子组未被证实安全。', '',
        'CYP2D6 底物任务本轮在重复切分中 AUPRC 提升，但 AUROC 从 0.800786 降至 0.793906；属于预设辅助指标容差内回退。VDss 某一折低相似度组 Spearman 从 0.610016 降至 0.600946，也接近但未超过 0.01 容差。PPBR 官方 valid 的 MAE 从 10.647318 降至 10.642767，但 Spearman 从 0.701343 降至 0.698801；重复切分 MAE 从 8.391696 降至 8.295546。此前采用的 DILI 子组回退和条件置信区间仍见 v9_delivery_report.md，未因新版本而消失。', '',
        '条件骨架 bootstrap 区间记录于 v10_uncertainty.csv：CYP2D6 底物 AUPRC 增益约 [-0.01113, 0.10747]，PPBR MAE 改善约 [-0.11098, 0.11088]，VDss Spearman 增益约 [-0.01040, 0.02476]。三者均跨过零，未校正选型偏差，不能称为已证实显著改善，更不能转换为平台总分区间。当前不能宣布达到 51.584929。', '',
        '## 文件', '',f'文件：{final.resolve()}', '',f'SHA256：{local["sha256"]}', '',
        '16,379 行、22 任务、6 列，UTF-8 BOM、CRLF、唯一表头。保留已成功文件的元数据和行序。继续使用上次成功的整文件上传方法；若刷新网页，需要重新设置临时分片阈值。没有自动上传。', '',
        '平台评分反馈将决定下一步是否继续模型迭代。没有按历史小幅增益线性外推总分。', '']
    Path('admet_diagnostics/v10_delivery_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(evidence,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
