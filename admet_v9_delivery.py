"""Check the cumulative release against the last scored CSV and record delivery evidence.

Standard library only. No model training, platform calls, or prediction changes.
"""
import csv
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    local = read_json('admet_release_v9/local_verification.json')
    assert local['status'] == 'VERIFIED_LOCAL_AND_REMOTE'
    v7 = Path('admet_submissions/v7/我从山北来带着兰花草-小分子药物预测技术创新赛-20260925.csv')
    v9 = Path(local['submission'])
    assert hashlib.sha256(v7.read_bytes()).hexdigest() == '88e81b9d16e48cb941f18c5fd6c4a49e067c73659de808e896e51dea8de5acf0'
    assert hashlib.sha256(v9.read_bytes()).hexdigest() == local['sha256']
    b, a = read_csv(v7), read_csv(v9)
    assert len(a) == len(b) == 16379
    changed = Counter()
    for old, new in zip(b, a):
        assert {k: v for k, v in old.items() if k != 'y_pred'} == {k: v for k, v in new.items() if k != 'y_pred'}
        if old['y_pred'] != new['y_pred']:
            changed[old['task']] += 1
    accepted = set()
    for version in [8, 9]:
        accepted.update(read_json(f'admet_release_v{version}/independent_verification.json')['accepted'])
    assert set(changed) == accepted
    cumulative = []
    rows8 = {r['task']: r for r in read_csv('admet_diagnostics/v8_comparison.csv')}
    rows9 = {r['task']: r for r in read_csv('admet_diagnostics/v9_comparison.csv')}
    assert set(rows8) == set(rows9) == {r['task'] for r in b}
    for task in sorted(rows8):
        p, q = rows8[task], rows9[task]
        for name in ['auroc', 'auprc', 'mae', 'spearman']:
            if p['after_' + name]:
                assert abs(float(p['after_' + name]) - float(q['before_' + name])) < 1e-12
        metric = p['metric']; key = metric.lower()
        before, after = float(p['before_' + key]), float(q['after_' + key])
        gain = before - after if metric == 'MAE' else after - before
        assert gain >= -1e-12
        if task not in accepted:
            assert gain == 0
        cumulative.append(dict(task=task, metric=metric, v7_valid=before, v9_valid=after,
                               favorable_change=gain, changed_prediction_rows=changed[task]))
    out = Path('admet_diagnostics/v9_vs_scored_v7.csv')
    with out.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cumulative[0])); writer.writeheader(); writer.writerows(cumulative)
    audit_count = 0
    for version in [8, 9]:
        for task in accepted:
            folder = Path(f'admet_model_outputs_v{version}') / task
            for path in folder.glob('*source_audit.json'):
                for audit in read_json(path):
                    assert audit['heldout_scaffold_overlap'] == audit['heldout_molecule_overlap'] == 0
                    audit_count += 1
    evidence = dict(status='VERIFIED_CUMULATIVE_V7_TO_V9', submission=str(v9.resolve()), sha256=local['sha256'],
        generated_at=datetime.now(timezone.utc).isoformat(), baseline_version=7, baseline_score_user_reported=49.343821,
        target_score=51.584929, gap_from_last_measured_score=2.241108, new_platform_score=None,
        target_achieved=False, awaiting='User-reported platform score for this exact v9 CSV',
        bytes=v9.stat().st_size, rows=16379, tasks=22, columns=6,
        changed_tasks=sorted(changed), changed_prediction_rows=dict(changed),
        unchanged_tasks=sorted(set(rows8)-set(changed)), original_metadata_unchanged=True,
        nonpromoted_prediction_strings_unchanged=True, valid_primary_declines=0,
        adopted_source_records_audited=audit_count, source_overlap_count=0,
        local_gpu_training=False, automatic_submission=False, remote_training_completed=True,
        caveat='Reused validation has selection bias; some fold/subgroup metrics declined within gates. No blind-score or no-regression guarantee.')
    write_json('admet_release_v9/cumulative_verification.json', evidence)
    run = read_json('admet_v9_remote_run.json')
    run.update(status='complete_verified_not_submitted', sha256=local['sha256'], submission=str(v9.resolve()),
        accepted=local['changed'], cumulative_changed_vs_v7=sorted(changed),
        unchanged_vs_v7=evidence['unchanged_tasks'], remote_contract_tests_passed=3,
        remote_independent_replay=True, local_template_verification=True,
        cumulative_v7_verification=True, training_processes_finished=True,
        awaiting_platform_feedback=True, target_achieved=False)
    write_json('admet_v9_remote_run.json', run)
    lines = ['# v9 交付：与已实测 v7 的累计对比', '',
        '建议只提交 v9 文件。它已合并 v8 的四项改进和 v9 的 DILI 改进，无需依次提交 v8。', '',
        '平台最后实测：49.343821；目标：51.584929；差距：2.241108。v9 尚无平台分数，不能宣布达标。', '',
        '## 完成的修改与训练', '',
        '- v8 增加多尺度分子核集成，以及带任务注意力和共享/私有分支的多任务网络。',
        '- v9 纠正小样本过度重复训练：每轮无放回遍历，35 轮，隐层 64，标签平滑和三个固定种子。',
        '- 两轮共完成 50 个任务/模型候选、150 次骨架折拟合，另完成入围方案的验证、重复切分和最终拟合。GPU 训练均在授权远程服务器。',
        '- 未采用未通过筛选的候选。原始 v7、v8 和重模型检查点保留，重模型留在远程。', '',
        '## 验证集累计结果', '',
        '以下是官方提供的 valid 数据结果，不是平台盲测分项成绩。仅列出采用新预测的五项；其他十七项预测字符串与 v7 一致。', '',
        '|任务|指标|v7|v9|', '|---|---|---:|---:|']
    for row in cumulative:
        if row['task'] in changed:
            lines.append(f"|{row['task']}|{row['metric']}|{row['v7_valid']:.6f}|{row['v9_valid']:.6f}|")
    lines += ['', '## 回退检查与证据限制', '',
        '每个采用方案均检查三折骨架验证、官方 valid、额外骨架切分和低相似度组。官方 valid 主指标要求不下降；各折及重复切分允许预先固定的小幅波动。低相似度门槛检查主指标，覆盖不足的组不能证明安全。', '',
        '这些数据被历史迭代反复使用，存在选择偏差。不能保证盲测总分上涨，也不能保证每个子群和辅助指标都不下降。', '',
        '例如 DILI 一个骨架折的 AUROC 从 0.802232 降至 0.801786；另一折低相似度组的辅助 AUPRC 从 0.875628 降至 0.855106。整体 valid AUROC 和 AUPRC 都改善，重复切分也改善，但这些局部回退仍需如实保留。', '',
        'DILI 的 valid 仅 48 条；骨架 bootstrap 的 AUROC 增益条件区间约 [-0.02024, 0.04609]，跨过零。v8 的 CYP2C9、CYP2D6、PPBR 增益区间也跨过零，因此不能将全部改进称为统计显著。区间未校正选型偏差。', '',
        '## 提交与格式', '',
        f'- 文件：{v9.resolve()}',
        f'- SHA256：{local["sha256"]}',
        '- 16,379 条预测、22 个任务、6 列，UTF-8 BOM、CRLF、唯一表头。元数据和行顺序逐行对齐官方模板；已重载远程保存模型复算预测。',
        '- 继续使用上次成功的整文件上传方式。若网页刷新，临时设置会恢复，需要重新使用 admet_upload_single_request.js；该脚本仅调整页面上传路由，不自动上传。',
        '- 上传后反馈平台分数，才能判断本轮盲测增益及是否达到目标。当前不依据验证集指标编造总分。', '',
        '更多证据：v8_result_report.md、v9_result_report.md、v8_uncertainty.csv、v9_uncertainty.csv，以及 admet_release_v9/cumulative_verification.json。', '']
    Path('admet_diagnostics/v9_delivery_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(evidence, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
