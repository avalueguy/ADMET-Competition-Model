"""Record human feedback as an overlay; preserve immutable release evidence."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    audit = read(ROOT / 'docs/v107_regression_audit_20261026.json')
    assert audit['status'] == 'AUDIT_COMPLETE' and audit['changed_tasks'] == {'herg': 130}
    feedback = dict(version='v107', official_score_user_reported=50.909,
                    baseline='v101', baseline_official_score_user_reported=50.932,
                    delta=-.023, source='user direct report', uploaded_byte_hash_confirmed=False,
                    delivered_sha256=audit['sha256'], per_task_scores_available=False,
                    status='USER_REPORTED_OFFICIAL_REGRESSION', recommended=False,
                    audit='docs/v107_regression_audit_20261026.json',
                    action='Retain v101 as development anchor; withdraw v107 recommendation.',
                    test_labels_read=False, new_submission_created=False)
    save(ROOT / 'docs/platform_feedback_v107_20261026.json', feedback)
    path = ROOT / 'CURRENT_STATUS.json'
    state = read(path)
    state.update(best_official_version='v101', best_official_version_user_reported='v101',
                 best_official_score_user_reported=50.932, official_score=50.932,
                 last_user_reported_official_version='v107', last_user_reported_official_score=50.909,
                 rollback_baseline='v101', recommended_submission_version='v101',
                 latest_completed_model_branch_is_proven_best=False,
                 active_experiment='v107_regression_audit', active_experiment_status='AUDIT_COMPLETE',
                 target_achieved=False, priority_next_step='Rebuild matched v101 validation controls before selecting further updates; v107 is rejected.')
    state['releases']['107'].update(status=feedback['status'], official_score=50.909,
                                    official_score_user_reported=50.909, recommended=False,
                                    feedback=feedback['audit'])
    for key in ('v107_result', 'active_iteration'):
        if key in state and state[key].get('version') == 'v107':
            state[key].update(status=feedback['status'], official_score=50.909, recommended=False)
    state['v107_feedback'] = feedback
    save(path, state)
    for path in (ROOT / 'docs/v107_local_delivery.json', Path('D:/bili_download/v107_count_difference_boost/v107_delivery.json')):
        delivery = read(path)
        delivery.update(status=feedback['status'], official_score=50.909,
                        official_score_user_reported=50.909, recommended=False,
                        best_official_version='v101', best_official_score=50.932,
                        platform_feedback=feedback)
        save(path, delivery)

    marker = '以下为历史记录；当前状态以本段与CURRENT_STATUS.json为准。'
    path = ROOT / 'README.md'
    tail = path.read_text(encoding='utf-8-sig').split(marker, 1)[1]
    top = '''# 当前官方最佳 v101（50.932）；v107 已退分并撤回推荐

用户报告 v107 得分 50.909，比 v101 低 0.023。v107 只改变 hERG 的 130 条预测，另外 21 个任务不变；开发基线恢复为 v101。本次没有生成或推荐新的提交文件。

独立重建 v101 和 v107 的 hERG 最终预测与交付文件逐值完全一致，CSV 格式检查通过。问题出在候选选择的证据强度：valid 只有 66 行（42 正、24 负），AUROC 增益 0.005456 的区间包含零；所谓四组提升中的两组用于选参，历史确认标签也经过多轮复用，基线仍从历史代理起步。无法由这些结果保证隐藏测试集收益。

50% 秩融合改变了测试预测中 694 / 8,646 对分子的相对排序，最大秩位移为 24.62 个百分点。这说明改动的影响范围比“只改一个任务”更大；该统计只使用预测，不使用测试标签。

后续选版须先重建与候选使用同一训练划分的完整 v101 对照，区分随机和骨架切分，使用预先冻结的比较方案；重划分不能把已反复使用过的标签变为全新盲测数据。暂不依据官方总分尝试权重扫描，也不将回退文件重命名为新版本。

[退分审计](docs/v107_regression_audit_20261026.json) · [官方反馈记录](docs/platform_feedback_v107_20261026.json)

'''
    path.write_text(top + marker + tail, encoding='utf-8')
    print(json.dumps(feedback, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
