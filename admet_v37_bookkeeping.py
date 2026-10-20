"""Persist the reported v36 result and promote the verified file as the anchor."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    dest = ROOT / 'admet_submissions/v36'
    csv = next(dest.glob('*.csv'))
    digest = hashlib.sha256(csv.read_bytes()).hexdigest()
    assert digest == '9f16e51d46f4b36d4c3bd262ca6c9053a8cd91009f66813ab29d9b02ea136ba9'
    record = dict(version='v36', reported_score=50.470960, previous_score=50.399027,
                  observed_display_gain=0.071933, submitted_sha256=digest,
                  reported_on='2026-10-06', source='User report following verified v36 delivery; platform hash unavailable',
                  leader_score_last_reported=52.723518, aggregation_formula='unknown', official_per_task_scores=None)
    write(dest / 'platform_score.json', record)
    path = ROOT / 'admet_diagnostics/platform_score_history.json'
    history = json.loads(path.read_text(encoding='utf-8'))
    if not any(row.get('version') == 'v36' for row in history):
        history.append(record)
    write(path, history)
    write(ROOT / 'admet_diagnostics/platform_score.json', record)
    path = ROOT / 'admet_current_candidate.json'
    state = json.loads(path.read_text(encoding='utf-8'))
    state.update(recommended_release='v36', submission=str(csv), sha256=digest,
                 platform_anchor_release='v36', platform_anchor_score=50.470960,
                 latest_displayed_score=50.470960, observed_display_gain=0.071933,
                 candidate_platform_score=50.470960, experimental_release='v37',
                 experimental_status='TRAINING_NOT_YET_STARTED', ready_for_platform_evaluation=False,
                 awaiting='Remote v37 expanded chemistry and metric-aligned model experiments', target_achieved=False)
    for key in ['experimental_submission', 'experimental_sha256']:
        state.pop(key, None)
    write(path, state)
    path = ROOT / 'admet_diagnostics/v36_report.md'
    report = path.read_text(encoding='utf-8')
    if '## 平台反馈 2026-10-06' not in report:
        path.write_text(report + '\n## 平台反馈 2026-10-06\n\n用户报告 v36 得分 50.470960，较 v32 增加 0.071933。已将 v36 提升为下一轮基线；上文待评测状态为交付时记录。平台没有公开分项得分，不能确定增益来自哪个任务。\n', encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
