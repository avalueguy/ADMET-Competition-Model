"""Record the user-reported v42 score without changing previous evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    dest = ROOT / 'admet_submissions/v42'
    file = next(dest.glob('*.csv'))
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    assert digest == 'd20a0e8e3b126ce8f8dd492b77d5343e30c35cc16cb0be30af50ba74c795b00d'
    record = dict(version='v42', reported_score=50.494738, previous_score=50.490698,
        observed_display_gain=0.004040, submitted_sha256=digest, reported_on='2026-10-08',
        source='User report following verified v42 delivery; platform file hash unavailable',
        leader_score_last_reported=52.723518, aggregation_formula='unknown', official_per_task_scores=None)
    write(dest / 'platform_score.json', record)
    path = ROOT / 'admet_diagnostics/platform_score_history.json'
    history = json.loads(path.read_text(encoding='utf-8'))
    if not any(x.get('version') == 'v42' for x in history):
        history.append(record)
    write(path, history)
    write(ROOT / 'admet_diagnostics/platform_score.json', record)
    path = ROOT / 'admet_current_candidate.json'
    state = json.loads(path.read_text(encoding='utf-8'))
    state.update(recommended_release='v42', submission=str(file), sha256=digest,
        platform_anchor_release='v42', platform_anchor_score=50.494738, latest_displayed_score=50.494738,
        observed_display_gain=0.004040, candidate_platform_score=50.494738, experimental_release='v43',
        experimental_status='PREPARING', ready_for_platform_evaluation=False, target_achieved=False,
        awaiting='Remote v43 model experiment')
    for key in ['experimental_submission', 'experimental_sha256', 'equivalent_internal_candidate']:
        state.pop(key, None)
    write(path, state)
    path = ROOT / 'admet_diagnostics/v42_report.md'
    content = path.read_text(encoding='utf-8')
    if '## 平台反馈 2026-10-08' not in content:
        path.write_text(content + '\n## 平台反馈 2026-10-08\n\n用户报告 v42 得分 50.494738，较 v39 提高 0.004040。已作为下一轮确认基线。上文待评测状态为交付时记录；未获得平台分项分数。\n', encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
