"""Persist the user-reported v39 score and promote its verified file."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    dest = ROOT / 'admet_submissions/v39'
    file = next(dest.glob('*.csv'))
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    assert digest == '7f14d29cd40562671a59536c40f4f6e9e9bf7113900323614fc808c4623e7bac'
    record = dict(version='v39', reported_score=50.490698, previous_score=50.470960,
        observed_display_gain=0.019738, submitted_sha256=digest, reported_on='2026-10-08',
        source='User report following verified v39 delivery; platform file hash unavailable',
        leader_score_last_reported=52.723518, aggregation_formula='unknown', official_per_task_scores=None)
    write(dest / 'platform_score.json', record)
    path = ROOT / 'admet_diagnostics/platform_score_history.json'
    history = json.loads(path.read_text(encoding='utf-8'))
    if not any(x.get('version') == 'v39' for x in history):
        history.append(record)
    write(path, history)
    write(ROOT / 'admet_diagnostics/platform_score.json', record)
    path = ROOT / 'admet_current_candidate.json'
    state = json.loads(path.read_text(encoding='utf-8'))
    state.update(recommended_release='v39', submission=str(file), sha256=digest,
        platform_anchor_release='v39', platform_anchor_score=50.490698, latest_displayed_score=50.490698,
        observed_display_gain=0.019738, candidate_platform_score=50.490698, experimental_release='v40',
        experimental_status='PREPARING', ready_for_platform_evaluation=False, target_achieved=False,
        awaiting='Remote v40 expanded chemical neural training')
    for key in ['experimental_submission', 'experimental_sha256', 'equivalent_internal_candidate']:
        state.pop(key, None)
    write(path, state)
    path = ROOT / 'admet_diagnostics/v39_report.md'
    content = path.read_text(encoding='utf-8')
    if '## 平台反馈 2026-10-08' not in content:
        path.write_text(content + '\n## 平台反馈 2026-10-08\n\n用户报告 v39 得分 50.490698，较 v36 提高 0.019738。已作为下一轮确认基线。上文待评测状态为交付时记录；未获得平台分项分数。\n', encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
