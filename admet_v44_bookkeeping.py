"""Record the user-confirmed v43 score and ranking before the next experiment."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    dest = ROOT / 'admet_submissions/v43'
    file = next(dest.glob('*.csv'))
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    assert digest == '1c426ece3cafbeb4294a44b66f11a4b5e0801779123af718d3e5a73831ec6220'
    record = dict(version='v43', reported_score=50.498381, previous_score=50.494738,
        observed_display_gain=0.003643, submitted_sha256=digest, reported_on='2026-10-12', ranking_user_reported=9,
        source='User confirmed v43 score; platform file hash unavailable',
        aggregation_formula='unknown', official_per_task_scores=None, leader_score_last_reported=52.723518)
    write(dest / 'platform_score.json', record)
    path = ROOT / 'admet_diagnostics/platform_score_history.json'
    history = json.loads(path.read_text(encoding='utf-8'))
    if not any(r.get('version') == 'v43' for r in history):
        history.append(record)
    write(path, history)
    write(ROOT / 'admet_diagnostics/platform_score.json', record)
    path = ROOT / 'admet_current_candidate.json'
    state = json.loads(path.read_text(encoding='utf-8'))
    state.update(recommended_release='v43', submission=str(file), sha256=digest, platform_anchor_release='v43',
        platform_anchor_score=50.498381, latest_displayed_score=50.498381, candidate_platform_score=50.498381,
        ranking_user_reported=9, observed_display_gain=0.003643, experimental_release='v44',
        experimental_status='PREPARING', ready_for_platform_evaluation=False,
        awaiting='Remote v44 learned molecular difference experiment')
    for key in ['experimental_submission', 'experimental_sha256']:
        state.pop(key, None)
    write(path, state)
    path = ROOT / 'admet_diagnostics/v43_report.md'
    old = path.read_text(encoding='utf-8')
    if '## 平台反馈 2026-10-12' not in old:
        path.write_text(old+'\n## 平台反馈 2026-10-12\n\n用户确认v43得分50.498381，较v42增加0.003643，排名第9。已固定为下一轮基线；以上待评测描述为交付时记录。\n', encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False))

if __name__ == '__main__':
    main()
