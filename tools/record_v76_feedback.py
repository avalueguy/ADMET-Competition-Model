"""Record user-reported scores and withdraw unsuccessful candidates."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    anchor = next((ROOT / 'admet_submissions/v73_v72').glob('*.csv'))
    digest = hashlib.sha256(anchor.read_bytes()).hexdigest()
    assert digest == 'beab85b972b336526a30c8baa6d181552927f0d004ac6360f42ecdbd96399763'
    feedback = dict(reported_date='2026-10-23', source='user direct reports',
        scores={'v73': 50.640, 'v74': 50.630, 'v75': 50.628,
                'v76_aggressive': 50.618, 'v76_conditional': 50.624},
        conservative_name_mapping='User conservative interpreted as delivered v76_conditional',
        uploaded_byte_hashes_independently_confirmed=False,
        tenth_place_last_user_reported=52.2, gap_to_tenth=1.56,
        best='v73', best_csv=anchor.relative_to(ROOT).as_posix(), best_sha256=digest,
        withdrawn=['v74', 'v75', 'v76_aggressive', 'v76_conditional', 'v76_v73'],
        known_implementation_issues=[
            'v76_v73 reapplied experts already contained in the v73 anchor.',
            'v76 final conditional trees hard-coded 120; evaluation and final model not fully aligned.',
            'Historical v57 proxy arrays do not constitute exact v73 OOF.'
        ],
        inference_limit='Aggregate feedback does not identify which task caused a loss.',
        next_experiment='v77: direct ranking vs same-row regression, no assay auxiliary features')
    save(ROOT / 'docs/platform_feedback_v76_20261023.json', feedback)
    status_path = ROOT / 'CURRENT_STATUS.json'
    status = json.loads(status_path.read_text(encoding='utf-8'))
    status.update(best_official_version='v73', best_official_score_user_reported=50.640,
        rollback_baseline='v73', target_achieved=False, active_experiment='v77',
        active_experiment_status='RUNNING_NOT_DELIVERED', tenth_place_last_user_reported=52.2,
        latest_feedback=feedback, recommended_submission_version='v73')
    for name in ('v76_aggressive', 'v76_conditional', 'v76_v73'):
        save(ROOT / 'admet_submissions' / name / 'WITHDRAWN.json', dict(
            status='WITHDRAWN_DO_NOT_SUBMIT', official_score_user_reported=feedback['scores'].get(name),
            restore_version='v73', restore_sha256=digest,
            reason='Below v73 or invalid duplicate-fusion construction; retained for audit.'))
    for name in ('74', '75'):
        if name in status.get('releases', {}):
            status['releases'][name].update(official_score_user_reported=feedback['scores']['v' + name],
                status='WITHDRAWN_BELOW_V73')
    save(status_path, status)
    readme_path = ROOT / 'README.md'
    readme = readme_path.read_text(encoding='utf-8')
    marker = '<!-- current-feedback-v76 -->'
    if marker not in readme:
        readme_path.write_text(marker + '\n# 当前推荐：v73（50.640）\n\n'
            '用户最新反馈：v76 激进版 50.618，保守版 50.624；v75 为 50.628。'
            '这些候选均撤回推荐。第十名最近报告为 52.2，仍差 1.560。'
            'v77 正在比较排序训练与同训练行的回归对照，尚无新提交文件。'
            '详情见 [反馈记录](docs/platform_feedback_v76_20261023.json)。\n\n' + readme,
            encoding='utf-8')
    print(json.dumps(dict(status='RECORDED', best='v73', sha256=digest)))


if __name__ == '__main__':
    main()
