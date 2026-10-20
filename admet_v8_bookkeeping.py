"""Record the user's v7 platform result without touching frozen predictions."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    submission = ROOT / 'admet_submissions/v7/我从山北来带着兰花草-小分子药物预测技术创新赛-20260925.csv'
    digest = hashlib.sha256(submission.read_bytes()).hexdigest()
    assert digest == '88e81b9d16e48cb941f18c5fd6c4a49e067c73659de808e896e51dea8de5acf0'
    result = dict(version=7, reported_score=49.343821, maximum_published=65.719642,
                  target=51.584929, gap_to_target=2.241108,
                  leader_score_user_reported=50.584929, previous_score=49.247501,
                  latest_improvement=0.096320, submitted_sha256=digest,
                  source='User reports successful original v7 upload after temporary whole-file setting, and platform score',
                  aggregation_formula='unknown', per_task_platform_scores='user confirmed unavailable',
                  upload_path='whole_file; user executed; server byte stream not independently captured')
    history_path = ROOT / 'admet_diagnostics/platform_score_history.json'
    history = json.loads(history_path.read_text(encoding='utf-8'))
    if not any(x.get('version') == 7 and x.get('reported_score') == result['reported_score'] for x in history):
        history.append(result)
    write(history_path, history)
    write(ROOT / 'admet_diagnostics/platform_score.json', result)
    run_path = ROOT / 'admet_v7_remote_run.json'
    run = json.loads(run_path.read_text(encoding='utf-8'))
    run.update(status='user_submitted_platform_scored', platform_score=49.343821,
               platform_score_source='user reported', recommended_upload=str(submission),
               recommended_upload_sha256=digest,
               recommended_upload_platform_acceptance='user reports successful whole-file upload and score 49.343821')
    write(run_path, run)
    report = ROOT / 'admet_diagnostics/v7_result_report.md'
    text = report.read_text(encoding='utf-8')
    text = text.replace('v7 平台分数尚未知。', '用户随后报告 v7 整文件上传成功，平台成绩为 **49.343821**，比 v6 提高 **0.096320**，距离目标尚差 **2.241108**。')
    report.write_text(text, encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
