"""Pin the user-reported v45 result before starting v46."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    file = next((ROOT/'admet_submissions/v45').glob('*.csv'))
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    assert digest == '49ffd97565da895035f407decedc30664c9efb3e3cf4e786e8eb8dd1d341effb'
    record = dict(version='v45', reported_score=50.682048, previous_score=50.582708,
        observed_display_gain=0.099340, submitted_sha256=digest, reported_on='2026-10-15',
        source='User report; platform file hash and latest/best display semantics unavailable',
        aggregation_formula='unknown', official_per_task_scores=None,
        leader_score_last_reported=52.723518)
    write(file.parent/'platform_score.json', record)
    history_path = ROOT/'admet_diagnostics/platform_score_history.json'
    history = json.loads(history_path.read_text(encoding='utf-8'))
    for version, score, previous in [('v44',50.582708,50.498381),('v45',50.682048,50.582708)]:
        if not any(r.get('version') == version for r in history):
            history.append(record if version=='v45' else dict(version=version,reported_score=score,
                previous_score=previous,source='User report from conversation',reported_on='2026-10-15'))
    write(history_path, history)
    write(ROOT/'admet_diagnostics/platform_score.json', record)
    state_path = ROOT/'admet_current_candidate.json'
    state = json.loads(state_path.read_text(encoding='utf-8'))
    state.update(recommended_release='v45',submission=str(file),sha256=digest,
        platform_anchor_release='v45',platform_anchor_score=50.682048,
        candidate_platform_score=50.682048,latest_displayed_score=50.682048,
        observed_display_gain=0.099340,report='admet_diagnostics/v45_report.md',
        experimental_release='v46',experimental_status='PREPARING',
        ready_for_platform_evaluation=False,awaiting='Remote task-adaptive graph fine-tuning and corrected baseline comparison')
    state.pop('experimental_submission',None)
    state.pop('experimental_sha256',None)
    write(state_path,state)
    report_path = ROOT/'admet_diagnostics/v45_report.md'
    report = report_path.read_text(encoding='utf-8')
    if '50.682048' not in report:
        report_path.write_text(report+'\n## 平台反馈 2026-10-15\n\n用户报告新分数50.682048，较v44增加0.099340；按最近交付v45关联，平台文件哈希及历史最高分显示语义未核实。\n',encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False))

if __name__=='__main__':main()
