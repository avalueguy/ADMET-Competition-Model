"""Record the human-reported platform result without modifying release evidence."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    save(ROOT/'docs/platform_feedback_v101_20261026.json',dict(version='v101',official_score_user_reported=50.932,previous_version='v94',previous_score=50.916,gain=.016,source='user direct report',uploaded_byte_hash_confirmed=False,per_task_scores_available=False,target_score=52.))
    p=ROOT/'CURRENT_STATUS.json';s=json.loads(p.read_text(encoding='utf-8-sig'))
    s.update(best_official_version='v101',best_official_version_user_reported='v101',best_official_score_user_reported=50.932,official_score=50.932,
        latest_completed_model_branch_is_proven_best=True,active_experiment='v102',active_experiment_status='COUNT_MARGIN_PREPARED',target_achieved=False)
    s['releases']['101'].update(official_score=50.932,status='USER_REPORTED_OFFICIAL_BEST')
    s['v101_result'].update(official_score=50.932,status='USER_REPORTED_OFFICIAL_BEST')
    save(p,s)
    p=ROOT/'README.md';text=p.read_text(encoding='utf-8-sig');text=text.replace('# 当前官方最佳 v94（50.916）；v101待平台验证','# 当前官方最佳 v101（50.932）；v102开发中',1)
    text=text.replace('目标52仍未达到。','用户报告v101官方得分50.932，比v94增加0.016；目标52仍未达到。',1);p.write_text(text,encoding='utf-8')
if __name__=='__main__':main()
