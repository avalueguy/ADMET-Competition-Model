"""Record the user supplied v145 scorecard without claiming a file hash match."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    a=load(ROOT/'docs/platform_scorecard_v130_20261027.json');z=load(ROOT/'docs/platform_scorecard_v144_20261027.json');keep={'ames','cyp2c9_veith','cyp2d6_veith'}
    tasks=[next(x for x in z['tasks'] if x['task']==r['task']) if r['task'] in keep else r for r in a['tasks']]
    # All 22 raw metrics and subtask scores in the latest pasted scorecard match this task routing.
    mean=sum(r['score'] for r in tasks)/22;assert round(mean,3)==51.166
    f=dict(version='v145',date='2026-10-27',submitted_at='2026-10-27 17:22',score=51.166,task_score_mean=mean,tasks=tasks,
        status='user_reported_official',source='User-pasted full 22-task scorecard, metrics match v145 task routing; platform file hash not exposed.',uploaded_byte_hash_confirmed=False,
        submission_sha256=load(ROOT/'docs/v145_local_delivery.json')['sha256'],baseline='v130',delta_task_mean=mean-a['task_score_mean'],summed_task_points_required_for_52=22*(52-mean))
    save(ROOT/'docs/platform_scorecard_v145_20261027.json',f)
    p=ROOT/'CURRENT_STATUS.json';s=load(p);s.update(latest_completed_model_branch='v145',latest_completed_model_branch_is_proven_best=True,best_official_version='v145',best_official_version_user_reported='v145',best_official_score_user_reported=51.166,
        official_score=51.166,last_user_reported_official_version='v145',last_user_reported_official_score=51.166,rollback_baseline='v145',aggregate_status='OFFICIAL_FEEDBACK_RECORDED',active_experiment='v146_to_v151',active_experiment_status='BASELINE_AUDIT_AND_NEW_REPRESENTATIONS')
    s['releases']['145'].update(official_score=51.166,status='USER_REPORTED_OFFICIAL_BEST');s['v145_result'].update(official_score=51.166,status='USER_REPORTED_OFFICIAL_BEST')
    s['latest_user_reported_official_score']=dict(version='v145',score=51.166,previous_version='v144',previous_score=51.099,observed_gain=.067,reported_date='2026-10-27',uploaded_byte_hash_confirmed=False)
    save(p,s);print(json.dumps(f,ensure_ascii=True)[:500])
if __name__=='__main__':main()
