"""Record visible official results without inferring uploaded file identities."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def save(p,value):
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    raw=json.loads((ROOT/'docs/platform_history_edge_20261027.json').read_text(encoding='utf-8'))
    tasks=raw['headers'][2:-3]
    records=[dict(position=int(r[0]),team=r[1],tasks={t:json.loads(r[i+2]) for i,t in enumerate(tasks)},score=float(r[-3]),certification=r[-2],submitted_at=r[-1]) for r in raw['rows']]
    assert len(records)==67 and len(tasks)==22
    baseline=next(r for r in records if r['submitted_at']=='2026/10/27 17:22')
    card=json.loads((ROOT/'docs/platform_scorecard_v145_20261027.json').read_text(encoding='utf-8'))
    assert baseline['score']==card['score']==51.166
    for row in card['tasks']:
        got=baseline['tasks'][row['task']]
        assert (got['metric'],got['raw_metric'],got['score'])==(row['metric'],row['raw'],row['score'])
    bests=[]
    for task in tasks:
        best=max(records,key=lambda r:r['tasks'][task]['score'])
        bests.append(dict(task=task,baseline=baseline['tasks'][task],historical_best=best['tasks'][task],best_submitted_at=best['submitted_at'],best_total=best['score'],task_score_gap=best['tasks'][task]['score']-baseline['tasks'][task]['score'],file_identity_confirmed=False))
    mean=sum(r['baseline']['score'] for r in bests)/len(bests);upper=sum(r['historical_best']['score'] for r in bests)/len(bests)
    summary=dict(source=raw['source'],records=records,baseline=dict(inferred_version='v145',score=baseline['score'],submitted_at=baseline['submitted_at'],all_22_metrics_match_local_scorecard=True,uploaded_bytes_hash_exposed=False),task_bests=bests,
        retrospective_task_max_mean=upper,retrospective_gain_over_baseline_mean=upper-mean,retrospective_mean_is_measured_submission=False,remaining_gap_to_52_even_at_task_maxima=52-upper,
        limitations='Official page exposes submission time and scores, not source filename or bytes. Version mapping requires archived files and cannot be established solely from time. Historical task maxima are not a measured combined submission.')
    save(ROOT/'docs/platform_history_edge_audit_20261027.json',summary)
    p=ROOT/'CURRENT_STATUS.json';state=json.loads(p.read_text(encoding='utf-8-sig'))
    state['automatic_platform_submission']=True
    state['platform_submission_authorization']=dict(source='User explicitly requests automatic submission through installed Edge extension and reporting each score with its corresponding file.',date='2026-10-27',site='http://www.aiinnovation.com.cn/',scope='Verified new ADMET competition CSVs, inspect receipt and official task scorecards.',new_terms_or_captcha_may_require_action=True)
    state['browser_submission_workflow']=dict(browser='Edge',login_verified=True,results_readable=True,file_input_found=True,history_rows=67,latest_visible_score=51.166,latest_visible_time=baseline['submitted_at'],new_file_uploaded=False)
    save(p,state)
    workflow={'browser':'Edge','logged_in':True,'userinfo_url':raw['url'],'entry_url':'http://www.aiinnovation.com.cn/#/trackDetail?id=159','observed_steps':['提交方法','上传','小分子药物预测技术创新赛数据集 row 上传','file chooser on input[type="file"]'],'file_input':'input.el-upload__input[type="file"]','upload_may_begin_on_file_choice':True,'verified_new_candidate_required':True,'authorization':'User explicitly authorized automatic submission of new model CSVs and reading official scores.','before_upload':['Verify CSV bytes, SHA256, original Drug_ID strings, task routing, model replay and local archive.','Save pre-upload official history.','Record version, local path, SHA256 and upload start time before selecting file.'],'after_upload':['Inspect visible upload response.','Read official history and identify newly added timestamp and 22-task scorecard.','Do not associate a score solely by rank or highest total.','If scoring is delayed, record pending.','Save screenshot of receipt/result and return CSV with actual score or pending status.'],'latest_new_upload':None}
    save(ROOT/'docs/edge_submission_workflow.json',workflow)
    print(json.dumps(dict(records=len(records),latest_score=baseline['score'],historical_gaps=[{k:r[k] for k in ('task','task_score_gap','best_submitted_at')} for r in bests if r['task_score_gap']>0]),ensure_ascii=False))
if __name__=='__main__':main()
