"""Freeze completed expensive-source experiments; fast source covers all 22 tasks."""
import json
import admet_v24_models as b
import admet_v33_run as r
assert str(r.ROOT)==r.P['remote_root']
parts=[json.loads((r.OUT/f'analysis_shard{j}.json').read_text()) for j in range(3)]
assert all(p['signature']==r.signature() for p in parts)
records={x['task']:x for p in parts for x in p['tasks']}
template=__import__('pandas').read_csv(r.ROOT/'admet_official/submission_template.csv')
for t in r.P['tasks']:
    if t not in records:
        records[t]=dict(task=t,metric=template.loc[template.task.eq(t),'metric'].iloc[0],selection=None,grid=[],
            rejection='Expensive source search ended before this task completed; v35 covers all tasks. Existing unfinished predictions excluded.')
ordered=[records[t] for t in r.P['tasks']]
b.save(r.OUT/'analysis.json',dict(signature=r.signature(),status='EVALUATED',baseline='v32',baseline_score=50.399027,
    tasks=ordered,selected_tasks=[x['task'] for x in ordered if x['selection']],official_score_prediction=None,
    completed_expensive_task_count=sum('screen_gain' in x for x in ordered),
    search_budget_note='Compute reassigned to full 22-task XGBoost source and selected final refits. Unfinished v33 candidates excluded.',
    limitations=r.P['limitations']))
print(json.dumps(dict(selected=[x['task'] for x in ordered if x['selection']],completed_tasks=sum('screen_gain' in x for x in ordered))))
