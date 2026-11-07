import json, pathlib, statistics
r=pathlib.Path('.')
x=json.loads((r/'docs/platform_scorecard_v111_20261027.json').read_text(encoding='utf-8'))
x.update(version='v114',score=51.091,submission_time_user_reported='2026/10/27 09:03',source='User-pasted official task scorecard')
updates={'bbb_martins':(50.032316,0.9289790494058787),'hia_hou':(50.143410,1.0),'herg':(49.677228,0.8522827687776141)}
for t in x['tasks']:
 if t['task'] in updates:t['score'],t['raw']=updates[t['task']]
x['task_score_mean']=statistics.mean(t['score'] for t in x['tasks'])
(r/'docs/platform_scorecard_v114_20261027.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('v114 task mean',x['task_score_mean']);print('to52 task gain sum',22*(52-x['task_score_mean']))
print('AMES v112 confirmed delta',50.059239-49.980787,'total contribution',(50.059239-49.980787)/22)
