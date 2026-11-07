"""Verify received remote evidence and build a task-level v385 update, no fits."""
from pathlib import Path
import csv,hashlib,io,json,tarfile
import numpy as np
from scipy.stats import spearmanr
from repack_upload_template import field_spans
from verify_submission import verify
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/regression9_v387'
PARTS=('fold0','fold1','fold2','valid')
TASKS=('caco2_wang','ld50_zhu','lipophilicity_astrazeneca','ppbr_az','solubility_aqsoldb','clearance_hepatocyte_az','clearance_microsome_az','half_life_obach','vdss_lombardo')
NAMES=('Caco2','LD50','脂溶性','PPBR','溶解度','肝细胞清除率','肝微粒体清除率','半衰期','VDss')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(b):return list(csv.reader(io.StringIO(b.decode('utf-8-sig'),newline='')))
def main():
 receipt=read(EXP/'result_archive.json');archive=ROOT/'releases/v387_remote_results.tar.gz';assert receipt['status']=='PASS'and sha(archive)==receipt['sha256']
 with tarfile.open(archive)as tf:
  seen=set()
  for member in tf.getmembers():
   path=(EXP/member.name).resolve();assert member.isfile()and path.is_relative_to(EXP.resolve())and member.name not in seen;seen.add(member.name)
   b=tf.extractfile(member).read()
   if path.exists():assert path.read_bytes()==b,member.name
   else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
 manifest=read(EXP/'result_manifest.json');sig=manifest['signature'];assert sig==receipt['signature']
 for r in manifest['files']:assert sha(EXP/r['path'])==r['sha256']
 inputs=read(EXP/'data_manifest.json');base=inputs['baseline'];assert inputs['baseline_csv']=='v385_baseline.csv';assert sha(EXP/'v385_baseline.csv')==inputs['baseline_csv_sha256']
 for n,k in (('official_rows.npz','official_rows_sha256'),('expert_features.npz','expert_features_sha256'),('stereo_keys.npz','stereo_keys_sha256')):assert sha(EXP/n)==inputs[k]
 for r in inputs['controls']:
  for k in ('control','design'):assert sha(EXP/r[k])==r[k+'_sha256']
 for r in read(EXP/'source_freeze.json')['files']:assert sha(EXP/r['name'])==r['sha256']==sha(ROOT/'tools'/r['name'])
 verified=read(EXP/'outputs/independent_verification.json');assert verified['status']=='PASS'and verified['signature']==sig
 assert sum(9 for r in verified['curve_records']if r['part']in PARTS[:2])==324
 dev=read(EXP/'outputs/development.json');conf=read(EXP/'outputs/confirmation.json');assert dev['signature']==conf['signature']==sig
 cf={r['task']:r for r in conf['records']};data=np.load(EXP/'official_rows.npz');audit=[];accepted=[];ablations=[]
 for r in dev['records']:
  t=r['task'];best=r['best'];c=cf.get(t);parts=[]
  for p in PARTS:
   if p in PARTS[2:]and c is None:continue
   a=np.load(EXP/'controls'/t/(p+'.npz'));z=np.load(EXP/'outputs/tasks'/t/p/best['config']['name']/'predictions.npz')
   for k in('fit','query'):np.testing.assert_array_equal(a[k],z[k])
   y=data['y'][a['query']];pred=(1-best['weight'])*a['p']+best['weight']*z['p'];old=np.abs(y-a['p']).mean();new=np.abs(y-pred).mean()
   item=dict(part=p,baseline=float(old),candidate=float(new),gain=float((old-new)/old))
   if t in TASKS[5:]:item['rank_gain']=float(spearmanr(y,pred).statistic-spearmanr(y,a['score_p']).statistic)
   report=next(x for x in(c['confirmation']if c else best['parts'])if x['part']==p)
   assert abs(item['gain']-report['gain'])<1e-12
   if 'rank_gain'in item:assert abs(item['rank_gain']-report['rank_gain'])<1e-12
   parts.append(item)
  positive=len(parts)==4 and all(x['gain']>0 and x.get('rank_gain',1)>0 for x in parts)
  strict=bool(positive and min(x['gain']for x in parts[:2])>.012 and np.mean([x['gain']for x in parts])>.012 and all(b['lower_positive']for b in c['bootstrap']))
  if c:assert c['experimental_eligible']==positive and c['strict_accepted']==strict
  if positive:accepted.append(c)
  audit.append(dict(task=t,best=best['config'],weight=best['weight'],parts=parts,accepted=positive,strict_accepted=strict,
   rejection_reasons=[] if positive else ([x['part']+':MAE不改善'for x in parts if x['gain']<=0]+[x['part']+':Spearman不改善'for x in parts if x.get('rank_gain',1)<=0]+(['开发门槛未过，未运行确认']if len(parts)<4 else[]))))
  byround=[]
  for horizon in(0,2,5,8):
   choices=[x for x in r['choices']if x['config']['round']==horizon];b=max(choices,key=lambda x:(x['objective'],-x['weight']))
   byround.append(dict(round=horizon,objective=b['objective'],config=b['config'],weight=b['weight'],gains=[x['gain']for x in b['parts']]))
  ablations.append(dict(task=t,by_round=byround,no_reselection=True))
 means={p:sum(next(x['gain']for x in r['confirmation']if x['part']==p)for r in accepted)/9 for p in PARTS}
 five={p:sum(next(x['gain']for x in r['confirmation']if x['part']==p)for r in accepted if r['task']in TASKS[:5])/5 for p in PARTS}
 result=dict(version='v387',status='NO_ELIGIBLE_CANDIDATE',baseline=base,signature=sig,official_best=51.308,new_official_score=None,
  csv=None,accepted_tasks=[r['task']for r in accepted],task_evidence=audit,
  nine_task_mean_relative_MAE_gain_by_part=means,five_scored_task_mean_relative_MAE_gain_by_part=five,
  goal_mean_5percent_achieved_local=all(x>=.05 for x in means.values()),five_scored_goal_achieved_local=all(x>=.05 for x in five.values()),
  full_development_solves=324,all_solves_verified=verified['models'],model_replay_max_error=verified['full_replay_max_error'],
  all_fitting_remote=True,strict_stability_confirmed=bool(accepted and all(r['strict_accepted']for r in accepted)),automatic_platform_submission=False,
  limitation='Repeatedly used public partitions, historical controls partly proxy. Four ranking MAEs use calibrated references. Positive direction is not proof of hidden-test improvement.')
 if accepted:
  raw=(EXP/'v385_baseline.csv').read_bytes();tab=table(raw);spans=field_spans(raw);replacements={};fusion=[]
  for r in accepted:
   t=r['task'];a=np.load(EXP/'controls'/t/'final.npz');z=np.load(EXP/'outputs/tasks'/t/'final'/r['config']['name']/'predictions.npz')
   for k in('fit','query'):np.testing.assert_array_equal(a[k],z[k])
   loc=[i for i,row in enumerate(tab)if i and row[0]==t];original=np.array([float(tab[i][4])for i in loc]);np.testing.assert_allclose(original,a['score_p'],rtol=0,atol=1e-10)
   assert np.isnan(data['y'][a['query']]).all()and [tab[i][3]for i in loc]==data['smiles'][data['mi'][a['query']]].tolist()
   w=r['weight'];pred=(1-w)*a['p']+w*z['p'];assert np.isfinite(pred).all()
   for i,v in zip(loc,pred):replacements[i]=format(float(v),'.17g').encode('ascii')
   fusion.append(dict(task=t,config=r['config'],weight=w,strict_accepted=r['strict_accepted'],rows=len(loc)))
  chunks=[];cursor=0
  for i,v in sorted(replacements.items()):lo,hi=spans[i][4];chunks.extend([raw[cursor:lo],v]);cursor=hi
  chunks.append(raw[cursor:]);payload=b''.join(chunks);folder=ROOT/'admet_submissions/v387_v385_correlated_metric_experimental';folder.mkdir(exist_ok=True)
  path=folder/'我从山北来带着兰花草-小分子药物预测技术创新赛-20261106.csv'
  if path.exists():assert path.read_bytes()==payload
  else:path.write_bytes(payload)
  check=verify(path,ROOT/'admet_official/submission_template.csv');changed={}
  for i,(old,new)in enumerate(zip(tab,table(payload))):
   assert old[:4]+old[5:]==new[:4]+new[5:]
   if i not in replacements:assert old==new
   if old!=new:changed[old[0]]=changed.get(old[0],0)+1
  result.update(status='VERIFIED_EXPERIMENTAL_DELIVERY',csv=path.relative_to(ROOT).as_posix(),csv_sha256=sha(path),fusion=fusion,changed_rows=changed,unchanged_tasks=22-len(accepted),format=check)
  save(path.with_suffix('.verification.json'),result)
 save(ROOT/'docs/v387_delivery.json',result);save(ROOT/'docs/v387_learning_curves.json',dict(signature=sig,ablations=ablations,all_fixed_horizons_completed=True,no_confirmation_reselection=True))
 lines=['# v387 结果：完整 v385 基线','','官方已知 v385 为51.308，新实验分数待用户提交。所有拟合在指定25870远端。',
  '','|任务|配置|权重|fold0 MAE降幅|fold1|fold2|valid|结论|','|---|---|---:|---:|---:|---:|---:|---|']
 for r in audit:
  gains={x['part']:f"{x['gain']:+.3%}"for x in r['parts']};label='严格通过'if r['strict_accepted']else'实验采用'if r['accepted']else'确认未通过'if len(r['parts'])==4 else'开发未通过'
  lines.append('|'+ '|'.join([NAMES[TASKS.index(r['task'])],r['best']['name'],str(r['weight']),*[gains.get(p,'未运行')for p in PARTS],label])+'|')
 lines+=['','未采用原因：']
 for r in audit:
  if r['rejection_reasons']:lines.append('- '+NAMES[TASKS.index(r['task'])]+'：'+'；'.join(r['rejection_reasons']))
 lines+=['','324次开发求解全部完成，再固定唯一候选作确认；没有使用确认结果改选候选。独立重放、查询标签扰动、输入/源码/模型哈希通过。',
  '',f'9项平均相对MAE降幅：{means}。5项直接MAE计分平均：{five}。拒绝项计零。',
  '', '9项平均5%目标已达到：'+str(result['goal_mean_5percent_achieved_local'])+'。',
  '', '实验提交：`'+str(result['csv'])+'`。新官方成绩未知。',
  '',result['limitation'],'','本轮独立比较完整v385，不与其他未评分候选叠加。']
 (ROOT/'docs/V387_RESULTS_AND_HANDOFF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items()if k not in('task_evidence','format','baseline')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
