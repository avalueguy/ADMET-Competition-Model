"""Receive either nine-task experiment and construct an explicit experimental CSV."""
from pathlib import Path
import argparse,csv,hashlib,io,json,tarfile
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from scipy.stats import rankdata,spearmanr
from repack_upload_template import field_spans
from verify_submission import verify
ROOT=Path(__file__).resolve().parents[1]
TASKS=('caco2_wang','ld50_zhu','lipophilicity_astrazeneca','ppbr_az','solubility_aqsoldb',
       'clearance_hepatocyte_az','clearance_microsome_az','half_life_obach','vdss_lombardo')


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--version',choices=['377','378','379','380','381'],required=True);args=ap.parse_args();v='v'+args.version
 exp=ROOT/('experiments/regression9_'+v);archive=ROOT/('releases/'+v+'_remote_results.tar.gz');receipt=read(exp/'result_archive.json');assert receipt['status']=='PASS' and sha(archive)==receipt['sha256']
 with tarfile.open(archive) as tf:
  names=set()
  for member in tf.getmembers():
   dest=(exp/member.name).resolve();assert member.isfile() and member.name not in names and dest.is_relative_to(exp.resolve());names.add(member.name)
   data=tf.extractfile(member).read()
   if dest.exists():assert dest.read_bytes()==data,member.name
   else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
 manifest=read(exp/'result_manifest.json');sig=manifest['signature'];assert sig==receipt['signature']
 for r in manifest['files']:assert sha(exp/r['path'])==r['sha256'],r['path']
 data_manifest=read(exp/'data_manifest.json')
 for name,key in (('official_rows.npz','official_rows_sha256'),('expert_features.npz','expert_features_sha256'),('stereo_keys.npz','stereo_keys_sha256')):
  assert sha(exp/name)==data_manifest[key],name
 for row in data_manifest['controls']:
  for key in ('control','design'):assert sha(exp/row[key])==row[key+'_sha256'],row[key]
 sources=['v371_mae_experts.py','v377_regression_experts.py']+(['v377_run_experts.py','v377_verify_experts.py'] if v=='v377' else ['v378_retrieval_transformer.py','v378_run_experts.py','v378_verify_experts.py'] if v=='v378' else ['v379_distribution_transformer.py','v379_run_experts.py','v379_verify_experts.py'] if v=='v379' else ['v380_histogram_median.py','v380_run_experts.py','v380_verify_experts.py'] if v=='v380' else ['v380_histogram_median.py','v381_oof_pareto.py','v381_run_experts.py','v381_verify_experts.py'])
 for name in sources:assert sha(ROOT/'tools'/name)==sha(exp/name),name
 checked=read(exp/'outputs/independent_verification.json');assert checked['status']=='PASS' and checked['signature']==sig
 development=read(exp/'outputs/development.json');confirmation=read(exp/'outputs/confirmation.json');assert development['signature']==confirmation['signature']==sig
 d=np.load(exp/'official_rows.npz');accepted=[];audit=[];conf={r['task']:r for r in confirmation['records']}
 lines=['# '+v+' 九项回归专家结果','','官方基线：用户显示51.271；关联v374本地文件。全部拟合仅指定25870远端CUDA服务器完成。','',
  '|任务|开发最优|权重|fold0 MAE降幅|fold1 MAE降幅|fold2 MAE降幅|valid MAE降幅|结果|',
  '|---|---|---:|---:|---:|---:|---:|---|']
 for rec in development['records']:
  t=rec['task'];best=rec['best'];c=conf.get(t);parts=[]
  for part in ('fold0','fold1','fold2','valid'):
   if part in ('fold2','valid') and c is None:continue
   a=dict(np.load(exp/'controls'/t/(part+'.npz')));z=np.load(exp/'outputs/tasks'/t/part/best['config']['name']/'predictions.npz')
   for k in ('fit','query'):np.testing.assert_array_equal(a[k],z[k])
   y=d['y'][a['query']];p=(1-best['weight'])*a['p']+best['weight']*z['p'];old=float(np.abs(y-a['p']).mean());new=float(np.abs(y-p).mean())
   item=dict(part=part,gain=(old-new)/old,baseline=old,candidate=new)
   if t in TASKS[5:]:
    official=p if v in ('v379','v380','v381') else (1-best['weight'])*rankdata(a['score_p'])/len(y)+best['weight']*rankdata(z['p'])/len(y)
    item.update(rank_gain=float(spearmanr(y,official).statistic-spearmanr(y,a['score_p']).statistic))
   reported=next(r for r in (c['confirmation'] if c else best['parts']) if r['part']==part)
   assert abs(item['gain']-reported['gain'])<1e-12
   if 'rank_gain' in item:assert abs(item['rank_gain']-reported['rank_gain'])<1e-12
   parts.append(item)
  positive=len(parts)==4 and all(r['gain']>0 and r.get('rank_gain',1)>0 for r in parts)
  strict=positive and min(r['gain'] for r in parts[:2])>.012 and np.mean([r['gain'] for r in parts])>.012 and all(b['lower_positive'] for b in c['bootstrap'])
  if c:assert c['experimental_eligible']==positive and c['strict_accepted']==bool(strict)
  if positive:accepted.append(c)
  gains={r['part']:f"{r['gain']:+.3%}" for r in parts};label='严格通过' if strict else '实验候选' if positive else '确认失败' if c else '开发失败'
  lines.append('|'+ '|'.join([t,best['config']['name'],str(best['weight']),*[gains.get(p,'未运行') for p in ('fold0','fold1','fold2','valid')],label])+'|')
  audit.append(dict(task=t,best=best['config'],weight=best['weight'],parts=parts,accepted=bool(positive)))
 # Full deployment averages include zero changes for every rejected task.
 means={p:float(sum(next(x['gain'] for x in r['confirmation'] if x['part']==p) for r in accepted)/9) for p in ('fold0','fold1','fold2','valid')}
 five_means={p:float(sum(next(x['gain'] for x in r['confirmation'] if x['part']==p) for r in accepted if r['task'] in TASKS[:5])/5) for p in means}
 achieved=all(value>=.05 for value in means.values());five_achieved=all(value>=.05 for value in five_means.values())
 final_states=[]
 for row in manifest['final_models']:
  remote=Path(row['remote_path']);suffix=remote.as_posix().split('/regression9_'+v+'/',1)[1];local=exp/suffix
  assert local.is_file() and sha(local)==row['sha256'] and local.stat().st_size==row['bytes'],str(local)
  final_states.append(dict(path=local.relative_to(ROOT).as_posix(),sha256=sha(local),bytes=local.stat().st_size))
 result=dict(status='NO_ELIGIBLE_CANDIDATE',version=v,signature=sig,baseline='v374_associated',official_best=51.271,new_official_score=None,csv=None,
   accepted_tasks=[r['task'] for r in accepted],nine_task_mean_relative_MAE_gain_by_part=means,five_scored_task_mean_relative_MAE_gain_by_part=five_means,
   goal_mean_5percent_achieved_local=achieved,five_scored_goal_achieved_local=five_achieved,all_fitting_remote=True,models_verified=checked['models'],model_replay_max_error=checked['full_replay_max_error'],final_model_states=final_states,automatic_platform_submission=False,
   limitation='Four ranking task MAE references are fit-label empirical-quantile calibrated score ranks, not recovered historical native predictions. Historical control partly proxy; public labels reused.',task_evidence=audit)
 if accepted:
  raw=(exp/'v374_baseline.csv').read_bytes();assert sha(exp/'v374_baseline.csv')==read(exp/'data_manifest.json')['baseline']['csv_sha256']
  table=list(csv.reader(io.StringIO(raw.decode('utf-8-sig'),newline='')));spans=field_spans(raw);replacements={};fusion=[]
  for r in accepted:
   t=r['task'];a=dict(np.load(exp/'controls'/t/'final.npz'));z=np.load(exp/'outputs/tasks'/t/'final'/r['config']['name']/'predictions.npz')
   for k in ('fit','query'):np.testing.assert_array_equal(a[k],z[k])
   loc=[i for i,row in enumerate(table) if i and row[0]==t];original=np.array([float(table[i][4]) for i in loc]);np.testing.assert_array_equal(original,a['score_p'])
   w=r['weight'];p=(1-w)*a['p']+w*z['p'] if v in ('v379','v380','v381') else (1-w)*original+w*z['p'] if t in TASKS[:5] else (1-w)*rankdata(original)/len(original)+w*rankdata(z['p'])/len(original)
   assert np.isnan(d['y'][a['query']]).all() and np.isfinite(p).all()
   assert [table[i][3] for i in loc]==d['smiles'][d['mi'][a['query']]].tolist()
   for i,value in zip(loc,p):replacements[i]=format(float(value),'.17g').encode('ascii')
   fusion.append(dict(task=t,config=r['config'],weight=w,strict_accepted=r['strict_accepted'],rows=len(loc)))
  cursor=0;chunks=[]
  for i,value in sorted(replacements.items()):lo,hi=spans[i][4];chunks.extend([raw[cursor:lo],value]);cursor=hi
  chunks.append(raw[cursor:]);payload=b''.join(chunks);date=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d')
  dest=ROOT/f'admet_submissions/{v}_regression9_experimental';dest.mkdir(parents=True,exist_ok=True);path=dest/f'我从山北来带着兰花草-小分子药物预测技术创新赛-{date}.csv'
  if path.exists():assert path.read_bytes()==payload
  else:path.write_bytes(payload)
  fmt=verify(path,ROOT/'admet_official/submission_template.csv');new=list(csv.reader(io.StringIO(payload.decode('utf-8-sig'),newline='')))
  for i,(old,row) in enumerate(zip(table,new)):
   assert old[:4]+old[5:]==row[:4]+row[5:]
   if i not in replacements:assert old==row
  result.update(status='VERIFIED_EXPERIMENTAL_DELIVERY',csv=path.relative_to(ROOT).as_posix(),csv_sha256=sha(path),fusion=fusion,format=fmt,unchanged_tasks=22-len(accepted))
  save(path.with_suffix('.verification.json'),result)
 lines+=['',f'采用全部任务后的9项平均相对MAE降幅：{means}。5项官方MAE平均：{five_means}。',
         '拒绝任务计入零改善，没有只对获胜任务求平均。四个分区均达到9项平均5%：'+str(achieved)+'。',
         '四项排序任务的MAE为拟合标签分位数校准参考；该误差基线不是旧模型原物理单位预测。排序提升另行检查。',
         '本轮'+('实验文件：`'+result['csv']+'`，官方成绩未知。' if result['csv'] else '没有合格更新，没有生成新CSV。'),
         '置信区间、单任务原MAE和排序增益见JSON。原严格拒绝保留，不能把实验方向一致误当稳定官方收益。']
 save(ROOT/f'docs/{v}_delivery.json',result);(ROOT/f'docs/V{args.version}_REGRESSION9_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in result.items() if k not in ('task_evidence','format')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
