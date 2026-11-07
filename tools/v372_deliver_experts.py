"""Receive expert evidence and create only a validated whole-task numeric blend."""
from pathlib import Path
import hashlib,json,tarfile,csv,io,sys
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from repack_upload_template import field_spans
from verify_submission import verify

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/mae_experts_v372'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def receive():
 receipt=read(EXP/'result_archive.json');path=ROOT/'releases/v372_remote_results.tar.gz';assert receipt['status']=='PASS' and sha(path)==receipt['sha256']
 with tarfile.open(path) as tf:
  names=set()
  for m in tf.getmembers():
   assert m.isfile() and m.name not in names;names.add(m.name);p=(EXP/m.name).resolve();assert p.is_relative_to(EXP.resolve());content=tf.extractfile(m).read()
   if p.exists():assert p.read_bytes()==content,m.name
   else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
 manifest=read(EXP/'result_manifest.json');assert manifest['signature']==receipt['signature']
 for p in manifest['files']:assert sha(EXP/p['path'])==p['sha256'],p['path']
 for n in ('v371_mae_experts.py','v372_crossfit_experts.py','v372_run_experts.py','v372_verify_experts.py'):assert sha(ROOT/'tools'/n)==sha(EXP/n),n
 save(ROOT/'docs/v372_receipt_verification.json',dict(status='PASS',signature=receipt['signature'],files=len(manifest['files']),archive_sha256=sha(path)))
def main():
 receive();out=EXP/'outputs';dev=read(out/'development.json');confirm=read(out/'confirmation.json');checked=read(out/'independent_verification.json');protocol=read(out/'protocol.json');manifest=read(EXP/'data_manifest.json');best=manifest['baseline'];sig=protocol['signature']
 assert all(r['signature']==sig for r in (dev,confirm,checked));assert checked['status']=='PASS' and confirm['status']=='COMPLETE'
 selected=[r for r in confirm['records'] if r['experimental_eligible']];lines=['# 五项MAE独立专家结果','',
  '目标为五项MAE各下降至少10%；对照v364，用户报告官方51.269。本轮全部拟合在端口25870远端RTX3090完成。','',
  '|任务|开发最优专家|权重|fold0 MAE下降|fold1 MAE下降|fold2 MAE下降|valid MAE下降|结果|',
  '|---|---|---:|---:|---:|---:|---:|---|']
 for r in dev['records']:
  b=r['best'];c=next((x for x in confirm['records'] if x['task']==r['task']),None);values=c['confirmation'] if c else b['parts'];g={v['part']:f"{v['gain']:+.3%}" for v in values};label='严格通过' if c and c['strict_accepted'] else '实验候选' if c and c['experimental_eligible'] else '确认失败' if c else '开发失败'
  lines.append('|'+ '|'.join([r['task'],b['config']['name'],str(b['weight']),*[g.get(p,'未运行') for p in ('fold0','fold1','fold2','valid')],label])+'|')
 reached=[r['task'] for r in confirm['records'] if r['ten_percent_all_parts']]
 lines+=['',f'四分区全部达到10%降幅的任务数：{len(reached)}/5。五项10%目标'+('已在本地验证达到；官方未知。' if len(reached)==5 else '未达到。'),
  'MAE下降为相对值，负数表示退步。模型与权重仅由fold0/1选定，确认失败未换配方。',
  '局部改善不能保证官方隐藏集提升；公开标签长期复用，历史集成OOF部分为代理对照。','']
 for r in confirm['records']:
  for b in r['bootstrap']:lines.append(f"{r['task']} {b['part']}：{b['groups']}组，MAE相对下降90%区间{b['ci90']}。")
 result=dict(status='NO_ELIGIBLE_CANDIDATE',signature=sig,baseline='v364',baseline_official_score=51.269,official_score=None,csv=None,
  target_five_tasks_ten_percent_achieved_local=len(reached)==5,ten_percent_all_parts_tasks=reached,approved_tasks=[r['task'] for r in selected],training_gpu_remote_only=True,all_fitting_remote=True,automatic_platform_submission=False,model_replay_max_error=checked['full_replay_max_error'])
 if selected:
  raw=(EXP/'v364_baseline.csv').read_bytes();assert sha(EXP/'v364_baseline.csv')==best['csv_sha256'];table=list(csv.reader(io.StringIO(raw.decode('utf-8-sig'),newline='')));spans=field_spans(raw);replace={};fusion=[]
  data=np.load(EXP/'official_rows.npz')
  for r in selected:
   t=r['task'];a=dict(np.load(EXP/'controls'/t/'final.npz'));z=np.load(out/'tasks'/t/'final'/r['config']['name']/'predictions.npz');assert str(z['signature'])==sig
   for k in ('fit','query'):np.testing.assert_array_equal(a[k],z[k])
   loc=[i for i,row in enumerate(table) if i and row[0]==t];old=np.array([float(table[i][4]) for i in loc]);np.testing.assert_array_equal(old,a['p']);values=(1-r['weight'])*old+r['weight']*z['p']
   assert [table[i][3] for i in loc]==data['smiles'][data['mi'][a['query']]].tolist();assert np.isfinite(values).all() and np.isnan(data['y'][a['query']]).all()
   for i,v in zip(loc,values):replace[i]=format(float(v),'.17g').encode('ascii')
   fusion.append(dict(task=t,config=r['config'],weight=r['weight'],strict_accepted=r['strict_accepted'],experimental_only=not r['strict_accepted'],rows=len(loc)))
  chunks=[];cursor=0
  for i,v in sorted(replace.items()):
   lo,hi=spans[i][4];chunks.extend([raw[cursor:lo],v]);cursor=hi
  chunks.append(raw[cursor:]);payload=b''.join(chunks);date=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d');label='experimental' if any(not r['strict_accepted'] for r in selected) else 'strict';dest=ROOT/f'admet_submissions/v372_mae_experts_{label}';dest.mkdir(parents=True,exist_ok=True);path=dest/f'我从山北来带着兰花草-小分子药物预测技术创新赛-{date}.csv'
  if path.exists():assert path.read_bytes()==payload
  else:path.write_bytes(payload)
  fmt=verify(path,ROOT/'admet_official/submission_template.csv');new=list(csv.reader(io.StringIO(payload.decode('utf-8-sig'),newline='')));newspans=field_spans(payload);cursor=0;restored=[]
  for i in range(1,len(table)):
   assert new[i][:4]+new[i][5:]==table[i][:4]+table[i][5:]
   if i not in replace:assert new[i]==table[i]
   lo,hi=newspans[i][4];alo,ahi=spans[i][4];restored.extend([payload[cursor:lo],raw[alo:ahi]]);cursor=hi
  restored.append(payload[cursor:]);assert b''.join(restored)==raw
  result.update(status='VERIFIED_'+label.upper()+'_DELIVERY',csv=path.relative_to(ROOT).as_posix(),csv_sha256=sha(path),fusion=fusion,format=fmt,unchanged_tasks=22-len(selected),all_nonprediction_bytes_exact=True,all_unchanged_predictions_exact=True)
  lines+=['',f'生成{label}提交：`{path.relative_to(ROOT).as_posix()}`。官方成绩未知。',f'SHA256：`{sha(path)}`。']
  save(path.with_suffix('.verification.json'),result)
 else:lines+=['','无符合条件的新候选，未制造新CSV。保留v364，官方最好51.269。']
 lines+=['','中位数SVR、核岭、GPU CatBoost MAE和XGBoost中位数专家，用拟合集内部骨架OOF学习单纯形权重与理化分区门控；内折独立归一化/列选择。失败记录保留。',f"独立重放核验模型数{checked['models']}，最大重放误差{checked['full_replay_max_error']}。"]
 save(ROOT/'docs/v372_delivery.json',result);save(out/'delivery.json',result);(ROOT/'docs/V372_MAE_EXPERT_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
