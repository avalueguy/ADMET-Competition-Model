"""Assemble verified, whole-task model outputs into exact v361 CSV bytes."""
from __future__ import annotations
from pathlib import Path
import csv, hashlib, io, json, tarfile
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np
from scipy.stats import rankdata
from verify_submission import verify
from repack_upload_template import field_spans

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/transformer_stage1_v364'
OUT=EXP/'outputs'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rows(raw):return list(csv.reader(io.StringIO(raw.decode('utf-8-sig'),newline='')))
def main():
 manifest=read(EXP/'result_manifest.json');verification=read(OUT/'independent_verification.json');confirmation=read(OUT/'confirmation.json');frozen=read(OUT/'frozen_selection.json');protocol=read(OUT/'protocol.json');data=read(EXP/'data_manifest.json')
 signature=protocol['signature'];assert verification['status']=='PASS' and confirmation['status']=='COMPLETE'
 assert all(x['signature']==signature for x in (manifest,verification,confirmation,frozen))
 for r in manifest['files']:assert sha(EXP/r['path'])==r['sha256'],r['path']
 assert sha(ROOT/'tools/v364_graph_transformer.py')==protocol['source_sha256']
 baseline=EXP/'v361_baseline.csv';raw=baseline.read_bytes();assert sha(baseline)==data['baseline']['csv_sha256']
 table=rows(raw);spans=field_spans(raw);assert len(table)==len(spans)==16380
 candidates=[r for r in confirmation['records'] if r['strict_accepted'] or r['experimental_eligible']]
 assert candidates,'No validated change is available; do not relabel the old CSV as an improved submission'
 replacement={};fusion_records=[]
 for r in candidates:
  t=r['task'];chosen=next(x for x in frozen['selected'] if x['task']==t);assert r['mode']==chosen['mode'] and r['weight']==chosen['weight']
  a=np.load(EXP/'controls'/t/'final.npz');p=np.load(OUT/'tasks'/t/'final'/r['mode']/'predictions.npz');assert str(p['signature'])==signature
  np.testing.assert_array_equal(a['query'],p['query']);np.testing.assert_array_equal(a['fit'],p['fit'])
  loc=[i for i,row in enumerate(table) if i and row[0]==t];old=np.array([float(table[i][4]) for i in loc]);np.testing.assert_array_equal(old,a['p'])
  values=(1-r['weight'])*old+r['weight']*p['p'] if r['metric']=='MAE' else (1-r['weight'])*rankdata(old)/len(old)+r['weight']*rankdata(p['p'])/len(old)
  assert len(values)==len(loc) and np.isfinite(values).all()
  for i,v in zip(loc,values):replacement[i]=format(float(v),'.17g').encode('ascii')
  fusion_records.append(dict(task=t,mode=r['mode'],weight=r['weight'],metric=r['metric'],rows=len(loc),strict_accepted=r['strict_accepted'],experimental_only=not r['strict_accepted'],prediction_sha256=sha(OUT/'tasks'/t/'final'/r['mode']/'predictions.npz')))
 chunks=[];cursor=0;changed={}
 for i,value in sorted(replacement.items()):
  lo,hi=spans[i][4];chunks.extend((raw[cursor:lo],value));cursor=hi
  if value!=raw[lo:hi]:changed[table[i][0]]=changed.get(table[i][0],0)+1
 chunks.append(raw[cursor:]);payload=b''.join(chunks);assert changed
 category='experimental' if any(not r['strict_accepted'] for r in candidates) else 'strict'
 dest=ROOT/f'admet_submissions/v364_graph_transformer_{category}';dest.mkdir(parents=True,exist_ok=True)
 date=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d');file=dest/f'我从山北来带着兰花草-小分子药物预测技术创新赛-{date}.csv'
 if file.exists():assert file.read_bytes()==payload,'Refuse overwriting a different release'
 else:file.write_bytes(payload)
 fmt=verify(file,ROOT/'admet_official/submission_template.csv');newrows=rows(payload);newspans=field_spans(payload)
 restore=[];cursor=0
 for i in range(1,len(table)):
  assert newrows[i][:4]+newrows[i][5:]==table[i][:4]+table[i][5:]
  if i not in replacement:assert newrows[i]==table[i]
  else:assert newrows[i][4].encode('ascii')==replacement[i]
  lo,hi=newspans[i][4];oldlo,oldhi=spans[i][4];restore.extend((payload[cursor:lo],raw[oldlo:oldhi]));cursor=hi
 restore.append(payload[cursor:]);assert b''.join(restore)==raw
 result=dict(status='VERIFIED_EXPERIMENTAL_DELIVERY' if category=='experimental' else 'VERIFIED_STRICT_DELIVERY',signature=signature,created_at_utc=datetime.now(timezone.utc).isoformat(),
  csv=file.relative_to(ROOT).as_posix(),csv_sha256=sha(file),baseline_version='v361',baseline_official_score=51.240,baseline_sha256=sha(baseline),
  official_score=None,official_gain_established=False,automatically_uploaded=False,all_training_gpu_remote_only=True,training_host=protocol['training_host'],
  strict_tasks=confirmation['strict_tasks'],experimental_tasks=confirmation['experimental_tasks'],changed_rows=changed,unchanged_tasks=22-len(changed),
  all_nonprediction_bytes_exact=True,all_unchanged_prediction_bytes_exact=True,fusion=fusion_records,format=fmt,
  development_selection_frozen_before_confirmation=True,confirmation_fallback=False,baseline_oof_limitation='Partial historical proxy; current adopted incremental layers exact',
  new_external_weights=False,new_external_data=False,user_authorized_experimental_csv=True)
 save(OUT/'delivery.json',result);save(ROOT/'docs/v364_stage1_delivery.json',result);save(file.with_suffix('.verification.json'),result)
 development=read(OUT/'development.json')
 rejected_dev=[r['task'] for r in development['records'] if r['best']['objective']<=0]
 lines=['# 第一阶段 Graph Transformer 实验交付','',f'对照：v361，用户报告官方总分 51.240。本文件官方分数未知，供手动提交。',
  '所有 GPU 训练均在指定的远端 RTX 3090 上完成；新路线未新增外部数据或外部预训练权重。',
  '6 层、192 维、6 头分子图 Transformer；比较从零训练、拟合集自监督和相关任务迁移，两个固定种子。',
  '每项开发 fold0/1 选择一次模式与融合权重，随后冻结，再检查 fold2 和 valid。','',
  '|任务|模式|融合权重|fold0 增益|fold1 增益|fold2 增益|valid 增益|验收|','|---|---|---:|---:|---:|---:|---:|---|']
 for r in confirmation['records']:
  gains=[x['gain'] for x in r['confirmation']];label='严格通过' if r['strict_accepted'] else '实验候选，稳定性未确认' if r['experimental_eligible'] else '拒绝，未写入提交'
  values=[f'{x:.3%}' if r['metric']=='MAE' else f'{x:+.6f}' for x in gains]
  lines.append('|'+ '|'.join([r['task'],r['mode'],str(r['weight']),*values,label])+'|')
 lines+=['','开发阶段未通过、继续沿用 v361 的任务：'+', '.join(rejected_dev)+'。',
  'MAE 增益是相对误差下降；其他指标为绝对增加。',
  '严格验收要求四分区正增益、实际增益门槛及两个确认分区的骨架 bootstrap 90% 区间下界均大于零。实验候选不表示严格通过。',
  '历史完整集成的部分 OOF 仍是代理对照；公开验证标签已长期使用，局部增益不能保证隐藏测试提分。',
  '未选中任务的预测字符串、全部任务标识、分子字段、行序和非预测字节均与 v361 完全一致。',
  '已重放模型、独立检查注意力计算、CPU 参考输出、查询标签扰动不变性和融合算术，并核验模板。',
  '此次没有达到可承诺 52 分的证据。以官方成绩单反馈决定是否替代 v361。',
  '全部训练检查点保留在指定远端服务器；本地结果清单记录最终检查点路径及 SHA256，交付包包含预测和验证证据。','',f'CSV SHA256：{sha(file)}','']
 note=dest/'第一阶段实验说明.md';note.write_text('\n'.join(lines),encoding='utf-8');(ROOT/'docs/V364_STAGE1_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
 progress=ROOT/'docs/v364_stage1_progress.json'
 old=read(progress) if progress.exists() else {}
 old.update(status='DELIVERED',official_score=None,new_submission_delivered=True,csv=result['csv'],csv_sha256=result['csv_sha256']);save(progress,old)
 # Readable release archive; all development checkpoints remain on the remote host.
 members=[file,note,file.with_suffix('.verification.json'),ROOT/'docs/V364_STAGE1_RESULTS.md',ROOT/'docs/v364_stage1_delivery.json',ROOT/'docs/v364_independent_input_audit.json',EXP/'data_manifest.json',EXP/'result_manifest.json']
 members += [OUT/name for name in ('protocol.json','smoke.json','development.json','frozen_selection.json','confirmation.json','independent_verification.json','delivery.json')]
 members += [ROOT/'tools'/name for name in ('v364_graph_transformer.py','v364_prepare_stage1.py','v364_run_stage1.py','v364_parallel_stage1.py','v364_parallel_confirmation.py','v364_verify_remote.py','v364_collect_remote.py','v364_after_training.py','v364_deliver_stage1.py')]
 for r in candidates:members += sorted((OUT/'tasks'/r['task']/'final'/r['mode']).glob('*'))
 members=sorted(set(p for p in members if p.is_file()))
 entries=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in members]
 release_manifest=dest/'release_manifest.json';save(release_manifest,dict(signature=signature,files=entries));members.append(release_manifest)
 archive=ROOT/f'releases/v364_graph_transformer_stage1_{date}.tar.gz'
 with tarfile.open(archive,'w:gz') as tf:
  for p in members:tf.add(p,arcname=p.relative_to(ROOT).as_posix())
 with tarfile.open(archive) as tf:
  for r in entries:assert hashlib.sha256(tf.extractfile(r['path']).read()).hexdigest()==r['sha256']
 save(ROOT/'docs/v364_release_archive.json',dict(status='PASS',path=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,files=len(members),all_member_hashes_verified=True))
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
