"""Verify remote evidence, fuse whole endpoints, preserve all other v364 bytes."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv, hashlib, io, json, tarfile
import numpy as np
from repack_upload_template import field_spans
from verify_submission import verify
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/transformer_stage3_v368'
OUT=EXP/'outputs'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rows(raw):return list(csv.reader(io.StringIO(raw.decode('utf-8-sig'),newline='')))
def main():
 manifest=read(EXP/'result_manifest.json');confirmation=read(OUT/'confirmation.json');frozen=read(OUT/'frozen_selection.json');protocol=read(OUT/'protocol.json');checked=read(OUT/'independent_verification.json');data=read(EXP/'data_manifest.json');development=read(OUT/'development.json')
 sig=protocol['signature'];assert all(x['signature']==sig for x in (manifest,confirmation,frozen,checked,development));assert checked['status']=='PASS' and confirmation['status']=='COMPLETE'
 for x in manifest['files']:assert sha(EXP/x['path'])==x['sha256'],x['path']
 assert sha(ROOT/'tools/v368_graph_transformer.py')==protocol['source_sha256'] and sha(ROOT/'tools/v364_graph_transformer.py')==protocol['architecture_source_sha256']
 raw=(EXP/'v364_baseline.csv').read_bytes();assert sha(EXP/'v364_baseline.csv')==data['baseline']['csv_sha256'];table=rows(raw);spans=field_spans(raw)
 selected=[r for r in confirmation['records'] if r['strict_accepted'] or r['experimental_eligible']]
 summary=['# 第三阶段 AMES、微粒体清除率、半衰期、VDss 结果','',
  '对照：v364，用户报告官方总分51.269。所有GPU训练均在指定远端RTX3090上完成。',
  '复用6层原子/键图Transformer，重新训练排除四项查询骨架的编码器；比较从零、自监督、相关任务迁移，双固定种子。',
  'AMES解冻末两层，BCE加成对排序；三项Spearman任务冻结主干并训练rank8适配器，拟合集Gaussian-rank加成对排序。','',
  '|任务|开发最优模式|权重|fold0指标增量|fold1指标增量|fold2指标增量|valid指标增量|结论|',
  '|---|---|---:|---:|---:|---:|---:|---|']
 for d in development['records']:
  t=d['task'];b=d['best'];c=next((r for r in confirmation['records'] if r['task']==t),None);parts=c['confirmation'] if c else b['parts']
  gains={r['part']:f"{r['gain']:+.6f}" for r in parts};label='严格通过' if c and c['strict_accepted'] else '实验候选，稳定性未确认' if c and c['experimental_eligible'] else '确认拒绝' if c else '开发拒绝'
  summary.append('|'+ '|'.join([t,b['mode'],str(b['weight']),*[gains.get(p,'未运行') for p in ('fold0','fold1','fold2','valid')],label])+'|')
 summary+=['','模型和融合权重在fold0/1选定后冻结，确认失败时不换配方。',
  '部分历史集成OOF仍为代理对照。长期复用公开验证标签，局部增益不保证隐藏测试提分。原严格拒绝记录全部保留。']
 for r in confirmation['records']:
  for b in r['bootstrap']:
   
   if b['ci90'] is None:summary.append(f"{r['task']} {b['part']}：{b['groups']}个骨架组，无法建立骨架bootstrap区间；严格验收不通过。")
   else:
    lo,hi=b['ci90'];summary.append(f"{r['task']} {b['part']}：{b['groups']}个骨架组，指标增量90%区间[{lo:+.6f}, {hi:+.6f}]。")
 if not selected:
  summary+=['','本轮没有可采用候选，未制造新提交文件，继续保留v364（51.269）。','']
  result=dict(status='NO_ELIGIBLE_CANDIDATE',signature=sig,baseline='v364',baseline_official_score=51.269,official_score=None,csv=None,new_submission_created=False,no_selection_fallback=True,training_gpu_remote_only=True)
  save(ROOT/'docs/v368_stage3_delivery.json',result);(ROOT/'docs/V368_STAGE3_RESULTS.md').write_text('\n'.join(summary),encoding='utf-8');print(json.dumps(result));return
 replace={};fusion=[]
 for r in selected:
  t=r['task'];p=np.load(OUT/'tasks'/t/'final'/r['mode']/'predictions.npz');a=np.load(EXP/'controls'/t/'final.npz');assert str(p['signature'])==sig
  np.testing.assert_array_equal(p['fit'],a['fit']);np.testing.assert_array_equal(p['query'],a['query']);assert r['weight']==next(s['weight'] for s in frozen['selected'] if s['task']==t)
  loc=[i for i,x in enumerate(table) if i and x[0]==t];old=np.array([float(table[i][4]) for i in loc]);np.testing.assert_array_equal(old,a['p']);
  from scipy.stats import rankdata
  values=(1-r['weight'])*rankdata(old)/len(old)+r['weight']*rankdata(p['p'])/len(old)
  assert len(loc)==len(values) and np.isfinite(values).all()
  for i,v in zip(loc,values):replace[i]=format(float(v),'.17g').encode('ascii')
  fusion.append(dict(task=t,mode=r['mode'],weight=r['weight'],strict_accepted=r['strict_accepted'],experimental_only=not r['strict_accepted'],rows=len(values),expert_sha256=sha(OUT/'tasks'/t/'final'/r['mode']/'predictions.npz')))
 cursor=0;pieces=[];changed={}
 for i,value in sorted(replace.items()):
  lo,hi=spans[i][4];pieces.extend((raw[cursor:lo],value));cursor=hi
  if value!=raw[lo:hi]:changed[table[i][0]]=changed.get(table[i][0],0)+1
 pieces.append(raw[cursor:]);payload=b''.join(pieces);assert changed
 label='experimental' if any(not r['strict_accepted'] for r in selected) else 'strict';dest=ROOT/f'admet_submissions/v368_graph_transformer_{label}';dest.mkdir(parents=True,exist_ok=True)
 date=datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d');file=dest/f'我从山北来带着兰花草-小分子药物预测技术创新赛-{date}.csv'
 if file.exists():assert file.read_bytes()==payload
 else:file.write_bytes(payload)
 fmt=verify(file,ROOT/'admet_official/submission_template.csv');new=rows(payload);newspans=field_spans(payload);back=[];cursor=0
 for i in range(1,len(table)):
  assert new[i][:4]+new[i][5:]==table[i][:4]+table[i][5:]
  if i not in replace:assert new[i]==table[i]
  else:assert new[i][4].encode('ascii')==replace[i]
  lo,hi=newspans[i][4];a,b=spans[i][4];back.extend((payload[cursor:lo],raw[a:b]));cursor=hi
 back.append(payload[cursor:]);assert b''.join(back)==raw
 result=dict(status='VERIFIED_EXPERIMENTAL_DELIVERY' if label=='experimental' else 'VERIFIED_STRICT_DELIVERY',signature=sig,
  csv=file.relative_to(ROOT).as_posix(),csv_sha256=sha(file),baseline_version='v364',baseline_official_score=51.269,baseline_sha256=sha(EXP/'v364_baseline.csv'),
  official_score=None,official_gain_established=False,automatically_uploaded=False,training_gpu_remote_only=True,training_host=protocol['training_host'],
  strict_tasks=confirmation['strict_tasks'],experimental_tasks=confirmation['experimental_tasks'],changed_rows=changed,unchanged_tasks=22-len(changed),all_nonprediction_bytes_exact=True,all_unchanged_predictions_exact=True,fusion=fusion,format=fmt,
  model_replay_max_error=checked['full_replay_max_error'],query_label_perturbation_checked=True,no_confirmation_fallback=True,all_full_weights_retained_remote=True)
 summary+=['',f'本次生成{label}提交，仅更新 '+', '.join(changed)+'。其余任务沿用v364。',
  '文件的官方分数未知，需手动提交测试，不能称为新最高分。',
  '16,379行、22项、6列格式核验通过，非预测字节与未变任务预测字符串完全保留；模型重载、CPU参考、注意力计算、查询标签扰动和融合算术检查通过。',
  '训练检查点保留于指定远端，最终权重路径和哈希见结果清单。','',f'CSV SHA256：{sha(file)}','']
 note=dest/'第三阶段实验说明.md';note.write_text('\n'.join(summary),encoding='utf-8');(ROOT/'docs/V368_STAGE3_RESULTS.md').write_text('\n'.join(summary),encoding='utf-8')
 save(ROOT/'docs/v368_stage3_delivery.json',result);save(OUT/'delivery.json',result);save(file.with_suffix('.verification.json'),result)
 members=[file,note,file.with_suffix('.verification.json'),EXP/'result_manifest.json',EXP/'data_manifest.json',ROOT/'docs/V368_STAGE3_RESULTS.md',ROOT/'docs/v368_stage3_delivery.json']
 members += [p for p in OUT.rglob('*') if p.is_file() and p.suffix not in ('.pt',)]
 members += [ROOT/'tools'/n for n in ('v364_graph_transformer.py','v368_graph_transformer.py','v368_prepare_stage3.py','v368_run_stage3.py','v368_verify_remote.py','v368_collect_remote.py','v368_deliver_stage3.py','v368_verify_local.py','v368_bootstrap.py')]
 members=sorted(set(members));entries=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in members];manifest=dest/'release_manifest.json';save(manifest,dict(signature=sig,files=entries));members.append(manifest)
 archive=ROOT/f'releases/v368_stage3_{date}.tar.gz'
 with tarfile.open(archive,'w:gz') as tf:
  for p in members:tf.add(p,arcname=p.relative_to(ROOT).as_posix())
 with tarfile.open(archive) as tf:
  for r in entries:assert hashlib.sha256(tf.extractfile(r['path']).read()).hexdigest()==r['sha256']
 save(ROOT/'docs/v368_release_archive.json',dict(status='PASS',path=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),files=len(members),bytes=archive.stat().st_size,all_member_hashes_verified=True))
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
