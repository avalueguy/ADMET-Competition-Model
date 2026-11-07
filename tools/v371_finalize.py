"""Persist actual outcomes and package reproducible expert evidence."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib,tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 exp=ROOT/'experiments/mae_experts_v371';delivery=read(ROOT/'docs/v371_delivery.json');verification=read(exp/'outputs/independent_verification.json');assert verification['status']=='PASS' and verification['signature']==delivery['signature']
 remote=read(exp/'completion_receipt.json');assert remote['no_training_processes_running']
 best=read(ROOT/'docs/current_best_after_v364_20261104.json');assert sha(ROOT/best['csv'])==best['csv_sha256']
 now=datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat();status=read(ROOT/'CURRENT_STATUS.json')
 progress=dict(status='COMPLETE',completed_at_hong_kong=now,version='v371',training_host='connect.nmb2.seetacloud.com:25870',official_best=51.269,target_five_MAE_reduced_ten_percent=delivery['target_five_tasks_ten_percent_achieved_local'],delivery=delivery,verification=verification['status'],remote_idle=remote,official_new_score=None)
 save(ROOT/'docs/v371_progress.json',progress)
 status.update(latest_completed_model_branch='v371',latest_completed_model_branch_is_proven_best=False,latest_aggregate_plan='v371',aggregate_status=delivery['status'],official_score=51.269,remote_training_running=False,local_gpu_training=False,current_experiment_progress='docs/v371_progress.json',latest_MAE_expert_round=progress)
 save(ROOT/'CURRENT_STATUS.json',status)
 old=(ROOT/'HANDOFF_CURRENT.md').read_text(encoding='utf-8');text='\n'.join(['# 最新补充：v371 五项MAE专家实验已完成','',f'完成时间（香港）：{now}。',
  '本轮按用户要求只优化Caco2、LD50、脂溶性、PPBR和溶解度，每项MAE下降10%为目标。GPU拟合全部在端口25870远端进行。',
  f"结果：{delivery['status']}。五项10%目标：{'本地达到，官方未知' if delivery['target_five_tasks_ten_percent_achieved_local'] else '未达到'}。官方最好仍为v364=51.269。",
  '完整结果见`docs/V371_MAE_EXPERT_RESULTS.md`；结构化交付见`docs/v371_delivery.json`。不把新版本编号或局部验证当作官方提分。',
  '已保存70个开发组合、确认预测、独立重放证据；模型检查点留在`/root/autodl-tmp/mae_experts_v371/outputs/tasks`。密码不在项目文件中。','', '---','',old])
 (ROOT/'HANDOFF_CURRENT.md').write_text(text,encoding='utf-8');(ROOT/'docs/HANDOFF_CURRENT.md').write_text(text,encoding='utf-8')
 files=[ROOT/'HANDOFF_CURRENT.md',ROOT/'docs/HANDOFF_CURRENT.md',ROOT/'tools/verify_submission.py',ROOT/'tools/repack_upload_template.py']
 files+=list((ROOT/'tools').glob('v371_*.py'))+list((ROOT/'docs').glob('v371_*.json'))+list((ROOT/'docs').glob('V371_*.md'))
 files += [p for p in exp.rglob('*') if p.is_file() and p.suffix in ('.json','.py','.npz','.csv') and '/kernels/' not in p.as_posix()]
 if delivery.get('csv'):files+=list((ROOT/delivery['csv']).parent.glob('*'))
 files=sorted(set(files));entries=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in files]
 manifest=ROOT/'docs/v371_review_package_manifest.json';save(manifest,dict(description='Official data, independent experts, predictions and audit evidence; no credentials; full models retained remote.',files=entries))
 archive=ROOT/'releases/v371_mae_expert_review.tar.gz'
 with tarfile.open(archive,'w:gz') as tf:
  for p in files+[manifest]:tf.add(p,arcname=p.relative_to(ROOT).as_posix())
 with tarfile.open(archive) as tf:
  for r in entries:assert hashlib.sha256(tf.extractfile(r['path']).read()).hexdigest()==r['sha256']
 save(ROOT/'docs/v371_review_package.json',dict(status='PASS',archive=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,files=len(entries)+1,all_member_hashes_verified=True))
 print(json.dumps(dict(status='PASS',csv=delivery.get('csv'),ten_percent_goal=delivery['target_five_tasks_ten_percent_achieved_local'],official_best=51.269)))


if __name__=='__main__':main()
