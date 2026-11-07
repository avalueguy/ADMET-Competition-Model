"""Persist results and reviewable evidence without overwriting official feedback."""
from pathlib import Path
import json,hashlib,tarfile
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    exp=ROOT/'experiments/mae_experts_v374';delivery=read(ROOT/'docs/v374_delivery.json');verify=read(exp/'outputs/independent_verification.json');remote=read(exp/'completion_receipt.json')
    assert verify['status']=='PASS' and verify['signature']==delivery['signature'] and remote['no_training_processes_running']
    best=read(ROOT/'docs/current_best_after_v371_20261105.json');assert sha(ROOT/best['csv'])==best['csv_sha256']
    progress=dict(status='COMPLETE',version='v374',completed_at_utc=datetime.now(timezone.utc).isoformat(),official_best=51.270,training_host='connect.nmb2.seetacloud.com:25870',
        delivery=delivery,verification=verify['status'],models_verified=verify['models'],remote_idle=remote,five_task_ten_percent_goal=delivery['target_five_tasks_ten_percent_achieved_local'],official_new_score=None)
    save(ROOT/'docs/v374_progress.json',progress)
    path=ROOT/'CURRENT_STATUS.json';s=read(path);s.update(latest_completed_model_branch='v374',latest_completed_model_branch_is_proven_best=False,latest_aggregate_plan='v374',aggregate_status=delivery['status'],
        remote_training_running=False,official_score=51.270,current_experiment_progress='docs/v374_progress.json',latest_MAE_expert_round=progress,
        latest_MAE_experimental_submission=delivery if delivery.get('csv') else read(ROOT/'docs/v371_delivery.json'),goal_five_MAE_ten_percent_achieved=delivery['target_five_tasks_ten_percent_achieved_local']);save(path,s)
    old=(ROOT/'HANDOFF_CURRENT.md').read_text(encoding='utf-8');section='\n'.join(['# 最新补充：v374 MAE专家继续实验已完成','',
        '官方新反馈：用户确认v371=51.270，LD50和溶解度小幅改善；完整卡见`docs/platform_scorecard_v371_20261105.json`。本地v371只改LD50，卡中溶解度也变；保留来源差异，不冒称平台字节已核验。',
        f"本轮结果：{delivery['status']}；五项各降10%：{'验证达到，官方未知' if delivery['target_five_tasks_ten_percent_achieved_local'] else '未达到'}。",
        'v374采用原计数/二值/前三视图核、重复测量次数及差异降权、真实内部OOF局部中位数和GPU残差树。全部拟合仅端口25870远端RTX3090进行。',
        '固定方案见`docs/V374_MAE_SPECIALIST_PLAN.md`，逐项结果见`docs/V374_MAE_EXPERT_RESULTS.md`，交付见`docs/v374_delivery.json`。',
        f"新CSV：`{delivery.get('csv')}`；官方分数未知，不能把验证当官方提分。模型检查点保留远端`/root/autodl-tmp/mae_experts_v374/outputs/tasks`和`outputs/residuals`。",
        '当前完整历史集成OOF仍部分proxy；本轮残差参考的内部OOF独立重训不代表历史全部已重建。公开标签反复使用限制保留。','', '---',''])
    if not old.startswith('# 最新补充：v374'):old=section+'\n'+old
    for p in (ROOT/'HANDOFF_CURRENT.md',ROOT/'docs/HANDOFF_CURRENT.md'):p.write_text(old,encoding='utf-8')
    files=[ROOT/'CURRENT_STATUS.json',ROOT/'HANDOFF_CURRENT.md',ROOT/'docs/HANDOFF_CURRENT.md',ROOT/'tools/v371_mae_experts.py',ROOT/'tools/verify_submission.py',ROOT/'tools/repack_upload_template.py']
    files+=list((ROOT/'tools').glob('v374_*.py'))+[ROOT/'tools/v373_record_feedback.py']+list((ROOT/'docs').glob('v374_*.json'))+list((ROOT/'docs').glob('V374_*.md'))
    files+=[ROOT/'docs/platform_scorecard_v371_20261105.json',ROOT/'docs/current_best_after_v371_20261105.json']
    files+=[p for p in exp.rglob('*') if p.is_file() and '/kernels/' not in p.as_posix() and ('/outputs/' in p.as_posix() or p.suffix in ('.json','.py','.log'))]
    if delivery.get('csv'):files+=list((ROOT/delivery['csv']).parent.glob('*'))
    files=sorted(set(files));files=[p for p in files if p.name not in ('v374_review_package.json','v374_review_package_manifest.json')]
    entries=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in files]
    manifest=ROOT/'docs/v374_review_package_manifest.json';save(manifest,dict(input_archive='v371_expert_inputs.tar.gz plus v374_delta_inputs.tar.gz',description='No credentials. Full model checkpoints remote; exact inputs through two hashed archives.',files=entries))
    archive=ROOT/'releases/v374_specialist_review.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for p in files+[manifest]:tf.add(p,arcname=p.relative_to(ROOT).as_posix())
    with tarfile.open(archive) as tf:
        for r in entries:assert hashlib.sha256(tf.extractfile(r['path']).read()).hexdigest()==r['sha256']
    save(ROOT/'docs/v374_review_package.json',dict(status='PASS',archive=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,files=len(entries)+1,all_member_hashes_verified=True))
    print(json.dumps(dict(status='PASS',csv=delivery.get('csv'),goal=progress['five_task_ten_percent_goal'],official_best=51.270)))
if __name__=='__main__':main()
