"""Package completed bounded rounds, preserve official baseline, and record unmet goals."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib,tarfile

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    exp=ROOT/'experiments/mae_experts_v372';v371=read(ROOT/'docs/v371_delivery.json');delivery=read(ROOT/'docs/v372_delivery.json');check=read(exp/'outputs/independent_verification.json')
    assert check['status']=='PASS' and check['signature']==delivery['signature']
    remote=read(exp/'completion_receipt.json');assert remote['no_training_processes_running']
    best=read(ROOT/'docs/current_best_after_v364_20261104.json');assert sha(ROOT/best['csv'])==best['csv_sha256']
    now=datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat();status=read(ROOT/'CURRENT_STATUS.json')
    progress=dict(status='COMPLETE',completed_at_hong_kong=now,version='v372',training_host='connect.nmb2.seetacloud.com:25870',official_best=51.269,
        target_five_MAE_reduced_ten_percent=delivery['target_five_tasks_ten_percent_achieved_local'],delivery=delivery,verification=check['status'],remote_idle=remote,official_new_score=None)
    save(ROOT/'docs/v372_progress.json',progress)
    preferred=delivery if delivery.get('csv') else v371
    status.update(latest_completed_model_branch='v372',latest_completed_model_branch_is_proven_best=False,latest_aggregate_plan='v372',aggregate_status=delivery['status'],
        official_score=51.269,remote_training_running=False,local_gpu_training=False,current_experiment_progress='docs/v372_progress.json',
        latest_MAE_expert_round=progress,latest_MAE_experimental_submission=preferred,goal_five_MAE_ten_percent_achieved=delivery['target_five_tasks_ten_percent_achieved_local'])
    save(ROOT/'CURRENT_STATUS.json',status)
    section='\n'.join(['# 最新补充：v372 内部交叉验证专家集成已完成','',f'完成时间（香港）：{now}。',
        '按用户要求优化Caco2、LD50、脂溶性、PPBR和溶解度，五项分别降低MAE至少10%为目标。全部拟合在端口25870远端RTX3090完成。',
        f"v372结果：{delivery['status']}。五项10%目标：{'本地达到，官方未知' if delivery['target_five_tasks_ten_percent_achieved_local'] else '未达到'}。官方最好保持v364=51.269。",
        '结果见`docs/V372_MAE_EXPERT_RESULTS.md`，固定方案见`docs/V372_CROSSFIT_MAE_EXPERT_PLAN.md`，结构化交付见`docs/v372_delivery.json`。',
        '四个基模型是SVR、核岭、GPU CatBoost MAE和GPU XGBoost中位数；三折内部骨架OOF学习凸组合及理化分区门控。内折归一化和列选择分别拟合，不用历史重代入预测学习权重。',
        '开发fold0/1选型冻结后才确认fold2/valid，无失败后补选。历史公开标签重复使用、基线OOF部分代理等限制依然存在。',
        f"当前可手动测试的新实验CSV：`{preferred.get('csv')}`。它并非官方已确认提升。v371只改LD501478行，其他21项保留v364；v372有自己的交付门槛。",
        '完整拟合检查点远端保留`/root/autodl-tmp/mae_experts_v372/outputs/base`，v371检查点保留原目录。项目不保存SSH密码。','', '---',''])
    old=(ROOT/'HANDOFF_CURRENT.md').read_text(encoding='utf-8')
    if not old.startswith('# 最新补充：v372'):
        old=section+'\n'+old
    (ROOT/'HANDOFF_CURRENT.md').write_text(old,encoding='utf-8');(ROOT/'docs/HANDOFF_CURRENT.md').write_text(old,encoding='utf-8')
    # Evidence package reuses the v371 frozen input package; no duplicate400MB training matrices.
    files=[ROOT/'HANDOFF_CURRENT.md',ROOT/'docs/HANDOFF_CURRENT.md',ROOT/'CURRENT_STATUS.json',ROOT/'tools/verify_submission.py',ROOT/'tools/repack_upload_template.py',ROOT/'tools/v371_mae_experts.py']
    files+=list((ROOT/'tools').glob('v372_*.py'))+list((ROOT/'docs').glob('v372_*.json'))+list((ROOT/'docs').glob('V372_*.md'))
    files+=[p for p in exp.rglob('*') if p.is_file() and ('/outputs/' in p.as_posix() or p.suffix in ('.py','.json','.log')) and '/kernels/' not in p.as_posix()]
    for r in (v371,delivery):
        if r.get('csv'):files+=list((ROOT/r['csv']).parent.glob('*'))
    files=sorted(set(files));entries=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in files]
    manifest=ROOT/'docs/v372_review_package_manifest.json'
    entries=[r for r in entries if r['path'] not in ('docs/v372_review_package_manifest.json','docs/v372_review_package.json')]
    save(manifest,dict(description='v372 sources, predictions and independent audit. Input matrices in v371_expert_inputs.tar.gz; all base checkpoints retained remote.',input_archive=read(ROOT/'docs/v371_input_package.json'),files=entries))
    archive=ROOT/'releases/v372_crossfit_expert_review.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for r in entries:tf.add(ROOT/r['path'],arcname=r['path'])
        tf.add(manifest,arcname=manifest.relative_to(ROOT).as_posix())
    with tarfile.open(archive) as tf:
        for r in entries:assert hashlib.sha256(tf.extractfile(r['path']).read()).hexdigest()==r['sha256']
    save(ROOT/'docs/v372_review_package.json',dict(status='PASS',archive=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,files=len(entries)+1,all_member_hashes_verified=True))
    print(json.dumps(dict(status='PASS',v372=delivery['status'],preferred_csv=preferred.get('csv'),ten_percent_goal=delivery['target_five_tasks_ten_percent_achieved_local'],official_best=51.269)))

if __name__=='__main__':main()
