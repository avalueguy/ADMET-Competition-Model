"""Record the three completed remote stages and preserve the proven best model."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    rounds = []
    for version, stage in ((368, 3), (369, 4), (370, 5)):
        exp = ROOT / f'experiments/transformer_stage{stage}_v{version}'
        delivery = read(ROOT / f'docs/v{version}_stage{stage}_delivery.json')
        verification = read(exp / 'outputs/independent_verification.json')
        receipt = read(ROOT / f'docs/v{version}_remote_receipt_verification.json')
        assert verification['status'] == receipt['status'] == 'PASS'
        assert delivery['signature'] == verification['signature'] == receipt['signature']
        development = read(exp / 'outputs/development.json')
        confirmation = read(exp / 'outputs/confirmation.json')
        rounds.append(dict(version=f'v{version}', stage=stage, delivery=delivery,
            tasks=[r['task'] for r in development['records']],
            development_jobs=len(list((exp/'outputs/tasks').glob('*/fold[01]/*/predictions.npz'))),
            strict_tasks=confirmation['strict_tasks'], experimental_tasks=confirmation['experimental_tasks'],
            report=f'docs/V{version}_STAGE{stage}_RESULTS.md', remote_root=f'/root/autodl-tmp/transformer_stage{stage}_v{version}',
            independent_replay_max_error=verification['full_replay_max_error']))
    best = read(ROOT / 'docs/current_best_after_v364_20261104.json')
    all_models = read(ROOT / 'experiments/transformer_stage5_v370/all_models_audit.json')
    assert all_models['status'] == 'PASS' and all_models['signature'] == rounds[-1]['delivery']['signature']
    assert hashlib.sha256((ROOT / best['csv']).read_bytes()).hexdigest() == best['csv_sha256']
    now = datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat()
    report = dict(completed_at_hong_kong=now, baseline=best, rounds=rounds,
        new_official_score=None, training_gpu_remote_only=True, stage5_all_models_audit=all_models,
        training_host='connect.nmb2.seetacloud.com:25870', target_52_achieved=False,
        no_confirmation_fallback=True, no_external_data_or_weights=True,
        limitations=['Partial historical proxy OOF', 'Public validation labels repeatedly reused',
                    'Experimental local gains do not guarantee hidden-test improvement'])
    save(ROOT / 'docs/v370_completed_round.json', report)
    status = read(ROOT / 'CURRENT_STATUS.json')
    status.update(latest_completed_model_branch='v370', latest_completed_model_branch_is_proven_best=False,
        rollback_baseline='v364', official_score=51.269, target_achieved=False, local_gpu_training=False, latest_aggregate_plan='v370',
        remote_optimization_current=report, aggregate_status=rounds[-1]['delivery']['status'],
        current_training_host='connect.nmb2.seetacloud.com:25870', remote_training_running=False)
    save(ROOT / 'CURRENT_STATUS.json', status)
    lines = ['# 新服务器优化实验结果', '', f'完成时间（香港）：{now}。', '',
        '官方已知最高仍为 v364，51.269。所有新训练都在用户指定的端口25870服务器 RTX3090 上完成。', '',
        '|阶段|任务|结果|', '|---|---|---|']
    for row in rounds:
        selected = row['strict_tasks'] + row['experimental_tasks']
        result = '无符合条件的新提交' if not selected else '通过严格条件：'+','.join(row['strict_tasks'])+'；实验候选：'+','.join(row['experimental_tasks'])
        lines.append(f"|{row['version']}|{', '.join(row['tasks'])}|{result}|")
    latest = rounds[-1]['delivery']
    if latest.get('csv'):
        lines += ['', f"本轮新实验 CSV：`{latest['csv']}`。官方成绩未知。", f"SHA256：`{latest['csv_sha256']}`。"]
    else:
        lines += ['', '本轮三组实验均未找到符合条件的新候选，没有生成新CSV。可继续使用此前官方最好文件：', '', f"`{best['csv']}`", '', f"SHA256：`{best['csv_sha256']}`。"]
    lines += ['', 'v368：Graph Transformer从零、自监督和任务家族迁移。AMES和半衰期开发改善，但valid退步；微粒体清除率和VDss开发未通过。',
        'v369：生物利用度和CYP底物小任务适配器。生物利用度开发改善，但fold2和valid退步；三项CYP底物开发未通过。',
        'v370：Graph Transformer增加4096维count指纹分支，比较AdamW和rho=.05的SAM，覆盖五项MAE任务。详细四分区结果见对应报告。', '',
        '不能把新增架构和更多组合视为必然提分。失败时保留原预测，未根据确认集反复换配方。新测试使用了长期复用的公开验证标签，历史控制预测部分是代理OOF。', '',
        '下一步优先核查训练与隐藏测试分布差异、测量单位及重复标签质量，建立更可信的分组对照；随后针对可解释误差训练有限任务专家。缺少新增信号时，继续堆叠同类Transformer不支持大幅提分预期。', '',
        '远端每个模型检查点均保留；本地保存输入、源码、预测、哈希清单及独立重放证据。新服务器没有前一服务器的旧训练目录；旧v364/v366最终权重已在此前GitHub交接包保留。密码不在文件中。', '']
    (ROOT / 'docs/V370_REMOTE_ROUND_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    handoff_path = ROOT / 'HANDOFF_CURRENT.md'
    previous = handoff_path.read_text(encoding='utf-8')
    banner = '\n'.join(['# 最新补充：新服务器 v368–v370 实验已结束', '',
        f'更新时间（香港）：{now}。GPU主机已改为 `connect.nmb2.seetacloud.com:25870`，覆盖下文旧端口。',
        '官方最好仍是v364=51.269。v366仍是官方未知的旧候选，本轮以v364对照。',
        f"v370状态：{latest['status']}。完整结论见 `docs/V370_REMOTE_ROUND_RESULTS.md`，结构化记录见 `docs/v370_completed_round.json`。",
        '本轮所有训练已结束。开发与确认失败记录保留，不得把实验编号当作官方最高分。', '', '---', '', previous])
    handoff_path.write_text(banner, encoding='utf-8')
    (ROOT / 'docs/HANDOFF_CURRENT.md').write_text(banner, encoding='utf-8')
    files = [ROOT / 'HANDOFF_CURRENT.md', ROOT / 'docs/HANDOFF_CURRENT.md']
    for pattern in ('v368_*.py', 'v369_*.py', 'v370_*.py'):
        files.extend((ROOT / 'tools').glob(pattern))
    files.extend([ROOT/'tools/v364_graph_transformer.py', ROOT/'tools/verify_submission.py', ROOT/'tools/repack_upload_template.py'])
    for pattern in ('v368_*.json','v369_*.json','v370_*.json','V368_*.md','V369_*.md','V370_*.md'):
        files.extend((ROOT / 'docs').glob(pattern))
    for version, stage in ((368,3),(369,4),(370,5)):
        exp = ROOT / f'experiments/transformer_stage{stage}_v{version}'
        for p in exp.rglob('*'):
            if p.is_file() and (p.suffix in ('.py','.json','.npz','.csv','.txt') or p.name=='graph_audit.json'):
                files.append(p)
    files = sorted(set(files))
    entries = [dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in files]
    package_manifest = ROOT/'docs/v370_round_package_manifest.json'
    save(package_manifest,dict(description='Source, official inputs, controls, prediction evidence and audits. No new adopted model; development checkpoints retained remotely.',files=entries))
    archive = ROOT/'releases/v370_remote_optimization_review.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for p in files+[package_manifest]:
            tf.add(p,arcname=p.relative_to(ROOT).as_posix())
    with tarfile.open(archive) as tf:
        for r in entries:
            assert hashlib.sha256(tf.extractfile(r['path']).read()).hexdigest()==r['sha256']
    save(ROOT/'docs/v370_round_package.json',dict(status='PASS',archive=archive.relative_to(ROOT).as_posix(),
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,files=len(entries)+1,
        all_member_hashes_verified=True,adopted_new_models=False))
    print(json.dumps(dict(status='PASS', official_best=51.269, candidate=latest.get('csv'), reports=len(rounds))))


if __name__ == '__main__':
    main()
