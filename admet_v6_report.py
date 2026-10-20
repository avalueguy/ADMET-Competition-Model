"""Collect measured gains, limitations and compact review artifacts."""
import json
import tarfile
from pathlib import Path
from admet_v6_features import sha

root=Path('admet_model_outputs_v6');out=Path('admet_diagnostics');out.mkdir(exist_ok=True)
decisions=[json.loads(p.read_text()) for p in sorted(root.glob('*/decision.json'))]
lines=['# v6 远程实验结果','',
       '已知平台基线：48.98128；用户提供榜首：50.584929；数值目标：51.584929。v6 平台成绩尚未知。',
       '', '新模型包括 D-MPNN 图神经网络、计数/功能/路径指纹核、残差神经网络及指标适配树模型。',
       '所有训练只在远程服务器完成；未读取盲测标签，也没有自动提交平台。','',
       '## 逐任务结果','',
       '下表列出候选实验指标；标为“否”的任务在最终文件中仍使用 v5，不采用表中退步预测。','',
       '|任务|计分指标|v5 valid|v6 候选 valid|采用|模型 / 新权重|原因|',
       '|---|---|---:|---:|---|---|---|']
for d in decisions:
    s=d.get('selection',{});metric=s.get('metric','')
    key=metric.lower();before=d.get('before',{}).get(key);after=d.get('after',{}).get(key)
    fmt=lambda x:'—' if x is None else f'{x:.6f}'
    lines.append(f"|{d['task']}|{metric}|{fmt(before)}|{fmt(after)}|{'是' if d['accepted'] else '否'}|{s.get('family','保留基线')} / {s.get('alpha',0)}|{d.get('reason','')}|")
lines += ['', '## 采用任务的辅助指标与最差折','',
          '|任务|辅助指标 valid 旧 → 新|三折主指标最差变化|额外骨架检查|',
          '|---|---|---:|---|']
for d in decisions:
    if not d['accepted']:continue
    s=d['selection'];metric=s['metric']
    secondary={'MAE':'spearman','Spearman':'mae','AUROC':'auprc','AUPRC':'auroc'}[metric]
    a=d['before'][secondary];b=d['after'][secondary]
    r=d['repeat_confirmation'];key=metric.lower();worst=s['worst_gain']
    if d.get('domain_guard',{}).get('passed'):
        from admet_v6_runner import deltas
        checks=d['domain_guard']['checks'];r=checks[-1]
        worst=min(deltas(metric,c['before'],c['after']) for c in checks[:3])
    lines.append(f"|{d['task']}|{secondary}: {a:.6f} → {b:.6f}|{worst:.6f}|{r['before'][key]:.6f} → {r['after'][key]:.6f}|")
lines += ['', 'MAE 任务的最差折变化按旧 MAE 的相对改进计算；其他任务为绝对指标改进。负数表示下降。',
          'HIA 若采用，官方 valid 为 1.0 → 1.0，并不代表该验证集上有提升。',
          'AMES、CYP2D6 抑制与 VDss 采用相似度回退。CYP2D6 最后一次重划分 AUPRC 有约 0.00194 小幅下降，符合预设容差；不存在所有指标绝对不降的承诺。']
lines += ['', '## 证据范围与风险','',
          '- 选择依据为训练集三折骨架 OOF，先锁定模型和权重，再检查官方 valid。OOF 与 valid 均存在历史复用。',
          '- 通过者另做固定新种子的骨架留出，属于重划分稳健性检查；样本仍来自已使用的训练集，不能称独立盲测。',
          '- 新模型至少改善平均 OOF 主指标且官方 valid 主指标不下降；折内主指标容差为排名指标 0.005 / MAE 1%，辅助指标容差为排名指标 0.015 / MAE 3%。',
          '- 完成选型后额外检查相似度低于 0.3 的分组。排名指标下降超过 0.01 或 MAE 恶化超过 3% 时，尝试该组退回 v5；若重查原门槛失败则整项退回。该检查也复用了验证数据，不是无偏估计。',
          '- 每项辅助指标可能小幅变化。隐藏测试集不退步、总分超过 51.584929 都无法保证。',
          '- 官方官网没有给出任务权重或总分计算公式，本报告不虚构平台预计得分。',
          '- 未采用任务沿用 v5。提交行序、ID、SMILES、metric 与官方模板逐项对齐，重载推理检查单独留档。',
          '- V6 推理依赖历史 v5 冻结预测；新模型保存了检查点，不能据此声称完全重建了整个历史模型服务。',
          '', '## 远程位置','', '`/root/autodl-tmp/admet_20260925`。本机只接收提交和轻量审计文件。']
(out/'v6_result_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
names=set()
for directory in ['admet_model_outputs_v6','admet_release_v6']:
    for p in Path(directory).rglob('*'):
        if p.is_file() and p.suffix in ['.csv','.json','.md'] and 'legacy_mtl' not in p.parts:names.add(p)
names.update(Path('.').glob('admet_v6*py'));names.update(Path('.').glob('test_admet_v6*py'))
names.add(Path('admet_v6_protocol.json'));names.add(out/'v6_result_report.md')
for p in out.glob('v6*'):
    if p.is_file():names.add(p)
for p in Path('admet_submissions').glob('*v6*'):
    if p.is_file():names.add(p)
for p in Path('admet_submissions/v6').glob('*'):
    if p.is_file():names.add(p)
for p in Path('logs').glob('v6*log'):names.add(p)
archive=Path('admet_v6_review.tar.gz')
with tarfile.open(archive,'w:gz') as tf:
    for p in sorted(names):tf.add(p,arcname=str(p),recursive=False)
print(json.dumps(dict(completed=len(decisions),accepted=[d['task'] for d in decisions if d['accepted']],
                     archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive))))
