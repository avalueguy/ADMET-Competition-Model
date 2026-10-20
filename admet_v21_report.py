"""Generate a concise evidence-based v21 delivery report from verified outputs."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    proof=json.loads((ROOT/'admet_submissions/v21/verification.json').read_text(encoding='utf-8'))
    local=json.loads((ROOT/'admet_submissions/v21/local_verification.json').read_text(encoding='utf-8'))
    changed=[a for a in proof['audits'] if a['selection']['weight']>0]
    lines=['# v21 远程微调实验结果','',
           'v20 官方得分由用户反馈为 **50.019727**，较 v19 的 49.993751 提高 0.025976。本轮以 v20 的原始提交文件作为基线。v21 尚未提交平台，没有官方分数。','',
           '## 模型变化','',
           '微调自监督预训练 MoLFormer 的最后两层 Transformer 和最终 LayerNorm，将池化分子表示与描述符/MACCS 支路融合。每个模型有 7,316,737 个可训练参数，使用两个随机种子、规范 SMILES 与随机 SMILES 两种视图，分类损失包含 BCE 与配对排序项，回归损失按任务指标设定。前十层固定，只缓存不含标签的表示。','',
           '训练轮数在每次拟合所用数据内部的 scaffold 划分上选择，之后重新使用全部拟合行训练。所有 GPU 训练、缓存及权重均在授权远程 RTX 4090 完成。盲测标签、外部实验标签均未使用，没有自动上传平台。','',
           '本轮固定测试 12 个任务。筛选目标为 0.6 × OOF 平均收益 + 0.4 × 验证收益 − 0.025 × 折间标准差 − 0.0002 × 新模型比例，允许单折或个别指标回落。该目标是内部代理分数，不是官方总分公式。','',
           '## 进入候选的任务','',
           '| 任务 | 新模型比例 | 融合方式 | OOF 平均收益 | 验证收益 | 重复骨架划分收益 |',
           '|---|---:|---|---:|---:|---:|']
    for a in changed:
        s=a['selection']; r=a.get('repeat',{}).get('gain')
        rv='未完成' if r is None else f'{r:+.6f}'
        lines.append(f"| {a['task']} | {s['weight']:.0%} | {s['mode']} | {s['oof_gain']:+.6f} | {s['valid_gain']:+.6f} | {rv} |")
    lines += ['', 'MAE 任务的收益是相对 MAE 降幅；AUROC、AUPRC、Spearman 的收益是指标绝对增量。三类数字不能直接相加得到平台分数。', '',
              f"训练覆盖 {proof['trained_tasks']} 个任务，最终改变 {len(changed)} 个任务、{local['changed_rows']} 行预测，其余任务完整沿用 v20。",'',
              '## 回落与不确定性','']
    for a in changed:
        s=a['selection']; negatives=[f'{i+1}: {v:+.6f}' for i,v in enumerate(s['fold_gains']) if v<0]
        ci=a['uncertainty']['percentile_95']
        lines.append(f"- {a['task']}：下降的 OOF 折为 {'、'.join(negatives) if negatives else '无'}；验证集 bootstrap 95% 分位区间 [{ci[0]:+.6f}, {ci[1]:+.6f}]。")
    lines += ['', '历史 OOF、官方验证集与重复骨架划分已在多轮实验中使用，存在选择偏差。配对 scaffold bootstrap 条件于当前已训练模型和已选方案，不涵盖模型选择误差。重复划分只作诊断，不据此重新调权重。即便均值上涨，也不能保证盲测总分上涨或达到 51 分。', '',
              '## 校验与复现','',
              '- 修复了长 SMILES 的旋转位置编码缓存随批次/精度变化的问题，推理使用 float32。受影响的初次实验归档到远程 admet_model_outputs_v21_attempt1，未参与候选。',
              '- 验证固定前缀表示与原预训练模型一致、微调层实际获得梯度更新；每折、验证和最终模型均保存后重载比较预测。',
              '- 检查 OOF 行顺序、标签和 fold 对齐，各 OOF 折的拟合行与留出行之间骨架、分子无交叉；模型元数据记录拟合行、标签哈希和代码/协议哈希。',
              '- 最终模型只使用官方 train+valid 标签。训练权重保留在远程，本机只接收预测和评估记录。',
              '- 提交 CSV 为 16,379 行、22 个任务、6 列、UTF-8 BOM、CRLF；非预测字段与模板逐行一致，预测有限且分类置信度位于 [0,1]。',
              '- 本机用独立实现重建原值融合/排名融合并核对所有改变任务，已通过。', '',
              '## 文件','', f"提交文件：`{local['submission']}`",'',f"SHA256：`{local['sha256']}`",'',
              f"字节数：{local['bytes']}。已验证基线为 v20，官方分数 50.019727。v21 官方分数仍需平台评测。"]
    report=ROOT/'admet_diagnostics/v21_delivery_report.md'
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(str(report))


if __name__=='__main__':main()
