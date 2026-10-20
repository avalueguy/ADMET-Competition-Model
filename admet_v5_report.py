"""Write a reviewable report from verified result artifacts, without training."""
import json
from pathlib import Path
import pandas as pd


def main():
    release=Path('admet_release_v5')
    v=json.loads((release/'verification.json').read_text())
    assert v['status']=='VERIFIED'
    accepted=v['accepted']
    labels=dict(ames='AMES',clearance_microsome_az='微粒体清除率',
                cyp2d6_substrate_carbonmangels='CYP2D6 底物',cyp2d6_veith='CYP2D6 抑制',
                herg='hERG',lipophilicity_astrazeneca='脂溶性',ppbr_az='血浆蛋白结合率')
    out=['# 远程 v5 实验结果','',
         '本轮在远程 RTX 4090 上完成 22 个任务的候选训练、骨架交叉验证和旧配方对照。'
         f'共 {len(accepted)} 个任务通过替换门槛，其余 {22-len(accepted)} 个任务逐字节保留已提交 v4 预测。','',
         '**已有平台成绩仍为 48.862655。新文件尚未评测；没有证据声称已达到或保证达到 60 分。**','',
         '## 接受的变化（官方 valid，非平台分数）','',
         '| 任务 | 指标 | 旧版 | 新候选融合 | 第二指标旧 → 新 |',
         '|---|---|---:|---:|---|']
    for task in accepted:
        r=json.loads((Path('admet_model_outputs_v5')/task/'result.json').read_text())
        metric=r['metric'];key=metric.lower();a,b=r['valid_before'],r['valid_blend']
        secondary=next(k for k in a if k!=key)
        out.append(f'| {labels.get(task,task)} | {metric} | {a[key]:.6f} | {b[key]:.6f} | {secondary}: {a[secondary]:.6f} → {b[secondary]:.6f} |')
    out += ['', '每个接受任务也通过了同一三折的旧配方重训对照：分类任务 AUROC/AUPRC 均不下降；回归任务 MAE 不升、Spearman 不降。候选家族由训练集内部 CV 选择，融合权重固定 25%。',
            '', '## 48.862655 分的诊断', '',
            '- 官方总分聚合公式与分项成绩未知，不能反推出准确扣分项，也不能把上表指标折算成加分。',
            '- 早期多轮复用相同 valid 和内部留出选择模型，存在选择偏差；部分 valid 很小。AMES valid 的 1,137 条样本仅覆盖一个 Murcko 骨架。',
            '- 9/22 个任务的 test 分子平均最近邻相似度低于 valid，存在任务特定的结构分布差异；其他任务相似度反而更高。',
            '- 旧迁移模型的辅助任务未严格排除留出骨架，本轮对照已排除。旧 v4 的新增头保存不完整是复现缺陷，但不能直接归因为已提交 CSV 的低分。',
            '', '## 新增模型与验证', '',
            '冻结 ChemBERTa MLM 表示、Morgan 指纹和 RDKit 描述符，比较三个种子的神经头、CatBoost 和适用小样本的 Tanimoto 核模型。三个预先固定骨架折用于选择和对照，含 MTL 的旧配方在远程 GPU 上重建。',
            '', 'CV 同时用于候选选择，因此仍不是独立盲测。valid 也曾反复使用。现有证据支持接受这些局部改进，无法保证隐藏集每项都不降，更无法保证总分 60+。`v5_uncertainty.csv` 的区间仅描述固定预测在骨架重采样下的波动，不校正模型选择偏差；单骨架任务不报告区间。',
            '', '## 提交与复现', '',
            '- 候选：`admet_submissions/admet-v5-20260925.csv`。官方模板 6 列、16,379 行；行序、任务、ID、SMILES 和 metric 与模板一致。',
            '- 分类输出均在 [0,1]，全部预测有限。官方模板中一个空 drug_id 原样保留。',
            '- 已在独立进程加载模型、重算验证和测试预测，核对模型与文件哈希。未采用任务的原预测字节保持不变。',
            '- 训练检查点、特征缓存保留在 `/root/autodl-tmp/admet_20260925`；本机只接收轻量审计包和提交。',
            '- 新头可重载；完整竞赛预测复现依赖冻结的 v4 预测，并非完全重建的任意新分子 v4 推理服务。',
            '- 复算发现 AvgIpc 在少数大分子上跨进程存在数值不稳定。验证重放固定使用原训练特征缓存，并保存缓存哈希；接受任务另从 SMILES 重算特征复核。后续新协议应剔除该描述符后重新训练，不能直接改动当前已训练模型的特征顺序。',
            '- 语言编码采用原始任务行序与 batch padding 复现。hERG 在只编码 test 的不同批次下有约 7.45e-7 的 embedding 差异，被树阈值放大为一条约 3.78e-5 的融合概率差异；冻结编码结果的重放完全通过。不能承诺不同 batch 形状逐位一致。',
            '', '提交 SHA256：`'+str(v['submission_sha256'])+'`。',
            '', '## 本机清理', '',
            '新增本地 GPU 环境、安装临时文件、重复下载缓存和一个冗余打包文件约 10.61 GiB 已移入回收站，尚未释放实际 C 盘空间。原数据、历史模型和提交保留。自动审批阻止了永久删除，未提供具体原因；需要用户清空相应回收站项目释放空间。',
            '', '## 全部任务决策', '', '| 任务 | 决策 | 原因 |','|---|---|---|']
    for p in sorted(release.glob('*/selection.json')):
        r=json.loads(p.read_text())
        reason='验证集及三折双指标通过' if r['accepted'] else ('验证集双指标未通过' if r.get('reason')=='valid gate rejected' else '三折中至少一项指标退化')
        out.append(f'| {r["task"]} | {"采用候选融合" if r["accepted"] else "保留 v4"} | {reason} |')
    dest=Path('admet_diagnostics/v5_result_report.md')
    dest.write_text('\n'.join(out)+'\n',encoding='utf8')
    print(dest)


if __name__=='__main__':main()
