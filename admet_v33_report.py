"""Readable release notes from actual saved predictions and verification."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def main():
    out=ROOT/'admet_model_outputs_v33';analysis=json.loads((out/'analysis.json').read_text(encoding='utf-8'))
    selected=[r for r in analysis['tasks'] if r['selection']]
    names=['邻域迁移神经网络','邻域属性+描述符提升树','邻域属性+片段提升树','神经网络+描述符提升树','神经网络+片段提升树']
    lines=['# v33 邻域实验属性迁移模型','',
        'v32 平台成绩为用户报告的 50.399027，比 v29 增加 0.012734。官方分项成绩与总分公式未知，本轮不预测平台分数。','',
        '本轮让其他分子的官方 ADMET 实验标签也提供信息：按两种化学指纹检索邻居，聚合各属性的实验值、相似度、标签分散度与有效邻居数；这些属性与描述符、ChemBERTa、指纹一起输入神经网络和两类 CatBoost。每个分支使用两个种子，神经网络使用两个快照。','',
        '目标属性的其他任务邻域块整体移除。自身任务邻居只使用当前拟合行的标签，并排除与查询分子同骨架的所有邻居；重复身份的标签先聚合。其他任务属性仅取官方 train（验证/CV）或 train+valid（最终）。未读取测试标签，未导入外部实验标签。','',
        '训练仅在 Qwen 服务器 `czp123@100.109.143.42:/home/czp123/admet_v23_20260928` 执行。没有本机 GPU 训练，没有自动上传比赛平台。','',
        f'共筛选 22 个任务，入选 {len(selected)} 个。MAE 改善为误差相对减少；AUROC、AUPRC、Spearman 为指标绝对增加。个别验证折允许回退。','',
        '| 任务 | 指标 | 模型 | 权重 | 完整 v32 历史 valid 改善 | 三折代理平均改善 |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for r in selected:
        s=r['selection'];a,c=s['valid_gain'],s['cv_mean'];mae=r['metric']=='MAE'
        a=f'{a*100:.3f}%' if mae else f'{a:+.6f}';c=f'{c*100:.3f}%' if mae else f'{c:+.6f}'
        lines.append(f"| {r['task']} | {r['metric']} | {names[s['variant']]} | {s['weight']:.2f} | {a} | {c} |")
    lines+=['','## 证据的边界','',
        'valid 与训练骨架折已在多轮开发中复用，不能当作全新独立验证。历史 valid 比较重放完整 v32，CV 比较重放 v32 新分支在 v24 神经分支上的融合，仍是代理集成，不是整个历史集成的 OOF。其他任务的公开标签可能覆盖查询骨架，这是有其他属性信息的条件预测。','',
        '三折平均与历史 valid 正收益不保证官方总分上涨。下面列出负收益折和选择稳定性结果，不把代理指标换算为比赛分数。','']
    for r in selected:
        negative=[f"第{j+1}折 {g:+.6f}" for j,g in enumerate(r['selection']['cv_gains']) if g<0]
        nested=sum(r.get('held_fold_selection_gains',[0]))/3
        if negative or nested<0:lines.append(f"- {r['task']}：{'；'.join(negative) or '固定选项无负收益折'}；留一折选型平均增量 {nested:+.6f}。")
    diagnostic=out/'uncertainty_and_shift.json'
    if diagnostic.exists():
        dd=json.loads(diagnostic.read_text(encoding='utf-8'));lines+=['','## 分布与不确定性检查','',
            '以下区间是选型之后、在复用 valid 上按骨架重采样的 95% 区间，没有消除模型选择偏差。相似度 <0.3 定义为结构较远。','',
            '| 任务 | valid 较远占比 | test 较远占比 | valid 增益重采样区间 |',
            '| --- | ---: | ---: | --- |']
        for r in dd['tasks']:
            lo,hi=r['selected_candidate_scaffold_bootstrap_95_interval']
            lines.append(f"| {r['task']} | {r['valid_distant_fraction']:.1%} | {r['test_distant_fraction']:.1%} | [{lo:+.6f}, {hi:+.6f}] |")
    dest=ROOT/'admet_submissions/v33'
    if dest.exists():
        path=next(dest.glob('*.csv'));info=json.loads((dest/'verification.json').read_text(encoding='utf-8'))
        local=json.loads((dest/'local_verification.json').read_text(encoding='utf-8'))
        assert local['sha256']==info['sha256'] and local['status']=='INDEPENDENT_MODEL_AND_CSV_REPLAY_PASSED'
        lines+=['','## 提交文件','',f'`{path.relative_to(ROOT).as_posix()}`','',
            f"16,379 行、22 个任务、6 列，{info['bytes']:,} 字节，UTF-8 BOM、CRLF。身份字段逐行等于官方模板。远程最终模型全部重载重算核验，本机独立重放融合并核验内容、哈希、编码与预测范围。",'',
            f"SHA256：`{info['sha256']}`。",'',
            '实际平台分数仍待提交后确认；v32 为最后确认得分的基线。']
    report=ROOT/'admet_diagnostics/v33_report.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(str(report))
if __name__=='__main__':main()
