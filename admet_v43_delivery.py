"""Deliver one verified candidate and retain the confirmed platform baseline."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    out = ROOT / 'admet_model_outputs_v43'
    dest = ROOT / 'admet_submissions/v43'
    file = next(dest.glob('*.csv'))
    verified = read(dest / 'local_verification.json')
    assert verified['status'] == 'INDEPENDENT_GRAPH_BOOST_FUSION_AND_CSV_PASSED'
    report = read(out / 'analysis.json')
    uncertainty = read(out / 'uncertainty.json')
    audit = read(out / 'scaffold_audit.json')
    assert report['signature'] == uncertainty['signature'] and report['status'] == 'EVALUATED'
    selected = [r for r in report['tasks'] if r['selection']]
    assert selected
    names = ['原始/主体结构与实验属性对照', '图表示与描述符/ChemBERTa', '图表示加结构及实验属性', '两图分支等权组合']
    lines = ['# v43 分子图表示与任务专用模型', '',
        '用户报告 v42 得分50.494738，较v39增加0.004040。v43是待平台评测的实验候选，不能预测或保证官方涨分。最后报告第一名52.723518，超过其一分的目标53.723518尚未达到。', '',
        '## 验证分布核查', '',
        f"重新从SMILES计算的原始Murcko骨架与缓存分组完全一致：{audit['raw_and_cached_groups_identical']}。二者均有{audit['raw_unique_groups']:,}个骨架组。所有任务内部train/valid/test骨架分离，没有分组代码错误证据。", '',
        'Ames验证集1,137行全部只有苯环骨架，测试集有449个骨架；LD50和溶解度验证集主要是无环结构，测试集则包括多种环状结构。这些差异限制官方valid涨幅对盲测收益的代表性。没有读取测试标签，不能用结构差异推断官方任务误差或总分公式。', '',
        '## 实际模型变化', '',
        '复用v27多任务有向化学键网络学习的分子表示，使用两个种子的最后固定快照。保留每个种子的128维坐标并串接成256维，不平均不同网络未对齐的隐藏坐标。提取过程不读取标签。', '',
        '每个验证分区使用该分区专属图网络：官方valid使用排除所有valid查询骨架的checkpoint，三折CV分别使用各折checkpoint，最终测试使用train+valid训练的checkpoint。验证期间，图网络的监督训练行与查询行的骨架和非立体异构身份全局隔离。最终源网络没有测试目标标签。', '',
        '随后按任务重新训练GPU XGBoost，两个种子、深度4/6、650/850棵树。训练三组：结构与其他任务实验属性对照；图表示+描述符/ChemBERTa；图表示+原始/主体结构+实验属性。另比较两个含图表示分支的等权组合。分类使用未加正类权重的logistic目标，MAE使用绝对误差，Spearman使用拟合行内高斯秩目标。全部预处理仅拟合当前训练分子。', '',
        '先冻结历史valid正收益候选，再计算三折骨架代理。CV收益为全折收益与有环/无环分层收益的各50%组合；分层权重只取自测试SMILES的有环比例，不用测试标签。valid与测试有环比例差超过0.5或valid只有一个骨架时，valid选择权重为0.15，其余0.45。目标为valid权重×valid增量+(1-valid权重)×调整后CV均值-0.04×CV标准差。要求valid增量>-0.005、调整后CV>0.0005且目标>0.0005。允许个别任务和折下降。', '',
        'GPU训练、图表示提取和模型重载全部在Qwen服务器czp123@100.109.143.42:/home/czp123/admet_v23_20260928运行，CPU线程限制3。本机仅整理轻量预测和检查CSV，没有下载模型checkpoint，没有本机GPU训练，没有自动上传比赛。', '',
        '## 冻结选型结果', '',
        'MAE增量为误差相对降低；AUROC/AUPRC/Spearman为指标绝对增加。均不等于平台分数。', '',
        '| 任务 | 指标 | 分支 | 新预测权重 | v42历史valid增量 | 三折代理平均增量 | 分层调整CV均值 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in selected:
        s = r['selection']
        fmt = (lambda a: f'{a*100:+.3f}%') if r['metric'] == 'MAE' else (lambda a: f'{a:+.6f}')
        lines.append(f"| {r['task']} | {r['metric']} | {names[s['variant']]} | {s['weight']:.2f} | {fmt(s['valid_gain'])} | {fmt(s['cv_mean'])} | {fmt(s['adjusted_cv_mean'])} |")
    lines += ['', '## 增量来源与不确定性', '',
        '含图表示分支和对照使用相同数据、种子和树参数。以下收益是在相同融合比例下含图赢家相对结构/实验属性对照的增量；对照分支入选时该值为零。它们是选型后诊断，不能把所有候选收益归因于图表示。', '',
        '| 任务 | 对照valid增量 | 对照CV平均增量 | valid骨架重采样95%区间 | 留一折选型平均收益 |',
        '| --- | ---: | ---: | --- | ---: |']
    for r in uncertainty['tasks']:
        fmt = (lambda a: f'{a*100:+.3f}%') if r['metric'] == 'MAE' else (lambda a: f'{a:+.6f}')
        interval = '单骨架，无法估计' if r['bootstrap_degenerate'] else '['+', '.join(fmt(a) for a in r['bootstrap_95'])+']'
        lines.append(f"| {r['task']} | {fmt(r['matched_control_valid_gain'])} | {fmt(r['matched_control_cv_mean'])} | {interval} | {fmt(r['held_fold_mean'])} |")
    lines += ['', '本轮三个入选任务的valid骨架重采样95%区间均跨零。P-gp留一折选型均值为负，说明它的融合权重对验证折的选择较敏感。已按预先冻结的允许局部回退规则保留该候选，但证据不足以声称稳健涨分或重大突破。']
    negatives = [(r['task'], j+1, gain) for r in selected for j, gain in enumerate(r['selection']['cv_gains']) if gain < 0]
    lines += ['', '出现负收益的原始CV折：', '']
    lines += [f'- {task} 第{fold}折：{gain:+.6f}' for task, fold, gain in negatives] or ['无。']
    lines += ['',
        '历史valid和CV训练标签已被多轮开发复用，存在搜索与选择偏差。CV比较对象仍是历史代理，未重训完整历史集成OOF；不能称为新独立外部验证。其他官方任务属性可能覆盖查询骨架，属于条件预测。环状/无环分层是粗略域调整，并非官方分数规则。300次骨架重采样和留一折诊断不修正这些偏差。', '',
        '## 交付核验', '', f'`{file.relative_to(ROOT).as_posix()}`', '',
        f"16,379行、22任务、6列，{verified['bytes']:,}字节；UTF-8 BOM与CRLF。所有非预测字段与官方模板逐行完全一致，任务/sample键唯一，预测有限且分类范围[0,1]。更新{len(selected)}个任务，其余与v42预测逐项一致。", '',
        '远程图checkpoint对查询分子重新提取表示，核对批次和排列不变性；任务专用树模型重新载入推理，核对保存的训练预处理器；污染隐藏测试标签及目标任务辅助属性不改变特征。本机用标准库独立重放原始源预测融合并核对源码和CSV哈希，全部通过。', '',
        f"SHA256：`{verified['sha256']}`。", '',
        '本轮只交付一个v43候选。v42确认成绩50.494738的原文件保留作为回退。实际涨分需要比赛平台评测。']
    path = ROOT / 'admet_diagnostics/v43_report.md'
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    state_path = ROOT / 'admet_current_candidate.json'
    state = read(state_path)
    assert state['platform_anchor_release'] == 'v42' and state['platform_anchor_score'] == 50.494738
    state.update(experimental_release='v43', experimental_submission=str(file), experimental_sha256=verified['sha256'],
        experimental_status='EXPERIMENTAL_AWAITING_PLATFORM', candidate_platform_score=None,
        awaiting='User platform evaluation of verified v43', report='admet_diagnostics/v43_report.md',
        verification='admet_submissions/v43/local_verification.json', remote_model_verification='admet_model_outputs_v43/model_replay.json',
        ready_for_platform_evaluation=True, changed_task_count=len(selected), automatically_uploaded=False,
        local_gpu_training=False, target_achieved=False, completed_experiments=['v43'], delivered_on='2026-10-08')
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(file))
    print(str(path))


if __name__ == '__main__':
    main()
