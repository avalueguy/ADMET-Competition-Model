"""Preserve the scored baseline and describe actual selected pair-model evidence."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
    out=ROOT/'admet_model_outputs_v44';dest=ROOT/'admet_submissions/v44'
    file=next(dest.glob('*.csv'))
    verified=read(dest/'local_verification.json');report=read(out/'analysis.json');uncertainty=read(out/'uncertainty.json')
    assert verified['status']=='INDEPENDENT_PAIR_FUSION_AND_CSV_PASSED'
    assert report['signature']==uncertainty['signature'] and report['status']=='EVALUATED' and uncertainty['status']=='COMPLETE'
    selected=[r for r in report['tasks'] if r['selection']]
    assert selected
    names=['直接结构回归对照','差值模型前4邻居加权均值','差值模型前16邻居加权均值',
           '差值模型前16邻居加权中位数','三个差值聚合等权','直接与差值等权']
    lines=['# v44 跨骨架分子属性差值学习','',
        '用户确认v43得分50.498381，较v42增加0.003643，排名第9。v44是尚待平台评测的新候选，不能保证大幅涨分。最后报告第一名52.723518，超过其一分目标53.723518尚未达到。','',
        '## 本轮变化','',
        '从当前任务的拟合行中按非立体异构分子身份聚合实验标签，回归取中位数。每个拟合分子用Morgan2/FCFP3相似度寻找12个跨骨架训练分子；输入两个分子的描述符、ChemBERTa表示、差和绝对差、512维拟合内筛选指纹的差与公共片段以及相似度，目标为两分子的属性差值。所有正向对加入反向对，标签取负。训练两个GPU XGBoost种子，深度5/6、500/650棵树，绝对误差目标。成对样本高度相关，不能把分子对数当作独立实验数。','',
        '推理时只查找当前拟合行提供的16个跨骨架实验锚点。模型预测差值为正向预测与反向预测之差的一半，确保交换两个分子后差值取负。对每个锚点，以已知实验值加上预测差值重建目标值，再按结构相似度四次方加权均值或中位数聚合。MAE在拟合内robust缩放空间学习并还原原单位；Spearman使用拟合标签内高斯秩空间，保留连续排名。','',
        '同时训练同种子、相同目标和训练标签的直接结构回归对照。共比较6个固定选项：直接对照；前4、前16锚点均值；前16锚点中位数；三个差值聚合等权；直接与差值等权。','',
        '九个连续任务全部训练官方valid和三折骨架模型，不用官方valid正收益作为CV前置淘汰条件。骨架组与非立体异构身份在目标任务拟合行/查询行中无重叠；训练分子对与查询锚点也排除同骨架。没有使用其他任务实验属性或外部标签，没有读取测试目标标签。','',
        '预设融合比例0.15/0.35/0.60/0.85/1.0。通常选择目标为0.45×valid收益+0.55×CV平均收益-0.06×CV标准差。valid/test有环比例差超过0.5时，valid权重降到0.15，CV收益用全折与有环/无环分层各50%组合，分层按无标签的测试结构比例加权。要求目标>0.001、CV均值>0.0005、至少两折调整收益为正、valid收益>-0.02；允许局部下降。选择全部冻结后才查看新分区诊断，不根据诊断重新选择赢家。','',
        'GPU训练、预测与模型重载都在Qwen服务器czp123@100.109.143.42:/home/czp123/admet_v23_20260928完成，输出独立存于v44目录，CPU线程限制3。nvidia-smi因NVML驱动版本不匹配报错；实际CUDA矩阵运算和XGBoost GPU训练经过功能测试并可用，没有改服务器驱动或中断Qwen进程。本机只编辑源码和核对轻量预测，没有本机GPU训练和自动比赛上传。','',
        '## 冻结选型','',
        'MAE收益为相对误差下降；Spearman收益为相关系数绝对增量。它们不等于官方总分。','',
        '| 任务 | 指标 | 分支 | 新分支权重 | v43历史valid增量 | 三折代理平均增量 | 调整后CV均值 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in selected:
        s=r['selection'];fmt=(lambda a:f'{100*a:+.3f}%') if r['metric']=='MAE' else (lambda a:f'{a:+.6f}')
        lines.append(f"| {r['task']} | {r['metric']} | {names[s['variant']]} | {s['weight']:.2f} | {fmt(s['valid_gain'])} | {fmt(s['cv_mean'])} | {fmt(s['adjusted_cv_mean'])} |")
    lines+=['','## 对照与选型后诊断','',
        '以下是相同比例融合时，冻结赢家相对直接回归对照的收益。另在冻结选择后，对train+valid按标签无关随机种子重新抽取20%骨架组，重新训练同类模型；只比较冻结源分支和直接对照，不使用原集成预测，因为它已经见过这些标签。该诊断是新的划分，不是新数据或独立外部验证。','',
        '| 任务 | valid相对直接对照增量 | CV相对直接对照均值 | 新划分相对直接对照收益 | 新划分查询数 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in uncertainty['tasks']:
        fmt=(lambda a:f'{100*a:+.3f}%') if r['metric']=='MAE' else (lambda a:f'{a:+.6f}')
        lines.append(f"| {r['task']} | {fmt(r['matched_control_valid_gain'])} | {fmt(r['matched_control_cv_mean'])} | {fmt(r['fresh_control_gain'])} | {r['fresh_query_rows']} |")
    lines+=['','| 任务 | valid骨架重采样95%区间 | 留一折选型平均收益 |','| --- | --- | ---: |']
    for r in uncertainty['tasks']:
        fmt=(lambda a:f'{100*a:+.3f}%') if r['metric']=='MAE' else (lambda a:f'{a:+.6f}')
        interval='无法估计' if r['bootstrap_95'] is None else '['+', '.join(fmt(a) for a in r['bootstrap_95'])+']'
        lines.append(f"| {r['task']} | {interval} | {fmt(r['held_fold_mean'])} |")
    lines+=['','原始CV的下降折：','']
    negative=[(r['task'],j+1,g) for r in selected for j,g in enumerate(r['selection']['cv_gains']) if g<0]
    lines+=[f'- {task} 第{fold}折：{gain:+.6f}' for task,fold,gain in negative] or ['无。']
    lines+=['',
        '历史valid和CV反复复用，存在搜索与选择偏差。CV对比对象是历史集成代理，未重训完整历史集成OOF。新划分仍用已被历史开发使用的标签，仅为与直接对照的补充诊断；重采样区间和留一折诊断不修正多轮选型偏差。有环比例调整不能完全消除验证与测试的结构差异。官方计分公式和任务分项仍未知，不能把这些数字换算为比赛分数或保证突破。','',
        '## 文件与核验','',f'`{file.relative_to(ROOT).as_posix()}`','',
        f"16,379行、22任务、6列，{verified['bytes']:,}字节。UTF-8 BOM、CRLF，非预测字段与官方模板逐行完全一致，键唯一，预测有限，分类范围[0,1]。本轮更新{len(selected)}个任务，其他任务与已得50.498381的v43逐项一致。",'',
        '所有最终直接/差值模型重载推理通过；锚点与训练预处理器独立重建一致，目标标签污染不影响输入，交换分子后的差值取负，单分子/批次推理一致。本机仅加载小型JSON预测数组，独立重放融合结果并核对源码和CSV哈希，全部通过。Spearman秩平均使用与远程相同的数组求均值顺序，避免1e-16量级舍入改变近似并列预测的排序。','',
        f"SHA256：`{verified['sha256']}`。",'',
        '只交付一个v44实验候选。v43原文件和50.498381成绩留存作为回退，官方涨分需要平台实测。']
    path=ROOT/'admet_diagnostics/v44_report.md';path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    statepath=ROOT/'admet_current_candidate.json';state=read(statepath)
    assert state['platform_anchor_release']=='v43' and state['platform_anchor_score']==50.498381
    state.update(experimental_release='v44',experimental_submission=str(file),experimental_sha256=verified['sha256'],
        experimental_status='EXPERIMENTAL_AWAITING_PLATFORM',candidate_platform_score=None,awaiting='User platform evaluation of verified v44',
        report='admet_diagnostics/v44_report.md',verification='admet_submissions/v44/local_verification.json',
        remote_model_verification='admet_model_outputs_v44/model_replay.json',ready_for_platform_evaluation=True,
        changed_task_count=len(selected),automatically_uploaded=False,local_gpu_training=False,target_achieved=False,
        completed_experiments=['v44'],delivered_on='2026-10-12')
    statepath.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(file));print(str(path))

if __name__=='__main__':main()
