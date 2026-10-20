"""Publish a verified experimental artifact without uploading to the platform."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'admet_model_outputs_v46'

def main():
    submission=next((ROOT/'admet_submissions/v46').glob('*.csv'))
    verify=json.loads((submission.parent/'local_verification.json').read_text(encoding='utf-8'))
    model=json.loads((OUT/'model_replay.json').read_text(encoding='utf-8'))
    baseline=json.loads((OUT/'baseline_file_audit.json').read_text(encoding='utf-8'))
    report=json.loads((OUT/'analysis.json').read_text(encoding='utf-8'))
    diagnostics=json.loads((OUT/'diagnostics.json').read_text(encoding='utf-8'))
    assert verify['status']==model['status']==baseline['status']=='PASS'
    assert report['status']=='FROZEN'
    assert verify['sha256']==hashlib.sha256(submission.read_bytes()).hexdigest()
    selected=[r for r in report['tasks'] if r['selection']]
    lines=['# v46 任务专用图网络微调','',
        '基线为用户报告50.682048的v45，较v44增加0.099340。该分数与最近交付文件关联，平台文件哈希和最高分/最近分显示语义未核实。',
        '',
        '先修正历史比较基线：v45搜索读取的是v44目录内代表v43的数组，没有重放v44当轮变动。本轮先按冻结权重重放v44，再按本机实际交付v45的export.json重放v45。远程旧v45导出与本机实际提交文件不同，因此以本机v45的SHA256固定输入，未覆盖历史远程文件；12个已交付任务的预测均由原始模型结果逐值精确重放通过。',
        '',
        '本轮对13个分类任务，从各分区v27多任务有向化学键网络的5500步checkpoint开始，分别做300步任务微调，使用无正类重加权BCE、0.15成对排序损失、0.02原模型logit蒸馏。编码器学习率0.00008，任务头0.0003，固定180/300步快照平均。双种子独立微调后平均；原图/原混合、微调图/微调混合以及两个固定平均分支共同比较。',
        '',
        '每个任务完成官方valid和三折骨架训练后选择；未用valid正收益提前淘汰。目标0.20×valid增量+0.80×CV平均增量−0.05×CV标准差，要求CV均值和目标均>0.0005，至少两折正收益。允许局部下降；不存在每任务不退步的保证。冻结后才重训train+valid并预测测试集。最终原图对照和微调均使用相同两个种子；历史v45的图源曾使用三个最终种子。',
        '',
        '训练、测试推理及checkpoint重载全部在Qwen服务器czp123@100.109.143.42的RTX 4090完成。新实验没有下载外部数据或读取测试目标；沿用历史图模型及结构特征来源。本机仅编辑源码、接收轻量预测和提交文件，以标准库复算融合及格式，没有本机GPU训练。',
        '',
        '| 任务 | 分支 | 新分支权重 | 历史valid指标增量 | 三折CV平均增量 | CV下降折 |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    for r in selected:
        s=r['selection']
        negative=', '.join(f'{i+1}: {g:+.6f}' for i,g in enumerate(s['cv_gains']) if g<0) or '无'
        lines.append(f"| {r['task']} | {s['name']} | {s['weight']:.2f} | {s['valid_gain']:+.6f} | {s['cv_mean']:+.6f} | {negative} |")
    lines+=['','所有指标增量均为AUROC/AUPRC绝对增量，不是官方总分。','',
        '| 任务 | 微调分支入选 | 相同权重下微调相对原图CV增量均值 | 留一折选型诊断均值 | valid骨架重采样95%区间 |',
        '| --- | --- | ---: | ---: | --- |']
    for r in diagnostics['tasks']:
        fine=sum(r['fine_tune_vs_original_cv_gains'])/3
        ci=r['valid_bootstrap_interval']
        text=f'[{ci[0]:+.6f}, {ci[1]:+.6f}]' if ci else '不可计算'
        lines.append(f"| {r['task']} | {r['fine_tune_selected']} | {fine:+.6f} | {r['leave_one_fold_mean_gain']:+.6f} | {text} |")
    lines+=['',
        '以上为冻结选择后的诊断，没有据此重新选择权重。历史valid与CV标签被多轮开发复用，基线CV也不是完整历史集成的重新OOF训练。重采样和留一折诊断不消除搜索偏差，不能视为独立验证；单骨架valid的重采样区间尤其缺乏泛化代表性。官方分项与总分公式未知，无法保证官方上涨或达到指定目标。',
        '',
        f"最终文件：{submission}",
        f"SHA256：{verify['sha256']}",
        f"格式：{verify['rows']}行、{verify['columns']}列、{verify['tasks']}任务，{verify['bytes']}字节，UTF-8 BOM、CRLF；非预测字段与官方模板逐行完全一致，任务/sample_idx键唯一，分类值在[0,1]且所有预测有限。修改{len(verify['changed_tasks'])}个任务、{verify['changed_rows']}行，其余任务与v45原预测字符串一致。",
        '',
        '独立核验包括：最终及验证分区所有入选任务的双种子、双快照重载；图源与微调查询骨架隔离；源预测精确比对；单样本/批次/排列不变性；污染隐藏查询标签不改变预测；标准库独立重放百分位秩融合。所有检查通过。大型checkpoint保留远程，未自动上传比赛。',
        '',
        'v45原文件及50.682048成绩保留作为回退。本次只交付一个v46实验候选，实际分数由平台评测确定。']
    path=ROOT/'admet_diagnostics/v46_report.md'
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    state_path=ROOT/'admet_current_candidate.json'
    state=json.loads(state_path.read_text(encoding='utf-8'))
    state.update(experimental_release='v46',experimental_status='EXPERIMENTAL_AWAITING_PLATFORM',
        ready_for_platform_evaluation=True,experimental_submission=str(submission),
        experimental_sha256=verify['sha256'],changed_task_count=len(verify['changed_tasks']),
        awaiting='User platform evaluation of verified v46',experimental_report=str(path),
        remote_model_verification=str(OUT/'model_replay.json'),local_gpu_training=False,
        automatically_uploaded=False,delivered_on='2026-10-15',
        completed_experiments=sorted(set(state.get('completed_experiments',[])+['v45','v46'])))
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(submission=str(submission),report=str(path),sha256=verify['sha256'],
        selected_tasks=list(verify['changed_tasks']),platform_score=None),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
