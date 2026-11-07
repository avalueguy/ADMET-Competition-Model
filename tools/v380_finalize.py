"""Finalize independently verified kernel result and complete handoff."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import tarfile

ROOT=Path(__file__).resolve().parents[1]
NAMES={'caco2_wang':'Caco2','ld50_zhu':'LD50','lipophilicity_astrazeneca':'脂溶性','ppbr_az':'PPBR',
       'solubility_aqsoldb':'溶解度','clearance_hepatocyte_az':'肝细胞清除率','clearance_microsome_az':'肝微粒体清除率',
       'half_life_obach':'半衰期','vdss_lombardo':'VDss'}


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    exp=ROOT/'experiments/regression9_v380'
    best=read(ROOT/'docs/current_best_after_v374_20261105.json')
    delivery=read(ROOT/'docs/v380_delivery.json')
    verified=read(exp/'outputs/independent_verification.json')
    completion=read(exp/'completion_receipt.json')
    assert sha(ROOT/best['csv'])==best['csv_sha256']
    assert verified['status']==completion['status']=='PASS'
    assert read(exp/'run_status.json')['stage']=='EVIDENCE_READY'
    assert verified['signature']==completion['signature']==delivery['signature']
    assert completion['no_training_processes_running']
    conf=read(exp/'outputs/confirmation.json')
    progress=dict(status='COMPLETE',version='v380',completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  official_best=51.271,new_official_score=None,delivery=delivery,
                  remote_completion_receipt=completion,all_fitting_remote=True,
                  goal_achieved_local=delivery['goal_mean_5percent_achieved_local'])
    save(ROOT/'docs/v380_progress.json',progress)
    lines=['# v377–v380 九项连续任务实验交付','',
           '官方已知最好仍为51.271。本次目标是9项相对MAE降幅等权平均至少5%，另报5项直接MAE计分；没有通过更新的任务计零。目标尚未达到，未进行平台自动提交。','',
           '实际任务范围：5项MAE、4项Spearman；其余13项分类任务的概率MAE下降不等价于官方AUROC/AUPRC提升。','',
           '|实验|完整开发预算|结果|','|---|---|---|',
           '|v377|54条树曲线、108完整种子训练，各3840轮|无采用更新|',
           '|v378|36条Transformer曲线、72完整种子训练，各4800步|无采用更新|',
           '|v379|36条小容量/EMA/分位数曲线、72完整种子训练，各6000步|无采用更新|',
           '|v380|104个凸核专家拟合，全部求解至收敛；6项进入固定确认|'+(', '.join(NAMES[t] for t in delivery['accepted_tasks']) or '无采用更新')+'|','',
           '所有模型拟合都在指定端口25870远端RTX3090服务器完成。没有依据短期验证提前终止曲线。每轮只由fold0/1选唯一配置与权重，再冻结确认fold2/valid，失败没有回退重选。','',
           'v380新比较：六片段计数L1归一化开平方，显式映射的多项式核或Hellinger映射高斯核；epsilon=0绝对误差SVR。四排序任务另比较固定加权log1p目标。该加权是原单位风险的局部近似。','',
           f"独立核验{verified['models']}个预测包，最大重放误差{verified['full_replay_max_error']:.4g}；核块重建、查询标签扰动、源码/输入/模型哈希通过。",'',
           '|任务|fold0 MAE降幅|fold1|fold2|valid|采用|',
           '|---|---:|---:|---:|---:|---|']
    for r in delivery['task_evidence']:
        vals={p['part']:f"{p['gain']:+.3%}" for p in r['parts']}
        lines.append('|'+ '|'.join([NAMES[r['task']],*[vals.get(p,'未进入确认') for p in ('fold0','fold1','fold2','valid')],
                                    '实验采用，稳定性未确认' if r['accepted'] else '否'])+'|')
    lines += ['',f"部署后9项平均相对MAE降幅：{' / '.join(f'{x:.3%}' for x in delivery['nine_task_mean_relative_MAE_gain_by_part'].values())}（fold0/1/2/valid）。",
              f"5项直接计分平均降幅：{' / '.join(f'{x:.3%}' for x in delivery['five_scored_task_mean_relative_MAE_gain_by_part'].values())}。",'']
    if delivery['csv']:
        assert sha(ROOT/delivery['csv'])==delivery['csv_sha256']
        lines += [f"交付CSV：`{delivery['csv']}`。",f"SHA256：`{delivery['csv_sha256']}`。",
                  f"仅替换{', '.join(NAMES[t] for t in delivery['accepted_tasks'])}，其他{delivery['unchanged_tasks']}项逐行保留原基线；模板核验通过。",
                  '这是已获用户授权的实验提交文件，未达到严格稳定性门槛，新官方分数未知。', '']
        for r in conf['records']:
            if r['experimental_eligible']:
                for b in r['bootstrap']:
                    lines.append(f"{NAMES[r['task']]} {b['part']}：{b['groups']}组，MAE相对降幅90%区间{b['ci90']}。")
    else:
        lines += ['无满足检查的新CSV，保留现有官方最好关联文件。','']
    lines += ['', '研究信号与提分必须区分：肝细胞、肝微粒体及VDss仍有四区参考MAE下降，但至少一处Spearman退步，未写入CSV。它们不能算作官方收益。',
              '四项排序任务的历史输出尺度任意，当前MAE参考来自拟合标签经验分位数校准的历史秩，并非恢复历史真实原单位预测。历史集成OOF部分为代理，公开验证反复使用，所有局部改进不能等同隐藏集改进。','',
              '材料覆盖：42份相关方法定义、13份早期结果、97项扩展源码声明索引及重点文件复读，另有17份精选原始论文/软件页面记录；这些集合重叠，不能相加为独立方法数，也不是穷尽世界研究。',
              '可用方法包括原单位中位数/分位数、物理残差、真实内部OOF局部校正、树、图与特征Transformer、自监督/迁移、检索、重复测量噪声、分子配对和融合。重复配置和失败结果保持，不将同算法重新命名为创新。','',
              '下一阶段优先补齐可重训强基线和完整分组OOF，再以同一预测检查MAE/排序的Pareto改进；v372已有四专家嵌套融合失败，下一轮须明确与它不同的可检验变化。对存在测量异质性的任务，只使用输入中可观测条件；不凭猜测删标签或编造pH。',
              '对末端仍改善的组合可以预先固定更长预算；对过拟合组合比较正则化及输入，不无限延长相同配置。单轮失败不等于永久放弃架构，但无证据不能承诺几分跃升。','',
              '分数条件算术：五项官方MAE同时降5%、其他17项固定约51.561；同幅降约12.1085%才约52。这不是本模型预估分。','']
    (ROOT/'docs/V380_COMPLETE_RESULTS_AND_HANDOFF.md').write_text('\n'.join(lines),encoding='utf-8')
    state=read(ROOT/'CURRENT_STATUS.json')
    state.update(latest_completed_model_branch='v380',latest_completed_model_branch_is_proven_best=False,
                 latest_aggregate_plan='v377_v378_v379_v380',aggregate_status='REGRESSION9_V380_COMPLETE',
                 remote_training_running=False,official_score=51.271,current_experiment_progress='docs/v380_progress.json',
                 goal_nine_task_mean_MAE_five_percent_achieved=delivery['goal_mean_5percent_achieved_local'],
                 goal_five_scored_mean_MAE_five_percent_achieved=delivery['five_scored_goal_achieved_local'])
    if delivery['csv']:state['latest_MAE_experimental_submission']=delivery
    save(ROOT/'CURRENT_STATUS.json',state)
    prefix='\n'.join(['# 最新补充：v380核专家完整实验已完成','',
                      '官方最好51.271，9项平均MAE降5%尚未达到。',
                      '实验交付与全部结果：`docs/V380_COMPLETE_RESULTS_AND_HANDOFF.md`。',
                      ('新实验CSV：`'+delivery['csv']+'`，官方未知。' if delivery['csv'] else '无新CSV。'),
                      '训练和核验已退出，空闲状态对应completion_receipt.json的记录时刻。','','---',''])
    old=(ROOT/'HANDOFF_CURRENT.md').read_text(encoding='utf-8')
    if not old.startswith('# 最新补充：v380核专家完整实验已完成'):old=prefix+'\n'+old
    for path in (ROOT/'HANDOFF_CURRENT.md',ROOT/'docs/HANDOFF_CURRENT.md'):path.write_text(old,encoding='utf-8')
    files=[ROOT/'CURRENT_STATUS.json',ROOT/'HANDOFF_CURRENT.md',ROOT/'docs/HANDOFF_CURRENT.md',
           ROOT/'releases/v379_complete_handoff.tar.gz',ROOT/'releases/v380_histogram_delta.tar.gz',
           ROOT/'tools/v377_deliver_regression.py',ROOT/'tools/v379_collect_auxiliary.py',ROOT/'tools/v379_unpack_auxiliary.py',
           ROOT/'tools/v378_research_vs_deployment.py']
    files+=list((ROOT/'tools').glob('v380_*.py'))+list((ROOT/'docs').glob('v380_*.json'))+list((ROOT/'docs').glob('V380_*.md'))
    files+=[p for p in exp.rglob('*') if p.is_file() and ('/outputs/' in p.as_posix() or p.suffix in ('.json','.py','.log'))]
    if delivery['csv']:files+=list((ROOT/delivery['csv']).parent.glob('*'))
    files=sorted(set(p for p in files if p.name not in ('v380_release.json','v380_release_manifest.json')))
    entries=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in files]
    manifest=ROOT/'docs/v380_release_manifest.json'
    save(manifest,dict(files=entries,credentials_included=False,inherited_inputs='Prior verified complete handoff nested.',nonselected_checkpoints_remain_remote=True))
    archive=ROOT/'releases/v380_complete_handoff.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for p in files+[manifest]:tf.add(p,arcname=p.relative_to(ROOT).as_posix())
    with tarfile.open(archive) as tf:
        for row in entries:assert hashlib.sha256(tf.extractfile(row['path']).read()).hexdigest()==row['sha256']
    release=dict(status='PASS',archive=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,
                 members=len(entries)+1,all_member_hashes_verified=True)
    save(ROOT/'docs/v380_release.json',release)
    print(json.dumps(dict(status='COMPLETE',csv=delivery['csv'],goal=delivery['goal_mean_5percent_achieved_local'],release=release)))


if __name__=='__main__':main()
