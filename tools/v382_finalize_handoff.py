"""Finalize verified phase and a small dependency-explicit research handoff."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parents[1]
NAMES={'caco2_wang':'Caco2','ld50_zhu':'LD50','lipophilicity_astrazeneca':'脂溶性','ppbr_az':'PPBR',
 'solubility_aqsoldb':'溶解度','clearance_hepatocyte_az':'肝细胞清除率','clearance_microsome_az':'肝微粒体清除率',
 'half_life_obach':'半衰期','vdss_lombardo':'VDss'}
PARTS=('fold0','fold1','fold2','valid')


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def main():
    exp=ROOT/'experiments/regression9_v381'
    combined=read(ROOT/'docs/v382_delivery.json');single=read(ROOT/'docs/v381_delivery.json')
    verified=read(exp/'outputs/independent_verification.json');receipt=read(exp/'completion_receipt.json')
    assert verified['status']==receipt['status']=='PASS' and verified['signature']==receipt['signature']==single['signature']
    assert receipt['no_training_processes_running'] and read(exp/'run_status.json')['stage']=='EVIDENCE_READY'
    assert sha(ROOT/combined['csv'])==combined['csv_sha256']
    assert not combined['goal_mean_5percent_achieved_local']
    assert sum(3 for p in verified['base_records'] if p['part'] in ('fold0','fold1'))==216
    assert len([r for r in verified['records'] if r['part'] in ('fold0','fold1')])==52
    for r in single['final_model_states']:assert sha(ROOT/r['path'])==r['sha256']
    status=dict(status='COMPLETE',phase='v381 training and v382 experimental assembly',completed_at_utc=datetime.now(timezone.utc).isoformat(),
          official_best=51.271,new_official_score=None,goal_achieved=False,all_fitting_remote=True,
          delivery=combined,remote_completion_receipt=receipt,base_models_verified=verified['base_models'],
          prediction_packages_verified=verified['models'],model_replay_max_error=verified['full_replay_max_error'])
    for version in ('381','382'):save(ROOT/f'docs/v{version}_progress.json',status)
    conf=read(exp/'outputs/confirmation.json')
    lines=['# v377–v382 MAE研究阶段交付','',
     '**目标未达到。** 官方已知分数仍为51.271，新文件尚未获官方评分。已完成v377、v378、v379、v380、v381五轮，交付v382合并实验CSV。',
     '', '## 指标和目标','',
     '22项中5项MAE、4项Spearman，共9项连续值研究任务，另外13项分类任务按AUROC/AUPRC计分。目标定义为9项逐任务相对MAE降幅等权平均至少5%，拒绝更新计零，另报5项。分类概率MAE不等于官方提分。',
     '', '四项Spearman历史输出尺度任意，研究MAE基线是拟合标签分位数校准的历史预测秩，不是历史模型真实原单位误差。新专家用同一原单位预测检查MAE/排序。完整历史集成OOF仍不齐，公开验证长期复用，因此局部结果不能当作隐藏集收益。',
     '', '## 实际完整预算','', '|轮次|方法和完整开发预算|交付变化|','|---|---|---|',
     '|v377|54条树曲线、108种子训练，各3840轮；640/1920/3840快照|无|',
     '|v378|36条Transformer/检索曲线、72种子训练，各4800步|无|',
     '|v379|36条小容量/EMA/分位数曲线、72种子训练，各6000步|无|',
     '|v380|104个中位数核专家拟合，至固定容差收敛|脂溶性实验|',
     '|v381|216次基础SVR开发拟合，三折分组OOF中位数/高斯秩融合，52开发预测包|PPBR实验|',
     '|v382|仅按整任务合并两份不相交实验输出，无重训练或重搜权重|脂溶性与PPBR|',
     '', '所有拟合均在用户指定25870远端RTX3090服务器完成，无本地拟合、外部数据或预训练。每轮完成预定开发预算后才选配置，固定确认失败不换配方。没有依据短期验证表现停止曲线。',
     '', f"v381独立核验{verified['base_models']}个基础模型、{verified['models']}个融合预测包，重放最大误差{verified['full_replay_max_error']:.4g}。全部内部OOF分组隔离、源码/输入/模型哈希、查询标签扰动、小块核重建及PSD检查、线性规划原始/对偶证书通过。最终训练和核验进程已退出，GPU空闲仅对应回执时刻。",
     '', '## v381确认及未通过原因','', '|任务|fold0 MAE降幅|fold1|fold2|valid|结果|','|---|---:|---:|---:|---:|---|']
    for r in single['task_evidence']:
        vals={p['part']:f"{p['gain']:+.3%}" for p in r['parts']}
        reason='实验采用，稳定性未确认' if r['accepted'] else '确认有退步' if len(r['parts'])==4 else '开发未通过'
        lines.append('|'+ '|'.join([NAMES[r['task']],*[vals.get(p,'未运行') for p in PARTS],reason])+'|')
    lines += ['', '肝细胞与肝微粒体四区参考MAE虽改善，确认Spearman退步。半衰期开发参考MAE下降34%–44%，但valid反而增加32.3%。这些结果不写入CSV，不用最好的开发数字预测官方成绩。',
     '', '## 最终推荐手动测试文件','', f"`{combined['csv']}`",'',f"SHA256：`{combined['csv_sha256']}`。",'',
     '保留模板BOM、CRLF、全部非预测字段、Drug_ID字符串和行顺序。脂溶性840行、PPBR559行与各自来源CSV逐行完全一致，其余20项与51.271关联v374基线逐行一致。未自动提交比赛或发布GitHub。',
     '', '|采用更新|fold0|fold1|fold2|valid|','|---|---:|---:|---:|---:|']
    for r in combined['task_evidence']:
        lines.append('|'+NAMES[r['task']]+'|'+'|'.join(f"{p['gain']:.3%}" for p in r['parts'])+'|')
    lines += ['|9项平均降幅|'+'|'.join(f"{combined['nine_task_mean_relative_MAE_gain_by_part'][p]:.3%}" for p in PARTS)+'|',
              '|5项直接计分平均|'+'|'.join(f"{combined['five_scored_task_mean_relative_MAE_gain_by_part'][p]:.3%}" for p in PARTS)+'|','',
              '两个采用更新均未达到严格稳定性门槛，这份是用户已授权的实验CSV。不能承诺超过51.271。置信区间见下表，骨架分组bootstrap90%。',
              '', '|任务|分区|MAE相对降幅90%区间|','|---|---|---|']
    for version in ('380','381'):
        cf=read(ROOT/f'experiments/regression9_v{version}/outputs/confirmation.json')
        for r in cf['records']:
            if r['experimental_eligible']:
                for b in r['bootstrap']:lines.append(f"|{NAMES[r['task']]}|{b['part']}|{b['ci90'][0]:.3%} 至 {b['ci90'][1]:.3%}|")
    lines += ['', '## 误差诊断与下一阶段依据','',
     '完成后只读诊断未改变选型：半衰期基线最难10%样本贡献约73%–82%的总绝对误差，且不同分区极端标签分布有差异。PPBR拟合数据相同立体结构的冲突测量，其组内中位数最小绝对损失均摊约1.61–1.85单位。这只适用于同结构同输出假设，不是所有输入下的不可约误差，不据此删除标签。',
     '', '优先路线：先建立可完整重训、可重放的结构专家强基线和统一分组OOF，缩小当前专家弱于历史集成的差距；再对长尾、边界堆积和测量条件分别作消融。两种有前景但本轮失败的机制（中位数/排序分工、长预算Transformer）保留权重和失败记录，后续需明确改变表示、正则化或可观测条件，避免重复扫描确认集。',
     '', '长预算需与评估强度一起增加：新机制预先固定完整多种子训练、短中长快照和消融；保持原确认区作历史比较，新增未参与选型的分组嵌套评估，不称已经完全未被历史研究使用的盲测。未见稳定信号时扩大合理训练范围，不凭早期一两个数永久放弃算法族；也不无限重复相同失败配置。',
     '', '目前材料复核含相关42份源码定义、13份早期结果、97项扩展源码声明索引及20份精选原始页面。这些集合重叠，不能相加为独立算法数，也不是全球资料穷尽。新增页面支持分组OOF融合和最优性检查，不能证明本赛必然改善。',
     '', '旧备份增量盘点从v67/v68归档恢复并逐项哈希验证8个v67最终检查点，保存在独立目录，不覆盖旧文件。未执行这些模型，尚未恢复完整上游权重或完整OOF，不能宣称历史全模型已精确重放。',
     '', '分数条件算术：五项官方MAE同时降5%、其他17项不变约51.561；同幅降约12.1085%才约52。这是指标公式的条件推演，不是模型预期。',
     '', '## 继续工作入口','',
     '`tools/v381_oof_pareto.py`及冻结源位于`experiments/regression9_v381/`。冻结源码不可原地改签名复用缓存。远端核缓存和未采用模型仍保留，回执及运行日志完整下载。最终选中模型包含PPBR的内层/全量专家与融合状态；核缓存能由输入重建而无需拟合模型。',
     '', '本轮增量交接包依赖已有`releases/v380_complete_handoff.tar.gz`，来源哈希见`docs/v382_handoff_dependencies.json`，避免再次复制已有1.55GB输入。两者共同构成阶段交接；增量包单独不含全部数据。没有包含SSH密码或令牌。','']
    (ROOT/'docs/V382_MAE_RESEARCH_RESULTS_AND_HANDOFF.md').write_text('\n'.join(lines),encoding='utf-8')
    state=read(ROOT/'CURRENT_STATUS.json')
    state.update(latest_completed_model_branch='v381',latest_completed_model_branch_is_proven_best=False,
       latest_aggregate_plan='v377_v378_v379_v380_v381_v382',aggregate_status='REGRESSION9_V382_PHASE_COMPLETE_GOAL_NOT_MET',
       remote_training_running=False,official_score=51.271,target_achieved=False,current_experiment_progress='docs/v382_progress.json',
       latest_MAE_experimental_submission=combined,goal_nine_task_mean_MAE_five_percent_achieved=False,
       goal_five_scored_mean_MAE_five_percent_achieved=False)
    save(ROOT/'CURRENT_STATUS.json',state)
    prefix='\n'.join(['# 最新补充：v381训练和v382实验CSV交付','',
        '目标未达到：官方已知51.271，9项平均MAE降5%尚未达到。',
        '结果与继续入口：`docs/V382_MAE_RESEARCH_RESULTS_AND_HANDOFF.md`。',
        '推荐手动测试CSV：`'+combined['csv']+'`。仅脂溶性、PPBR更新，20项保持基线，稳定性未确认，官方未知。',
        '25870远端训练/核验完成。所有模型拟合在指定远端，未自动提交。','', '---',''])
    old=(ROOT/'HANDOFF_CURRENT.md').read_text(encoding='utf-8')
    if not old.startswith('# 最新补充：v381训练和v382实验CSV交付'):old=prefix+'\n'+old
    for p in (ROOT/'HANDOFF_CURRENT.md',ROOT/'docs/HANDOFF_CURRENT.md'):p.write_text(old,encoding='utf-8')
    dependency=read(ROOT/'docs/v380_release.json')
    assert dependency['status']=='PASS' and sha(ROOT/dependency['archive'])==dependency['sha256']
    save(ROOT/'docs/v382_handoff_dependencies.json',dict(required_base_archive=dependency,
         delta_alone_is_not_full_input_package=True,final_kernel_cache_rebuild_requires_authorized_remote_cuda=True,
         remote_root='/root/autodl-tmp/regression9_v381',all_fitting_remote=True,credentials_included=False))
    files=[ROOT/'CURRENT_STATUS.json',ROOT/'HANDOFF_CURRENT.md',ROOT/'docs/HANDOFF_CURRENT.md',
       ROOT/'tools/v377_deliver_regression.py',ROOT/'tools/v378_research_vs_deployment.py',ROOT/'tools/v379_collect_auxiliary.py',ROOT/'tools/v379_unpack_auxiliary.py',
       ROOT/'releases/v381_oof_delta.tar.gz']
    files+=list((ROOT/'tools').glob('v381_*.py'))+list((ROOT/'tools').glob('v382_*.py'))
    files+=list((ROOT/'docs').glob('v381_*.json'))+list((ROOT/'docs').glob('v382_*.json'))+list((ROOT/'docs').glob('V381_*.md'))+list((ROOT/'docs').glob('V382_*.md'))
    files += [p for p in exp.rglob('*') if p.is_file() and ('/outputs/' in p.as_posix() or p.suffix in ('.json','.py','.log'))]
    files += [p for p in (ROOT/'experiments/v381_legacy_checkpoint_recovery').rglob('*') if p.is_file()]
    files += list((ROOT/combined['csv']).parent.glob('*'))+list((ROOT/single['csv']).parent.glob('*'))
    files=sorted(set(p for p in files if p.name not in ('v382_release.json','v382_release_manifest.json')))
    entries=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in files]
    manifest=ROOT/'docs/v382_release_manifest.json';save(manifest,dict(files=entries,credentials_included=False,depends_on='docs/v382_handoff_dependencies.json'))
    archive=ROOT/'releases/v382_handoff_delta.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for p in files+[manifest]:tf.add(p,arcname=p.relative_to(ROOT).as_posix())
    with tarfile.open(archive) as tf:
        for row in entries:assert hashlib.sha256(tf.extractfile(row['path']).read()).hexdigest()==row['sha256']
    release=dict(status='PASS',archive=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,
                 members=len(entries)+1,all_member_hashes_verified=True,required_base_archive=dependency['archive'])
    save(ROOT/'docs/v382_release.json',release)
    print(json.dumps(dict(status='PHASE_COMPLETE_GOAL_NOT_MET',csv=combined['csv'],sha256=combined['csv_sha256'],release=release),ensure_ascii=False))


if __name__=='__main__':main()
