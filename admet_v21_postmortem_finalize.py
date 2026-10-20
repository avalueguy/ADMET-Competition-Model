"""Record the reported outcome and produce an audit, without retraining."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'admet_diagnostics/v21_postmortem'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d): Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')

files=read(OUT/'files.json')
data=read(OUT/'remote_data.json')
remote=read(OUT/'remote_metrics.json')
proof=read(ROOT/'admet_diagnostics/v21_summary.json')
init=read(OUT/'initialization.json')
assert files['changed_rows']==822 and len(data['tasks'])==22
assert sum(len(r['oof_fold_checks']) for r in data['tasks'])==66
assert not init['cold_equals_warm'] and init['warm_equals_warm']
weak=[r for r in remote if 'pure_candidate_valid' in r and
      (r['pure_candidate_valid']>r['baseline_valid'] if r['metric']=='MAE' else r['pure_candidate_valid']<r['baseline_valid'])]
assert len(weak)==11

outcome=dict(version='v21',score=50.019727,previous_version='v20',previous_score=50.019727,
             observed_display_gain=0.0,source='user report',
             submission_sha256=files['candidate_sha256'],file_association='inferred from conversation; server file hash unavailable',
             platform_display_semantics='unverified: latest submission versus historical best/aggregate',
             release_status='NO_OBSERVED_GAIN_INSUFFICIENT_VALIDATION_EVIDENCE',
             recorded_on='2026-09-27',audit='admet_diagnostics/v21_postmortem/report.md')
write(ROOT/'admet_submissions/v21/platform_score.json',outcome)
current=read(ROOT/'admet_current_candidate.json')
backup=OUT/'candidate_state_before_audit.json'
if not backup.exists(): write(backup,current)
current.update(recommended_release='v20',submission=files['baseline'],sha256=files['baseline_sha256'],
               platform_anchor_release='v20',platform_anchor_score=50.019727,candidate_platform_score=50.019727,
               observed_display_gain=0.,experimental_release='v21',experimental_submission=files['candidate'],
               experimental_sha256=files['candidate_sha256'],experimental_status=outcome['release_status'],
               awaiting='Evidence meeting the preregistered experiment gate; no new upload candidate',
               report='admet_diagnostics/v21_postmortem/report.md',verification='admet_diagnostics/v21_postmortem/files.json',
               score_source='user report; platform display semantics unverified',target_achieved=False)
write(ROOT/'admet_current_candidate.json',current)
historypath=ROOT/'admet_diagnostics/platform_score_history.json'
history=read(historypath)
if not any(str(r.get('version')) in ['21','v21'] for r in history):
    history.append(dict(outcome,reported_score=50.019727,latest_improvement=0.,aggregation_formula='unknown'))
    write(historypath,history)

lines=['# v21 无可确认提升：模型与评测审计','',
       '**结论：撤回 v21 作为升级推荐，保留 v20（用户报告 50.019727）作为本地基线。本轮没有训练或生成新提交。**','',
       '用户报告 v21 后显示值仍为 50.019727，观察到的显示增量为零。不能据此确定盲测指标逐项相同，也不能反推具体任务失分；平台没有公开总分变换/权重，且本次服务器评测记录未取得。此前发布判断偏重重复使用的验证集均值，证据不足，应先纠正评测方法。','',
       '## 1. 提交身份与格式','',
       f'- v20 SHA256：`{files["baseline_sha256"]}`。',
       f'- v21 SHA256：`{files["candidate_sha256"]}`；{files["candidate_bytes"]:,} 字节。',
       '- 原 v21 目录中的 CSV 在审计开始时缺失，已从本地审查归档恢复精确字节，哈希与原交付记录相同。缺失原因未知，不能据此认定用户传错文件。',
       '- 16,379 行、22 任务、6 列、UTF-8 BOM、CRLF；非预测字段逐行等于官方模板；sample_idx 连续且唯一；预测有限，分类范围合格。',
       '- 改动仅 3 项、822 行，其余 19 项原样沿用。独立重算融合与恢复文件一致。','',
       '| 任务 | 变化行 | 平均绝对预测变化 | 严格排序翻转对数 | 新旧 Spearman |',
       '|---|---:|---:|---:|---:|']
for r in files['tasks']:
    if r['changed']:
        lines.append(f'| {r["task"]} | {r["changed"]} | {r["mean_abs_delta"]:.6f} | {r["strict_pair_reversals"]} | {r["spearman"]:.6f} |')
lines+=['','这些不是相同文件，也不是单纯单调缩放；排序确实变化。排序翻转不说明方向正确，不能推出官方加分。','',
        '## 2. 模型效果与筛选缺陷','',
        '新微调模型单独使用时，12 项中 11 项在官方 valid 上弱于基线，只有 PPBR 的 MAE 更好。较弱模型有时能提供互补性，但不能把微调参数更多等同于模型变强。valid 已参与历次筛选，以下只能作为诊断。','',
        '| 任务 | 指标 | v20 valid | 微调单模型 valid |', '|---|---|---:|---:|']
for r in remote:
    if 'pure_candidate_valid' in r:
        lines.append(f'| {r["task"]} | {r["metric"]} | {r["baseline_valid"]:.6f} | {r["pure_candidate_valid"]:.6f} |')
lines+=['','MAE 越低越好，其余指标越高越好。以下是最终融合方案的证据：','',
        '| 任务 | valid 收益 | 换一次骨架划分的收益 | valid 条件 bootstrap 95% 区间 |',
        '|---|---:|---:|---|']
for r in proof['audits']:
    if r['selection']['weight']:
        lo,hi=r['uncertainty']['percentile_95']
        lines.append(f'| {r["task"]} | {r["selection"]["valid_gain"]:+.6f} | {r["repeat"]["gain"]:+.6f} | [{lo:+.6f}, {hi:+.6f}] |')
lines+=['',
        '- CYP2C9 的重复复核 AUPRC 从 0.355781 降到 0.329176；PPBR 的 MAE 从 8.198796 升到 8.295319。三项区间均跨零，且未计入多轮筛选误差。',
        '- 生物利用度 valid 只有 64 行（43 正、21 负），AUROC +0.001107 等于 1/(43×21)，仅净改善一对正负样本次序。两个 OOF 折同时下降。',
        '- 筛选复用了历史 3 折和官方 valid，并在 12 个任务中搜索融合模式与权重；重复复核被明确设置成“不否决”。这些合起来会放大偶然收益。用户允许单项下降并不意味着可以忽略整体证据不稳。',
        '- CYP2C9 基线为排名置信度，新模型为 sigmoid 输出，直接 raw 融合的有效比例受尺度影响。候选 valid 标准差 0.128863、test 0.183536；这不是已证实的错误，但固定 30% 权重不代表稳定的排名贡献。',
        '- 两种子的最佳训练轮数不稳定：生物利用度 valid 为 [2,2]，最终为 [22,4]；PPBR valid 为 [6,8]，最终为 [18,20]。全量重训改变分布、早停切分和轮数，valid 收益不能无条件迁移。',
        '- 本次独立融合计算发现极近的排名并列对运算顺序敏感，生物利用度 fold0 AUROC 有约 0.0001295 的重算差异。它不足以解释平台不变，但说明应统一 tie 处理与按导出字节回放指标，不能追逐万分位变化。','',
        '## 3. 全部 22 项数据与划分','',
        '实际包为 53,695 条 train 标签、11,735 条 valid 标签、16,379 条盲测行，共 22 个独立任务。不是说明中“十万独立分子×完整标签矩阵”。跨任务存在分子复用；本审计从未读取盲测标签。',
        '22 项 train/valid/test 两两之间的规范分子、非立体分子和当前骨架标识交集均为零，盲测 SMILES/ID 与模板对应。远程 66 个 OOF 折的规范分子和当前骨架标识均无交叉；但这不等于严格低 Tanimoto 相似度。','',
        '| 任务 | train / valid / test | train 重复规范分子行 | train 标签冲突分子组 | valid 基线指标 |',
        '|---|---|---:|---:|---:|']
metrics={r['task']:r for r in remote}
for r in data['tasks']:
    t=r['splits']['train']; v=r['splits']['valid']; te=r['splits']['test']
    lines.append(f'| {r["task"]} | {t["rows"]} / {v["rows"]} / {te["rows"]} | {t["duplicate_canonical_rows"]} | {t["canonical_conflicting_label_groups"]} | {metrics[r["task"]]["baseline_valid"]:.6f} |')
lines+=['',
        '- PPBR train 1,952 行只有 1,282 个规范分子，481 个分子组带有不同观测值；valid 279 行只有 172 个规范分子。只输入分子结构的确定性模型无法同时精确拟合同分子的相互冲突观测。按 valid 分子组使用事后最优中位数得到的 MAE 下界为 2.377061，此为诊断下界，不能用于预测或作为可达成绩。',
        '- 肝细胞清除率 train 848 行有 136 行规范分子重复且存在观测冲突。这类噪声应建模并做消融；不能随意删标签，也不能改动评估行权重。',
        '- 非立体 SMILES 会消除手性。v21 的文本支路使用非立体化是匹配预训练表示的选择，但应通过专门的手性指纹/图支路补充，不能简单换成预训练词表不匹配的字符串就宣称提升。',
        '- 发现当前 OOF 的“ACYCLIC+立体规范式”分组允许非立体完全相同的立体异构体跨折：AMES 各折 42/42/40 个非立体键，Caco2 为 1/0/1，CYP3A4 Veith 为 3/3/2。对不识别手性的支路属于验证隔离漏洞；三项均未在 v21 新提交中改动，不能解释本次结果，但必须在下一轮分组时修复。',
        '- 12 项 token 缓存仅 VDSS 有 1 行在两个视图超过 256 tokens；该任务未入选。v21 改动的三项没有截断，所以“截断导致本次失败”不受证据支持。','',
        '## 4. 训练实现与指标细节','',
        '- 远程只读检查覆盖 54 个训练阶段的权重/预处理/预测缓存哈希与元数据。保存时的权重重载回放记录通过；本轮没有重新运行 GPU 推理或训练。OOF 行、Y、fold 顺序均核对。',
        '- 特征标准化和目标变换在每次 fit 内拟合；内部骨架留出只用拟合行；早停后重训全部 fit。未发现本轮直接把 OOF/valid 的目标放入参数拟合的问题。存在的主要问题是外层反复选型，而非这里的显式行混入。',
        '- CPU 上复现了初始化缺陷：相同 seed=21271，首次加载和热缓存加载的预测头不同，最大权重差 0.0666062；热加载重复一致。原因是 pretrained() 调用了 torch.manual_seed(21200)。正式训练首个 CYP2D6 任务可能受影响，但该任务最终权重为零；入选三项的 final/repeat 入口又提前初始化，不能将此缺陷说成平台结果原因。',
        '- 协议称“residual prediction head”，实现是普通拼接 MLP，没有残差连接；encoder_dropout 字段未被直接应用，但实际预训练层 dropout 都是 0.1，与声明一致。属于可复现/文档问题，不应包装为创新。',
        '- RobustTarget 的 Spearman 逆变换在训练标签区间两端饱和，可能造成并列；本次没有 Spearman 项被换入提交，影响限于未来实验。',
        '- 前缀缓存 key 没有包含完整生成代码/精度/依赖哈希，训练缓存主要约束顶层源文件。未来改 helper 后复用旧缓存有风险，需要完整依赖清单。',
        '- AUPRC 使用 sklearn average_precision_score。已只读核查 TDC 当前公开 evaluator.py，其 pr-auc 同样映射到 average_precision_score。没有证据支持本轮靠替换成梯形 PR-AUC 解决问题；赛事后端实现仍未公开。','',
        '## 5. 为什么显示分数完全一样：尚未获得的证据','',
        '公开前端读取后端返回的 score/total_score；一条排名展示路径用 toFixed(3)，另一条总分路径直接显示 total_score。用户提供的是六位小数，不能用三位小数的展示函数解释这次相同。前端没有公开后端如何选取最新/最高成绩、分数合成和每次评测状态。',
        '本轮读取可用浏览器状态时没有打开的页面，也未取得登录态提交历史，因此没有核实服务器收到的文件哈希或本次任务完成状态。缺少这些证据时，不归因于用户传错文件、后台故障或必然退步。能确认的是“未观察到显示提升”，以及本地支持涨分的证据不足。','',
        '## 6. 下一轮采用的判定标准','',
        '已保存 admet_next_experiment_protocol.json，仅是事前设计，不是已训练新版本。先修随机种子、分组隔离、缓存溯源和评分回放，再做两个固定预算的消融：PPBR 小容量/噪声处理，以及稀疏分类的手性支路与统一融合尺度。',
        '使用预先固定的 5 个种子×3 个外层骨架折；基线和候选在同一拟合行上重训，所有融合选型进入内层。官方 valid 退出调参，只作历史诊断。新切分仍来自旧标签，不能假称全新独立验证集。',
        '允许个别任务下降，但要求预先定义的组合收益在至少 4/5 个外层种子均值中为正，按骨架分组并对固定候选数校正后的条件区间下界为正；完整报告所有任务和失败实验。总分公式未知，组合指标必须标为代理指标，不能折算为“必涨 0.6”或“保证 51”。',
        '当前 v21 不满足以上证据要求，因此不制作 v22 上传文件。下一轮若仍未通过就保留 v20，不通过更换文件名、继续微调同一验证集来宣布升级。','',
        '## 本轮产物与复现','',
        '- admet_v21_postmortem.py：文件/数据/只读远程指标/CPU初始化/公开前端片段审计。',
        '- files.json、data.json、remote_data.json、remote_metrics.json、initialization.json：逐项证据。',
        '- public_sources/provenance.json：公开 TDC 评估实现和当前官网 JS 来源/哈希；当前官网 JS 与此前缓存一致。',
        '- admet_submissions/v21/platform_score.json：用户反馈登记，明确成绩显示语义未核实。',
        '- admet_current_candidate.json：恢复 v20 推荐，v21 标为无已观察提升。',
        '- 未使用外部 assay 标签、盲测标签；未上传平台；未做本机或远程新增 GPU 训练。远程检查结束时 GPU 为 0%，显存 1 MiB。','']
(OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
write(OUT/'decision.json',dict(recommendation='keep_v20',new_submission_created=False,training_performed=False,
      insufficient_evidence_tasks=proof['changed'],official_gain_guaranteed=False,
      forecast=None,observed_display_gain=0.,report='report.md'))
print(json.dumps(dict(report=str(OUT/'report.md'),recommendation='v20',new_submission=False),ensure_ascii=False))
