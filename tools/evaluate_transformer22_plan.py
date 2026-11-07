"""Read-only model feasibility calculations; writes a plan, never trains models."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from v184_score_objective_audit import normalized

OUT = ROOT / "docs/transformer22_plan.json"
DOC = ROOT / "docs/TRANSFORMER_22_TASK_PLAN.md"
RULES = ROOT / "docs/transformer22_rule_check.json"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rules():
    url = "http://www.aiinnovation.com.cn/api/app/getTrack"
    req = urllib.request.Request(url, data=urllib.parse.urlencode({"id": 159}).encode(),
                                 headers={"User-Agent": "Mozilla/5.0", "Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            x = json.load(response)
        d = x["data"]
        assert x["code"] == 20000 and d["id"] == 159
        plain = " ".join(re.sub("<[^>]+>", " ", d["describe"]).split())
        assert "不允许使用其他外部数据" in plain
        at = plain.index("不允许使用其他外部数据")
        return dict(status="LIVE_PUBLIC_API_CHECKED", url=url, method="POST form id=159",
                    title=d["road_title"], external_data_rule=plain[max(0, at-30):at+40],
                    edit_time=d.get("edit_time"), external_pretrained_weights_explicitly_authorized=False,
                    interpretation="Do not introduce external pretrained weights into the proposed new route without an explicit organizer clarification. No clarification requested or assumed.")
    except Exception as exc:
        return dict(status="LIVE_REFRESH_UNAVAILABLE", error_type=type(exc).__name__,
                    fallback="docs/official_rules_20261020.json", external_pretrained_weights_explicitly_authorized=False,
                    external_data_rule="Local recorded official rules prohibit other external data.")


# Independent task designs, all sharing audited data tools and optional encoders.
# These are experimental hypotheses, not implemented trainers or measured gains.
DESIGNS = {
    "ames": ("P2", "原子/键Graph Transformer＋反应性片段残差头", "BCE＋成对排序；同一分子等价视图一致性", "中等样本，解冻末2层；与SVC专家比较互补性"),
    "bbb_martins": ("P3", "冻结Graph Transformer＋脂溶性/极性描述符小头", "BCE＋排序", "当前AUROC .929，数学总分空间有限；小适配器"),
    "bioavailability_ma": ("P2", "Graph Transformer适配器＋吸收/代谢上下文分支", "BCE＋排序", "448条训练；上下文分支独立消融；不将HIA/PPBR测量视为必然可用"),
    "caco2_wang": ("P1", "Graph Transformer＋理化性质残差头", "原尺度L1/中位数回归；可选成对性质差辅助项", "637条训练；冻结主干，只训练小适配器和头；成对差路线需单独证明增量"),
    "clearance_hepatocyte_az": ("P1", "Graph Transformer适配器＋清除率任务专用头", "Gaussian-rank回归＋成对排序", "848条训练；同微粒体任务共享只作迁移初始化，独立微调；测量上下文独立消融"),
    "clearance_microsome_az": ("P2", "Graph Transformer适配器＋微粒体清除率专用头", "Gaussian-rank回归＋成对排序", "770条训练；不直接强制与肝细胞标签数值相同"),
    "cyp2c9_substrate_carbonmangels": ("P2", "底物家族迁移的Graph Transformer适配器", "BCE＋可微AP辅助项的单独消融", "仅91个训练正例；不以重采样后更高AP当原分布收益；保留当前强基线"),
    "cyp2c9_veith": ("P2", "CYP抑制家族Graph Transformer＋任务头", "BCE为对照，另测AP辅助损失", "8463条训练；可解冻末2层；比较结构Transformer相对当前核专家的增量"),
    "cyp2d6_substrate_carbonmangels": ("P2", "底物家族迁移适配器＋条件测量小分支", "BCE＋AP辅助项单独消融", "465条训练、134正例；小头、双种子；不能照搬大抑制任务全量微调"),
    "cyp2d6_veith": ("P1", "Graph Transformer＋CYP2D6抑制专用头", "BCE对照＋AP辅助目标消融", "9191条训练，最适合首批检查Transformer学习能力的任务之一"),
    "cyp3a4_substrate_carbonmangels": ("P2", "底物家族Graph Transformer小适配器", "BCE＋成对排序", "468条训练；必须对比当前恢复分支，不能重用已退步v354组合"),
    "cyp3a4_veith": ("P2", "抑制家族Graph Transformer＋CYP3A4头", "BCE＋AP辅助目标消融", "8628条训练；现有AUPRC .899，优先级低于CYP2D6抑制"),
    "dili": ("P3", "冻结Graph Transformer＋反应性片段小头", "BCE＋排序", "331条训练，官方AUROC .935；谨慎分配预算"),
    "half_life_obach": ("P2", "Graph Transformer适配器＋半衰期专用双目标头", "log1p与Gaussian-rank双目标＋排序", "465条训练；当前双目标树已有效，先检查新增表示互补性"),
    "herg": ("P1", "Graph Transformer小适配器＋电荷/芳香片段头", "BCE＋成对排序", "457条训练；新表示先冻结；旧模型已用理化特征，单加描述符不算新机制"),
    "hia_hou": ("KEEP", "保留v361完整输出；仅建立独立登记", "不启动新训练", "官方AUROC已1.0，没有正向分数空间"),
    "ld50_zhu": ("P2", "Graph Transformer＋毒性专用回归头", "原尺度L1为主；稳健尾部处理单独消融", "3970条训练；先核对单位和标签变换，不能仅追求训练集RMSE"),
    "lipophilicity_astrazeneca": ("P1", "Graph Transformer＋残差MAE头", "原尺度L1；以现有物理/检索专家为对照", "2940条训练；验证表示互补，校准仅用拟合/内层预测"),
    "pgp_broccatelli": ("P3", "冻结Graph Transformer＋P-gp小头", "BCE＋排序", "851条训练，官方AUROC .950；极限总分收益仅约.058"),
    "ppbr_az": ("P2", "Graph Transformer＋有界中位数回归头", "原尺度L1；[0,100]输出或固定边界裁剪", "1952条训练；直接优化PPBR误差，不能以变换尺度的损失替代最终MAE"),
    "solubility_aqsoldb": ("P1", "Graph Transformer＋溶解度MAE头", "原尺度L1；重复测量不确定性作为可选消融", "5045条训练；不能把结构增强当作新增独立标签；标签检索已试过，必须检验新增结构表示"),
    "vdss_lombardo": ("P2", "Graph Transformer适配器＋VDss排序头", "Gaussian-rank回归＋成对排序", "791条训练；理化性质/PPBR关系作为辅助，不强加未经证实的确定方程"),
}


def main():
    assert not OUT.exists() and not DOC.exists() and not RULES.exists(), "Inspect existing plan before rewriting."
    card_path = ROOT / "docs/platform_scorecard_v361_20261104.json"
    card = json.loads(card_path.read_text(encoding="utf-8-sig"))
    tasks = card["tasks"]
    assert len(tasks) == len(DESIGNS) == 22
    base = sum(r["score"] for r in tasks) / 22
    exact_formula_base = sum(normalized(r["metric"], r["raw"], r["reference_score"]) for r in tasks) / 22
    errors = [abs(normalized(r["metric"], r["raw"], r["reference_score"]) - r["score"]) for r in tasks]
    assert max(errors) <= .00000051
    unique_train, unique_dev = set(), set()
    labelled_train, labelled_dev = 0, 0
    task_plans, dependencies = [], [card_path]
    for r in tasks:
        task_id = r["task"]
        counts = {}
        for part in ("train", "valid"):
            path = ROOT / "admet_competition_data/data_benchmark" / task_id / (part+".csv")
            with path.open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            counts[part] = len(rows)
            if r["metric"] in ("AUROC", "AUPRC"):
                values = [float(x["Y"]) for x in rows]
                assert set(values) <= {0., 1.}
                counts[part+"_positives"] = sum(values)
            sm = {x["Drug"] for x in rows}
            unique_dev.update(sm)
            labelled_dev += len(rows)
            if part == "train":
                unique_train.update(sm)
                labelled_train += len(rows)
            dependencies.append(path)
        tier, model, objective, note = DESIGNS[task_id]
        best_score = normalized(r["metric"], 0 if r["metric"] == "MAE" else 1, r["reference_score"])
        task_plans.append(dict(task=task_id, metric=r["metric"], counts=counts, priority=tier,
                               current_raw=r["raw"], current_task_score=r["score"],
                               proposed_model=model, objective=objective, notes=note,
                               mathematical_total_headroom=(best_score-r["score"])/22,
                               headroom_is_attainable_forecast=False, trained=False))
    scenarios = []
    scenario_defs = [
        ("no_new_signal", "未发现可用新信号，保留v361", 0., 0., 0.),
        ("small_joint_gains", "小幅联合改善情景", .03, .005, .02),
        ("target_52", "冲52的联合改善情景", .08, .015, .05),
        ("stretch_52_6", "进取联合改善情景", .12, .025, .08),
        ("target_53_plus", "53分以上需要的强改善情景", .20, .03, .10),
        ("target_54_plus", "54分以上需要的极强改善情景", .25, .05, .15),
    ]
    for name, description, mae_red, cls_delta, rank_delta in scenario_defs:
        result, gain_by_metric = [], {}
        for r in tasks:
            m, raw = r["metric"], r["raw"]
            new = raw*(1-mae_red) if m == "MAE" else min(1., raw+(rank_delta if m == "Spearman" else cls_delta))
            score = normalized(m, new, r["reference_score"])
            gain_by_metric[m] = gain_by_metric.get(m, 0.)+(score-r["score"])/22
            result.append(dict(task=r["task"], raw=new, score=score))
        mean = sum(r["score"] for r in result)/22
        scenarios.append(dict(id=name, description=description, MAE_uniform_relative_reduction=mae_red,
                              classification_uniform_absolute_increase=cls_delta,
                              Spearman_uniform_absolute_increase=rank_delta, scores_capped_at_raw_one=True,
                              hypothetical_total=mean, hypothetical_total_gain=mean-base,
                              gain_by_metric=gain_by_metric, tasks=result,
                              type="CONDITIONAL_ARITHMETIC_NOT_PREDICTION", success_probability=None))
    mae_only = []
    for target in (52., 53., 54.):
        lo, hi = 0., 1.
        for _ in range(80):
            mid = (lo+hi)/2
            mean = sum(normalized(r["metric"], r["raw"]*(1-mid), r["reference_score"])
                       if r["metric"] == "MAE" else r["score"] for r in tasks)/22
            if mean < target:
                lo = mid
            else:
                hi = mid
        mae_only.append(dict(target_total=target, required_all_five_MAE_relative_reduction=hi,
                             other_17_task_scores_held_fixed=True))
    four = [r for r in task_plans if r["task"] in ("herg", "bbb_martins", "dili", "pgp_broccatelli")]
    live = rules()
    RULES.write_text(json.dumps(live, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    encoder = dict(status="PROPOSED_NOT_IMPLEMENTED", type="atom_bond_graph_transformer", external_weights=False,
                   layers=6, hidden=192, heads=6, target_parameter_budget_millions=[2, 5],
                   parameter_budget_is_estimate=True, dropout=.15,
                   attention_bias=["bond_type", "topological_distance", "ring_and_charge_features"],
                   pretraining=["fit-only masked atom/bond reconstruction", "same-molecule graph/SMILES view consistency"],
                   training_structures="Official fit molecules only; no query/test structures for pretraining in the primary experiment.",
                   small_task_finetuning="Freeze backbone, train low-rank adapter rank8 and small head; unfreeze only after measured evidence.",
                   larger_task_finetuning="Compare frozen encoder to unfreezing final two layers.",
                   auxiliary_context="Separate optional branch with current target assay value/mask excluded; report structure-only and conditional modes separately.",
                   claim="Candidate engineering combination; global novelty and score gain not established.")
    ablations = ["A: current v361 output plus reproducible current structure control; preserve upstream OOF limitations",
                 "B: graph Transformer from scratch, target head only",
                 "C: fit-only self-supervised graph Transformer plus target adapter; compare to B",
                 "D: C plus target-family supervised initialization; compare to C, global query groups excluded",
                 "E: strongest frozen developmental C/D plus optional target-excluded context, separate hypothesis",
                 "F: fixed coarse blend of expert with v361, include zero weight; freeze before confirmation"]
    plan = dict(status="DESIGN_AND_SCORE_ANALYSIS_ONLY", user_reported_current=51.240, current_exact_mean=base,
                formula_reconstructed_mean=exact_formula_base, formula_max_residual=max(errors),
                formula_provenance="Empirical score identity verified against reported official rows; not fetched published full specification.",
                no_new_training=True, no_new_submission=True, no_new_app_threads=True,
                recommendation="22 independent endpoint experiment projects sharing fold-safe molecular representation and data infrastructure; retain per-task baselines. Do not train22 large encoders independently.",
                data=dict(train_label_rows=labelled_train, train_unique_exact_smiles=len(unique_train),
                          development_label_rows=labelled_dev, development_unique_exact_smiles=len(unique_dev),
                          counts_are_exact_smiles_not_canonical_molecule_counts=True,
                          note="Every outer fold uses a smaller fitting-only subset; full counts are not allowed in every validation encoder."),
                rules_check=live, tasks=task_plans, encoder=encoder, ablations=ablations,
                pilot_tasks=[r["task"] for r in task_plans if r["priority"] == "P1"],
                prior_transformer_results=[
                    dict(route="v58/v60", result="ChemBERTa fine-tuning on12 tasks; selected DILI source was the structure control head, sequence-fusion branch not adopted", evidence="docs/history/v60_report.md"),
                    dict(route="v238/v239", result="Six endpoint molecular-view Transformers failed original development selection; no confirmed model or CSV", evidence="docs/v239_final_status.json"),
                    dict(route="v159/v161", result="Target-focused MLP multitask transfer and gradient projection already tried; no strict accepted task", evidence="docs/ITERATION_V161_20261027.md"),
                ],
                validation=dict(global_identity_scaffold_isolation=True, pretraining_refit_each_outer_split=True,
                                query_labels_excluded_from_all_task_encoders=True,
                                historical_public_labels_reused=True, new_split_is_independent_new_data=False,
                                existing_v361_full_OOF_recoverability="partial; report where only approximate controls available",
                                developer_selection="Bounded hypotheses on fold0/1, freeze before fold2/valid; no confirmation-driven retries",
                                strict_gate="Retain existing four-point improvement and scaffold-bootstrap gate; label experimental alternatives separately, never rewrite rejected status",
                                no_claim_per_task_nonregression_on_hidden_test=True),
                scenarios=scenarios, mae_only_target_requirements=mae_only,
                four_high_AUROC_tasks_perfect_mathematical_total_gain=sum(r["mathematical_total_headroom"] for r in four),
                estimate=dict(empirical_forecast=None, numeric_success_probability=None,
                              reason="No new Transformer trained or independently validated; prior experiments do not justify a calibrated forecast.",
                              planned_milestone=52., stretch_milestone=53., guaranteed_multi_point_jump=False),
                compute=dict(available_GPU_this_turn="not inspected", wall_time_forecast=None,
                             pilot_design="6 tasks, shared fold-specific encoder, bounded ablations; benchmark first",
                             independent_experts_reference_models=21*4*2,
                             note="168 expert fits per hypothesis if21 trainable endpoints,4 development partitions,2seeds; encoders,controls,final fits and ablations add work. HIA is keep-only."),
                evidence=[dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p)) for p in dependencies])
    OUT.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = ["# Transformer 与22项独立专家方案评估", "", "当前官方显示51.240，精确分项均值51.239528409。建议结合两个思路：共享原子/键分子Transformer表示，22项分别建立实验、训练头和适配器、选择模型及融合；HIA维持当前输出。方案已设计，尚未实现新训练器或启动训练。", "", "## 可行性结论", "", "22个独立实验项目可行；22套完全独立的大Transformer不适合本数据规模。hERG457、DILI331、CYP底物约465–468条训练标签，容易过拟合。CYP抑制约8千–9千条，更适合先检验表征学习。独立模型增加任务专用性，但跨任务共享能给小样本提供信息。", "", f"现有官方训练数据共{labelled_train:,}条标签行、{len(unique_train):,}条不同原始SMILES；train+valid共{labelled_dev:,}条标签行、{len(unique_dev):,}条不同原始SMILES。这里不是规范化去重分子计数，更不是百万级预训练数据。每个验证编码器只能使用其拟合集子集。", "", "既有失败证据：v58做过ChemBERTa微调，未采用序列融合；v238是指纹视图token Transformer，六项未通过开发筛选；v159也试过多任务迁移和梯度冲突处理。因此“换Transformer”“独立任务”“排序损失”本身均不是新的提分证据。原子/键层的编码与拟合集自监督适配，是本方案要检验的新组合假设。", "", "## 数据范围", "", f"规则核对状态：{live['status']}。官网公开接口仍载明“不允许使用其他外部数据”。本新路线采用官方拟合数据学习表示，不新增外部预训练权重。规则没有明确授权外部权重；如以后考虑ChemBERTa/MoLFormer/Uni-Mol等，先取得明确适用解释。历史v361含早期预训练来源的事实不等于新路线已获授权。", "", "## 模型设计", "", "候选主干：6层、192维、6头的原子/键Graph Transformer，加入键类型、图距离、电荷/环特征的注意力偏置。参数预算暂定约2–5百万，待实现实测。编码器在每个外层拟合集独立做掩码原子/键重建与同分子不同视图一致性训练；查询骨架从结构预训练和所有任务标签中一起排除。", "", "小样本任务冻结主干，训练rank8低秩适配器及小头；较大任务比较冻结与解冻末两层。各任务从共享初始化复制模型，随后独立优化，不在一个目标下互相拖动。其他官方测量上下文作为可选独立分支，目标值及掩码严格排除，并与纯结构方案分别报告。", "", "若新表示有增量，再考虑与原标签检索或现有树/核专家融合；这些机制已有历史实验，不将重新组合自动视为创新或收益。3D仅作为第二阶段消融，构象生成与实际溶液构象的不确定性需评估。", "", "## 22项独立实验设计", "", "P1为首批六项；P2在首批证明表示有效后扩展；P3预算较低；KEEP维持当前输出。具体算法是候选假设，不是已训练结果。", "", "|任务|train/valid|优先级|专用模型|目标与限制|", "|---|---:|---|---|---|"]
    for r in task_plans:
        lines.append(f"|{r['task']}|{r['counts']['train']}/{r['counts']['valid']}|{r['priority']}|{r['proposed_model']}|{r['objective']}；{r['notes']}|")
    lines += ["", "## 实验和项目组织", "", "采用同一代码仓库下22个任务实验目录，每项独立配置、基线哈希、开发预测、选型记录、检查点、最终预测与报告；共享数据加载、骨架划分、编码器和CSV组装代码。最终按task/sample_idx映射组装模板，只写y_pred。无需复制22份代码，也无需创建22个Codex聊天。", "", "首批六项：水溶解度、脂溶性、Caco-2、CYP2D6抑制、肝细胞清除率、hERG。覆盖MAE、AUPRC、Spearman和AUROC，也覆盖数据较充足与小样本场景。", "", "分阶段消融："]
    lines += [str(i+1)+". "+s for i, s in enumerate(ablations)]
    lines += ["", "使用开发fold0/1做有界选择，冻结后检查fold2及官方valid，保留原严格验收规则；实验候选可按既有授权单独标明，但不能改写严格拒绝记录。所有公开标签长期复用，新骨架划分与bootstrap不是全新独立样本。旧完整集成OOF缺失的任务需保留“代理对照”标注，不能借新方案抹去这一限制。", "", "每个任务保留基线和有限个假设，以骨架验证及增量为停止条件；“一直训练到最好”不存在可验证终点，反复观察同一验证集会选中噪声。各项独立训练可以独立选择，但不能承诺隐藏测试每项均不退步。", "", "## 计分情景与大幅提升门槛", "", "下表是把假设原始指标代入已由成绩单复核的计分关系所得的条件算术，**不是模型成绩预测、置信区间或成功概率**。AUROC/AUPRC增加为绝对值，MAE下降为相对比例，Spearman增加为绝对值；达到1的任务封顶。每一行假设相应组内全部任务都达到改进且无回退。", "", "|情景|5项MAE下降|未满分类指标增加|4项Spearman增加|条件总分|", "|---|---:|---:|---:|---:|"]
    for s in scenarios:
        lines.append(f"|{s['description']}|{100*s['MAE_uniform_relative_reduction']:.1f}%|{s['classification_uniform_absolute_increase']:.3f}|{s['Spearman_uniform_absolute_increase']:.3f}|{s['hypothetical_total']:.3f}|")
    lines += ["", "未发现有效新信号时维持51.240是可能结果；未经有效选型的新提交也可能低于基线。52是应首先检验的里程碑，53是需要更强证据的进取目标，54及以上不能作为现阶段可信预期。", "", "只提升5个MAE任务、其余17项固定时："]
    lines += [f"- 达到{s['target_total']:.0f}：五项MAE均需下降约{100*s['required_all_five_MAE_relative_reduction']:.2f}%。" for s in mae_only]
    lines += ["", f"hERG、BBB、DILI和P-gp即使全部达到AUROC=1，总分最多也只增加约{sum(r['mathematical_total_headroom'] for r in four):.3f}。低平台分不等于大幅可提升空间；HIA已1.0，无法贡献新增分数。五项MAE应得到重点预算。", "", "换算关系：MAE分项=100*reference/(raw+reference)；分类分项=100*raw/(raw+reference)；Spearman分项=100*(raw+1)/(raw+reference+2)；总分为22项均值。当前22行复核误差不超过0.00000051。该关系来自已观测成绩的经验复核，不冒称已获取官方完整计分文档。", "", "## 资源与交付边界", "", "本次没有检查服务器当前GPU状态，不能给可靠工期。先对代表任务测实测吞吐和显存，再排预算。21项可训练任务×4个开发分区×2种子=168次专家拟合/假设，尚不含编码器预训练、消融、对照和最终重训；因此不一次盲跑22套大模型。", "", "详细数值、样本量、任务设计和输入哈希见 [JSON](transformer22_plan.json)，最新规则核对见 [记录](transformer22_rule_check.json)。本轮交付方案及算术分析，没有启动新训练、生成新提交或创建新聊天。"]
    DOC.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(dict(status=plan["status"], tasks=len(task_plans), pilot_tasks=plan["pilot_tasks"],
                          formula_max_residual=max(errors), scenarios={s["id"]:s["hypothetical_total"] for s in scenarios},
                          rules_status=live["status"], document=str(DOC)), ensure_ascii=False))


if __name__ == "__main__":
    main()
