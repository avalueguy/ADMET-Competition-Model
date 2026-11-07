"""Document adopted v361 endpoint routes without training or changing submissions."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/CURRENT_V361_TASK_MODEL_INVENTORY.md"
REPORT = ROOT / "docs/current_v361_task_model_inventory.json"


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predictions(path):
    result, keys = {}, []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result.setdefault(row["task"], []).append(row["y_pred"])
            keys.append(tuple((k, v) for k, v in row.items() if k != "y_pred"))
    return result, keys


def csv_version(version):
    choices = sorted((ROOT / "admet_submissions").glob(version + "_*/*.csv"))
    assert len(choices) == 1, (version, choices)
    return choices[0]


def array(data, task):
    return np.asarray(data[task], dtype=float)


def rank(values):
    return rankdata(values, method="average") / len(values)


def task(task_id, name, model, details, sources, limitations=""):
    return dict(task=task_id, name_zh=name, adopted_model=model, details=details,
                sources=sources, lineage_completeness="partial_legacy" if limitations else "adopted_route_confirmed",
                limitations=limitations)


def inventory():
    legacy = "旧组合输出可核对，但更早成员、全部配比和完整 OOF 未能从现存本地记录还原。"
    return [
        task("ames", "AMES 致突变", "65% 历史 v20 +35% 乘积核 SVC（v139，C=1）；秩融合",
             "v124 恢复 v112 的 v20 AMES；v145 保留 v139 product/C1 专家。v20 有残差树/早期表征相关源码，缺少该输出完整选型，不能认定为单一 CatBoost。",
             ["tools/v145_feedback_repair.py", "admet_model_outputs_v144/evidence.json", "admet_model_outputs_v124/evidence.json", "admet_submissions/v112_history_expansion/evidence.json", "admet_v20_protocol.json"], legacy),
        task("bbb_martins", "血脑屏障 BBB", "历史 v29 组合；精确内部成员及权重待恢复",
             "v113 保留 v112 的完整 v29 BBB 输出。v29 的搜索菜单不能直接当作最终采用列表。",
             ["admet_submissions/v112_history_expansion/evidence.json", "admet_submissions/v113_v111_restore_keep_bbb_hia/evidence.json"], legacy),
        task("bioavailability_ma", "口服生物利用度", "历史基线 + 中性母体/原分子测量上下文 CatBoost 残差（v96，深3、100树、双种子）",
             "采用 assay_d3/residual/step100；选择权重1意味着使用完整“基线偏置+残差”输出，不意味着消除历史基线。更早含 v81 深3/100树条件残差层。",
             ["tools/v96_normalized_assays.py", "admet_model_outputs_v96/release.json", "admet_model_outputs_v101/release.json", "tools/v81_export.py"], legacy),
        task("caco2_wang", "Caco-2 通透性", "历史 v23 完整输出；不能确认只是多任务神经网络",
             "v111 从历史 v23 整任务复制；同一输出也存在于历史 v20。v24 协议明确历史 v23 完整集成不可用，v23 神经组件重训不能代表整套模型。",
             ["admet_submissions/v111_historical_task_blend/evidence.json", "admet_v24_protocol.json"], legacy),
        task("clearance_hepatocyte_az", "肝细胞清除率", "历史排序基线 + 其他测量上下文的岭回归残差（v85，alpha=1、strength=1）",
             "选中 assay_a1。通过 Gaussian rank 基线偏置与线性残差生成预测；不是把 v84 邻居树再简单叠加一次。",
             ["tools/v85_smooth_residual.py", "admet_model_outputs_v85/release.json", "tools/v125_low_task_baseline.py"], legacy),
        task("clearance_microsome_az", "肝微粒体清除率", "历史 v46 中继承的连续任务组合；内部配比待恢复",
             "v111 恢复历史 v46 输出。v46 本轮只更新分类任务，因此不能称该连续任务采用 v46 图网络微调。",
             ["admet_submissions/v111_historical_task_blend/evidence.json", "admet_submissions/v112_history_expansion/evidence.json", "docs/history/v46_report.md"], legacy),
        task("cyp2c9_substrate_carbonmangels", "CYP2C9 底物", "历史基线 + 其他测量上下文 CatBoost 残差（v82，深2、100树、3种子）",
             "官方其他任务测量值、缺失掩码和历史预测 rank-logit 为输入；目标任务及其掩码排除；均值原始 logit 后 sigmoid。",
             ["tools/v82_export.py", "admet_model_outputs_v82_final/release.json", "tools/v83_residual.py"], legacy),
        task("cyp2c9_veith", "CYP2C9 抑制", "65% 条件残差基线 +35% 加和核 SVC（v139，C=10）；秩融合",
             "基线保留 v80 深4/300树、3种子 CatBoost 偏置残差。v139 sum 核由结构指纹、增强理化/构象/药效团与其他测量上下文组成。",
             ["admet_model_outputs_v144/evidence.json", "tools/v139_classification_kernels.py", "admet_model_outputs_v80_final/release.json", "tools/verify_v80_models.py", "tools/v83_residual.py"], legacy),
        task("cyp2d6_substrate_carbonmangels", "CYP2D6 底物", "75% 条件残差基线 +25% XGBoost（v86，深3、200树）；秩融合",
             "v86 aux_logloss_d3_t200，用二分类 logloss，不能因文件名 ranking 宣称选中的是排序损失。基线有 v82 深2/50树、3种子 CatBoost。",
             ["admet_model_outputs_v86/release.json", "tools/v86_metric_ranking.py", "tools/v82_export.py", "tools/v125_low_task_baseline.py"], legacy),
        task("cyp2d6_veith", "CYP2D6 抑制", "65% 多层树/核基线 +35% XGBoost（v135，深6、900树、双种子）；秩融合",
             "基线包括 v81 深4/300树 CatBoost 条件残差，v92 多指纹 Tanimoto 核岭回归25%（alpha=.3），以及 v100 2D/3D加和核岭回归25%（alpha=.1）。各层重新计算秩，不能把嵌套比例平铺相乘。",
             ["admet_model_outputs_v144/evidence.json", "tools/v135_broad_boosting.py", "admet_model_outputs_v101/release.json", "admet_model_outputs_v92/release.json", "tools/v92_full_kernel.py", "tools/v100_shape_kernel.py", "tools/v81_export.py"], legacy),
        task("cyp3a4_substrate_carbonmangels", "CYP3A4 底物", "50% 共享多任务神经网络（v67）+50% CatBoost（v127，深4、900树）；秩融合",
             "v67 指纹+RDKit描述符，2种子、5500/7000步快照；v68 核分支该任务未通过，所以历史源不是核SVM。v127 physchem 包含理化特征和目标排除的其他测量上下文。v354 的168/210/224组合已撤回。",
             ["admet_v67_protocol.json", "admet_model_outputs_v67/fresh.json", "admet_model_outputs_v68/selection.json", "tools/v125_low_task_baseline.py", "tools/v127_low_classification.py", "admet_model_outputs_v130/evidence.json", "tools/v361_herg_feedback_repair.py"]),
        task("cyp3a4_veith", "CYP3A4 抑制", "25% 历史基线 +75% v75 专家；专家内部50%条件 CatBoost +50%混合核 SVC",
             "v75 conditional_kernel3：树深6、851树、3种子；SVC C=10，85%化学核+15%理化RBF。两级均为秩融合。协议变量 anchor_weight=.75 在实际融合中是专家权重，应按源码语义解释。",
             ["admet_model_outputs_v75/release.json", "admet_v75_protocol.json", "admet_v75_experiments.py", "tools/select_v75.py", "tools/verify_v75_models.py", "admet_submissions/v111_historical_task_blend/evidence.json"], legacy),
        task("dili", "药物性肝损伤 DILI", "75% 历史基线 +25% 多视图 CatBoost（v73，深6、391树、双种子）；秩融合",
             "v73 multiview：多种结构计数指纹、整分子与母体描述符；之后各采用版本未再改变该任务。",
             ["admet_model_outputs_v73/release.json", "admet_v73_experts.py", "tools/finalize_v73.py"], legacy),
        task("half_life_obach", "半衰期", "50% v68 神经网络/SVR组合 +50% 双目标 CatBoost（v119，深4、900树、双种子）",
             "v68 内部65%共享多任务NN秩+35%混合结构核SVR秩（C=3，85%指纹核+15%描述符RBF）。v119 MultiRMSE 同时拟合log1p及Gaussian rank，逐输出取秩再融合。",
             ["tools/v119_matched_half_life.py", "tools/v118_exact_half_life.py", "admet_model_outputs_v124/evidence.json", "admet_v68_protocol.json", "admet_v68_kernel.py"]),
        task("herg", "hERG 心脏毒性", "60% v346基线 +各13⅓%稀疏逻辑回归、监督多核SVC、GroupDRO神经标签检索；秩融合",
             "三专家：v168 strength1 的TF-IDF/NB-logratio+L2逻辑回归；v233 C=3监督核加权SVC；v333 8化学域GroupDRO标签检索。v346基线hERG=50%v85理化平滑残差+50%v127 CatBoost。v361保留v354完整hERG输出。原严格bootstrap未通过记录仍保留；本次官方反馈确认本轮公开评测小幅改善。",
             ["admet_model_outputs_v354/frozen_before_final.json", "tools/v349_diverse_model_combinations.py", "tools/v168_sparse_linear.py", "tools/v233_supervised_chemical_similarity.py", "tools/v333_chemical_domain_dro.py", "admet_model_outputs_v130/evidence.json", "admet_submissions/v114_herg_history52/evidence.json", "tools/v85_smooth_residual.py", "tools/v361_herg_feedback_repair.py"], legacy),
        task("hia_hou", "人体肠道吸收 HIA", "共享多任务神经网络（v67，经v68输出继承）",
             "v67 gated候选在fresh检查下降，final_weight=0，实际退回shared；v68核候选也未通过。v113保留v112从v68恢复的输出。双种子与5500/7000步快照；当前官方AUROC=1，无上升空间。",
             ["admet_v67_protocol.json", "admet_model_outputs_v67/fresh.json", "admet_model_outputs_v68/selection.json", "admet_submissions/v112_history_expansion/evidence.json"]),
        task("ld50_zhu", "急性毒性 LD50", "历史 v44 连续任务组合；最终内部模型和配比待恢复",
             "当前预测字符串与历史v44/v46/v56均一致。v44曾比较成对性质差树与直接树，但缺少该任务最终选型，不能直接宣称采用全部这些候选。",
             ["admet_v44_protocol.json", "docs/history/v46_report.md", "admet_model_outputs_v56/release.json"], legacy),
        task("lipophilicity_astrazeneca", "脂溶性", "80% v56物理残差神经网络组合 +20% v304可学习标签检索；数值加权",
             "v56内部65%更早输出+35%物理残差神经分支（variant1），随后仿射校准1.025082651919374*x-.033204669702501285。v304近邻24、width128/key32，双种子、直接MAE；当前外层比例由v313固定。",
             ["admet_model_outputs_v56/release.json", "admet_model_outputs_v56/fusion_replay.json", "docs/history/v56_report.md", "tools/v304_learned_label_retrieval.py", "docs/v313_experimental_delivery.json", "tools/v314_current_baseline.py"], legacy),
        task("pgp_broccatelli", "P-gp", "历史基线 + 其他测量上下文 CatBoost 残差（v81，深3、100树、3种子）",
             "用基线rank-logit及其他测量值/缺失掩码，原始logit均值后sigmoid；v125有当前采用分支的精确最终预测重放记录。",
             ["tools/v81_export.py", "admet_model_outputs_v81_final/release.json", "tools/v125_low_task_baseline.py"], legacy),
        task("ppbr_az", "血浆蛋白结合率 PPBR", "55% v83残差校准组合 +45% v304可学习标签检索；数值加权",
             "v83内部50%旧基线+50%其他测量上下文CatBoost MAE残差（深3/250树、3种子），旧基线含v56偏差校准。v304近邻24、width128/key32，2种子，直接MAE，输出限制[0,100]。",
             ["admet_model_outputs_v83/release.json", "tools/v83_residual.py", "admet_model_outputs_v56/release.json", "tools/v304_learned_label_retrieval.py", "docs/v313_experimental_delivery.json", "tools/v314_current_baseline.py"], legacy),
        task("solubility_aqsoldb", "水溶解度", "80% 历史 v44输出 +20% v329 SAM神经标签检索；数值加权",
             "v329继承v304标签检索结构，SAM半径.05配合AdamW；近邻24、双种子，直接MAE。历史v44内部选型不完整，不能仅凭v44研究菜单断言采用成对回归。",
             ["admet_submissions/v111_historical_task_blend/evidence.json", "admet_v44_protocol.json", "tools/v329_sharpness_retrieval.py", "tools/v346_experimental_solubility.py", "docs/v346_experimental_delivery.json"], legacy),
        task("vdss_lombardo", "稳态分布容积 VDss", "历史 v46 中继承的连续任务组合；内部配比待恢复",
             "v112恢复来源记录表明同历史v46输出；本地逐字符串复核一致。该任务不是v46当轮分类图微调分支。",
             ["admet_submissions/v112_history_expansion/evidence.json", "docs/history/v46_report.md"], legacy),
    ]


def main():
    assert not DOC.exists() and not REPORT.exists(), "Review previous inventory before overwriting."
    card = read("docs/platform_scorecard_v361_20261104.json")
    best = read("docs/current_best_after_v361_20261104.json")
    file = ROOT / card["csv"]
    assert sha(file) == card["submission_sha256"] == best["csv_sha256"]
    data, identity = predictions(file)
    versions = {n: predictions(csv_version(n))[0] for n in ("v68", "v124", "v130", "v145", "v313", "v346", "v354")}
    b346 = versions["v346"]
    assert [t for t in data if data[t] != b346[t]] == ["herg"]
    assert data["herg"] == versions["v354"]["herg"]
    assert identity == predictions(csv_version("v346"))[1]
    checks, dependencies = [], {file, csv_version("v346"), csv_version("v354")}

    def record(task_id, expected, label, paths):
        error = float(np.max(np.abs(array(data, task_id) - expected)))
        assert error < 1e-12, (task_id, label, error)
        checks.append(dict(task=task_id, check=label, max_abs_error=error, status="PASS"))
        dependencies.update(paths)

    def cached(pattern):
        found = list(ROOT.glob(pattern))
        assert len(found) == 1, (pattern, found)
        path = found[0]
        dependencies.add(path)
        with np.load(path, allow_pickle=False) as archive:
            return archive["p"].copy()

    for t, w in (("ppbr_az", .45), ("lipophilicity_astrazeneca", .2)):
        expert = cached(f"admet_model_outputs_v313/expert/models/{t}/final/*/predictions.npz")
        record(t, (1-w)*array(versions["v145"], t)+w*expert, "v304 arithmetic blend", [csv_version("v145")])
    t = "solubility_aqsoldb"
    expert = cached(f"admet_model_outputs_v346/expert/models/{t}/final/*/predictions.npz")
    record(t, .8*array(versions["v313"], t)+.2*expert, "v329 arithmetic blend", [csv_version("v313")])
    experts = [cached(f"admet_model_outputs_v{n}/models/herg/final/*/predictions.npz") for n in (168, 233, 333)]
    record("herg", .6*rank(array(b346, "herg"))+.4*np.mean([rank(p) for p in experts], axis=0), "three experts and baseline rank blend", [csv_version("v346")])
    t = "cyp3a4_substrate_carbonmangels"
    expert = cached(f"admet_model_outputs_v127/models/{t}/final/physchem/predictions.npz")
    record(t, .5*rank(array(versions["v124"], t))+.5*rank(expert), "v127 rank blend", [csv_version("v124")])
    expert = cached("admet_model_outputs_v119/models/final/full/predictions.npz")
    t = "half_life_obach"
    record(t, .5*rank(array(versions["v68"], t))+.5*rank(expert), "v119 rank blend", [csv_version("v68")])
    t = "cyp2d6_veith"
    expert = cached(f"admet_model_outputs_v135/models/{t}/final/full_d6/predictions.npz")
    record(t, .65*rank(array(versions["v130"], t))+.35*rank(expert), "v135 rank blend", [csv_version("v130")])

    historical = {
        "caco2_wang": "recovered_history/official_scorecard_sources/admet_submissions/v23",
        "bbb_martins": "recovered_history/official_scorecard_sources/admet_submissions/v29",
        "clearance_microsome_az": "recovered_history/official_scorecard_sources/admet_submissions/v46",
        "ld50_zhu": "recovered_history/official_scorecard_sources/admet_submissions/v44",
        "vdss_lombardo": "recovered_history/official_scorecard_sources/admet_submissions/v46",
        "hia_hou": "admet_submissions/v68_official_only",
        "dili": "admet_submissions/v73_v72",
        "pgp_broccatelli": "admet_submissions/v83_matched_residual",
        "cyp2c9_substrate_carbonmangels": "admet_submissions/v83_matched_residual",
        "clearance_hepatocyte_az": "admet_submissions/v85_smooth_residual",
        "cyp3a4_veith": "admet_submissions/v75_v73",
    }
    for t, folder in historical.items():
        path = sorted((ROOT/folder).glob("*.csv"))[0]
        other, _ = predictions(path)
        assert data[t] == other[t], (t, path)
        dependencies.add(path)
        checks.append(dict(task=t, check="exact historical prediction strings", source=path.relative_to(ROOT).as_posix(), status="PASS"))
    source56 = sorted((ROOT/"admet_submissions/v56").glob("*.csv"))[0]
    p56, _ = predictions(source56)
    assert versions["v145"]["lipophilicity_astrazeneca"] == p56["lipophilicity_astrazeneca"]
    dependencies.add(source56)

    rows = inventory()
    assert len(rows) == 22 and {r["task"] for r in rows} == set(data)
    metrics = {r["task"]: r for r in card["tasks"]}
    assert [r["task"] for r in rows] == [r["task"] for r in card["tasks"]]
    for r in rows:
        r.update(metric=metrics[r["task"]]["metric"], official_raw_metric=metrics[r["task"]]["raw"], official_task_score=metrics[r["task"]]["score"], submission_rows=len(data[r["task"]]))
        for source in r["sources"]:
            path = ROOT / source
            assert path.exists(), source
            dependencies.add(path)
    for source in ("docs/platform_scorecard_v361_20261104.json", "docs/current_best_after_v361_20261104.json", "tools/audit_v361_task_inventory.py"):
        dependencies.add(ROOT / source)
    precise = float(np.mean([r["score"] for r in card["tasks"]]))
    gap = 52 - precise
    report = dict(status="PASS_DOCUMENTARY_AND_PREDICTION_AUDIT", version="v361", user_reported_official=51.240,
                  exact_task_mean=precise, gap_to_52=gap, task_score_sum_gain_to_52=22*gap,
                  csv=card["csv"], csv_sha256=sha(file), platform_uploaded_byte_hash_confirmed=False,
                  association=card["association"], rows=rows, checks=checks,
                  evidence=[dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p)) for p in sorted(dependencies)],
                  no_training=True, no_new_submission=True, no_platform_access=True,
                  limitations=["This documents adopted routes, not all attempted models.", "Historical baseline internal composition remains partly unavailable; source scripts alone do not prove every searched component was adopted.", "Recent additions had no new external data or weights, but early anchors contain historical pretrained sources.", "Historical validation labels were repeatedly reused; some upstream OOF is approximate. Public platform feedback used for whole-task selection is not independent private-test validation.", "Numeric checks replay cached expert predictions and fusion; they do not establish fresh full checkpoint inference of every historical model."])
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = ["# v361 当前采用的 22 项任务模型清单", "", "依据用户最新成绩单（2026/11/04 13:06）：官方显示 **51.240**，22分项均值 **51.239528409**。与v361交付和分项吻合；平台未提供上传文件字节哈希。", "", "本清单只列进入当前预测的路线。不是候选算法清单，也没有在本次盘点启动训练。标注“旧组合”的行，其现存输出可核对，但内部成员与全部权重仍有缺失。", "", "## 逐任务清单", "", "|任务|指标|当前模型及融合|官方分项|", "|---|---|---|---:|"]
    for r in rows:
        lines.append(f"|{r['name_zh']} (`{r['task']}`)|{r['metric']}|{r['adopted_model']}|{r['official_task_score']:.6f}|")
    lines += ["", "## 融合、输入和模型含义", "", "数值加权：直接混合MAE任务预测值。秩融合：先把每个分支输出转为平均并列秩百分位，再加权；嵌套秩融合不能平铺为原始模型权重。", "", "树模型包括CatBoost和XGBoost；SVC/SVR是支持向量分类/回归；核岭回归与SVC是不同算法；多任务神经网络共享跨任务结构表征；可学习标签检索从拟合集近邻结构和标签估计新样本。", "", "近年采用专家主要使用官方结构计算的指纹、RDKit理化描述符，部分增加构象、药效团。条件模型另使用官方训练/验证中其他任务已知测量，排除当前目标任务及其掩码。它们评估的是带可用测量上下文的条件预测。", "", "近期新增专家没有新下载外部标签或预训练权重；早期历史基线含预训练表征，不能把整套v361称为完全从零训练。", "", "## 各行细节及证据", ""]
    for r in rows:
        lines += [f"### {r['name_zh']} — {r['task']}", "", r["details"], "", f"官方原始{r['metric']}：{r['official_raw_metric']:.12g}；预测行数：{r['submission_rows']}。", ""]
        if r["limitations"]:
            lines += ["还原限制："+r["limitations"], ""]
        lines += ["源码/记录："] + [f"- [{s}](../{s})" for s in r["sources"]] + [""]
    lines += ["## 本次核验范围", "", "当前CSV SHA256：`"+sha(file)+"`。相对v346仅hERG改变；hERG完整输出与v354一致，其余21项及非预测字段与v346一致。", "", "从归档专家预测复算了7项外层融合，最大误差不超过1e-12；另外11项与指定历史源的预测字符串完全一致，并核对v145脂溶性与v56一致。这是缓存/输出与源码层的盘点，不是所有旧模型权重的完整重推理。", "", "本次完整证据路径、SHA256和各项误差见 [JSON](current_v361_task_model_inventory.json)。原严格筛选拒绝的实验仍保留拒绝状态，官方小幅改善不能改写为严格验证通过。", "", "## 最新成绩的含义及改进建议评估方法", "", "v361相对v346只有hERG原始AUROC从0.857584683提高到0.861119293，分项从49.832263到49.935090；22项均值增加0.004673955。CYP3A4底物保持0.746044304，v354退步分支已经撤回。", "", f"到52还差{gap:.9f}总分，等价于22项平台分项合计增加{22*gap:.6f}。单项平台分增加1，总分约增加1/22=0.045455；不能把AUROC提高0.01解释成总分提高0.01。", "", "HIA官方AUROC已经1，应保留。其他任务能否明显改善取决于互补信息和泛化，低平台分不自动等于最好优化。现有小幅加权收益不能证明下一步会大幅提分。", "", "用户可按“任务—建议模型/改法—预期解决的问题—是否需要外部数据或预训练”提出建议。逐项评价将包括：", "", "1. 机制是否解决实际误差，是否只重复已尝试但未采用的路线。", "2. 对应任务指标是否匹配：AUROC/AUPRC关注排序，Spearman关注连续排序，MAE关注绝对数值；统一单调校准通常不能改善排序指标。", "3. 数据可用性、同源重复及查询标签隔离；需要外部数据的方案另核对比赛要求。", "4. 现有硬件和代码改动可行性、训练/推理成本。", "5. 冻结配置后在骨架划分与确认分区对比当前模型；公开分区长期复用，不能把bootstrap当成全新独立证据。", "6. 以可观察结果评估小幅/大幅改善可能性，报告不确定性；未实验前不给没有依据的提分概率或保证。", "", "用户尚未提出本轮具体方法，因此本清单没有替用户虚构改进意见或声称已评估它们。"]
    DOC.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(dict(status="PASS", tasks=len(rows), fusion_checks=7, historical_checks=11, report=str(REPORT), document=str(DOC), exact_task_mean=precise), ensure_ascii=False))


if __name__ == "__main__":
    main()
