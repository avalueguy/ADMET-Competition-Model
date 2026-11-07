"""Summarize completed frozen search and verified delivery without changing either."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main():
    development = read("admet_model_outputs_v349/development.json")
    confirmation = read("admet_model_outputs_v349/confirmation.json")
    protocol = read("admet_model_outputs_v349/protocol.json")
    delivery = read("docs/v354_delivery.json")
    local = read("docs/v354_local_delivery_verification.json")
    archive = read("docs/v354_archive.json")
    assert confirmation["status"] == "COMPLETE"
    assert local["status"] == archive["status"] == "PASS"
    assert local["csv_sha256"] == delivery["csv_sha256"]
    assert hashlib.sha256((ROOT / delivery["csv"]).read_bytes()).hexdigest() == delivery["csv_sha256"]
    path = ROOT / "docs/V354_MODEL_COMBINATION_REVIEW.md"
    assert not path.exists(), "Preserve existing reports; inspect before rewriting"
    available = sum(r["available"] for r in protocol["registry"])
    selected = set(delivery["selected_tasks"])
    lines = [
        "# v354 多模型组合搜索及实验交付记录", "",
        "当前用户报告的官方最佳仍为 **v346：51.235**。v354 尚未提交，官方分数未知。", "",
        f"本轮登记 {len(protocol['registry'])} 条路线，其中服务器可用 {available} 条；"
        f"对 {len(development['records'])} 个任务比较 {development['total_combinations']:,} 个跨模型家族的双模型、三模型组合。"
        f"签名或划分不匹配记录 {len(development['rejected_alignment_or_signature'])} 条。"
        "这些是现有代码及预测缓存中的有限搜索，不能代表全球所有模型。", "",
        "先用 fold0、fold1 选定每项唯一赢家并冻结，再检查 fold2、valid。"
        "所有组合固定为 60% v346 基线 + 40% 专家等权平均；排序指标先各自转平均秩，MAE 保留原数值。"
        "未扫描官方反馈权重，没有使用测试标签、逐样本手工修改或新增外部预训练权重。", "",
        "实验导出规则在剩余确认结果产生前固定：四处分数增益均为正、平均增益超过原阈值，"
        "且两份确认分区均含多个骨架并可计算区间。严格规则另要求两份 90% 骨架重采样区间下界都大于零。"
        "实验导出不撤销严格拒绝记录。", "",
        "## 本轮确认结果", "",
        "AUROC、AUPRC、Spearman 填写指标绝对增益；MAE 填写相对误差降幅，正值为改善。", "",
        "| 任务 | fold0 | fold1 | fold2 | valid | 导出 | 严格通过 |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for r in confirmation["records"]:
        gains = " | ".join(f"{p['gain']:+.6f}" for p in r["confirmation"])
        lines.append(f"| {r['task']} | {gains} | {'是' if r['task'] in selected else '否'} | {'是' if r['strict_accepted'] else '否'} |")
    lines += ["", "## 导出组合及不确定性", ""]
    for r in confirmation["records"]:
        if r["task"] not in selected:
            continue
        members = "、".join(f"v{m['version']}（{m['family']}）" for m in r["members"])
        ci = "；".join(f"{b['part']}：[{b['gain_ci90'][0]:+.6f}, {b['gain_ci90'][1]:+.6f}]" for b in r["bootstrap"])
        lines += [f"- **{r['task']}**：{members}。90% 骨架重采样增益区间：{ci}。"]
    lines += ["", f"共更新 {len(selected)} 项任务，另外 {delivery['unchanged_tasks']} 项任务预测及所有非预测字节与 v346 完全一致。"
        "远端完成模型独立重载、非拟合目标标签扰动隔离和融合复算；本地另完成模型文件哈希、独立融合计算、逐字节保留及官方模板检查。", "",
        "导出过程中 v233 核模型的中间系数重算出现约 8.7×10⁻¹⁹ 的差异，被原逐位相等比较拦截。"
        "原失败日志和冻结文件保留，另用带审计的数值复核继续：浮点变换 rtol=2e-13、atol=2e-14，"
        "整数、字符串及划分字段仍完全相等；最终预测保留原 atol=1e-12，额外要求排序完全一致。"
        "没有改动模型源码、配置、种子、权重或筛选结果。记录见 `admet_model_outputs_v354/resume_audit.json`。", "",
        "公开验证标签已在历史实验中重复使用，上游 OOF 特征为近似方案；大规模组合搜索存在选择偏差。"
        "四处点估计改善不能证明隐藏测试集提分，区间跨零的组合仍只属于实验候选。"
        "不能据此承诺大幅提分或达到 52。", "",
        "## 文件", "",
        f"- 手动测试 CSV：`{delivery['csv']}`",
        f"- CSV SHA256：`{delivery['csv_sha256']}`",
        f"- 模型及审计归档：`{archive['path']}`",
        "- 本地独立核验：`docs/v354_local_delivery_verification.json`",
        "- 冻结规则：`docs/v352_combination_delivery_policy.json`",
        "- 完整确认记录：`docs/v349_combination_result.json`", "",
        "保留 v346 为当前最佳。收到本次完整官方成绩单后，再确认实际增益及具体任务变化；不自动提交平台。", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(report=path.as_posix(), combinations=development["total_combinations"], available_routes=available,
        selected_tasks=delivery["selected_tasks"], strict_tasks=delivery["strict_accepted_tasks"], official_best=51.235,
        official_v354_score=None), ensure_ascii=True))


if __name__ == "__main__":
    main()
