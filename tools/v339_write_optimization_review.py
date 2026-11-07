"""Summarize terminal SAM and domain-DRO evidence without changing old records."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8-sig"))


def main():
    registry = read("docs/current_best_after_v321_20261101.json")
    anchor = ROOT / registry["csv"]
    assert hashlib.sha256(anchor.read_bytes()).hexdigest() == registry["csv_sha256"]
    gap = read("docs/v338_current_objective_audit.json")
    rows, confirmations, summaries = [], [], []
    for train, verify, label in ((329, 330, "SAM 抑制过拟合"), (333, 334, "化学结构域稳健训练")):
        status = read(f"docs/v{train}_final_status.json")
        verification = read(f"docs/v{verify}_verification.json")
        local = read(f"docs/v{verify}_local_archive_verification.json")
        assert verification["status"] == local["status"] == "PASS"
        assert status["accepted_tasks"] == verification["accepted_tasks"]
        for record in status["development_best"]:
            values = [z["gain"] for z in record["parts"]]
            fmt = (lambda x:f"{100*x:+.3f}%") if record["metric"] == "MAE" else (lambda x:f"{x:+.6f}")
            passed = record["objective"] > (.012 if record["metric"] == "MAE" else .003)
            rows.append(f"| v{train} | {record['task']} | {record['weight']:.2f} | {fmt(values[0])} | {fmt(values[1])} | {'进入确认' if passed else '开发筛除'} |")
        for record in status["confirmation"]:
            values = [z["gain"] for z in record["confirmation"]]
            intervals = [z["gain_ci90"] for z in record["bootstrap"]]
            confirmations.append(dict(version=f"v{train}", task=record["task"], metric=record["metric"],
                fixed_weight=record["weight"], gains=values, scaffold_bootstrap_ci90=intervals,
                accepted=record["accepted"]))
        summaries.append(dict(training=f"v{train}", verification=f"v{verify}", method=label,
            accepted_tasks=verification["accepted_tasks"], csv_created=verification["csv_created"],
            replay_partitions=len(verification["checks"]), archive_sha256=local["sha256"],
            archive_members=local["members"], dependency_hashes_checked=local["dependency_hashes_checked"]))

    accepted = [t for s in summaries for t in s["accepted_tasks"]]
    at = datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()
    outcome = "LOCAL_CANDIDATE_REQUIRES_MANUAL_OFFICIAL_TEST" if accepted else "COMPLETE_NO_CONFIRMED_NEW_MODEL"
    report = dict(status=outcome, at_hong_kong=at, best=registry, target=52., gap_to52=gap["gap_to52"],
        experiments=summaries, confirmations=confirmations, accepted_tasks=accepted,
        official_score_changed=False, platform_submission=False,
        next_direction="If no candidate survives, any next mechanism must be justified by a new fit-only diagnostic or genuinely distinct modeling objective; do not sweep weights/seeds of these rejected mechanisms.",
        limitations="Repeated public-label reuse and approximate inherited historical OOF remain. Numerical replay verifies implementation, not hidden-test improvement. 52 is not achieved.")
    jp = ROOT / "docs/v339_optimization_review.json"
    md = ROOT / "docs/OPTIMIZATION_V329_V333_20261101.md"
    assert not jp.exists() and not md.exists()
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    text = f"""# v329 / v333 优化与核验结论

记录时间（香港）：{at}

当前官方最高仍为 **v313：51.205**；22 项分数精确均值 {registry['exact_task_mean']:.9f}，距 52 约 {gap['gap_to52']:.6f}。本轮状态：`{outcome}`。{'存在待用户手动测试的本地候选，官方分数未知。' if accepted else '本轮没有通过原稳定性门槛的任务，没有生成新 CSV。'}

## v321 退步的核对

v321 仅改变 CYP3A4 底物。用户返回的 AUROC 从 0.746044304 降至 0.741410488，分项从 50.706702 降至 50.550963，平均总分下降约 0.007079。其余 21 项相同，CSV 构建与任务映射未发现错误。局部分区正增益并未迁移至隐藏测试；两份骨架置信区间此前已跨零。该版本不再作为升级或新实验基线。

## 本轮机制

- v329：同一检索网络的 SAM 训练，每次更新在梯度方向扰动权重，再以同一随机掩码计算第二次梯度，精确恢复后更新。五个 MAE 任务从头训练，种子与更新次数沿用原配置。
- v333：只用拟合集结构拟合八个化学域，完整骨架归入同一域；按域均衡采样，动态对困难域加权。三项分类任务从头训练，未重试 CYP3A4。
- 两轮均在评分前锁定源码与配置；原开发选择、0.20/0.45 融合、确认和骨架 bootstrap 门槛未改变。被拒任务不回退配置或换种子。

## 开发结果

MAE 数值是相对误差下降，分类数值是原指标绝对增加；正数越好。以下均为本地验证，不是官方成绩预测。

| 实验 | 任务 | 冻结融合 | fold0 增益 | fold1 增益 | 结果 |
| --- | --- | ---: | ---: | ---: | --- |
""" + "\n".join(rows) + "\n\n## 确认结果\n\n"
    for c in confirmations:
        fmt = (lambda x:f"{100*x:+.3f}%") if c["metric"] == "MAE" else (lambda x:f"{x:+.6f}")
        text += f"- {c['version']} / {c['task']}：四区增益 " + "、".join(fmt(z) for z in c["gains"]) + "；fold2 / valid 骨架 90% 区间 " + " / ".join("[" + ", ".join(fmt(z) for z in ci) + "]" for ci in c["scaffold_bootstrap_ci90"]) + f"。{'通过' if c['accepted'] else '拒绝'}。\n"
    text += "\n## 独立核验与计分差距\n\n"
    for s in summaries:
        text += f"- {s['training']}：{s['replay_partitions']} 个分区重载、特征重建、查询标签扰动、注意力公式与顺序检查通过；归档 {s['archive_members']} 个成员及 {s['dependency_hashes_checked']} 个输入依赖在本地复核。SHA256：`{s['archive_sha256']}`。\n"
    text += f"""
按用户成绩单匹配的计分公式，五个 MAE 同时下降 10% 只贡献约 {gap['all_five_MAE_minus10percent_total_gain']:.6f} 总分；若其他 17 项保持不变，需要这五项同时约下降 {100*gap['all_five_MAE_uniform_relative_reduction_needed_if_all_other_tasks_identical']:.2f}% 才能补足差距。这是算术情景，不是能力预测。

公开验证标签多次复用，上游历史 OOF 仍有近似。重载与归档 PASS 只说明实现一致，不证明官方提分。未上传平台，未更新官方最高分，未达到 52。

证据：[终态汇总](v339_optimization_review.json)、[v329 终态](v329_final_status.json)、[v330 核验](v330_verification.json)、[v333 终态](v333_final_status.json)、[v334 核验](v334_verification.json)、[计分差距](v338_current_objective_audit.json)。
"""
    md.write_text(text, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
