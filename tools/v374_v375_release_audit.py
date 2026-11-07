"""Audit the two completed rounds and build a self-contained, hashed handoff.

This performs file/array arithmetic only. It does not fit models or submit files.
"""
from pathlib import Path
import csv
import hashlib
import io
import json
import math
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    best = read(ROOT / "docs/current_best_after_v371_20261105.json")
    baseline = ROOT / best["csv"]
    assert sha(baseline) == best["csv_sha256"]
    rows = list(csv.reader(io.StringIO(baseline.read_bytes().decode("utf-8-sig"), newline="")))
    reports = []
    checkpoints = []
    prediction_changes = []
    files = set()
    for version in (374, 375):
        name = f"v{version}"
        exp = ROOT / f"experiments/mae_experts_{name}"
        delivery = read(ROOT / f"docs/{name}_delivery.json")
        evidence = read(exp / "outputs/independent_verification.json")
        status = read(exp / "run_status.json")
        receipt = read(exp / "completion_receipt.json")
        manifest = read(exp / "result_manifest.json")
        assert status["stage"] == "EVIDENCE_READY"
        assert evidence["status"] == "PASS" and receipt["no_training_processes_running"]
        assert status["signature"] == delivery["signature"] == evidence["signature"] == manifest["signature"]
        final = manifest["final_models"]
        final += [r for r in manifest.get("residual_models", []) if "/final/" in r["remote_path"]]
        for record in final:
            prefix = f"/root/autodl-tmp/mae_experts_{name}/"
            assert record["remote_path"].startswith(prefix)
            p = exp / record["remote_path"][len(prefix):]
            assert sha(p) == record["sha256"] and p.stat().st_size == record["bytes"]
            checkpoints.append(dict(version=name, path=p.relative_to(ROOT).as_posix(), sha256=sha(p)))
        if delivery["csv"]:
            p = ROOT / delivery["csv"]
            assert sha(p) == delivery["csv_sha256"]
            actual = list(csv.reader(io.StringIO(p.read_bytes().decode("utf-8-sig"), newline="")))
            assert len(actual) == len(rows)
            changed = {}
            deltas = {}
            for old, new in zip(rows, actual):
                assert old[:4] + old[5:] == new[:4] + new[5:]
                if old != new:
                    changed[old[0]] = changed.get(old[0], 0) + 1
                    delta = float(new[4]) - float(old[4])
                    assert math.isfinite(delta)
                    deltas.setdefault(old[0], []).append(delta)
            assert set(changed) == set(delivery["approved_tasks"])
            prediction_changes.append(dict(version=name, changed_rows=changed,
                delta_summary={task: dict(mean_delta=sum(v)/len(v), mean_absolute_delta=sum(abs(x) for x in v)/len(v),
                              max_absolute_delta=max(abs(x) for x in v)) for task, v in deltas.items()}))
            files.update(p.parent.glob("*"))
        else:
            assert not delivery["approved_tasks"] and not final
        dev = read(exp / "outputs/development.json")
        confirmation = read(exp / "outputs/confirmation.json")
        reports.append(dict(version=name, delivery=delivery, verification_models=evidence["models"],
                            development=dev, confirmation=confirmation))
        files.update(p for p in exp.rglob("*") if p.is_file() and "/kernels/" not in p.as_posix()
                     and ("/outputs/" in p.as_posix() or p.suffix in (".json", ".py", ".log")))
        files.update((ROOT / "tools").glob(f"{name}_*.py"))
        files.update((ROOT / "docs").glob(f"{name}_*.json"))
        files.update((ROOT / "docs").glob(f"V{version}_*.md"))
    lines = ["# v374 / v375 五项 MAE 专家实验交付", "",
             "官方最好：用户确认 v371 为 51.270。以下为验证实验，不是新的官方成绩。",
             "两轮全部拟合均在用户指定端口 25870 的远端 RTX 3090 完成，本地只准备输入、复核和打包。", "",
             "|版本|任务|专家|权重|fold0 降幅|fold1 降幅|fold2 降幅|valid 降幅|结果|",
             "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for report in reports:
        conf = {r["task"]: r for r in report["confirmation"]["records"]}
        for record in report["development"]["records"]:
            b = record["best"]
            c = conf.get(record["task"])
            gains = {v["part"]: f"{v['gain']:+.3%}" for v in (c["confirmation"] if c else b["parts"])}
            label = "严格通过" if c and c["strict_accepted"] else "实验候选" if c and c["experimental_eligible"] else "确认失败" if c else "开发失败"
            lines.append("|" + "|".join([report["version"], record["task"], b["config"]["name"], str(b["weight"]),
                         *[gains.get(k, "未运行") for k in ("fold0", "fold1", "fold2", "valid")], label]) + "|")
    count = len(set(r["task"] for x in reports for r in x["confirmation"]["records"] if r["ten_percent_all_parts"]))
    lines += ["", f"四个分区均下降至少 10% 的任务：{count}/5。五项各降 10% 目标" + ("已在这些验证分区达到。" if count == 5 else "尚未达到。"),
              "负数表示 MAE 退步。模型配置和权重只由 fold0/1 决定，确认失败后未改配方。", ""]
    for x in reports:
        d = x["delivery"]
        lines += [f"## {x['version']}", "", f"独立重放核验 {x['verification_models']} 个模型包，最大预测差 {d['model_replay_max_error']:.3g}。"]
        if d["csv"]:
            lines += [f"实验 CSV：`{d['csv']}`。", f"SHA256：`{d['csv_sha256']}`。",
                      "仅更新通过四分区方向检查的任务，未达到严格稳定性门槛的候选明确标为实验。"]
        else:
            lines += ["没有符合交付门槛的新候选，未生成新 CSV。"]
        for r in x["confirmation"]["records"]:
            for b in r["bootstrap"]:
                lines.append(f"{r['task']} {b['part']}：{b['groups']} 组，MAE 相对降幅 90% 区间 {b['ci90']}。")
        lines.append("")
    scenarios = read(ROOT / "docs/v374_v375_conditional_scores.json")
    noise = read(ROOT / "experiments/mae_experts_v375/fit_noise_diagnostic.json")
    ppbr = next(x for x in noise["records"] if x["task"] == "ppbr_az")
    lines += ["## 分数目标与限制", "",
              "按现有官方公式，若其他 17 项保持不变，五项 MAE 同幅下降 10% 对应总分约 51.866；同幅下降约 12.115% 才对应 52。仅为条件算术，不是模型预测。",
              "当前历史完整集成的 OOF 仍部分为代理，公开标签长期反复用于研究；新专家的独立重放只验证实现一致，不证明隐藏集提升。",
              f"仅训练数据诊断：PPBR 的 {ppbr['train_rows']} 行中有 {ppbr['rows_in_duplicate_groups']} 行属于重复立体结构，{ppbr['conflicting_groups']} 个重复组的标签不同。强制同结构同预测时，逐组训练中位数的经验 MAE 下限约 {ppbr['empirical_structure_only_floor_all_train']:.3f}。这不是隐藏集噪声下界，其他测量上下文可能区分相同结构的行，也不能据此声称 10% 目标不可能。",
              "用户已确认成绩归属 v371，但本地 v371 只改 LD50，官方卡还显示溶解度变化。保留这项来源差异，不声称上传字节哈希已确认。",
              "每份 CSV 独立对照精确本地 v371。不要将本轮未经官方评估的候选视为官方最优。", "",
              "## 复现", "",
              "冻结输入按顺序来自 v371_expert_inputs.tar.gz、v374_delta_inputs.tar.gz、v375_delta_inputs.tar.gz。",
              "复现 v374 时只应用前两个输入包，再用本交付中的 v374_verify_experts.py 覆盖初始版本；source_amendment.json 记录了只涉及核验的修订。",
              "复现 v375 时依次应用三个输入包。训练仍须在指定远端运行。最终采用模型权重及 v374 局部残差状态已下载并核对哈希，其他试验权重仍保留远端。", ""]
    doc = ROOT / "docs/V374_V375_MAE_RESULTS_AND_HANDOFF.md"
    doc.write_text("\n".join(lines), encoding="utf-8")
    audit = ROOT / "docs/v374_v375_release_verification.json"
    save(audit, dict(status="PASS", official_best=51.270, new_official_score=None,
                     ten_percent_tasks=count, checkpoints=checkpoints, conditional_scores=scenarios,
                     prediction_changes=prediction_changes,
                     models_verified=sum(x["verification_models"] for x in reports),
                     submissions=[x["delivery"] for x in reports if x["delivery"]["csv"]]))
    files.update((doc, audit, baseline, ROOT / "tools/v374_v375_release_audit.py",
                  ROOT / "tools/v371_mae_experts.py", ROOT / "tools/v373_record_feedback.py",
                  ROOT / "tools/repack_upload_template.py", ROOT / "tools/verify_submission.py",
                  ROOT / "CURRENT_STATUS.json", ROOT / "HANDOFF_CURRENT.md", ROOT / "docs/HANDOFF_CURRENT.md",
                  ROOT / "docs/v371_input_package.json", ROOT / "docs/v374_v375_conditional_scores.json",
                  ROOT / "docs/platform_scorecard_v371_20261105.json", ROOT / "docs/current_best_after_v371_20261105.json"))
    files.add(ROOT / "tools/v374_v375_completion.py")
    for name in ("v371_expert_inputs", "v374_delta_inputs", "v375_delta_inputs"):
        p = ROOT / f"releases/{name}.tar.gz"
        receipt = read(ROOT / f"docs/{name.split('_')[0]}_input_package.json")
        assert sha(p) == receipt["sha256"]
        files.add(p)
    files = sorted(p for p in files if p.is_file())
    entries = [dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p), bytes=p.stat().st_size) for p in files]
    manifest_path = ROOT / "docs/v374_v375_handoff_manifest.json"
    save(manifest_path, dict(files=entries, description="Frozen inputs, sources, results, adopted checkpoints, and independent submission files."))
    archive = ROOT / "releases/v374_v375_complete_handoff.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=1) as tf:
        for p in files + [manifest_path]:
            tf.add(p, arcname=p.relative_to(ROOT).as_posix())
    with tarfile.open(archive) as tf:
        for r in entries:
            h = hashlib.sha256()
            with tf.extractfile(r["path"]) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            assert h.hexdigest() == r["sha256"]
    save(ROOT / "docs/v374_v375_handoff_package.json", dict(status="PASS", archive=archive.relative_to(ROOT).as_posix(),
          sha256=sha(archive), bytes=archive.stat().st_size, files=len(entries) + 1, all_member_hashes_verified=True))
    print(json.dumps(dict(status="PASS", models_verified=sum(x["verification_models"] for x in reports),
                         ten_percent_tasks=count, archive=archive.relative_to(ROOT).as_posix())))


if __name__ == "__main__":
    main()
