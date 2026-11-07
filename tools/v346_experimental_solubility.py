"""One frozen solubility experiment for manual official testing; stability unconfirmed."""
from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import v329_sharpness_retrieval as model
import v330_verify_sharpness as independent
from repack_upload_template import field_spans
from verify_submission import verify

v, b = model.v, model.b
OUT = ROOT / "admet_model_outputs_v346"
TASK = "solubility_aqsoldb"
WEIGHT = .2


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "delivery.json").exists(), "Inspect completed delivery before rerun"
    original_out = model.OUT
    frozen = json.loads((original_out / "frozen_selection.json").read_text())
    confirmation = json.loads((original_out / "confirmation.json").read_text())
    verified = json.loads((ROOT / "docs/v330_verification.json").read_text())
    plan_path = ROOT / "docs/v346_experimental_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    assert plan["user_authorized_experimental_csv"] is True
    assert plan["source_sha256"] == b.sha(__file__)
    assert plan["model_source_sha256"] == b.sha(model.__file__)
    assert plan["verifier_sha256"] == b.sha(independent.__file__)
    assert plan["runner_sha256"] == b.sha(model.runner.__file__)
    assert plan["task"] == TASK and plan["weight"] == WEIGHT
    assert verified["status"] == "PASS" and verified["accepted_tasks"] == []
    assert frozen["signature"] == confirmation["signature"] == verified["signature"] == model.sig()
    choice = next(r for r in frozen["selected"] if r["task"] == TASK)
    rejected = next(r for r in confirmation["records"] if r["task"] == TASK)
    assert choice["config"] == model.CONFIGS[0] and choice["weight"] == WEIGHT
    assert rejected["accepted"] is False and min(r["gain"] for r in rejected["confirmation"]) > 0
    assert rejected["bootstrap"][0]["lower_positive"] is True
    assert rejected["bootstrap"][1]["lower_positive"] is False
    assert plan["choice"] == choice and plan["strict_confirmation"] == rejected
    assert not (OUT / "frozen_before_final.json").exists()
    freeze = dict(status="USER_AUTHORIZED_EXPERIMENTAL_ONLY", at_utc=datetime.now(timezone.utc).isoformat(),
        model_signature=model.sig(), model_source_sha256=b.sha(model.__file__), source_sha256=b.sha(__file__),
        verifier_sha256=b.sha(independent.__file__), runner_sha256=b.sha(model.runner.__file__),
        baseline_sha256=v.ANCHOR_SHA, plan_sha256=b.sha(plan_path),
        choice=choice, strict_confirmation=rejected, strict_gate_pass=False,
        no_retuning=True, official_score=None, manual_submission_only=True,
        target="Any positive official gain over v313; unconfirmed until user feedback")
    b.save(OUT / "frozen_before_final.json", freeze)
    # New final models live separately; all original validation artifacts remain intact.
    model.OUT = OUT / "expert"
    model.OUT.mkdir(exist_ok=True)
    d = model.load()
    k = d["tasks"].index(TASK)
    model.optimizer_reference()
    expert = model.predict(d, k, "final", choice["config"])
    print(json.dumps(dict(stage="final_training_complete", task=TASK, rows=len(expert))), flush=True)
    check = independent.check_partition(d, TASK, "final")
    replayed = model.replay(d, k, "final", choice["config"])
    np.testing.assert_array_equal(expert, replayed)
    base = v.baseline(d, k, "final")
    fused = v.mix("MAE", base, expert, WEIGHT)
    np.testing.assert_array_equal(fused, (1. - WEIGHT) * base + WEIGHT * replayed)
    check["independent_final_arithmetic_fusion"] = True
    print(json.dumps(dict(stage="final_independent_verification", **check)), flush=True)

    original = v.ANCHOR.read_bytes()
    assert hashlib.sha256(original).hexdigest() == v.ANCHOR_SHA
    rows, spans = b.table(v.ANCHOR), field_spans(original)
    assert len(spans) == len(rows) + 1
    pieces, cursor, count, changed = [], 0, 0, 0
    for i, row in enumerate(rows, 1):
        if row["task"] != TASK:
            continue
        value = format(float(fused[count]), ".17g").encode("ascii")
        count += 1
        lo, hi = spans[i][4]
        pieces.extend((original[cursor:lo], value))
        cursor = hi
        changed += value != original[lo:hi]
    pieces.append(original[cursor:])
    assert count == len(fused) and changed > 0
    today = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y%m%d")
    dest = ROOT / "admet_submissions/v346_experimental_solubility"
    dest.mkdir(parents=True, exist_ok=True)
    file = dest / ("我从山北来带着兰花草-小分子药物预测技术创新赛-" + today + ".csv")
    assert not file.exists()
    file.write_bytes(b"".join(pieces))
    format_check = verify(file, ROOT / "admet_official/submission_template.csv")
    new_rows = b.table(file)
    assert len(new_rows) == len(rows)
    np.testing.assert_array_equal(np.array([float(r["y_pred"]) for r in new_rows if r["task"] == TASK]), fused)
    for old, new in zip(rows, new_rows):
        if old["task"] != TASK:
            assert old == new
    raw, cursor, restored = file.read_bytes(), 0, []
    new_spans = field_spans(raw)
    for i in range(1, len(rows) + 1):
        lo, hi = new_spans[i][4]
        old_lo, old_hi = spans[i][4]
        restored.extend((raw[cursor:lo], original[old_lo:old_hi]))
        cursor = hi
    restored.append(raw[cursor:])
    assert b"".join(restored) == original
    (dest / "实验提交说明.md").write_text(
        "# 实验文件：官方效果未知，未经稳定性确认\n\n"
        "目标改为任何可验证的官方提升。以用户报告 51.205 的 v313 为基线，"
        "仅更新溶解度，沿用 v329 在开发阶段冻结的 SAM 模型、原种子及 0.20 算术融合。\n\n"
        "四分区相对 MAE 下降为 2.079%、2.495%、2.511%、0.234%；"
        "fold2 骨架 90% 区间为 [1.852%, 9.435%]，valid 为 [-0.252%, 0.763%]。"
        "严格稳定性结论仍为拒绝。选择这一候选是在查看既有验证结果之后作出的，"
        "公开验证标签已反复使用；局部正增益不能保证隐藏测试提分。\n\n"
        "该文件供用户手动官方测试，可能升分、持平或退步。尚不能称为新最高分。"
        "建议与 v313 成绩单逐项对照，仅溶解度的成绩应发生变化。\n\n"
        "其余 21 项、Drug_ID、行序及所有非预测字节与 v313 一致。"
        "最终特征重建、模型重载、标签隔离、注意力公式、算术融合和模板检查通过。\n\n"
        f"CSV SHA256：{b.sha(file)}\n", encoding="utf-8")
    result = dict(status="VERIFIED_EXPERIMENTAL_DELIVERY", at_utc=datetime.now(timezone.utc).isoformat(),
        purpose="User-authorized single-task manual official experiment; any positive gain sought, unconfirmed",
        user_authorized_experimental_csv=True, strict_gate_pass=False,
        strict_confirmation_remains_rejected=True, strict_accepted_tasks=[],
        experimental_tasks=[TASK], weights={TASK: WEIGHT}, checks=[check],
        csv=file.relative_to(ROOT).as_posix(), csv_sha256=b.sha(file), csv_created=True,
        format_verification=format_check, changed_rows={TASK: changed}, unchanged_tasks=21,
        all_other_task_prediction_bytes_exact=True, all_nonprediction_bytes_exact=True,
        baseline_sha256=v.ANCHOR_SHA, official_best=51.205, official_score=None,
        target="Any positive official improvement", target_achieved=False, platform_submission=False,
        official_gain_established=False, no_retuning=True, retrospective_candidate_choice=True)
    b.save(OUT / "delivery.json", result)
    b.save(ROOT / "docs/v346_experimental_delivery.json", result)
    b.save(file.with_suffix(".verification.json"), result)

    files = [p for p in OUT.rglob("*") if p.is_file()]
    files += list(dest.glob("*"))
    files += [Path(__file__).resolve(), Path(model.__file__), Path(model.original.__file__), Path(independent.__file__),
              ROOT / "tools/v314_current_baseline.py", ROOT / "tools/v314_v313_experiment_runner.py",
              plan_path, ROOT / "docs/v346_experimental_delivery.json", ROOT / "docs/v330_verification.json",
              ROOT / "docs/platform_scorecard_v313_20261101.json", ROOT / "docs/current_best_after_v321_20261101.json",
              original_out / "frozen_selection.json", original_out / "confirmation.json", original_out / "frozen_before_scores.json"]
    assert len(files) == len(set(files))
    manifest = dict(files=[dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=b.sha(p))
                           for p in files], dependencies=d["sources"], baseline_sha256=v.ANCHOR_SHA)
    mp = OUT / "manifest.json"
    b.save(mp, manifest)
    files.append(mp)
    archive = ROOT / "releases/v346_experimental_solubility_20261101.tar.gz"
    assert not archive.exists()
    with tarfile.open(archive, "w:gz") as tf:
        for path in files:
            tf.add(path, arcname=path.relative_to(ROOT).as_posix())
    with tarfile.open(archive) as tf:
        for entry in manifest["files"]:
            assert hashlib.sha256(tf.extractfile(entry["path"]).read()).hexdigest() == entry["sha256"]
    summary = dict(status="PASS", path=archive.relative_to(ROOT).as_posix(), sha256=b.sha(archive),
                   files=len(files), all_member_hashes_verified=True, experimental_only=True)
    b.save(ROOT / "docs/v346_archive.json", summary)
    print(json.dumps(dict(delivery=result, archive=summary)), flush=True)


if __name__ == "__main__":
    main()
