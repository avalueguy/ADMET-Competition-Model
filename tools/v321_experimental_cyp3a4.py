"""Frozen experimental CYP3A4 substrate update; statistical gate stays rejected."""
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
import v315_rank_label_retrieval as model
import v316_verify_rank_retrieval as independent
from repack_upload_template import field_spans
from verify_submission import verify

v, b = model.v, model.b
OUT = ROOT / "admet_model_outputs_v321"
TASK = "cyp3a4_substrate_carbonmangels"
WEIGHT = .2


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "delivery.json").exists(), "Inspect completed delivery before rerun"
    original_out = model.OUT
    frozen = json.loads((original_out / "frozen_selection.json").read_text())
    confirmation = json.loads((original_out / "confirmation.json").read_text())
    verified = json.loads((ROOT / "docs/v316_verification.json").read_text())
    plan = json.loads((ROOT / "docs/v321_experimental_plan.json").read_text(encoding="utf-8-sig"))
    assert plan["user_authorized_experimental_csv"] is True
    assert plan["source_sha256"] == b.sha(__file__)
    assert plan["model_source_sha256"] == b.sha(model.__file__)
    assert plan["runner_sha256"] == b.sha(model.runner.__file__)
    assert plan["task"] == TASK and plan["weight"] == WEIGHT
    assert verified["status"] == "PASS" and verified["accepted_tasks"] == []
    assert frozen["signature"] == confirmation["signature"] == verified["signature"] == model.sig()
    choice = next(r for r in frozen["selected"] if r["task"] == TASK)
    rejected = next(r for r in confirmation["records"] if r["task"] == TASK)
    assert choice["config"] == model.CONFIGS[0] and choice["weight"] == WEIGHT
    assert rejected["accepted"] is False and min(r["gain"] for r in rejected["confirmation"]) > 0
    model.OUT = OUT / "expert"
    model.OUT.mkdir(exist_ok=True)
    d = model.load()
    freeze = dict(status="USER_AUTHORIZED_EXPERIMENTAL_ONLY", at_utc=datetime.now(timezone.utc).isoformat(),
        model_signature=model.sig(), model_source_sha256=b.sha(model.__file__), source_sha256=b.sha(__file__),
        runner_sha256=b.sha(model.runner.__file__), baseline_sha256=v.ANCHOR_SHA,
        choice=choice, strict_confirmation=rejected, strict_gate_pass=False,
        no_retuning=True, official_score=None, manual_submission_only=True)
    b.save(OUT / "frozen_before_final.json", freeze)
    k = d["tasks"].index(TASK)
    expert = model.predict(d, k, "final", choice["config"])
    check = independent.check_partition(d, TASK, "final")
    replayed = model.replay(d, k, "final", choice["config"])
    np.testing.assert_array_equal(expert, replayed)
    base = v.baseline(d, k, "final")
    fused = v.mix("AUROC", base, expert, WEIGHT)
    from scipy.stats import rankdata
    np.testing.assert_array_equal(fused, (1. - WEIGHT) * rankdata(base) / len(base) + WEIGHT * rankdata(replayed) / len(replayed))
    check["independent_final_rank_fusion"] = True
    original = v.ANCHOR.read_bytes()
    assert hashlib.sha256(original).hexdigest() == v.ANCHOR_SHA
    rows = b.table(v.ANCHOR)
    spans = field_spans(original)
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
    assert count == len(fused)
    today = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y%m%d")
    dest = ROOT / "admet_submissions/v321_experimental_cyp3a4"
    dest.mkdir(parents=True, exist_ok=True)
    file = dest / ("我从山北来带着兰花草-小分子药物预测技术创新赛-" + today + ".csv")
    assert not file.exists()
    file.write_bytes(b"".join(pieces))
    format_check = verify(file, ROOT / "admet_official/submission_template.csv")
    new_rows = b.table(file)
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
        "# 实验提交：未经稳定性确认\n\n"
        "以用户报告 51.205 的 v313 为基线，仅更新 CYP3A4 底物，冻结秩融合权重 0.20。"
        "其余 21 项保持 v313 字节，包括 PPBR 和脂溶性的改进。\n\n"
        "四分区 AUROC 改善：+0.007724、+0.007910、+0.002220、+0.014337；"
        "两份确认骨架 bootstrap 90% 区间均跨零，严格门槛仍拒绝。"
        "按用户此前选择的实验 CSV 手动测试方式交付，官方效果未知，未达到 52 的证据。\n\n"
        "最终特征重建、模型重载、注意力参考、标签隔离、秩融合、模板与字节检查通过。\n\n"
        f"CSV SHA256：{b.sha(file)}\n", encoding="utf-8")
    result = dict(status="VERIFIED_EXPERIMENTAL_DELIVERY", at_utc=datetime.now(timezone.utc).isoformat(),
        purpose="User-authorized experimental CSV for manual official testing; stability not confirmed",
        user_authorized_experimental_csv=True, strict_gate_pass=False,
        strict_confirmation_remains_rejected=True, strict_accepted_tasks=[],
        experimental_tasks=[TASK], weights={TASK: WEIGHT}, checks=[check],
        csv=file.relative_to(ROOT).as_posix(), csv_sha256=b.sha(file), csv_created=True,
        format_verification=format_check, changed_rows={TASK: changed}, unchanged_tasks=21,
        all_other_task_prediction_bytes_exact=True, all_nonprediction_bytes_exact=True,
        baseline_sha256=v.ANCHOR_SHA, official_best=51.205, official_score=None,
        target_score=52., target_achieved=False, platform_submission=False,
        official_gain_established=False, no_retuning=True)
    b.save(OUT / "delivery.json", result)
    b.save(ROOT / "docs/v321_experimental_delivery.json", result)
    b.save(file.with_suffix(".verification.json"), result)
    files = [p for p in OUT.rglob("*") if p.is_file()]
    files += list(dest.glob("*"))
    files += [Path(__file__).resolve(), Path(model.__file__), Path(independent.__file__),
              ROOT / "tools/v314_current_baseline.py", ROOT / "tools/v314_v313_experiment_runner.py",
              ROOT / "docs/v321_experimental_plan.json", ROOT / "docs/v321_experimental_delivery.json",
              ROOT / "docs/v316_verification.json", original_out / "frozen_selection.json",
              original_out / "confirmation.json", original_out / "frozen_before_scores.json"]
    manifest = dict(files=[dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=b.sha(p))
                           for p in files], dependencies=d["sources"], baseline_sha256=v.ANCHOR_SHA)
    mp = OUT / "manifest.json"
    b.save(mp, manifest)
    files.append(mp)
    archive = ROOT / "releases/v321_experimental_cyp3a4_20261101.tar.gz"
    assert not archive.exists()
    with tarfile.open(archive, "w:gz") as tf:
        for path in files:
            tf.add(path, arcname=path.relative_to(ROOT).as_posix())
    with tarfile.open(archive) as tf:
        for entry in manifest["files"]:
            assert hashlib.sha256(tf.extractfile(entry["path"]).read()).hexdigest() == entry["sha256"]
    summary = dict(status="PASS", path=archive.relative_to(ROOT).as_posix(), sha256=b.sha(archive),
                   files=len(files), all_member_hashes_verified=True, experimental_only=True)
    b.save(ROOT / "docs/v321_archive.json", summary)
    print(json.dumps(dict(delivery=result, archive=summary)), flush=True)


if __name__ == "__main__":
    main()
