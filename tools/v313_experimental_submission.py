"""User-authorized experimental v304 PPBR/lipophilicity, strict gates stay rejected."""
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
import v304_learned_label_retrieval as model
import v305_verify_learned_retrieval as independent
from repack_upload_template import field_spans
from verify_submission import verify

v, b = model.v, model.b
OUT = ROOT / "admet_model_outputs_v313"
TASKS = ("ppbr_az", "lipophilicity_astrazeneca")
WEIGHTS = {"ppbr_az": .45, "lipophilicity_astrazeneca": .2}


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "delivery.json").exists(), "Inspect completed delivery before rerun."
    original_out = model.OUT
    frozen = json.loads((original_out / "frozen_selection.json").read_text())
    confirmation = json.loads((original_out / "confirmation.json").read_text())
    verified = json.loads((ROOT / "docs/v305_verification.json").read_text())
    plan = json.loads((ROOT / "docs/v313_experimental_plan.json").read_text(encoding="utf-8-sig"))
    assert plan["user_authorized_experimental_csv"] is True
    assert plan["source_sha256"] == b.sha(__file__)
    assert plan["model_source_sha256"] == b.sha(model.__file__)
    assert plan["runner_sha256"] == b.sha(model.runner.__file__)
    assert plan["tasks"] == list(TASKS) and plan["weights"] == WEIGHTS
    assert verified["status"] == "PASS" and verified["accepted_tasks"] == []
    assert frozen["signature"] == confirmation["signature"] == verified["signature"] == model.sig()
    choices, rejected = {}, {}
    for task in TASKS:
        choices[task] = next(r for r in frozen["selected"] if r["task"] == task)
        rejected[task] = next(r for r in confirmation["records"] if r["task"] == task)
        assert choices[task]["config"] == model.CONFIGS[0]
        assert choices[task]["weight"] == WEIGHTS[task]
        assert rejected[task]["accepted"] is False
        assert min(r["gain"] for r in rejected[task]["confirmation"]) > 0
    # Dedicated final models preserve every original scored partition and archive.
    model.OUT = OUT / "expert"
    model.OUT.mkdir(exist_ok=True)
    d = model.load()
    freeze = dict(status="USER_AUTHORIZED_EXPERIMENTAL_ONLY", at_utc=datetime.now(timezone.utc).isoformat(),
        model_signature=model.sig(), model_source_sha256=b.sha(model.__file__), source_sha256=b.sha(__file__),
        runner_sha256=b.sha(model.runner.__file__), baseline_sha256=v.ANCHOR_SHA,
        choices=choices, strict_confirmation=rejected, strict_gate_pass=False,
        no_retuning=True, official_score=None, manual_submission_only=True)
    b.save(OUT / "frozen_before_final.json", freeze)
    replacements, checks = {}, []
    for task in TASKS:
        k = d["tasks"].index(task)
        expert = model.predict(d, k, "final", choices[task]["config"])
        check = independent.check_partition(d, task, "final")
        replayed = model.replay(d, k, "final", choices[task]["config"])
        np.testing.assert_array_equal(expert, replayed)
        base = v.baseline(d, k, "final")
        fused = v.mix("MAE", base, expert, WEIGHTS[task])
        np.testing.assert_array_equal(fused, (1. - WEIGHTS[task]) * base + WEIGHTS[task] * replayed)
        replacements[task] = fused
        checks.append(dict(**check, independent_final_fusion=True))
        print(json.dumps(dict(stage="experimental_final_verified", **checks[-1])), flush=True)
    original = v.ANCHOR.read_bytes()
    assert hashlib.sha256(original).hexdigest() == v.ANCHOR_SHA
    rows = b.table(v.ANCHOR)
    spans = field_spans(original)
    assert len(spans) == len(rows) + 1
    pieces, cursor, changed = [], 0, {t: 0 for t in TASKS}
    counters = {t: 0 for t in TASKS}
    for i, row in enumerate(rows, 1):
        task = row["task"]
        if task not in replacements:
            continue
        value = format(float(replacements[task][counters[task]]), ".17g").encode("ascii")
        counters[task] += 1
        lo, hi = spans[i][4]
        pieces.extend((original[cursor:lo], value))
        cursor = hi
        changed[task] += value != original[lo:hi]
    pieces.append(original[cursor:])
    assert all(counters[t] == len(replacements[t]) for t in TASKS)
    today = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y%m%d")
    dest = ROOT / "admet_submissions/v313_experimental_ppbr_lipo"
    dest.mkdir(parents=True, exist_ok=True)
    file = dest / ("我从山北来带着兰花草-小分子药物预测技术创新赛-" + today + ".csv")
    assert not file.exists()
    file.write_bytes(b"".join(pieces))
    format_check = verify(file, ROOT / "admet_official/submission_template.csv")
    new_rows = b.table(file)
    assert len(new_rows) == len(rows)
    counters = {t: 0 for t in TASKS}
    for old, new in zip(rows, new_rows):
        task = old["task"]
        if task not in replacements:
            assert old == new
        else:
            assert float(new["y_pred"]) == replacements[task][counters[task]]
            counters[task] += 1
    raw, cursor, restored = file.read_bytes(), 0, []
    new_spans = field_spans(raw)
    for i in range(1, len(rows) + 1):
        lo, hi = new_spans[i][4]
        old_lo, old_hi = spans[i][4]
        restored.extend((raw[cursor:lo], original[old_lo:old_hi]))
        cursor = hi
    restored.append(raw[cursor:])
    assert b"".join(restored) == original
    readme = dest / "实验提交说明.md"
    readme.write_text(
        "# 实验提交：未经稳定性确认\n\n"
        "用户明确选择另做实验 CSV 供手动测试官方分数。此文件不属于通过原严格门槛的模型。\n\n"
        "仅替换 PPBR（v304 权重 0.45）和脂溶性（v304 权重 0.20），配置与两个种子保持冻结。"
        "其余 20 项保持实际 v145 的预测字节。两项四处验证 MAE 均降低，但确认的骨架置信区间跨零，"
        "原严格拒绝状态保持不变。\n\n"
        "最终模型独立重建特征、重载、标签隔离、注意力参考计算、行序及融合复算均通过；"
        "16,379 行、22 项、Drug_ID 和非预测字节核验通过。官方分数未知，不能宣称已提分或达到 52。\n\n"
        f"CSV SHA256：{b.sha(file)}\n", encoding="utf-8")
    result = dict(status="VERIFIED_EXPERIMENTAL_DELIVERY", at_utc=datetime.now(timezone.utc).isoformat(),
        purpose="User-authorized manual official evaluation; statistical stability NOT confirmed",
        user_authorized_experimental_csv=True, strict_gate_pass=False,
        strict_confirmation_remains_rejected=True, strict_accepted_tasks=[],
        experimental_tasks=list(TASKS), weights=WEIGHTS, checks=checks,
        csv=file.relative_to(ROOT).as_posix(), csv_sha256=b.sha(file), csv_created=True,
        format_verification=format_check, changed_rows=changed, unchanged_tasks=20,
        all_other_task_prediction_bytes_exact=True, all_nonprediction_bytes_exact=True,
        baseline_sha256=v.ANCHOR_SHA, official_best=51.166, official_score=None,
        target_score=52., target_achieved=False, platform_submission=False,
        official_gain_established=False, no_retuning=True)
    b.save(OUT / "delivery.json", result)
    b.save(ROOT / "docs/v313_experimental_delivery.json", result)
    b.save(file.with_suffix(".verification.json"), result)
    files = [p for p in OUT.rglob("*") if p.is_file()]
    files += list(dest.glob("*"))
    files += [Path(__file__).resolve(), Path(model.__file__), Path(independent.__file__),
              ROOT / "docs/v313_experimental_plan.json", ROOT / "docs/v313_experimental_delivery.json",
              ROOT / "docs/v305_verification.json", original_out / "frozen_selection.json",
              original_out / "confirmation.json", original_out / "frozen_before_scores.json"]
    manifest = dict(files=[dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=b.sha(p))
                           for p in files], dependencies=d["sources"], baseline_sha256=v.ANCHOR_SHA)
    mp = OUT / "manifest.json"
    b.save(mp, manifest)
    files.append(mp)
    archive = ROOT / "releases/v313_experimental_ppbr_lipo_20261101.tar.gz"
    assert not archive.exists()
    with tarfile.open(archive, "w:gz") as tf:
        for path in files:
            tf.add(path, arcname=path.relative_to(ROOT).as_posix())
    with tarfile.open(archive) as tf:
        for entry in manifest["files"]:
            assert hashlib.sha256(tf.extractfile(entry["path"]).read()).hexdigest() == entry["sha256"]
    summary = dict(status="PASS", path=archive.relative_to(ROOT).as_posix(), sha256=b.sha(archive),
                   files=len(files), all_member_hashes_verified=True, experimental_only=True)
    b.save(ROOT / "docs/v313_archive.json", summary)
    print(json.dumps(dict(delivery=result, archive=summary)), flush=True)


if __name__ == "__main__":
    main()
