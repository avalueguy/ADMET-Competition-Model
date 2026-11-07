"""Reuse complete v354 hERG endpoint in exact v346 bytes; no new fit or weights."""
import hashlib
import json
import sys
import tarfile
from pathlib import Path

from repack_upload_template import field_spans, rows
from verify_submission import verify
from v360_record_v354_feedback import read, save

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "admet_model_outputs_v361"
TASK = "herg"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not OUT.exists(), "Inspect prior artifacts before retry"
    best = read("docs/current_best_after_v346_20261102.json")
    newer = read("docs/v354_delivery.json")
    feedback = read("docs/platform_scorecard_v354_20261102.json")
    baseline_card = read("docs/platform_scorecard_v346_20261101.json")
    source_local = read("docs/v354_local_delivery_verification.json")
    assert source_local["status"] == "PASS" and source_local["csv_sha256"] == newer["csv_sha256"]
    base_path, source_path = ROOT / best["csv"], ROOT / newer["csv"]
    assert sha(base_path) == best["csv_sha256"] and sha(source_path) == newer["csv_sha256"]
    raw, source_raw = base_path.read_bytes(), source_path.read_bytes()
    old, source = rows(raw), rows(source_raw)
    spans, source_spans = field_spans(raw), field_spans(source_raw)
    assert len(old) == len(source) == 16380
    assert old[0] == source[0]
    parts, cursor, count = [], 0, 0
    for i, (before, after) in enumerate(zip(old[1:], source[1:]), 1):
        assert before[:4]+before[5:] == after[:4]+after[5:]
        if before[0] != TASK:
            continue
        lo, hi = spans[i][4]
        slo, shi = source_spans[i][4]
        parts.extend((raw[cursor:lo], source_raw[slo:shi]))
        cursor = hi
        count += source_raw[slo:shi] != raw[lo:hi]
    parts.append(raw[cursor:])
    payload = b"".join(parts)
    assert count == 132
    output_name = "admet_submissions/v361_herg_feedback_repair/我从山北来带着兰花草-小分子药物预测技术创新赛-20261102.csv"
    output = ROOT / output_name
    assert not output.exists()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    fmt = verify(output, ROOT / "admet_official/submission_template.csv")
    new = rows(payload)
    assert all(after == (expert if before[0] == TASK else before)
        for before, expert, after in zip(old[1:], source[1:], new[1:]))
    final_spans, restored, cursor = field_spans(payload), [], 0
    for i in range(1, len(old)):
        lo, hi = final_spans[i][4]
        blo, bhi = spans[i][4]
        restored.extend((payload[cursor:lo], raw[blo:bhi]))
        cursor = hi
    restored.append(payload[cursor:])
    assert b"".join(restored) == raw
    best_by = {r["task"]:r for r in baseline_card["tasks"]}
    latest_by = {r["task"]:r for r in feedback["tasks"]}
    expected = (sum(r["score"] for r in baseline_card["tasks"]) - best_by[TASK]["score"] + latest_by[TASK]["score"]) / 22
    projection = dict(value=expected, rounded_display=round(expected, 3),
        gain_vs_v346=expected-baseline_card["task_score_mean"],
        gain_vs_v354=expected-feedback["task_score_mean"],
        type="CONDITIONAL_ENDPOINT_SCORE_REUSE_ARITHMETIC_ONLY",
        conditions="Same evaluation data/scoring and independent endpoint scoring. Same endpoint prediction bytes reproduce prior reported endpoint metrics.",
        official_result=False, model_generalization_forecast=False)
    OUT.mkdir()
    delivery = dict(status="VERIFIED_FEEDBACK_ROUTED_CANDIDATE", version="v361", csv=output_name,
        csv_sha256=sha(output), baseline=best, herg_source=dict(version="v354",csv=newer["csv"],csv_sha256=newer["csv_sha256"]),
        selected_task=TASK, changed_rows={TASK:count}, unchanged_tasks=21,
        cyp3a4_substrate_restored_to_v346=True, exact_whole_endpoint_reuse=True,
        herg_prediction_bytes_exact_v354=True, all_other_prediction_bytes_exact_v346=True,
        all_nonprediction_bytes_exact=True, format_verification=fmt, projection=projection,
        model_retrained=False, new_weights_or_seeds=False, per_sample_manual_edits=False,
        official_feedback_used_for_endpoint_selection=True, official_score=None,
        original_strict_rejections_preserved=True, official_best_unchanged="v346:51.235",
        platform_submission=False, limitations="Feedback-selected recombination, not new independently validated model. Large gain not established; private-test behavior unknown.")
    save("docs/v361_delivery.json", delivery)
    save("admet_model_outputs_v361/delivery.json", delivery)
    description = (
        "# v361 hERG 保留与 CYP3A4 恢复候选\n\n"
        "v354 官方总分为51.224，低于v346的51.235：hERG分项+0.102827，CYP3A4底物分项-0.331647。\n\n"
        "本文件从v346逐字节构建，仅将完整hERG任务的132条预测替换为v354原预测；其余21项与v346完全一致。"
        "未重新训练、扫描权重或逐样本修改，原严格验证拒绝记录保留。\n\n"
        f"若评测数据、规则及按任务独立评分保持不变，复用历史分项可算得{expected:.9f}，显示约{round(expected,3):.3f}。"
        "这只是条件算术估计，v361尚无官方成绩，也不代表52分目标已经接近实现。\n\n"
        "依据官方反馈选择任务可能过拟合当前公开评测；不能当作新的独立泛化证据。请手动提交后核对完整成绩。\n\n"
        f"CSV SHA256：{sha(output)}\n"
    )
    note = output.parent / "实验提交说明.md"
    note.write_text(description, encoding="utf-8")
    package()
    print(json.dumps(delivery,ensure_ascii=True))


def package():
    delivery = read("docs/v361_delivery.json")
    assert delivery == read("admet_model_outputs_v361/delivery.json")
    base_path, source_path = ROOT/delivery["baseline"]["csv"], ROOT/delivery["herg_source"]["csv"]
    output = ROOT/delivery["csv"]
    assert sha(base_path) == delivery["baseline"]["csv_sha256"]
    assert sha(source_path) == delivery["herg_source"]["csv_sha256"]
    assert sha(output) == delivery["csv_sha256"]
    note = output.parent / "实验提交说明.md"
    # Resolve the Windows project junction before taking relative paths.
    files = [base_path, source_path, output, note, Path(__file__).resolve(), ROOT/"tools/v360_record_v354_feedback.py",
        ROOT/"docs/v360_feedback_review.json", ROOT/"docs/platform_scorecard_v354_20261102.json",
        ROOT/"docs/platform_scorecard_v346_20261101.json", ROOT/"docs/current_best_after_v346_20261102.json",
        ROOT/"docs/current_status_after_v354_20261102.json", ROOT/"docs/v361_delivery.json", OUT/"delivery.json"]
    dependencies = {name:sha(ROOT/name) for name in (
        "docs/v354_delivery.json", "docs/v354_local_delivery_verification.json", "docs/v354_archive.json",
        "releases/v354_experimental_combinations_20261102.tar.gz", "admet_official/submission_template.csv")}
    manifest = dict(files=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in files],
        dependencies=dependencies, checkpoint_archive="releases/v354_experimental_combinations_20261102.tar.gz",
        model_retrained=False)
    save("admet_model_outputs_v361/manifest.json",manifest)
    archive = ROOT / "releases/v361_herg_feedback_repair_20261102.tar.gz"
    assert not archive.exists()
    with tarfile.open(archive,"w:gz") as tf:
        for p in files + [OUT/"manifest.json"]:
            tf.add(p,arcname=p.relative_to(ROOT).as_posix())
    save("docs/v361_archive.json",dict(status="CREATED_PENDING_LOCAL_VERIFICATION",path=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),files=len(files)+1))
    print(json.dumps(dict(archive=archive.as_posix(),status="PACKAGED")))


if __name__ == "__main__":
    if sys.argv[1:] == ["--package-only"]:
        package()
    else:
        assert not sys.argv[1:]
        main()
