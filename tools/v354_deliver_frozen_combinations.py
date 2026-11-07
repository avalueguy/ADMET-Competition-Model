"""Build a manually tested experimental CSV under the pre-frozen v352 rule."""
from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import v349_diverse_model_combinations as experiment
import v351_verify_combination_models as verifier
from repack_upload_template import field_spans
from verify_submission import verify

OUT = ROOT / "admet_model_outputs_v354"


def eligible(record):
    return (min(r["gain"] for r in record["confirmation"]) > 0
        and np.mean([r["gain"] for r in record["confirmation"]]) > experiment.v.threshold(record["metric"])
        and all(r["groups"] > 1 and r["gain_ci90"] is not None for r in record["bootstrap"]))


def main():
    assert not OUT.exists(), "Inspect any partial delivery before retry"
    source = experiment.OUT / "confirmation.json"
    result = json.loads(source.read_text())
    policy_path = ROOT / "docs/v352_combination_delivery_policy.json"
    policy = json.loads(policy_path.read_text())
    assert result["status"] == "COMPLETE"
    assert policy["model_search_source_sha256"] == experiment.sha(experiment.__file__)
    assert policy["selection_sha256"] == experiment.sha(experiment.OUT / "frozen_selection.json")
    selected = [r for r in result["records"] if eligible(r)]
    OUT.mkdir()
    freeze = dict(status="FROZEN_BEFORE_FINAL_FITS", at_hong_kong=experiment.now(),
        tasks=[r["task"] for r in selected], strict_tasks=[r["task"] for r in selected if r["strict_accepted"]],
        experimental_tasks=[r["task"] for r in selected if not r["strict_accepted"]],
        policy_sha256=experiment.sha(policy_path), confirmation_sha256=experiment.sha(source),
        source_sha256=experiment.sha(__file__), verifier_sha256=experiment.sha(verifier.__file__),
        baseline="v346", official_best=51.235, official_score=None,
        chosen_records=selected, no_weight_seed_or_config_fallback=True)
    experiment.save(OUT / "frozen_before_final.json", freeze)
    if not selected:
        experiment.save(ROOT / "docs/v354_delivery.json", dict(status="NO_ELIGIBLE_COMBINATION", csv_created=False,
            platform_submission=False, official_gain_established=False, freeze=freeze))
        print("NO_ELIGIBLE_COMBINATION", flush=True)
        return
    mods = experiment.modules()
    required = {m["version"] for r in selected for m in r["members"]}
    d = verifier.load_data(required, mods)
    checks, replacements, dependencies = [], {}, dict(d["sources"])
    for choice in selected:
        task, metric = choice["task"], choice["metric"]
        k = d["tasks"].index(task)
        for part in experiment.PARTS + ("final",):
            experts = []
            for member in choice["members"]:
                model = mods[member["version"]]
                model.predict(d, k, part, member["config"])
                check = verifier.check(d, k, part, member, mods)
                checks.append(check)
                expert, provenance = experiment.cached(d, k, part, member, mods)
                experts.append(expert)
                dependencies.update(check["model_files"])
                dependencies[provenance["path"]] = provenance["sha256"]
                print(json.dumps(dict(stage="independent_member_verified", task=task, part=part,
                    version=member["version"])), flush=True)
            base = experiment.current(d, k, part)
            prediction = experiment.fused(metric, base, experts)
            query = experiment.v.split(d, k, part)[1]
            if part != "final":
                cached = np.load(experiment.OUT / "predictions" / task / (part + ".npz"))
                np.testing.assert_array_equal(cached["query"], query)
                np.testing.assert_allclose(cached["p"], prediction, rtol=0, atol=1e-12)
                expected = next(r for r in choice["confirmation"] if r["part"] == part)
                assert abs(experiment.v.r83.gain(metric, d["y"][query], base, prediction) - expected["gain"]) < 1e-12
            else:
                replacements[task] = prediction
                (OUT / "final").mkdir(exist_ok=True)
                np.savez_compressed(OUT / "final" / (task + ".npz"), query=query, p=prediction,
                    base=base, experts=np.stack(experts), metric=metric)
    best_path = ROOT / "docs/current_best_after_v346_20261102.json"
    best = json.loads(best_path.read_text())
    anchor = ROOT / best["csv"]
    assert experiment.sha(anchor) == best["csv_sha256"]
    raw = anchor.read_bytes()
    rows, spans = experiment.v.b.table(anchor), field_spans(raw)
    pieces, cursor, counters, changed = [], 0, {t:0 for t in replacements}, {}
    for i, row in enumerate(rows, 1):
        task = row["task"]
        if task not in replacements:
            continue
        value = format(float(replacements[task][counters[task]]), ".17g").encode("ascii")
        counters[task] += 1
        lo, hi = spans[i][4]
        pieces.extend((raw[cursor:lo], value))
        cursor = hi
        changed[task] = changed.get(task, 0) + (value != raw[lo:hi])
    pieces.append(raw[cursor:])
    assert all(counters[t] == len(replacements[t]) and changed[t] > 0 for t in counters)
    date = experiment.now()[:10].replace("-", "")
    folder = ROOT / "admet_submissions/v354_experimental_model_combinations"
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / ("我从山北来带着兰花草-小分子药物预测技术创新赛-" + date + ".csv")
    assert not file.exists()
    payload = b"".join(pieces)
    file.write_bytes(payload)
    fmt = verify(file, ROOT / "admet_official/submission_template.csv")
    new_rows = experiment.v.b.table(file)
    for before, after in zip(rows, new_rows):
        if before["task"] not in replacements:
            assert before == after
    new_spans, restored, cursor = field_spans(payload), [], 0
    for i in range(1, len(rows) + 1):
        lo, hi = new_spans[i][4]
        original_lo, original_hi = spans[i][4]
        restored.extend((payload[cursor:lo], raw[original_lo:original_hi]))
        cursor = hi
    restored.append(payload[cursor:])
    assert b"".join(restored) == raw
    delivery = dict(status="VERIFIED_EXPERIMENTAL_COMBINATION_DELIVERY", at_hong_kong=experiment.now(),
        baseline=best, csv=file.relative_to(ROOT).as_posix(), csv_sha256=experiment.sha(file),
        changed_rows=changed, unchanged_tasks=22-len(replacements), selected_tasks=list(replacements),
        strict_accepted_tasks=freeze["strict_tasks"], experimental_tasks=freeze["experimental_tasks"],
        checks=checks, format_verification=fmt, csv_created=True,
        all_nonprediction_bytes_exact=True, unchanged_task_prediction_bytes_exact=True,
        policy_sha256=experiment.sha(policy_path), official_score=None, official_gain_established=False,
        platform_submission=False, retrospective_public_label_reuse=True,
        limitations="Experimental. Many combinations searched, public labels reused, inherited OOF approximate. Positive local points do not establish hidden-test gain. Strict rejections preserved; no official score estimate.")
    experiment.save(OUT / "delivery.json", delivery)
    experiment.save(ROOT / "docs/v354_delivery.json", delivery)
    (folder / "实验提交说明.md").write_text(
        "# v354 多模型组合实验\n\n"
        "当前最佳仍为 v346（51.235）。该文件供用户手动官方测试，官方效果未知，可能退步。\n\n"
        "基于固定规则搜索 10611 个双模型、三模型组合；先冻结每项唯一开发赢家，再验证。"
        "最终仅包含四处分数增益为正、平均增益过原阈值且两份确认区间可计算的任务。"
        "严格稳定性拒绝记录保持不变；未通过双区间检验的任务仍明确标为实验。\n\n"
        "更新任务：" + "、".join(replacements) + "。其余任务及所有非预测字节与 v346 完全一致。\n\n"
        "组合：60% 基线、40% 专家均值；回归用原预测算术平均，排序指标用各模型的平均秩。"
        "没有按官方反馈扫描权重，未使用测试标签或新增外部数据。\n\n"
        "独立模型重载、非拟合目标标签隔离、融合复算和官方模板检查通过。"
        "公开验证标签已重复使用，大规模组合搜索存在选择偏差。仅凭本地结果不能认定提分。\n\n"
        f"CSV SHA256：{experiment.sha(file)}\n", encoding="utf-8")
    # Package the current result and member checkpoints; never rewrite old releases.
    files = [p for p in OUT.rglob("*") if p.is_file()]
    files += [p for p in experiment.OUT.rglob("*") if p.is_file()]
    files += list(folder.iterdir())
    files += [Path(__file__), Path(verifier.__file__), Path(experiment.__file__), policy_path, best_path,
        ROOT / "docs/platform_scorecard_v346_20261101.json", ROOT / "docs/v354_delivery.json",
        ROOT / "tools/v350_run_combination_confirmation.py", ROOT / "tools/v352_freeze_combination_delivery_policy.py",
        ROOT / "tools/v353_prefit_frozen_combination_tail.py", anchor]
    files += [ROOT / name for name in dependencies if "/models/" in name and name in {p for c in checks for p in c["model_files"]}]
    files += [Path(mods[n].__file__) for n in required]
    files = sorted(set(p.resolve() for p in files))
    manifest = dict(files=[dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
        sha256=experiment.sha(p)) for p in files], dependencies=d["sources"],
        baseline_sha256=best["csv_sha256"], model_dependency_hashes=dependencies)
    manifest_path = OUT / "manifest.json"
    experiment.save(manifest_path, manifest)
    archive = ROOT / "releases" / ("v354_experimental_combinations_" + date + ".tar.gz")
    assert not archive.exists()
    with tarfile.open(archive, "w:gz") as tf:
        for path in files + [manifest_path]:
            tf.add(path, arcname=path.relative_to(ROOT).as_posix())
    with tarfile.open(archive) as tf:
        for record in manifest["files"]:
            assert hashlib.sha256(tf.extractfile(record["path"]).read()).hexdigest() == record["sha256"]
    info = dict(status="PASS", path=archive.relative_to(ROOT).as_posix(), sha256=experiment.sha(archive),
        files=len(files)+1, all_member_hashes_verified=True, experimental_only=True)
    experiment.save(ROOT / "docs/v354_archive.json", info)
    print(json.dumps(dict(stage="delivery_complete", csv=delivery["csv"], sha256=delivery["csv_sha256"],
        changed_rows=changed, archive=info)), flush=True)


if __name__ == "__main__":
    main()
