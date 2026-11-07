"""Independently reconstruct differentiable trees and conditionally deliver."""
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
import v307_differentiable_trees as model
from repack_upload_template import field_spans
from verify_submission import verify

v, b = model.v, model.b
OUT = ROOT / "admet_model_outputs_v308"


def independent_tree(net, x):
    logits = net.feature_logits.detach().numpy()
    selector = model.entmax_reference(logits)
    projected = np.einsum("bf,tdf->btd", x.astype(float), selector)
    threshold = net.threshold.detach().numpy()
    temperature = np.clip(np.exp(net.log_temperature.detach().numpy()), .03, 3.)
    z = (projected - threshold) / temperature
    yes = np.exp(-np.logaddexp(0., -z))
    bits = net.bits.numpy().astype(bool)
    probability = np.ones((len(x), logits.shape[0], len(bits)))
    for leaf, direction in enumerate(bits):
        for level, branch in enumerate(direction):
            probability[:, :, leaf] *= yes[:, :, level] if branch else 1. - yes[:, :, level]
    np.testing.assert_allclose(probability.sum(-1), 1., atol=1e-12, rtol=0)
    return (probability * net.leaves.detach().numpy()[None]).sum(-1).mean(-1) + float(net.bias.detach())


def check_partition(d, task, part):
    k = d["tasks"].index(task)
    c = model.CONFIGS[0]
    prediction = model.replay(d, k, part, c)
    a = model.design(d, k, part, c)
    saved = d["y"].copy()
    try:
        d["y"][(d["tid"] == k) & ~np.isin(np.arange(len(saved)), a["state"]["fit"])] = 999999.
        d.pop("_v96_assays", None)
        altered = model.design(d, k, part, c)
        model.same(a["state"], altered["state"])
        np.testing.assert_array_equal(a["x"], altered["x"])
        np.testing.assert_array_equal(a["qx"], altered["qx"])
    finally:
        d["y"][:] = saved
        d.pop("_v96_assays", None)
    folder = model.OUT / "models" / task / part / model.key(c)
    meta = json.loads((folder / "metadata.json").read_text())
    for name, digest in meta["model_hashes"].items():
        assert b.sha(folder / name) == digest
    nets = [model.make_loaded(folder / f"model_{seed}.pt", a["x"].shape[1], c) for seed in model.SEEDS]
    raw = [model.infer_cpu(net, a["qx"]) for net in nets]
    reference = model.output(np.mean(raw, axis=0), a["state"], task)
    np.testing.assert_array_equal(reference, prediction)
    reverse = np.mean([model.infer_cpu(net, a["qx"][::-1].copy()) for net in nets], axis=0)
    np.testing.assert_allclose(model.output(reverse, a["state"], task)[::-1], prediction, atol=1e-12, rtol=0)
    for net, result in zip(nets, raw):
        np.testing.assert_allclose(independent_tree(net, a["qx"][:3]), result[:3], atol=1e-12, rtol=0)
    return dict(task=task, part=part, rows=len(prediction), feature_rebuild=True,
        checkpoint_reload=True, nonfit_target_label_isolation=True,
        independent_entmax_bisection_and_tree_path_reference=True,
        probability_conservation=True, reversed_query_tolerance=1e-12)

def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "verification.json").exists(), "Inspect existing results before rerun."
    conf = json.loads((model.OUT / "confirmation.json").read_text())
    frozen = json.loads((model.OUT / "frozen_selection.json").read_text())
    before = json.loads((model.OUT / "frozen_before_scores.json").read_text())
    assert conf["status"] == "COMPLETE"
    assert conf["signature"] == frozen["signature"] == before["signature"] == model.sig()
    assert before["source_sha256"] == b.sha(model.__file__)
    assert before["baseline_signature"] == v.sig()
    assert before["runner_sha256"] == b.sha(model.runner.__file__)
    d = model.load()
    checks = []
    for task in model.TASKS:
        for part in ("fold0", "fold1"):
            checks.append(check_partition(d, task, part))
            print(json.dumps(dict(stage="independent_replay", **checks[-1])), flush=True)
    accepted = {}
    for record in conf["records"]:
        task = record["task"]
        k = d["tasks"].index(task)
        choice = next(c for c in frozen["selected"] if c["task"] == task)
        assert model.runner.confirm(d, k, model.predict, choice) == record
        # Rejected confirmation models are replayed too; never retuned here.
        for part in ("fold2", "valid"):
            checks.append(check_partition(d, task, part))
            print(json.dumps(dict(stage="confirmation_replay", **checks[-1])), flush=True)
        if record["accepted"]:
            accepted[task] = record
            checks.append(check_partition(d, task, "final"))
    assert sorted(accepted) == sorted(conf["accepted"])
    result = dict(status="PASS", signature=model.sig(), checks=checks,
        accepted_tasks=list(accepted), csv_created=False, platform_submission=False,
        official_best=51.166, target_score=52., official_gain_established=False)
    replacements, changed = {}, {}
    for task, record in accepted.items():
        k = d["tasks"].index(task)
        replacements[task] = v.mix(record["metric"], v.baseline(d, k, "final"),
                                   model.predict(d, k, "final", record["config"]), record["weight"])
    if replacements:
        original = v.ANCHOR.read_bytes()
        rows = b.table(v.ANCHOR)
        spans = field_spans(original)
        assert len(spans) == len(rows) + 1
        pieces, cursor = [], 0
        counters = {task: 0 for task in replacements}
        for i, row in enumerate(rows, 1):
            task = row["task"]
            if task not in replacements:
                continue
            val = format(float(replacements[task][counters[task]]), ".17g").encode("ascii")
            counters[task] += 1
            lo, hi = spans[i][4]
            pieces.extend((original[cursor:lo], val))
            cursor = hi
            changed[task] = changed.get(task, 0) + (val != original[lo:hi])
        assert all(counters[t] == len(replacements[t]) for t in counters)
        pieces.append(original[cursor:])
        today = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y%m%d")
        dest = ROOT / "admet_submissions/v308_differentiable_trees"
        dest.mkdir(parents=True, exist_ok=True)
        file = dest / ("我从山北来带着兰花草-小分子药物预测技术创新赛-" + today + ".csv")
        assert not file.exists()
        file.write_bytes(b"".join(pieces))
        verification = verify(file, ROOT / "admet_official/submission_template.csv")
        new = b.table(file)
        for old, row in zip(rows, new):
            if old["task"] not in accepted:
                assert old == row
        raw = file.read_bytes()
        new_spans = field_spans(raw)
        restored, cursor = [], 0
        for i, row in enumerate(rows, 1):
            lo, hi = new_spans[i][4]
            old_lo, old_hi = spans[i][4]
            restored.extend((raw[cursor:lo], original[old_lo:old_hi]))
            cursor = hi
        restored.append(raw[cursor:])
        assert b"".join(restored) == original
        result.update(outcome="VERIFIED_LOCAL_CANDIDATE_OFFICIAL_SCORE_UNKNOWN", csv_created=True,
            csv=str(file.relative_to(ROOT)), csv_sha256=b.sha(file), changed_rows=changed,
            format_verification=verification, nonprediction_and_unchanged_task_bytes_exact=True)
    else:
        result["outcome"] = "NO_CONFIRMED_NEW_MODEL"
    result["at_utc"] = datetime.now(timezone.utc).isoformat()
    b.save(OUT / "verification.json", result)
    b.save(ROOT / "docs/v308_verification.json", result)
    files = [p for p in model.OUT.rglob("*") if p.is_file()]
    files += [Path(model.__file__), Path(__file__).resolve(), ROOT / "tools/v309_finalize_differentiable_trees.py", OUT / "verification.json",
              ROOT / "docs/v307_final_status.json", ROOT / "docs/v308_verification.json"]
    for path in (ROOT / "tools/v307_differentiable_trees.preflight1.py",
                 ROOT / "v307_smoke.preflight1.log", ROOT / "docs/v307_training_plan.json"):
        if path.exists():
            files.append(path)
    if accepted:
        files += list((ROOT / "admet_submissions/v308_differentiable_trees").glob("*"))
    manifest = dict(files=[dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                               sha256=b.sha(p)) for p in files], baseline_sha256=v.ANCHOR_SHA,
                    baseline_signature=v.sig(), dependencies=d["sources"])
    mp = OUT / "manifest.json"
    b.save(mp, manifest)
    files.append(mp)
    archive = ROOT / "releases/v308_differentiable_trees_20261101.tar.gz"
    assert not archive.exists()
    with tarfile.open(archive, "w:gz") as tf:
        for file in files:
            tf.add(file, arcname=file.relative_to(ROOT).as_posix())
    with tarfile.open(archive) as tf:
        for entry in manifest["files"]:
            assert hashlib.sha256(tf.extractfile(entry["path"]).read()).hexdigest() == entry["sha256"]
    summary = dict(status="PASS", path=archive.relative_to(ROOT).as_posix(), sha256=b.sha(archive),
                   files=len(files), all_member_hashes_verified=True)
    b.save(ROOT / "docs/v308_archive.json", summary)
    print(json.dumps(dict(verification=result, archive=summary)), flush=True)


if __name__ == "__main__":
    main()
