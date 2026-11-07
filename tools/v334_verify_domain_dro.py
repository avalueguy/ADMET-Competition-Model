"""Independently reconstruct learned label retrieval and conditionally deliver."""
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
import v333_chemical_domain_dro as model
from repack_upload_template import field_spans
from verify_submission import verify

v, b = model.v, model.b
OUT = ROOT / "admet_model_outputs_v334"


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
    reference = model.output(np.mean([model.infer_cpu(net, a) for net in nets], axis=0), a["state"], task)
    np.testing.assert_allclose(reference, prediction, rtol=0, atol=1e-12)
    state = a["state"]
    reverse_state = dict(state, query_ix=state["query_ix"][::-1].copy(),
                         query_similarity=state["query_similarity"][::-1].copy())
    reverse_a = dict(a, qx=a["qx"][::-1].copy(), state=reverse_state)
    reverse = np.mean([model.infer_cpu(net, reverse_a) for net in nets], axis=0)
    reverse = model.output(reverse, a["state"], task)
    np.testing.assert_allclose(reverse[::-1], prediction, rtol=0, atol=1e-12)
    # Independently evaluate reference keys, attention, values and head for three rows.
    for net in nets:
        n = min(3, len(a["qx"]))
        ids = state["query_ix"][:n]
        with model.torch.inference_mode():
            qh, qk = net.encoded(model.torch.as_tensor(a["qx"][:n], dtype=model.torch.float64))
            rk = net.encoded(model.torch.as_tensor(a["x"][ids].reshape(-1, a["x"].shape[1]), dtype=model.torch.float64))[1].reshape(n, c["neighbors"], -1)
            labels = model.torch.as_tensor(state["target"][ids], dtype=model.torch.float64)
            similarity = state["query_similarity"][:n]
            difference = qk[:, None] - rk
            scores = -difference.square().mean(dim=-1).numpy() + 4. * np.log(np.maximum(similarity.astype(float), 1e-4))
            weight = np.exp(scores - scores.max(axis=1, keepdims=True))
            weight /= weight.sum(axis=1, keepdims=True)
            values = net.label(labels[..., None]) + net.difference(difference)
            context = (values.numpy() * weight[..., None]).sum(axis=1)
            raw = net.head(model.torch.cat([qh, model.torch.as_tensor(context)], dim=1)).squeeze(-1).numpy()
            raw += float(net.context_coefficient) * (weight * labels.numpy()).sum(axis=1)
        np.testing.assert_allclose(raw, model.infer_cpu(net, a)[:n], rtol=0, atol=1e-12)
    fit = state["fit"]
    groups = d["groups"][fit]
    domain = state["domain"]
    assert all(len(np.unique(domain[groups == group])) == 1 for group in np.unique(groups))
    np.testing.assert_allclose(state["domain_prior"], np.bincount(domain) / len(domain), rtol=0, atol=0)
    assert state["domain_centers"].shape == (c["domains"], 4096)
    assert meta["fit_only_domain_construction"]
    assert meta["domain_counts"] == np.bincount(domain).tolist()
    for history in meta["histories"]:
        for entry in history["history"]:
            np.testing.assert_allclose(sum(entry["q"]), 1., rtol=0, atol=1e-12)
            assert np.isfinite(entry["q"]).all() and min(entry["q"]) > 0
    for rows, ids in ((fit, state["train_ix"]), (state["query"], state["query_ix"])):
        assert np.all(d["canonical"][rows, None] != d["canonical"][fit][ids])
        assert np.all(d["groups"][rows, None] != d["groups"][fit][ids])
    rebuilt_ix, rebuilt_sim = model.neighbors(d, fit, state["query"][::-1], c["neighbors"])
    np.testing.assert_array_equal(rebuilt_ix, state["query_ix"][::-1])
    np.testing.assert_array_equal(rebuilt_sim, state["query_similarity"][::-1])
    return dict(task=task, part=part, rows=len(prediction), feature_rebuild=True,
        checkpoint_reload=True, nonfit_target_label_isolation=True,
        independent_attention_reference=True, reference_labels_fit_only=True,
        reference_identity_and_scaffold_disjoint=True, neighbor_reconstruction=True,
        reversed_query_tolerance=1e-12, fit_only_domains=True, scaffold_domain_integrity=True)


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
    model.objective_reference()
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
        official_best=51.205, target_score=52., official_gain_established=False)
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
        dest = ROOT / "admet_submissions/v334_chemical_domain_dro"
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
    b.save(ROOT / "docs/v334_verification.json", result)
    files = [p for p in model.OUT.rglob("*") if p.is_file()]
    files += [Path(model.__file__), Path(__file__).resolve(), ROOT / "tools/v335_finalize_domain_dro.py",
              ROOT / "tools/v315_rank_label_retrieval.py", ROOT / "tools/v314_current_baseline.py", ROOT / "tools/v314_v313_experiment_runner.py",
              ROOT / "docs/platform_scorecard_v313_20261101.json", ROOT / "docs/v314_feedback_review.json", OUT / "verification.json",
              ROOT / "docs/v333_final_status.json", ROOT / "docs/v334_verification.json"]
    for path in (ROOT / "tools/v333_chemical_domain_dro.preflight1.py",
                 ROOT / "v333_smoke.preflight1.log", ROOT / "docs/v333_training_plan.json"):
        if path.exists():
            files.append(path)
    if accepted:
        files += list((ROOT / "admet_submissions/v334_chemical_domain_dro").glob("*"))
    manifest = dict(files=[dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                               sha256=b.sha(p)) for p in files], baseline_sha256=v.ANCHOR_SHA,
                    baseline_signature=v.sig(), dependencies=d["sources"])
    mp = OUT / "manifest.json"
    b.save(mp, manifest)
    files.append(mp)
    archive = ROOT / "releases/v334_chemical_domain_dro_20261101.tar.gz"
    assert not archive.exists()
    with tarfile.open(archive, "w:gz") as tf:
        for file in files:
            tf.add(file, arcname=file.relative_to(ROOT).as_posix())
    with tarfile.open(archive) as tf:
        for entry in manifest["files"]:
            assert hashlib.sha256(tf.extractfile(entry["path"]).read()).hexdigest() == entry["sha256"]
    summary = dict(status="PASS", path=archive.relative_to(ROOT).as_posix(), sha256=b.sha(archive),
                   files=len(files), all_member_hashes_verified=True)
    b.save(ROOT / "docs/v334_archive.json", summary)
    print(json.dumps(dict(verification=result, archive=summary)), flush=True)


if __name__ == "__main__":
    main()
