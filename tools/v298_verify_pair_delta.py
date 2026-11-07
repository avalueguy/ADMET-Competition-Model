"""Independently replay v297 and deliver only tasks passing frozen v148 gates."""
from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import v297_pair_delta_regression as model
from repack_upload_template import field_spans
from verify_submission import verify

v, b = model.v, model.b
OUT = ROOT / "admet_model_outputs_v298"


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "verification.json").exists(), "Inspect an existing verification before rerunning."
    conf = json.loads((model.OUT / "confirmation.json").read_text())
    frozen = json.loads((model.OUT / "frozen_selection.json").read_text())
    assert conf["status"] == "COMPLETE" and conf["signature"] == frozen["signature"] == model.sig()
    before = json.loads((model.OUT / "frozen_before_scores.json").read_text())
    assert before["source_sha256"] == b.sha(model.__file__)
    assert v.sig() == before["baseline_signature"]
    assert b.sha(model.runner.__file__) == before["runner_sha256"]
    d = model.load()
    checks = []
    for task in model.TASKS:
        k = d["tasks"].index(task)
        for part in ("fold0", "fold1"):
            p = model.replay(d, k, part, model.CONFIGS[0])
            # Rebuild features and fit pairs with every non-fit target altered.
            a = model.design(d, k, part)
            saved = d["y"].copy()
            try:
                d["y"][(d["tid"] == k) & ~np.isin(np.arange(len(saved)), a["fit"])] = 999999.
                d.pop("_v96_assays", None)
                z = model.design(d, k, part)
                for name in a:
                    np.testing.assert_array_equal(a[name], z[name])
            finally:
                d["y"][:] = saved
                d.pop("_v96_assays", None)
            folder = model.OUT / "models" / task / part / model.key(model.CONFIGS[0])
            meta = json.loads((folder / "metadata.json").read_text())
            for filename, digest in meta["model_hashes"].items():
                assert b.sha(folder / filename) == digest
            # Build query neighbors again in reverse row order, including ties.
            reverse_ix, reverse_w = model.neighbors(d, a["fit"], a["query"][::-1], False)
            np.testing.assert_array_equal(reverse_ix, a["query_ix"][::-1])
            np.testing.assert_array_equal(reverse_w, a["query_w"][::-1])
            reversed_a = dict(a, query=a["query"][::-1], qx=a["qx"][::-1],
                              query_ix=reverse_ix, query_w=reverse_w)
            nets = [model.CatBoostRegressor().load_model(str(folder / f"model_{seed}.cbm"))
                    for seed in model.SEEDS]
            reversed_p = model.infer(d, reversed_a, nets)[::-1]
            if task == "ppbr_az":
                reversed_p = reversed_p.clip(0, 100)
            np.testing.assert_array_equal(p, reversed_p)
            checks.append(dict(task=task, part=part, rows=len(p),
                               feature_rebuild=True, checkpoint_reload=True,
                               row_order_invariant=True, nonfit_target_label_isolation=True))
            print(json.dumps(dict(stage="independent_replay", **checks[-1])), flush=True)
    accepted = {r["task"]: r for r in conf["records"] if r["accepted"]}
    result = dict(status="PASS", signature=model.sig(), checks=checks,
                  accepted_tasks=list(accepted), csv_created=False,
                  platform_submission=False, official_best=51.166, target_score=52.)
    changed, replacements = {}, {}
    for task, record in accepted.items():
        k = d["tasks"].index(task)
        choice = next(r for r in frozen["selected"] if r["task"] == task)
        assert model.runner.confirm(d, k, model.predict, choice) == record
        for part in ("fold2", "valid", "final"):
            model.replay(d, k, part, record["config"])
        expert = model.predict(d, k, "final", record["config"])
        replacements[task] = v.mix(record["metric"], v.baseline(d, k, "final"), expert, record["weight"])
    if accepted:
        raw = v.ANCHOR.read_bytes()
        rows = b.table(v.ANCHOR)
        spans = field_spans(raw)
        assert len(spans) == len(rows) + 1
        cursors = {task: 0 for task in replacements}
        pieces, cursor = [], 0
        for i, row in enumerate(rows, 1):
            task = row["task"]
            if task not in replacements:
                continue
            value = format(float(replacements[task][cursors[task]]), ".17g").encode("ascii")
            cursors[task] += 1
            lo, hi = spans[i][4]
            pieces.extend((raw[cursor:lo], value))
            cursor = hi
            changed[task] = changed.get(task, 0) + (value != raw[lo:hi])
        pieces.append(raw[cursor:])
        today = datetime.now().strftime("%Y%m%d")
        dest = ROOT / "admet_submissions/v298_pair_delta"
        dest.mkdir(parents=True, exist_ok=True)
        file = dest / ("我从山北来带着兰花草-小分子药物预测技术创新赛-" + today + ".csv")
        assert not file.exists()
        file.write_bytes(b"".join(pieces))
        verification = verify(file, ROOT / "admet_official/submission_template.csv")
        new = b.table(file)
        for old, row in zip(rows, new):
            if old["task"] not in accepted:
                assert old == row
        # Bytewise inverse replacement proves every other byte stayed exact.
        newraw = file.read_bytes()
        newspans = field_spans(newraw)
        restored, cursor = [], 0
        for i, row in enumerate(rows, 1):
            lo, hi = newspans[i][4]
            tlo, thi = spans[i][4]
            restored.extend((newraw[cursor:lo], raw[tlo:thi]))
            cursor = hi
        restored.append(newraw[cursor:])
        assert b"".join(restored) == raw
        verification.update(nonprediction_bytes_exact=True, unchanged_tasks_exact=True,
                            baseline_sha256=v.ANCHOR_SHA, changed_tasks=changed)
        b.save(file.with_suffix(".verification.json"), verification)
        result.update(csv_created=True, csv=file.relative_to(ROOT).as_posix(),
                      csv_sha256=b.sha(file), changed_tasks=changed,
                      csv_verification=verification)
    else:
        result["outcome"] = "NO_CONFIRMED_NEW_MODEL"
    result["at_utc"] = datetime.now(timezone.utc).isoformat()
    b.save(OUT / "verification.json", result)
    b.save(ROOT / "docs/v298_verification.json", result)
    # Archive all actual experiment inputs, checkpoints and predictions with a
    # manifest. Large immutable historical data/model dependencies stay in place.
    files = [p for p in model.OUT.rglob("*") if p.is_file()]
    files += [Path(model.__file__), Path(__file__).resolve(), OUT / "verification.json",
              ROOT / "docs/v297_final_status.json", ROOT / "docs/v298_verification.json"]
    if accepted:
        files += list((ROOT / "admet_submissions/v298_pair_delta").glob("*"))
    manifest = dict(files=[dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                               sha256=b.sha(p)) for p in files], baseline_sha256=v.ANCHOR_SHA,
                    baseline_signature=v.sig(), dependencies=d["sources"])
    mp = OUT / "manifest.json"
    b.save(mp, manifest)
    files.append(mp)
    prefix = ROOT / "releases/v298_pair_delta_20261101.tar.gz"
    assert not prefix.exists()
    with tarfile.open(prefix, "w:gz") as tf:
        for path in files:
            tf.add(path, arcname=path.relative_to(ROOT).as_posix())
    with tarfile.open(prefix) as tf:
        for entry in manifest["files"]:
            assert hashlib.sha256(tf.extractfile(entry["path"]).read()).hexdigest() == entry["sha256"]
    archive = dict(status="PASS", sha256=b.sha(prefix), files=len(files),
                   all_member_hashes_verified=True, path=prefix.relative_to(ROOT).as_posix())
    b.save(ROOT / "docs/v298_archive.json", archive)
    print(json.dumps(dict(verification=result, archive=archive)), flush=True)


if __name__ == "__main__":
    main()
