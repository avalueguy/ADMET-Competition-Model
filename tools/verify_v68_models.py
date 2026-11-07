"""Reload persisted kernel models and replay the complete v68 submission."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import admet_v67_data as b
import admet_v68_kernel as kernel
from verify_submission import read, verify


def main():
    d = b.build()
    out = kernel.OUT
    release = json.loads((out / "release.json").read_text(encoding="utf-8"))
    release67 = json.loads((b.OUT / "release.json").read_text(encoding="utf-8"))
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    assert release["signature"] == selection["signature"] == kernel.signature()
    source = ROOT / release67["file"]
    assert b.sha(source) == release67["sha256"] == release["source67_sha256"]
    file = ROOT / release["file"]
    original, exported = read(source), read(file)
    checks, accepted = [], []
    for k, record in enumerate(selection["tasks"]):
        task, metric = record["task"], record["metric"]
        assert task == d["tasks"][k] and metric == d["metrics"][k]
        positions = [i for i, row in enumerate(original[1:], 1) if row[0] == task]
        if not record["pass_gate"]:
            assert all(original[i] == exported[i] for i in positions)
            continue
        assert record["fresh_gain"] > 0 and np.mean(record["held_gains"]) > .001
        chosen = record["selection"]
        assert chosen["mean"] > .002 and sum(g > 0 for g in chosen["gains"]) >= 2
        fit = np.flatnonzero((d["tid"] == k) & (d["split"] < 2))
        query = np.flatnonzero((d["tid"] == k) & (d["split"] == 2))
        b.assert_excluded(d, fit, query)
        dest = out / task / "final"
        with np.load(dest / "predictions.npz", allow_pickle=False) as cache:
            assert str(cache["signature"]) == kernel.signature()
            np.testing.assert_array_equal(cache["fit"], fit)
            np.testing.assert_array_equal(cache["query"], query)
            stored = cache["p"].copy()
        predictions = []
        for family, (_, matrix) in enumerate(kernel.kernels(d, fit, query)):
            matrix = matrix.astype(np.float64)
            for c in kernel.P["c_values"]:
                checkpoint = dest / f"family{family}_C{c:g}.joblib"
                state = joblib.load(checkpoint)
                assert state["signature"] == kernel.signature()
                assert state["family"] == family and state["c"] == c
                np.testing.assert_array_equal(state["rows"], fit)
                model = state["model"]
                p = model.decision_function(matrix) if metric in ("AUROC", "AUPRC") else model.predict(matrix)
                if metric == "MAE":
                    p = p * state["target"]["scale"] + state["target"]["center"]
                error = float(np.max(np.abs(p - stored[len(predictions)])))
                np.testing.assert_allclose(p, stored[len(predictions)], atol=1e-9, rtol=1e-9)
                predictions.append(p)
                checks.append(dict(task=task, family=family, c=c, max_error=error,
                                   checkpoint_sha256=b.sha(checkpoint), model_reload_passed=True))
        baseline = np.array([float(original[i][4]) for i in positions])
        expected = b.blend(metric, baseline, predictions[chosen["variant"]], chosen["weight"])
        if task == "ppbr_az":
            expected = expected.clip(0, 100)
        actual = np.array([float(exported[i][4]) for i in positions])
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)
        accepted.append(task)
        b.emit("v68_final_reload", task=task, models=4, status="PASS")
    assert accepted == release["changed_tasks"]
    audit = verify(file, ROOT / "admet_official/submission_template.csv")
    assert audit["sha256"] == release["sha256"]
    b.save(file.with_suffix(".verification.json"), audit)
    b.save(out / "model_replay.json", dict(status="PASS", checks=checks,
           changed_tasks=accepted, source67_sha256=b.sha(source), sha256=audit["sha256"],
           unchanged_tasks_exact=True, exported_predictions_replayed=True,
           external_data=False, pretrained_weights=False))
    b.emit("v68_independent_replay", status="PASS", models=len(checks), sha256=audit["sha256"])


if __name__ == "__main__":
    main()
