"""Preserve v313 two-task update with exact original frozen blend controls."""
import hashlib
import json
from pathlib import Path

import numpy as np
import v146_current_baseline as old
import v304_learned_label_retrieval as retrieval

ROOT = Path(__file__).resolve().parents[1]
b, r83 = old.b, old.r83
ANCHOR = next((ROOT / "admet_submissions/v313_experimental_ppbr_lipo").glob("*.csv"))
ANCHOR_SHA = "681e9d23100c7ea39b4e6003dd6de4fac738bc467254782b7b8c4b7bc1d2941b"
PARTS = old.PARTS
WEIGHTS = {"ppbr_az": .45, "lipophilicity_astrazeneca": .2}
OUT = ROOT / "admet_model_outputs_v314"


def sig():
    return hashlib.sha256((b.sha(__file__) + old.sig() + retrieval.sig() + ANCHOR_SHA).encode()).hexdigest()


def load():
    d = old.load()
    assert b.sha(ANCHOR) == ANCHOR_SHA
    d["v313_rows"] = b.table(ANCHOR)
    return d


def split(d, k, part):
    return old.split(d, k, part)


def baseline(d, k, part):
    task = d["tasks"][k]
    if part == "final":
        return np.array([float(row["y_pred"]) for row in d["v313_rows"] if row["task"] == task])
    base = old.baseline(d, k, part)
    if task not in WEIGHTS:
        return base
    path = retrieval.OUT / "models" / task / part / retrieval.key(retrieval.CONFIGS[0]) / "predictions.npz"
    assert path.exists(), "Never retrain inherited controls silently"
    z = np.load(path)
    assert str(z["signature"]) == retrieval.sig()
    np.testing.assert_array_equal(z["query"], split(d, k, part)[1])
    return mix("MAE", base, z["p"], WEIGHTS[task])


def mix(metric, base, expert, weight):
    return old.mix(metric, base, expert, weight)


def threshold(metric):
    return old.threshold(metric)


def tasks(d):
    card = json.loads((ROOT / "docs/platform_scorecard_v313_20261101.json").read_text())
    return [r["task"] for r in sorted(card["tasks"], key=lambda r:r["score"]) if r["task"] != "hia_hou"]
