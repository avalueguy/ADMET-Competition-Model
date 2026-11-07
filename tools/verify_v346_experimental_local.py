"""Independent local arithmetic-fusion and exact-byte checks for v346."""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from repack_upload_template import field_spans
from verify_submission import verify

ROOT = Path(__file__).resolve().parents[1]


def main():
    delivery = json.loads((ROOT / "docs/v346_experimental_delivery.json").read_text(encoding="utf-8-sig"))
    plan = json.loads((ROOT / "docs/v346_experimental_plan.json").read_text(encoding="utf-8-sig"))
    assert delivery["status"] == "VERIFIED_EXPERIMENTAL_DELIVERY"
    assert delivery["strict_gate_pass"] is False and delivery["strict_confirmation_remains_rejected"] is True
    assert delivery["official_score"] is None and delivery["platform_submission"] is False
    assert plan["user_authorized_experimental_csv"] is True
    file = ROOT / delivery["csv"]
    assert file.resolve().is_relative_to((ROOT / "admet_submissions/v346_experimental_solubility").resolve())
    raw = file.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == delivery["csv_sha256"]
    baseline = next((ROOT / "admet_submissions/v313_experimental_ppbr_lipo").glob("*.csv"))
    original = baseline.read_bytes()
    assert hashlib.sha256(original).hexdigest() == delivery["baseline_sha256"] == plan["baseline_csv_sha256"]
    table = lambda value:list(csv.reader(io.StringIO(value.decode("utf-8-sig"), newline="")))
    old, new = table(original), table(raw)
    assert len(old) == len(new) == 16380 and old[0] == new[0]
    task, weight = "solubility_aqsoldb", .2
    assert delivery["weights"] == {task:weight} and plan["weight"] == weight
    counts, changed = {}, {}
    for before, after in zip(old[1:], new[1:]):
        t = before[0]
        assert before[:4] + before[5:] == after[:4] + after[5:]
        counts[t] = counts.get(t, 0) + 1
        if t != task:
            assert before == after
        elif before[4] != after[4]:
            changed[t] = changed.get(t, 0) + 1
    assert changed == delivery["changed_rows"] and counts[task] == 1997 and len(counts) == 22
    predictions = list((ROOT / f"admet_model_outputs_v346/expert/models/{task}/final").glob("*/predictions.npz"))
    assert len(predictions) == 1
    saved = np.load(predictions[0])
    expert = saved["p"]
    assert len(saved["query"]) == len(expert) == counts[task]
    base = np.array([float(r[4]) for r in old[1:] if r[0] == task])
    actual = np.array([float(r[4]) for r in new[1:] if r[0] == task])
    np.testing.assert_array_equal(actual, (1. - weight) * base + weight * expert)
    old_spans, spans = field_spans(original), field_spans(raw)
    restored, cursor = [], 0
    for i in range(1, len(old)):
        lo, hi = spans[i][4]
        old_lo, old_hi = old_spans[i][4]
        if old[i][0] != task:
            assert raw[lo:hi] == original[old_lo:old_hi]
        restored.extend((raw[cursor:lo], original[old_lo:old_hi]))
        cursor = hi
    restored.append(raw[cursor:])
    assert b"".join(restored) == original
    meta = json.loads((predictions[0].parent / "metadata.json").read_text(encoding="utf-8"))
    for name, digest in meta["model_hashes"].items():
        assert hashlib.sha256((predictions[0].parent / name).read_bytes()).hexdigest() == digest
    freeze = json.loads((ROOT / "admet_model_outputs_v346/frozen_before_final.json").read_text(encoding="utf-8"))
    assert str(saved["signature"]) == freeze["model_signature"]
    assert freeze["choice"] == plan["choice"] and freeze["strict_gate_pass"] is False
    for name, field in (("tools/v346_experimental_solubility.py", "source_sha256"),
                        ("tools/v329_sharpness_retrieval.py", "model_source_sha256"),
                        ("tools/v330_verify_sharpness.py", "verifier_sha256"),
                        ("tools/v314_v313_experiment_runner.py", "runner_sha256")):
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == plan[field] == freeze[field]
    result = dict(status="PASS", experimental_only=True, strict_gate_pass=False,
        csv=delivery["csv"], sha256=delivery["csv_sha256"], rows=16379, tasks=len(counts),
        changed_rows=changed, unchanged_tasks=21, local_arithmetic_fusion_recomputed=True,
        saved_model_hashes_verified=True, frozen_source_hashes_verified=True,
        all_nonprediction_bytes_exact=True, unchanged_task_prediction_bytes_exact=True,
        baseline_v313_sha256=delivery["baseline_sha256"], official_score=None,
        official_gain_established=False, platform_submission=False,
        format_verification=verify(file, ROOT / "admet_official/submission_template.csv"))
    path = ROOT / "docs/v346_local_delivery_verification.json"
    assert not path.exists()
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
