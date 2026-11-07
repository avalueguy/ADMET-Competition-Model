"""Independent local rank-fusion, protected tasks and template checks for v321."""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from repack_upload_template import field_spans
from verify_submission import verify

ROOT = Path(__file__).resolve().parents[1]


def main():
    delivery = json.loads((ROOT / "docs/v321_experimental_delivery.json").read_text(encoding="utf-8-sig"))
    plan = json.loads((ROOT / "docs/v321_experimental_plan.json").read_text(encoding="utf-8-sig"))
    assert delivery["status"] == "VERIFIED_EXPERIMENTAL_DELIVERY"
    assert delivery["strict_gate_pass"] is False and delivery["strict_confirmation_remains_rejected"] is True
    assert delivery["official_score"] is None and delivery["platform_submission"] is False
    assert plan["user_authorized_experimental_csv"] is True
    file = ROOT / delivery["csv"]
    assert file.resolve().is_relative_to((ROOT / "admet_submissions/v321_experimental_cyp3a4").resolve())
    raw = file.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == delivery["csv_sha256"]
    baseline = next((ROOT / "admet_submissions/v313_experimental_ppbr_lipo").glob("*.csv"))
    original = baseline.read_bytes()
    assert hashlib.sha256(original).hexdigest() == delivery["baseline_sha256"]
    table = lambda value:list(csv.reader(io.StringIO(value.decode("utf-8-sig"), newline="")))
    old, new = table(original), table(raw)
    assert len(old) == len(new) == 16380
    task, weight = "cyp3a4_substrate_carbonmangels", .2
    assert delivery["weights"] == {task:weight}
    counts, changed = {}, {}
    for before, after in zip(old[1:], new[1:]):
        t = before[0]
        assert before[:4] + before[5:] == after[:4] + after[5:]
        counts[t] = counts.get(t, 0) + 1
        if t != task:
            assert before == after
        elif before[4] != after[4]:
            changed[t] = changed.get(t, 0) + 1
    assert changed == delivery["changed_rows"]
    pred = next((ROOT / f"admet_model_outputs_v321/expert/models/{task}/final").glob("*/predictions.npz"))
    expert = np.load(pred)["p"]
    base = np.array([float(r[4]) for r in old[1:] if r[0] == task])
    actual = np.array([float(r[4]) for r in new[1:] if r[0] == task])
    np.testing.assert_array_equal(actual, (1. - weight) * rankdata(base) / len(base) + weight * rankdata(expert) / len(expert))
    old_spans, spans = field_spans(original), field_spans(raw)
    restored, cursor = [], 0
    for i in range(1, len(old)):
        lo, hi = spans[i][4]
        old_lo, old_hi = old_spans[i][4]
        restored.extend((raw[cursor:lo], original[old_lo:old_hi]))
        cursor = hi
    restored.append(raw[cursor:])
    assert b"".join(restored) == original
    result = dict(status="PASS", experimental_only=True, strict_gate_pass=False,
        csv=delivery["csv"], sha256=delivery["csv_sha256"], rows=16379, tasks=len(counts),
        changed_rows=changed, unchanged_tasks=21, local_rank_fusion_recomputed=True,
        ppbr_and_lipophilicity_preserved=True, all_nonprediction_bytes_exact=True,
        unchanged_task_prediction_bytes_exact=True, official_score=None, platform_submission=False,
        format_verification=verify(file, ROOT / "admet_official/submission_template.csv"))
    (ROOT / "docs/v321_local_delivery_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
