"""Independent local fusion, frozen-rule and byte-level delivery verification."""
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result_file = ROOT / "docs/v354_local_delivery_verification.json"
    assert not result_file.exists()
    delivery = json.loads((ROOT / "docs/v354_delivery.json").read_text(encoding="utf-8"))
    assert delivery["status"] == "VERIFIED_EXPERIMENTAL_COMBINATION_DELIVERY"
    assert delivery["official_score"] is None and delivery["platform_submission"] is False
    freeze = json.loads((ROOT / "admet_model_outputs_v354/frozen_before_final.json").read_text(encoding="utf-8"))
    policy_path = ROOT / "docs/v352_combination_delivery_policy.json"
    assert sha(policy_path) == freeze["policy_sha256"] == delivery["policy_sha256"]
    conf_file = ROOT / "admet_model_outputs_v349/confirmation.json"
    assert sha(conf_file) == freeze["confirmation_sha256"]
    conf = json.loads(conf_file.read_text(encoding="utf-8"))
    threshold = lambda m:.012 if m == "MAE" else .008 if m == "Spearman" else .003
    eligible = [r for r in conf["records"] if min(p["gain"] for p in r["confirmation"]) > 0
        and np.mean([p["gain"] for p in r["confirmation"]]) > threshold(r["metric"])
        and all(b["groups"] > 1 and b["gain_ci90"] is not None for b in r["bootstrap"])]
    assert eligible == freeze["chosen_records"]
    assert [r["task"] for r in eligible] == freeze["tasks"] == delivery["selected_tasks"]
    assert freeze["source_sha256"] == sha(ROOT / "tools/v354_deliver_frozen_combinations.py")
    assert freeze["verifier_sha256"] == sha(ROOT / "tools/v351_verify_combination_models.py")
    best = delivery["baseline"]
    anchor, csv_file = ROOT / best["csv"], ROOT / delivery["csv"]
    assert sha(anchor) == best["csv_sha256"] == "bef15e3a892afddd661173e3496b671078067e03ee2022b2abbfde02da45a906"
    assert sha(csv_file) == delivery["csv_sha256"]
    old_raw, raw = anchor.read_bytes(), csv_file.read_bytes()
    table = lambda b:list(csv.reader(io.StringIO(b.decode("utf-8-sig"), newline="")))
    old, new = table(old_raw), table(raw)
    assert old[0] == new[0] and len(old) == len(new) == 16380
    changed, predictions = {}, {}
    for record in eligible:
        task, metric = record["task"], record["metric"]
        saved = np.load(ROOT / "admet_model_outputs_v354/final" / (task + ".npz"))
        base = np.array([float(row[4]) for row in old[1:] if row[0] == task])
        np.testing.assert_array_equal(saved["base"], base)
        experts = saved["experts"]
        assert len(experts) == len(record["members"]) in (2, 3)
        if metric == "MAE":
            expected = .6 * base + .4 * np.mean(experts, axis=0)
        else:
            def ranked(p):
                # Independent scalar ranks, explicit tie averaging.
                order = np.argsort(p, kind="stable")
                ranks, lo = np.empty(len(p), dtype=float), 0
                while lo < len(p):
                    hi = lo + 1
                    while hi < len(p) and p[order[hi]] == p[order[lo]]:
                        hi += 1
                    ranks[order[lo:hi]] = ((lo + 1) + hi) / 2 / len(p)
                    lo = hi
                np.testing.assert_array_equal(ranks, rankdata(p) / len(p))
                return ranks
            expected = .6 * ranked(base) + .4 * np.mean([ranked(p) for p in experts], axis=0)
        actual = np.array([float(row[4]) for row in new[1:] if row[0] == task])
        np.testing.assert_array_equal(saved["p"], expected)
        np.testing.assert_array_equal(actual, expected)
        predictions[task] = len(actual)
    for before, after in zip(old[1:], new[1:]):
        assert before[:4] + before[5:] == after[:4] + after[5:]
        if before[0] not in predictions:
            assert before == after
        if before[4] != after[4]:
            changed[before[0]] = changed.get(before[0], 0) + 1
    assert changed == delivery["changed_rows"]
    spans, old_spans, restored, cursor = field_spans(raw), field_spans(old_raw), [], 0
    for i in range(1, len(old)):
        lo, hi = spans[i][4]
        before_lo, before_hi = old_spans[i][4]
        restored.extend((raw[cursor:lo], old_raw[before_lo:before_hi]))
        cursor = hi
    restored.append(raw[cursor:])
    assert b"".join(restored) == old_raw
    for check in delivery["checks"]:
        assert check["saved_model_reload"] and check["nonfit_target_label_isolation"] and check["exact_split_alignment"]
        if check["version"] == 233:
            assert check["all_integer_string_and_split_fields_exact"] and check["all_prediction_ranks_exact"]
            assert check["transform_float_rtol"] == 2e-13 and check["transform_float_atol"] == 2e-14
            assert check["prediction_atol"] == 1e-12 and check["prediction_max_abs_error"] <= 1e-12
        for name, digest in check["model_files"].items():
            assert sha(ROOT / name) == digest, name
    audit = json.loads((ROOT / "admet_model_outputs_v354/resume_audit.json").read_text(encoding="utf-8"))
    assert audit["original_freeze_sha256"] == sha(ROOT / "admet_model_outputs_v354/frozen_before_final.json")
    assert audit["runner_sha256"] == sha(ROOT / "tools/v359_resume_frozen_delivery.py")
    assert audit["adapter_sha256"] == sha(ROOT / "tools/v358_kernel_replay_adapter.py")
    assert audit["previous_log_sha256"] == sha(ROOT / "v354_delivery.log")
    assert audit["model_source_config_seed_weight_and_selection_unchanged"]
    result = dict(status="PASS", csv=delivery["csv"], csv_sha256=delivery["csv_sha256"],
        experimental_only=True, baseline="v346", official_best=51.235, official_score=None,
        independent_fusion_recomputed=True, independent_scalar_tie_ranks=True,
        frozen_policy_and_confirmation_verified=True, saved_model_hashes_verified=True,
        numerical_replay_adapter_and_original_failure_audited=True,
        changed_rows=changed, unchanged_tasks=22-len(predictions), all_nonprediction_bytes_exact=True,
        unchanged_task_predictions_exact=True, platform_submission=False,
        format_verification=verify(csv_file, ROOT / "admet_official/submission_template.csv"))
    result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
