"""Record user-reported v321 regression and independently audit changed CSV bytes."""
import csv
import hashlib
import io
import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    old = json.loads((ROOT / "docs/platform_scorecard_v313_20261101.json").read_text(encoding="utf-8"))
    delivery = json.loads((ROOT / "docs/v321_experimental_delivery.json").read_text(encoding="utf-8"))
    previous = json.loads((ROOT / "docs/v313_experimental_delivery.json").read_text(encoding="utf-8"))
    task = "cyp3a4_substrate_carbonmangels"
    raw, points = .7414104882459313, 50.550963
    before = next(row for row in old["tasks"] if row["task"] == task)
    card = deepcopy(old)
    for row in card["tasks"]:
        if row["task"] == task:
            row.update(raw=raw, score=points)
        x, ref = row["raw"], row["reference_score"]
        computed = (100*ref/(x+ref) if row["metric"] == "MAE" else
                    100*(x+1)/(x+ref+2) if row["metric"] == "Spearman" else 100*x/(x+ref))
        assert abs(computed-row["score"]) <= .00000051
    base_file, new_file = ROOT / previous["csv"], ROOT / delivery["csv"]
    base_bytes, new_bytes = base_file.read_bytes(), new_file.read_bytes()
    assert hashlib.sha256(base_bytes).hexdigest() == previous["csv_sha256"] == delivery["baseline_sha256"]
    assert hashlib.sha256(new_bytes).hexdigest() == delivery["csv_sha256"]
    table = lambda value:list(csv.reader(io.StringIO(value.decode("utf-8-sig"), newline="")))
    base, new = table(base_bytes), table(new_bytes)
    assert len(base) == len(new) == 16380 and base[0] == new[0]
    changed = {}
    for left, right in zip(base[1:], new[1:]):
        assert left[:4]+left[5:] == right[:4]+right[5:]
        if left[4] != right[4]:
            changed[left[0]] = changed.get(left[0], 0) + 1
    assert changed == {task:135}
    mean = sum(r["score"] for r in card["tasks"])/22
    delta = mean-old["task_score_mean"]
    assert abs(delta-(points-before["score"])/22) < 1e-12
    card.update(version="v321", date="2026-11-01", submitted_at="2026-11-01 18:35",
        timezone="Asia/Hong_Kong", score=51.198, task_score_mean=mean,
        status="user_reported_official", platform_displayed_certification="未认证",
        source="User-pasted complete 22-task scorecard explicitly identified as v321 in this conversation.",
        baseline="v313", delta_task_mean=delta, summed_task_points_required_for_52=(52.-mean)*22., csv=delivery["csv"],
        submission_sha256=delivery["csv_sha256"], uploaded_byte_hash_confirmed=False,
        association="User explicitly identifies v321; exactly its one changed task differs. Platform upload hash unavailable.",
        timestamp_note="Platform timestamp18:35 retained verbatim; it precedes local delivery log18:48:56. Exact upload clock/file identity not independently authenticated.",
        current_best=False, route_outcome="OFFICIAL_REGRESSION_DO_NOT_PROMOTE")
    report = dict(status="PASS", version="v321", official_user_reported=51.198,
        prior_best_user_reported=51.205, exact_mean=mean, exact_mean_delta=delta,
        changed_task=task, old_raw=before["raw"], new_raw=raw, raw_delta=raw-before["raw"],
        old_task_score=before["score"], new_task_score=points,
        task_score_delta=points-before["score"], unchanged_official_tasks=21,
        local_csv_changed_rows=changed, local_byte_identity_verified=True,
        format_or_task_mapping_error_found=False, strict_confirmation_was_rejected=True,
        interpretation="Four local point gains did not transfer to hidden official test. Exact statistical cause is not established by this scorecard. No test labels inferred.",
        current_best_version="v313", current_best_score=51.205,
        action="Do not promote or stack v321. Preserve v313 baseline; no feedback-driven blend sweep.")
    best = dict(version="v313", official_user_reported=51.205,
        exact_task_mean=old["task_score_mean"], csv=previous["csv"], csv_sha256=previous["csv_sha256"],
        source="docs/platform_scorecard_v313_20261101.json", rejected_later_version="v321",
        reason="v321 user-reported score51.198 is lower. Historical delivery records preserved.",
        platform_uploaded_byte_hash_confirmed=False)
    for name, value in (("platform_scorecard_v321_20261101.json", card),
                        ("v328_feedback_review.json", report),
                        ("current_best_after_v321_20261101.json", best)):
        path = ROOT / "docs" / name
        assert not path.exists()
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
