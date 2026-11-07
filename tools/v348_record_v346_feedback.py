"""Record the user-pasted v346 official feedback without rewriting prior evidence."""
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from v184_score_objective_audit import normalized

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    assert not path.exists(), f"Preserve existing record: {path}"
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    old = json.loads((ROOT / "docs/platform_scorecard_v313_20261101.json").read_text(encoding="utf-8"))
    delivery = json.loads((ROOT / "docs/v346_experimental_delivery.json").read_text(encoding="utf-8"))
    csv = ROOT / delivery["csv"]
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == delivery["csv_sha256"]
    tasks = copy.deepcopy(old["tasks"])
    changed = next(r for r in tasks if r["task"] == "solubility_aqsoldb")
    before = copy.deepcopy(changed)
    # The user's complete scorecard agrees with v313 for all other21 endpoints.
    changed.update(raw=0.7127214187464622, score=50.886239)
    assert len(tasks) == 22
    assert max(abs(normalized(r["metric"], r["raw"], r["reference_score"]) - r["score"]) for r in tasks) < .00000051
    mean = sum(r["score"] for r in tasks) / len(tasks)
    assert round(mean, 3) == 51.235
    delta = mean - old["task_score_mean"]
    assert abs(delta - (changed["score"] - before["score"]) / 22) < 1e-12
    at = datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()
    card = dict(version="v346", date="2026-11-01", submitted_at="2026/11/01 22:53",
        recorded_at_hong_kong=at, score=51.235, task_score_mean=mean, tasks=tasks,
        status="user_reported_official", source="User-pasted complete22-task v346 official scorecard in this conversation.",
        uploaded_byte_hash_confirmed=False, submission_sha256=delivery["csv_sha256"],
        baseline="v313", baseline_score=old["score"], delta_task_mean=delta,
        changed_tasks=["solubility_aqsoldb"], unchanged_tasks=21,
        solubility_relative_mae_reduction=(before["raw"] - changed["raw"]) / before["raw"],
        timezone="Asia/Hong_Kong", platform_displayed_certification="未认证",
        association="User explicitly identifies v346; all21 unchanged metrics match v313. Platform upload hash not independently available.",
        csv=delivery["csv"], original_local_strict_rejection_unchanged=True)
    source = "docs/platform_scorecard_v346_20261101.json"
    best = dict(version="v346", official_user_reported=51.235, exact_task_mean=mean,
        recorded_at_hong_kong=at, csv=delivery["csv"], csv_sha256=delivery["csv_sha256"],
        source=source, previous_best="v313", previous_best_official=old["score"],
        delta_task_mean=delta, platform_uploaded_byte_hash_confirmed=False,
        reason="User-reported official solubility improvement; other21 endpoints match. Historical strict-validation rejection is preserved.")
    save(ROOT / source, card)
    save(ROOT / "docs/current_best_after_v346_20261102.json", best)
    report = dict(status="PASS_USER_REPORTED_OFFICIAL_IMPROVEMENT", recorded_at_hong_kong=at,
        best=best, task_before=before, task_after=changed, unchanged_tasks=21,
        solubility_relative_mae_reduction=card["solubility_relative_mae_reduction"],
        official_total_delta_from_rounded_display=.030,
        exact_task_mean_delta=delta, gap_to52=52. - mean,
        platform_submission=False, hidden_labels_used=False, prior_artifacts_modified=False)
    save(ROOT / "docs/v348_feedback_review.json", report)
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
