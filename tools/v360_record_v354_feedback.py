"""Record all22 user-supplied endpoint results; do not rewrite past best records."""
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from v184_score_objective_audit import normalized

ROOT = Path(__file__).resolve().parents[1]
RAW = (
    .8769096712291213, .9289790494058787, .7948121050881277, .27592992051587734,
    .565208996561239, .6756845436053647, .5862085536546348, .8349408577894758,
    .7583244699983189, .7439447168745902, .7362115732368897, .8991915344045904,
    .935, .6509691357439866, .861119293078056, 1., .55547536869963,
    .42089320332796315, .9504798720341242, 6.665484533215087, .7127214187464622,
    .7145137427047057,
)
SCORES = (
    50.141941, 50.032316, 52.391603, 50.689546, 50.211228, 50.495262,
    56.936749, 51.364323, 50.358383, 50.846574, 50.375055, 50.422543,
    50.715743, 52.539161, 49.93509, 50.14341, 52.05411, 52.909067,
    50.320749, 52.376135, 50.886239, 50.792751,
)


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def save(name, value):
    path = ROOT / name
    assert not path.exists(), f"Preserve existing artifact: {path}"
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    baseline = read("docs/platform_scorecard_v346_20261101.json")
    best = read("docs/current_best_after_v346_20261102.json")
    delivery = read("docs/v354_delivery.json")
    csv_path = ROOT / delivery["csv"]
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == delivery["csv_sha256"]
    tasks = copy.deepcopy(baseline["tasks"])
    assert len(tasks) == len(RAW) == len(SCORES) == 22
    changes = []
    for task, raw, score, old in zip(tasks, RAW, SCORES, baseline["tasks"]):
        task.update(raw=raw, score=score)
        assert abs(normalized(task["metric"], raw, task["reference_score"]) - score) < .00000051
        if task != old:
            changes.append(dict(task=task["task"], metric=task["metric"],
                raw_before=old["raw"], raw_after=raw, raw_delta=raw-old["raw"],
                score_before=old["score"], score_after=score,
                task_score_delta=score-old["score"], total_score_delta=(score-old["score"])/22))
    assert {r["task"] for r in changes} == {"herg", "cyp3a4_substrate_carbonmangels"}
    mean = sum(SCORES) / 22
    assert round(mean, 3) == 51.224
    delta = mean - baseline["task_score_mean"]
    assert abs(sum(r["total_score_delta"] for r in changes) - delta) < 1e-12
    at = datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()
    source = "docs/platform_scorecard_v354_20261102.json"
    card = dict(version="v354", submitted_at="2026/11/02 15:56", date="2026-11-02",
        recorded_at_hong_kong=at, timezone="Asia/Hong_Kong", score=51.224,
        task_score_mean=mean, tasks=tasks, status="user_reported_official",
        source="User-pasted complete22-task scorecard after v354 experimental CSV delivery.",
        csv=delivery["csv"], submission_sha256=delivery["csv_sha256"],
        uploaded_byte_hash_confirmed=False, platform_displayed_certification="未认证",
        association="Inferred v354 from conversation and exact20 unchanged tasks. Platform upload hash is unavailable.",
        baseline="v346", baseline_score=51.235, changed_tasks=[r["task"] for r in changes],
        unchanged_tasks=20, delta_task_mean=delta, original_local_strict_rejections_preserved=True)
    save(source, card)
    report = dict(status="CONFIRMED_USER_REPORTED_V354_REGRESSION", source=source,
        recorded_at_hong_kong=at, official_best_unchanged=best, v354_score=51.224,
        v354_exact_task_mean=mean, delta_vs_v346=delta, changes=changes,
        next_action="Whole-endpoint reuse only: v346 for21tasks, exact v354 hERG for1task. No retraining, weight sweep or per-sample editing.",
        decision_uses_official_feedback=True, new_candidate_requires_official_confirmation=True,
        limitations="Endpoint selection based on reused public leaderboard feedback can overfit that evaluation. No private-test gain claim.",
        platform_submission=False, hidden_labels_used=False, prior_artifacts_modified=False)
    save("docs/v360_feedback_review.json", report)
    save("docs/current_status_after_v354_20261102.json", dict(
        confirmed_best=best, latest_official=card, next_candidate="v361", candidate_official_score=None,
        reason="v354 hERG improves, CYP3A4 substrate declines; preserve v346 as confirmed best until recombination is submitted."))
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
