"""Record the user-pasted full scorecard; immutable local candidate association."""
import hashlib
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    old = json.loads((ROOT / "docs/platform_scorecard_v145_20261027.json").read_text(encoding="utf-8"))
    delivery = json.loads((ROOT / "docs/v313_experimental_delivery.json").read_text(encoding="utf-8"))
    scorecard = deepcopy(old)
    replacements = {
        "lipophilicity_astrazeneca": (.42089320332796315, 52.909067, .472895),
        "ppbr_az": (6.665484533215087, 52.376135, 7.330617),
    }
    references = [.871945,.927779,.722248,.283647,.55204,.642814,.44337,.790586,
                  .747531,.719172,.725249,.884121,.908609,.49139,.863358,.99428,
                  .603071,.472895,.938363,7.330617,.738443,.660995]
    improvements = []
    for row, reference in zip(scorecard["tasks"], references):
        task = row["task"]
        row.update(reference_score=reference, direction="lower" if row["metric"] == "MAE" else "higher", status="OK")
        if task in replacements:
            raw, score, reference2 = replacements[task]
            assert reference == reference2
            improvements.append(dict(task=task, old_raw=row["raw"], new_raw=raw,
                relative_mae_reduction=1. - raw / row["raw"], old_score=row["score"],
                new_score=score, task_point_gain=score-row["score"], total_point_gain=(score-row["score"])/22.))
            row.update(raw=raw, score=score)
        metric, raw = row["metric"], row["raw"]
        computed = (100*reference/(raw+reference) if metric == "MAE" else
                    100*(raw+1)/(raw+reference+2) if metric == "Spearman" else
                    100*raw/(raw+reference))
        assert abs(computed-row["score"]) <= .00000051
    csv = ROOT / delivery["csv"]
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == delivery["csv_sha256"]
    mean = sum(row["score"] for row in scorecard["tasks"])/22
    scorecard.update(version="v313", date="2026-11-01", submitted_at="2026-11-01 18:02",
        timezone="Asia/Hong_Kong", score=51.205, task_score_mean=mean,
        status="user_reported_official", platform_displayed_certification="未认证",
        source="User-pasted complete 22-task official scorecard in current conversation; no platform access.",
        baseline="v145", delta_task_mean=mean-old["task_score_mean"],
        summed_task_points_required_for_52=(52.-mean)*22.,
        submission_sha256=delivery["csv_sha256"], uploaded_byte_hash_confirmed=False,
        association="User feedback follows v313 delivery; only its two changed tasks differ from v145. Platform upload hash unavailable.",
        timestamp_note="Platform timestamp18:02 is retained verbatim; it precedes local delivery log18:15:46. Exact upload time/file identity not independently confirmed.",
        csv=delivery["csv"])
    output = ROOT / "docs/platform_scorecard_v313_20261101.json"
    assert not output.exists()
    output.write_text(json.dumps(scorecard, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    report = dict(status="PASS", official_best_user_reported=51.205, task_score_mean=mean,
        prior_best=51.166, exact_mean_gain=mean-old["task_score_mean"],
        target_score=52., remaining_mean_gap=52.-mean, improvements=improvements,
        unchanged_tasks=20, all22_score_formula_rounding_checks=True,
        platform_file_hash_confirmed=False, new_test_labels_inferred=False,
        feedback_driven_weight_search=False)
    (ROOT / "docs/v314_feedback_review.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
