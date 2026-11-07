"""Freeze the experimental export rule before all combination confirmations finish."""
import json
from pathlib import Path

import v349_diverse_model_combinations as experiment

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "docs/v352_combination_delivery_policy.json"
    assert not path.exists()
    assert not (experiment.OUT / "confirmation.json").exists()
    frozen = json.loads((experiment.OUT / "frozen_selection.json").read_text())
    progress = experiment.OUT / "confirmation.progress.json"
    observed = json.loads(progress.read_text())["records"] if progress.exists() else []
    policy = dict(status="FROZEN_BEFORE_REMAINING_CONFIRMATIONS", at_hong_kong=experiment.now(),
        source_sha256=experiment.sha(__file__), model_search_source_sha256=experiment.sha(experiment.__file__),
        selection_sha256=experiment.sha(experiment.OUT / "frozen_selection.json"),
        frozen_selected_tasks=[r["task"] for r in frozen["selected"]],
        already_observed_confirmation_tasks=[r["task"] for r in observed],
        strict_rule="Unchanged: four positive gains, mean practical threshold, both paired scaffold90 lower bounds positive.",
        experimental_rule="Exactly the already-frozen per-task winner. Four positive gains, mean above original practical threshold, both confirmation partitions have more than one scaffold and nonnull intervals. All such tasks included; no weight/config/seed fallback. Both-CI strict rejection remains recorded.",
        no_delivery_if_zero_eligible_tasks=True, require_independent_member_replay=True,
        require_nonfit_target_label_isolation=True, require_exact_baseline_preservation=True,
        baseline="v346", official_best=51.235, official_gain_established=False,
        user_authorized_experimental_csv=True, manual_submission_only=True,
        rationale="User asks for broad model combinations and any real gain. A finite fixed export rule prevents choosing a different failed combination after confirmation. Previously observed AMES is excluded because valid contains only one scaffold; its strict rejection is unchanged.",
        limitations="No claim of statistical certainty or official improvement. Many historical models/public labels repeatedly reused, selection multiplicity and approximate inherited OOF remain. File may regress.")
    experiment.save(path, policy)
    print(json.dumps(policy, ensure_ascii=True))


if __name__ == "__main__":
    main()
