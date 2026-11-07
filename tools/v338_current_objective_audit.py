"""Reconcile supplied v313/v321 scorecards and quantify arithmetic target gap."""
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from v184_score_objective_audit import normalized

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = "docs/platform_scorecard_v313_20261101.json"
    card = json.loads((ROOT / source).read_text(encoding="utf-8"))
    tasks = card["tasks"]
    assert len(tasks) == 22 and len({r["task"] for r in tasks}) == 22
    errors = [abs(normalized(r["metric"], r["raw"], r["reference_score"]) - r["score"]) for r in tasks]
    assert max(errors) < .00000051
    current = sum(r["score"] for r in tasks) / len(tasks)
    assert abs(current - card["task_score_mean"]) < 1e-12
    gap = 52. - current
    sensitivity = []
    for r in tasks:
        m, raw, ref = r["metric"], r["raw"], r["reference_score"]
        scenario = raw * .9 if m == "MAE" else min(1., raw + .1) if m == "Spearman" else min(1., raw + .01)
        gain = normalized(m, scenario, ref) - normalized(m, raw, ref)
        maximum = normalized(m, 0. if m == "MAE" else 1., ref) - normalized(m, raw, ref)
        sensitivity.append(dict(task=r["task"], metric=m, scenario_raw=scenario,
            scenario="MAE -10%" if m == "MAE" else "Spearman +0.1" if m == "Spearman" else "AUROC/AUPRC +0.01",
            scenario_total_gain=gain / 22, perfect_task_total_gain=maximum / 22))

    def mae_gain(fraction):
        return sum((normalized(r["metric"], r["raw"] * (1. - fraction), r["reference_score"]) -
                    normalized(r["metric"], r["raw"], r["reference_score"])) / 22
                   for r in tasks if r["metric"] == "MAE")

    lo, hi = 0., .99
    for _ in range(80):
        mid = (lo + hi) / 2.
        if mae_gain(mid) < gap:
            lo = mid
        else:
            hi = mid
    report = dict(status="PASS_ARITHMETIC_ONLY", at_hong_kong=datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat(),
        baseline="v313", official_user_reported=card["score"], exact_task_mean=current,
        gap_to52=gap, summed_task_point_gain_required=gap * 22,
        formula_max_rounding_residual=max(errors), all_five_MAE_minus10percent_total_gain=mae_gain(.1),
        all_five_MAE_uniform_relative_reduction_needed_if_all_other_tasks_identical=(lo + hi) / 2.,
        sensitivities=sensitivity, source=source, model_selection_changed=False,
        test_labels_used=False, limitations="Arithmetic scenarios, not model forecasts. Scoring identities empirically match supplied scorecards; not a newly fetched official specification. No hidden labels inferred, no blend tuning and no new submission produced.")
    path = ROOT / "docs/v338_current_objective_audit.json"
    assert not path.exists()
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "sensitivities"}))


if __name__ == "__main__":
    main()
