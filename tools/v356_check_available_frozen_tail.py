"""Score already-frozen tail combinations when all four cached parts are available."""
import json

import v349_diverse_model_combinations as experiment


def main():
    frozen = json.loads((experiment.OUT / "frozen_selection.json").read_text())
    assert frozen["source_sha256"] == experiment.sha(experiment.__file__)
    mods, d, records = experiment.modules(), experiment.v.load(), []
    for choice in frozen["selected"][8:]:
        task, metric = choice["task"], choice["metric"]
        k = d["tasks"].index(task)
        if not all((mods[m["version"]].OUT / "models" / task / part / m["key"] / "predictions.npz").exists()
                   for part in experiment.PARTS for m in choice["members"]):
            continue
        parts = []
        for part in experiment.PARTS:
            experts = [experiment.cached(d, k, part, m, mods)[0] for m in choice["members"]]
            base = experiment.current(d, k, part)
            pred = experiment.fused(metric, base, experts)
            query = experiment.v.split(d, k, part)[1]
            parts.append(dict(part=part, gain=experiment.v.r83.gain(metric, d["y"][query], base, pred)))
        records.append(dict(task=task, parts=parts, frozen_choice_unchanged=True))
    report = dict(status="PARTIAL_CACHED_POINT_RESULTS_ONLY", at_hong_kong=experiment.now(), records=records,
        used_for_reselection=False, weights_changed=False, official_score_estimate=None)
    experiment.save(experiment.ROOT / "docs/v356_available_point_results.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
