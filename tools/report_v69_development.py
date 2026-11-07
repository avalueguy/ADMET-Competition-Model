"""Recompute descriptive development metrics for the frozen aggressive recipe."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import admet_v67_data as b
import admet_v69_aggressive as model


def main():
    d = b.build()
    path = model.OUT / "selection.json"
    frozen = b.sha(path)
    selection = json.loads(path.read_text(encoding="utf-8"))
    assert selection["signature"] == model.signature()
    tasks = []
    for k, record in enumerate(selection["tasks"]):
        partitions = []
        for part in model.P["splits"]:
            query, old, sources = model.source_predictions(d, k, part)
            new = model.apply(record["metric"], old, sources, record["recipe"])
            partitions.append(dict(partition=part, rows=len(query),
                baseline=b.metric_score(record["metric"], d["y"][query], old),
                candidate=b.metric_score(record["metric"], d["y"][query], new),
                gain=b.gain(record["metric"], d["y"][query], old, new)))
        np.testing.assert_allclose([p["gain"] for p in partitions], record["development"]["gains"], atol=1e-12)
        tasks.append(dict(task=record["task"], metric=record["metric"],
            changed=bool(record["recipe"]), recipe=record["recipe"], partitions=partitions))
    assert b.sha(path) == frozen
    valid = [t["partitions"][-1]["gain"] for t in tasks]
    result = dict(status="DEVELOPMENT_DIAGNOSTIC_ONLY", signature=model.signature(),
        selection_sha256=frozen, changed_tasks=sum(t["changed"] for t in tasks),
        valid_improved=sum(g > 1e-12 for g in valid), valid_worsened=sum(g < -1e-12 for g in valid),
        valid_unchanged=sum(abs(g) <= 1e-12 for g in valid), tasks=tasks,
        official_valid_used_for_tuning=True, independent_generalization_estimate=False,
        official_score=None, official_score_gain=None)
    b.save(model.OUT / "development_metrics.json", result)
    b.emit("v69_development_report", changed_tasks=result["changed_tasks"],
        improved=result["valid_improved"], worsened=result["valid_worsened"],
        unchanged=result["valid_unchanged"], independent_generalization_estimate=False)


if __name__ == "__main__":
    main()
