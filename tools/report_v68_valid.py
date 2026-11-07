"""Report official-valid kernel changes after the final selection is frozen."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import admet_v67_data as b
import admet_v68_kernel as kernel


def main():
    d = b.build()
    release = json.loads((kernel.OUT / "release.json").read_text(encoding="utf-8"))
    replay = json.loads((kernel.OUT / "model_replay.json").read_text(encoding="utf-8"))
    assert replay["status"] == "PASS" and replay["sha256"] == release["sha256"]
    frozen = b.sha(kernel.OUT / "selection.json")
    choices = json.loads((kernel.OUT / "selection.json").read_text())["tasks"]
    records67 = json.loads((b.OUT / "fresh.json").read_text())["tasks"]
    query = np.flatnonzero(d["split"] == 1)
    fit = np.flatnonzero((d["split"] == 0) & ~np.isin(d["rg"], np.unique(d["rg"][query])))
    records = []
    for k, choice in enumerate(choices):
        record67 = records67[k]
        record67["selection"]["weight"] = record67["final_weight"]
        q = query[d["tid"][query] == k]
        old = kernel.neural(d, q, "valid", record67)
        new = old.copy()
        if choice["pass_gate"]:
            chosen = choice["selection"]
            rr = fit[d["tid"][fit] == k]
            raw = kernel.fit(d, rr, q, choice["task"], "valid_report")[chosen["variant"]]
            new = b.blend(choice["metric"], old, raw, chosen["weight"])
        metric = choice["metric"]
        records.append(dict(task=choice["task"], metric=metric, rows=len(q),
                            changed=choice["pass_gate"], v67=b.metric_score(metric, d["y"][q], old),
                            v68=b.metric_score(metric, d["y"][q], new),
                            gain=b.gain(metric, d["y"][q], old, new)))
    assert frozen == b.sha(kernel.OUT / "selection.json")
    assert b.sha(ROOT / release["file"]) == release["sha256"]
    b.save(kernel.OUT / "valid_metrics.json", dict(status="REPORT_ONLY", tasks=records,
           used_for_selection_or_veto=False, global_scaffold_overlap=0, canonical_overlap=0,
           selection_sha256=frozen, release_sha256=release["sha256"]))
    b.emit("v68_valid_report", tasks=records)


if __name__ == "__main__":
    main()
