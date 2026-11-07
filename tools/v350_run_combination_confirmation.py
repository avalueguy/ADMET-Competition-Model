"""Load required historical feature stores and execute the frozen v349 confirmation."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import v349_diverse_model_combinations as experiment


def main():
    frozen = json.loads((experiment.OUT / "frozen_selection.json").read_text())
    assert frozen["source_sha256"] == experiment.sha(experiment.__file__)
    assert not (experiment.OUT / "confirmation.json").exists()
    mods = experiment.modules()
    required = {m["version"] for r in frozen["selected"] for m in r["members"]}
    d = experiment.v.load()
    for version in (168, 221, 250):
        if version not in required:
            continue
        extra = mods[version].load()
        for key in ("tasks", "metrics", "y", "tid", "split", "mi", "groups", "canonical"):
            np.testing.assert_array_equal(d[key], extra[key])
        for key, value in extra.items():
            if key not in d:
                d[key] = value
        print(json.dumps(dict(stage="feature_store_ready", version=version)), flush=True)
    experiment.sam.original.initialize()
    if 178 in required:
        mods[178].initialize()
    # Keep signatures and model sources immutable; only reuse one fully loaded dataset.
    experiment.v.load = lambda:d
    experiment.confirm()


if __name__ == "__main__":
    main()
