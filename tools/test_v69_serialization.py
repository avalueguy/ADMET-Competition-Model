"""GPU regression: persist and replay a Spearman target with array metadata."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import admet_v67_data as b
import admet_v69_aggressive as model


def main():
    model.OUT = ROOT / "admet_v69_serialization_smoke"
    d = b.build()
    k = d["tasks"].index("half_life_obach")
    assert d["metrics"][k] == "Spearman"
    profile = model.P["families"][0]
    p = model.fit_tree(d, k, "fold0", profile, iterations=5)
    metadata = json.loads((model.OUT / d["tasks"][k] / "fold0" /
                           profile["name"] / str(model.P["development_seed"]) /
                           "metadata.json").read_text(encoding="utf-8"))
    assert isinstance(metadata["target"]["knots"], list)
    assert isinstance(metadata["target"]["z"], list)
    rows, _ = model.splits(d, k, "fold0")
    np.testing.assert_allclose(b.encode(d["y"][rows], metadata["target"]),
                               b.encode(d["y"][rows], b.target(d["y"][rows], "Spearman")))
    np.testing.assert_array_equal(p, model.fit_tree(d, k, "fold0", profile, iterations=5))
    b.emit("v69_spearman_metadata_regression", status="PASS", signature=model.signature())


if __name__ == "__main__":
    main()
