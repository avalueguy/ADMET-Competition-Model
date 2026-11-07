"""Training-only descriptive limits of structure-only deterministic predictions.

An empirical within-identity MAE floor is not a generalization bound or a score
forecast. Other available assay context can distinguish rows with the same graph.
"""
from pathlib import Path
import json
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    d = dict(np.load(root / "official_rows.npz"))
    keys = dict(np.load(root / "stereo_keys.npz"))["isomeric"]
    tasks = json.loads((root / "data_manifest.json").read_text())["tasks"]
    records = []
    for name in ("caco2_wang", "ld50_zhu", "lipophilicity_astrazeneca", "ppbr_az", "solubility_aqsoldb"):
        rows = np.flatnonzero((d["tid"] == tasks.index(name)) & (d["split"] == 0))
        y = d["y"][rows]
        assert np.isfinite(y).all()
        _, inverse, counts = np.unique(keys[d["mi"][rows]], return_inverse=True, return_counts=True)
        errors = np.zeros(len(rows))
        conflicts = 0
        for k, count in enumerate(counts):
            ids = np.flatnonzero(inverse == k)
            errors[ids] = np.abs(y[ids] - np.median(y[ids]))
            conflicts += int(count > 1 and np.ptp(y[ids]) > 1e-8)
        duplicate = counts[inverse] > 1
        records.append(dict(task=name, train_rows=len(rows), identity_groups=len(counts),
                            duplicate_identity_groups=int((counts > 1).sum()), conflicting_groups=conflicts,
                            rows_in_duplicate_groups=int(duplicate.sum()),
                            empirical_structure_only_floor_all_train=float(errors.mean()),
                            empirical_structure_only_floor_duplicate_rows=float(errors[duplicate].mean()) if duplicate.any() else None,
                            note="Same stereo graph forced to same prediction; per-group training median is an oracle statistic only. Assay context may distinguish duplicate rows."))
    value = dict(training_labels_only=True, no_model_change=True, no_test_labels=True,
                 limitation="Empirical descriptive floor only, not a held-out noise estimate, not a bound for context-conditioned experts.", records=records)
    (root / "fit_noise_diagnostic.json").write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(value))


if __name__ == "__main__":
    main()
