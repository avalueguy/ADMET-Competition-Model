"""Fit-only adaptive histogram trees; ensemble-distribution medians for MAE."""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "4"
import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import v146_current_baseline as v
import v148_experiment_runner as runner

b, r83 = v.b, v.r83
OUT = ROOT / "admet_model_outputs_v299"
TASKS = ("caco2_wang", "ppbr_az", "lipophilicity_astrazeneca", "ld50_zhu", "solubility_aqsoldb")
CONFIGS = (dict(bins=24, trees=700, depth=5, fingerprint_columns=512),)
SEEDS = (2991101, 2991102)
METHOD = (
    "Official-only adaptive histogram conditional-distribution trees for five MAE tasks. "
    "24 fit-quantile target cells with duplicate/empty cells removed. CatBoost MultiClass "
    "proper log-loss learns categorical conditional probabilities using augmented chemistry, "
    "target-excluded other assays, and 512 fit-variance-selected count features. Feature "
    "winsorization at fit-only .005/.995 quantiles; native missingness. Average two fixed-seed "
    "probability distributions and invert their CDF using the empirical fit outcome quantile "
    "inside the selected cell. No query labels define bins or transformations. Unlike v115 "
    "ordinal clearance ranking or v181 squared-loss forest leaf neighborhoods, this fits a "
    "categorical conditional density and targets its ensemble median for raw MAE. No neural "
    "location-scale mixture, molecular pair differences, historical prediction training inputs, "
    "external training data, or changes to v145/v148 splits, weights or confirmation gates."
)


def sig():
    versions = "|".join(importlib.metadata.version(p) for p in ("numpy", "catboost"))
    return hashlib.sha256((b.sha(__file__) + v.sig() + b.sha(runner.__file__) + versions).encode()).hexdigest()


def key(c):
    assert c == CONFIGS[0]
    return "adaptive_histogram24"


def load():
    threadpool_limits(limits=4)
    return v.load()


def target_cells(y, bins):
    y = np.asarray(y, dtype=np.float64)
    cuts = np.unique(np.quantile(y, np.arange(1, bins) / bins))
    raw_labels = np.searchsorted(cuts, y, side="right")
    # Repeated labels and boundary atoms can leave some cells empty.
    occupied, labels = np.unique(raw_labels, return_inverse=True)
    values = np.concatenate([np.sort(y[labels == j], kind="stable") for j in range(len(occupied))])
    offsets = np.r_[0, np.cumsum(np.bincount(labels))]
    assert len(occupied) >= 2 and offsets[-1] == len(y)
    return dict(cuts=cuts, occupied=occupied, labels=labels, values=values, offsets=offsets)


def distribution_median(probabilities, values, offsets):
    p = np.asarray(probabilities, dtype=np.float64)
    assert p.ndim == 2 and p.shape[1] + 1 == len(offsets)
    assert np.isfinite(p).all() and np.all(p >= 0)
    p = p / p.sum(axis=1, keepdims=True)
    cdf = np.cumsum(p, axis=1)
    cell = np.minimum((cdf < .5).sum(axis=1), p.shape[1] - 1)
    before = np.where(cell > 0, cdf[np.arange(len(p)), np.maximum(cell - 1, 0)], 0.)
    mass = p[np.arange(len(p)), cell]
    assert np.all(mass > 0)
    fraction = np.clip((.5 - before) / mass, 0., 1.)
    result = np.empty(len(p), dtype=np.float64)
    for j in np.unique(cell):
        mask = cell == j
        result[mask] = np.quantile(values[offsets[j]:offsets[j + 1]], fraction[mask], method="linear")
    return result


def design(d, k, part, c):
    assert d["tasks"][k] in TASKS and d["metrics"][k] == "MAE"
    fit, query, x, qx, source = v.old.features(d, k, part, "augmented")
    fp = d["views"]["multiview"][:, :12288]
    variance = fp[d["mi"][fit]].astype(np.float64).var(axis=0)
    columns = np.sort(np.argsort(-variance, kind="stable")[:c["fingerprint_columns"]])
    raw = np.c_[x, fp[d["mi"][fit]][:, columns]].astype(np.float64)
    query_raw = np.c_[qx, fp[d["mi"][query]][:, columns]].astype(np.float64)
    lower = np.empty(raw.shape[1])
    upper = np.empty(raw.shape[1])
    for j, col in enumerate(raw.T):
        good = col[np.isfinite(col)]
        lower[j], upper[j] = np.quantile(good, [.005, .995]) if len(good) else (0., 0.)
    def apply(a):
        a = np.where(np.isfinite(a), np.clip(a, lower, upper), np.nan)
        # Keep historical extreme descriptor sentinels finite in float32.
        return np.clip(a, -1e30, 1e30).astype(np.float32)
    cells = target_cells(d["y"][fit], c["bins"])
    np.testing.assert_array_equal(fit, v.split(d, k, part)[0])
    assert not set(fit) & set(query)
    assert np.all(d["tid"][source] != k)
    assert np.all(d["split"][source] < (2 if part == "final" else 1))
    assert part == "final" or not set(d["groups"][fit]) & set(d["groups"][query])
    return dict(fit=fit, query=query, source=source, columns=columns, lower=lower, upper=upper,
                x=apply(raw), qx=apply(query_raw), **cells)


def estimator(c, seed, smoke=False):
    return CatBoostClassifier(loss_function="MultiClass", iterations=7 if smoke else c["trees"],
        depth=c["depth"], learning_rate=.035, l2_leaf_reg=20., random_seed=seed,
        thread_count=4, verbose=False, allow_writing_files=False,
        bootstrap_type="Bernoulli", subsample=.85, border_count=64,
        task_type="GPU", devices="0")


def infer(a, models):
    members = []
    for model in models:
        np.testing.assert_array_equal(model.classes_, np.arange(len(a["offsets"]) - 1))
        members.append(model.predict_proba(a["qx"], thread_count=4))
    probability = np.mean(members, axis=0)
    return distribution_median(probability, a["values"], a["offsets"]), probability


FIELDS = ("fit", "query", "source", "columns", "lower", "upper", "cuts", "occupied", "labels", "values", "offsets")


def replay(d, k, part, c):
    folder = OUT / "models" / d["tasks"][k] / part / key(c)
    a = design(d, k, part, c)
    z = np.load(folder / "predictions.npz")
    assert str(z["signature"]) == sig()
    for name in FIELDS:
        np.testing.assert_array_equal(a[name], z[name])
    models = [CatBoostClassifier().load_model(str(folder / f"model_{s}.cbm")) for s in SEEDS]
    p, probabilities = infer(a, models)
    if d["tasks"][k] == "ppbr_az":
        p = p.clip(0, 100)
    np.testing.assert_array_equal(p, z["p"])
    np.testing.assert_array_equal(probabilities, z["probabilities"])
    return p


def predict(d, k, part, c):
    if d["tasks"][k] not in TASKS:
        return v.baseline(d, k, part)
    folder = OUT / "models" / d["tasks"][k] / part / key(c)
    path = folder / "predictions.npz"
    if path.exists():
        z = np.load(path)
        assert str(z["signature"]) == sig()
        np.testing.assert_array_equal(z["query"], v.split(d, k, part)[1])
        return z["p"].copy()
    assert not folder.exists(), "Inspect partial artifacts before retry."
    started = time.monotonic()
    a = design(d, k, part, c)
    folder.mkdir(parents=True)
    models, hashes = [], {}
    for seed in SEEDS:
        model = estimator(c, seed).fit(a["x"], a["labels"])
        mp = folder / f"model_{seed}.cbm"
        model.save_model(str(mp))
        loaded = CatBoostClassifier().load_model(str(mp))
        np.testing.assert_array_equal(model.predict_proba(a["qx"][:17]), loaded.predict_proba(a["qx"][:17]))
        hashes[mp.name] = b.sha(mp)
        models.append(loaded)
    p, probabilities = infer(a, models)
    if d["tasks"][k] == "ppbr_az":
        p = p.clip(0, 100)
    assert len(p) == len(a["query"]) and np.isfinite(p).all()
    np.savez_compressed(path, signature=sig(), p=p, probabilities=probabilities,
                        **{name: a[name] for name in FIELDS})
    b.save(folder / "metadata.json", dict(signature=sig(), model_hashes=hashes,
        fit_rows=len(a["fit"]), query_rows=len(p), classes=len(a["offsets"]) - 1,
        features=a["x"].shape[1], seconds=time.monotonic() - started))
    print(json.dumps(dict(stage="histogram_train", task=d["tasks"][k], part=part,
                         seconds=round(time.monotonic() - started))), flush=True)
    return p


def smoke(d):
    # Check inversion, duplicate/empty-cell handling and label isolation before scoring.
    p = distribution_median(np.array([[.25, .75], [.75, .25], [.5, .5]]),
                            np.array([0., 2., 10., 14.]), np.array([0, 2, 4]))
    np.testing.assert_allclose(p, [10. + 4. / 3., 4. / 3., 2.], rtol=0, atol=1e-12)
    cells = target_cells(np.r_[np.zeros(20), np.ones(20), np.full(20, 2.)], 24)
    assert len(cells["offsets"]) == 4 and np.all(np.diff(cells["offsets"]) == 20)
    k = d["tasks"].index("caco2_wang")
    a = design(d, k, "fold0", CONFIGS[0])
    saved = d["y"].copy()
    try:
        d["y"][(d["tid"] == k) & ~np.isin(np.arange(len(saved)), a["fit"])] = 999999.
        d.pop("_v96_assays", None)
        alternative = design(d, k, "fold0", CONFIGS[0])
        for name in a:
            np.testing.assert_array_equal(a[name], alternative[name])
    finally:
        d["y"][:] = saved
        d.pop("_v96_assays", None)
    model = estimator(CONFIGS[0], SEEDS[0], True).fit(a["x"], a["labels"])
    path = OUT / "smoke.cbm"
    model.save_model(str(path))
    other = CatBoostClassifier().load_model(str(path))
    original, _ = infer(a, [model])
    repeated, _ = infer(a, [other])
    np.testing.assert_array_equal(original, repeated)
    reverse, _ = infer(dict(a, qx=a["qx"][::-1]), [other])
    np.testing.assert_array_equal(original, reverse[::-1])
    b.save(OUT / "smoke.json", dict(status="PASS", signature=sig(),
        median_reference=True, empty_cell_reference=True, nonfit_target_label_isolation=True,
        independent_checkpoint_reload=True, reversed_query_invariant=True,
        query_scores_computed=False, final_predictions_computed=False))
    print("SMOKE_PASS", flush=True)


def finish():
    dev = json.loads((OUT / "development.json").read_text())
    conf = json.loads((OUT / "confirmation.json").read_text())
    assert dev["status"] == conf["status"] == "COMPLETE"
    best = [max((r for r in dev["records"] if r["task"] == task), key=lambda r: r["objective"]) for task in TASKS]
    b.save(ROOT / "docs/v299_final_status.json", dict(
        status="REQUIRES_INDEPENDENT_DELIVERY_VERIFICATION" if conf["accepted"] else "COMPLETE_NO_CONFIRMED_NEW_MODEL",
        at_utc=datetime.now(timezone.utc).isoformat(), signature=sig(), baseline="v145",
        official_best=51.166, target_score=52., development_best=best,
        accepted_tasks=conf["accepted"], confirmation=conf["records"],
        csv_created=False, platform_submission=False, official_gain_established=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "run", "report"))
    stage = parser.parse_args().stage
    OUT.mkdir(exist_ok=True)
    if stage == "report":
        finish()
        return
    d = load()
    if stage == "smoke":
        smoke(d)
    else:
        assert json.loads((OUT / "smoke.json").read_text())["signature"] == sig()
        assert not (OUT / "development.json").exists(), "Never silently repeat a scored experiment."
        b.save(OUT / "frozen_before_scores.json", dict(signature=sig(),
            source_sha256=b.sha(__file__), baseline_signature=v.sig(), runner_sha256=b.sha(runner.__file__),
            at_utc=datetime.now(timezone.utc).isoformat(), tasks=TASKS, configs=CONFIGS, seeds=SEEDS,
            packages={p: importlib.metadata.version(p) for p in ("numpy", "catboost")},
            no_query_scores_read_for_configuration=True))
        runner.run(sys.modules[__name__], d)
        finish()


if __name__ == "__main__":
    main()
