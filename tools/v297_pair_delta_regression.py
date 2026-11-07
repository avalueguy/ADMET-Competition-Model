"""Scaffold-excluded molecular difference learning for the five MAE endpoints.

Only official fit labels train pair differences. Query prediction is a weighted
median of fit anchors plus antisymmetric learned molecular differences. This
is a new expert blended with immutable v145 controls by the unchanged runner.
"""
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
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import v146_current_baseline as v
import v148_experiment_runner as runner

b, r83 = v.b, v.r83
OUT = ROOT / "admet_model_outputs_v297"
TASKS = ("caco2_wang", "ppbr_az", "lipophilicity_astrazeneca", "ld50_zhu", "solubility_aqsoldb")
SEEDS = (2971101, 2971102)
CONFIGS = (dict(neighbors=8, fingerprint_columns=256, trees=700, depth=5),)
METHOD = (
    "Official-only pair-difference regression for all five MAE endpoints. "
    "Eight nearest fit molecular anchors by count Tanimoto; fit pairs exclude "
    "same canonical identity and scaffold. Original physchem and target-excluded "
    "other-assay inputs plus 256 fit-variance-selected count columns. Fit-only "
    "median/IQR transforms; directed difference and midpoint features, symmetric "
    "pair targets, two fixed CatBoost MAE seeds. Antisymmetric pair predictions "
    "plus observed fit anchor labels, similarity^4 weighted median aggregation. "
    "No query target labels, historical prediction fitting offsets, external "
    "data, or previous rejected experts. Single configuration; original v148 "
    "weights, splits, selection and scaffold confirmation unchanged."
)


def sig():
    packages = "|".join(importlib.metadata.version(p) for p in ("numpy", "scipy", "catboost"))
    return hashlib.sha256((b.sha(__file__) + v.sig() + b.sha(runner.__file__) + packages).encode()).hexdigest()


def key(c):
    assert c == CONFIGS[0]
    return "scaffold_pair_delta8"


def load():
    threadpool_limits(limits=4)
    return v.load()


def neighbors(d, fit, rows, train, take=8):
    """Structure-only neighbor selection, stable global-row tie ordering."""
    raw = d["views"]["multiview"][:, :4096]
    ref = raw[d["mi"][fit]].astype(np.float32)
    refnorm = np.einsum("ij,ij->i", ref, ref)
    indices, weights = [], []
    for lo in range(0, len(rows), 128):
        rr = rows[lo:lo + 128]
        x = raw[d["mi"][rr]].astype(np.float32)
        dot = x @ ref.T
        sim = np.clip(dot / np.maximum(np.einsum("ij,ij->i", x, x)[:, None] + refnorm[None] - dot, 1e-10), 0, 1)
        if train:
            excluded = ((d["groups"][rr, None] == d["groups"][fit][None]) |
                        (d["canonical"][rr, None] == d["canonical"][fit][None]))
            sim[excluded] = -1
        ix = np.argsort(-sim, axis=1, kind="stable")[:, :take]
        values = np.take_along_axis(sim, ix, axis=1)
        assert np.all(values >= 0), "Not enough scaffold-disjoint anchors"
        indices.append(ix)
        weights.append(np.maximum(values, 1e-6) ** 4)
    return np.concatenate(indices), np.concatenate(weights)


def transform(x, qx):
    # Historical conformer columns can contain extreme finite sentinels. Use
    # float64 while finding scale/variation; transformed inputs are clipped.
    x, qx = x.astype(np.float64), qx.astype(np.float64)
    median = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0. for col in x.T])
    tx = np.where(np.isfinite(x), x, median)
    scale = np.subtract(*np.quantile(tx, [.75, .25], axis=0)).clip(.1)
    keep = np.flatnonzero(np.std(tx, axis=0) > .001)
    # Missingness is retained explicitly, including query-only missing values.
    def apply(raw):
        values = np.clip((np.where(np.isfinite(raw), raw, median) - median) / scale, -8., 8.)
        return np.c_[values[:, keep], np.isfinite(raw[:, keep])].astype(np.float32)
    return apply(x), apply(qx), dict(median=median, scale=scale, keep=keep)


def pair_features(left, right):
    return np.c_[left - right, (left + right) * .5].astype(np.float32)


def design(d, k, part, c=None):
    c = c or CONFIGS[0]
    assert d["tasks"][k] in TASKS and d["metrics"][k] == "MAE"
    fit, query, x, qx, source = v.old.features(d, k, part, "augmented")
    np.testing.assert_array_equal(fit, v.split(d, k, part)[0])
    fp = d["views"]["multiview"][:, :12288]
    variance = fp[d["mi"][fit]].var(axis=0)
    columns = np.sort(np.argsort(-variance, kind="stable")[:c["fingerprint_columns"]])
    x = np.c_[x, fp[d["mi"][fit]][:, columns]]
    qx = np.c_[qx, fp[d["mi"][query]][:, columns]]
    tx, tq, state = transform(x, qx)
    train_ix, train_w = neighbors(d, fit, fit, True, c["neighbors"])
    query_ix, query_w = neighbors(d, fit, query, False, c["neighbors"])
    assert not set(fit) & set(query) and np.all(d["tid"][source] != k)
    assert np.all(d["split"][source] < (2 if part == "final" else 1))
    assert part == "final" or not set(d["groups"][fit]) & set(d["groups"][query])
    y = d["y"][fit]
    target_scale = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05)
    left = np.repeat(np.arange(len(fit)), c["neighbors"])
    right = train_ix.ravel()
    assert np.all(d["groups"][fit[left]] != d["groups"][fit[right]])
    delta = (y[left] - y[right]) / target_scale
    features = pair_features(tx[left], tx[right])
    reverse = pair_features(tx[right], tx[left])
    # Each molecule contributes equal mass; neighbors share it by similarity.
    w = (train_w / train_w.sum(1, keepdims=True)).ravel()
    weights = np.r_[w, w] * c["neighbors"]
    return dict(fit=fit, query=query, source=source, columns=columns,
                x=tx, qx=tq, train_ix=train_ix, query_ix=query_ix,
                query_w=query_w, pair_x=np.r_[features, reverse],
                target=np.r_[delta, -delta], weights=weights,
                target_scale=target_scale, **state)


def weighted_median(values, weights):
    order = np.argsort(values, axis=1, kind="stable")
    ys = np.take_along_axis(values, order, axis=1)
    ws = np.take_along_axis(weights, order, axis=1)
    pos = (np.cumsum(ws, axis=1) < ws.sum(1, keepdims=True) * .5).sum(1)
    return ys[np.arange(len(ys)), np.minimum(pos, ys.shape[1] - 1)]


def infer(d, a, models):
    parts = []
    n = a["query_ix"].shape[1]
    for lo in range(0, len(a["query"]), 128):
        ids = a["query_ix"][lo:lo + 128]
        left = np.repeat(a["qx"][lo:lo + len(ids)], n, axis=0)
        right = a["x"][ids.ravel()]
        forward, reverse = pair_features(left, right), pair_features(right, left)
        delta = np.mean([.5 * (net.predict(forward, thread_count=4) -
                               net.predict(reverse, thread_count=4)) for net in models], axis=0)
        anchored = d["y"][a["fit"]][ids] + delta.reshape(ids.shape) * a["target_scale"]
        parts.append(weighted_median(anchored, a["query_w"][lo:lo + len(ids)]))
    return np.concatenate(parts)


def estimator(c, seed, smoke=False):
    return CatBoostRegressor(loss_function="MAE", iterations=9 if smoke else c["trees"],
        depth=c["depth"], learning_rate=.035, l2_leaf_reg=20., random_seed=seed,
        bootstrap_type="Bernoulli", subsample=.85, border_count=64,
        thread_count=8, verbose=False, allow_writing_files=False,
        task_type="GPU", devices="0")


def replay(d, k, part, c):
    folder = OUT / "models" / d["tasks"][k] / part / key(c)
    a = design(d, k, part, c)
    saved = np.load(folder / "predictions.npz")
    assert str(saved["signature"]) == sig()
    for field in ("fit", "query", "source", "columns", "train_ix", "query_ix", "query_w", "median", "scale", "keep"):
        np.testing.assert_array_equal(a[field], saved[field])
    nets = [CatBoostRegressor().load_model(str(folder / f"model_{seed}.cbm")) for seed in SEEDS]
    p = infer(d, a, nets)
    if d["tasks"][k] == "ppbr_az":
        p = p.clip(0, 100)
    np.testing.assert_array_equal(p, saved["p"])
    return p


def predict(d, k, part, c):
    if d["tasks"][k] not in TASKS:
        return v.baseline(d, k, part)
    folder = OUT / "models" / d["tasks"][k] / part / key(c)
    file = folder / "predictions.npz"
    if file.exists():
        saved = np.load(file)
        assert str(saved["signature"]) == sig()
        np.testing.assert_array_equal(saved["query"], v.split(d, k, part)[1])
        return saved["p"].copy()
    assert not folder.exists(), "Inspect partial training artifact before retry"
    start = time.monotonic()
    a = design(d, k, part, c)
    folder.mkdir(parents=True)
    nets, hashes = [], {}
    for seed in SEEDS:
        net = estimator(c, seed)
        net.fit(a["pair_x"], a["target"], sample_weight=a["weights"])
        path = folder / f"model_{seed}.cbm"
        net.save_model(str(path))
        loaded = CatBoostRegressor().load_model(str(path))
        np.testing.assert_array_equal(net.predict(a["pair_x"][:31], thread_count=1),
                                      loaded.predict(a["pair_x"][:31], thread_count=1))
        hashes[path.name] = b.sha(path)
        nets.append(loaded)
    p = infer(d, a, nets)
    if d["tasks"][k] == "ppbr_az":
        p = p.clip(0, 100)
    assert np.isfinite(p).all()
    fields = {name: a[name] for name in ("fit", "query", "source", "columns", "train_ix", "query_ix", "query_w", "median", "scale", "keep", "target_scale")}
    np.savez_compressed(file, signature=sig(), p=p, **fields)
    b.save(folder / "metadata.json", dict(signature=sig(), model_hashes=hashes,
        fit_rows=len(a["fit"]), query_rows=len(a["query"]), pairs=len(a["target"]),
        features=a["pair_x"].shape[1], query_target_labels_used=False,
        same_scaffold_training_pairs=False, seconds=time.monotonic() - start))
    print(json.dumps(dict(stage="pair_delta_train", task=d["tasks"][k], part=part,
        pairs=len(a["target"]), seconds=round(time.monotonic() - start))), flush=True)
    return p


def smoke(d):
    k = d["tasks"].index("caco2_wang")
    a = design(d, k, "fold0")
    saved = d["y"].copy()
    try:
        d["y"][(d["tid"] == k) & ~np.isin(np.arange(len(saved)), a["fit"])] = 999999.
        d.pop("_v96_assays", None)
        alternative = design(d, k, "fold0")
        for name in a:
            np.testing.assert_array_equal(a[name], alternative[name])
    finally:
        d["y"][:] = saved
        d.pop("_v96_assays", None)
    net = estimator(CONFIGS[0], SEEDS[0], True)
    net.fit(a["pair_x"], a["target"], sample_weight=a["weights"])
    path = OUT / "smoke.cbm"
    net.save_model(str(path))
    other = CatBoostRegressor().load_model(str(path))
    p, q = infer(d, a, [net]), infer(d, a, [other])
    np.testing.assert_array_equal(p, q)
    np.testing.assert_array_equal(pair_features(a["x"][:5], a["x"][:5])[:, :a["x"].shape[1]], 0.)
    np.testing.assert_array_equal(weighted_median(np.array([[1., 3., 2.], [0., 5., 8.]]),
                                                  np.array([[1., 1., 1.], [1., 5., 1.]])), [2., 5.])
    b.save(OUT / "smoke.json", dict(status="PASS", signature=sig(),
        nonfit_target_label_isolation=True, original_scaffold_pair_exclusion=True,
        independent_checkpoint_reload=True, weighted_median_reference=True,
        scores_computed=False, production_query_predictions_computed=False))
    print("SMOKE_PASS", flush=True)


def finish():
    conf = json.loads((OUT / "confirmation.json").read_text())
    dev = json.loads((OUT / "development.json").read_text())
    assert conf["status"] == dev["status"] == "COMPLETE"
    best = [max((r for r in dev["records"] if r["task"] == task), key=lambda r: r["objective"]) for task in TASKS]
    b.save(ROOT / "docs/v297_final_status.json", dict(
        status="REQUIRES_INDEPENDENT_DELIVERY_VERIFICATION" if conf["accepted"] else "COMPLETE_NO_CONFIRMED_NEW_MODEL",
        at_utc=datetime.now(timezone.utc).isoformat(), signature=sig(), baseline="v145",
        official_best=51.166, target_score=52., development_best=best,
        accepted_tasks=conf["accepted"], confirmation=conf["records"],
        csv_created=False, platform_submission=False, official_gain_established=False))


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("stage", choices=("smoke", "run", "report"))
    stage = args.parse_args().stage
    OUT.mkdir(exist_ok=True)
    if stage == "report":
        finish()
        return
    d = load()
    if stage == "smoke":
        smoke(d)
    else:
        assert json.loads((OUT / "smoke.json").read_text())["signature"] == sig()
        assert not (OUT / "development.json").exists(), "Never silently repeat an experiment"
        b.save(OUT / "frozen_before_scores.json", dict(signature=sig(),
            source_sha256=b.sha(__file__), baseline_signature=v.sig(), runner_sha256=b.sha(runner.__file__),
            at_utc=datetime.now(timezone.utc).isoformat(), tasks=TASKS, configs=CONFIGS, seeds=SEEDS,
            packages={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "catboost")},
            no_query_scores_read_for_configuration=True))
        runner.run(sys.modules[__name__], d)
        finish()


if __name__ == "__main__":
    main()
