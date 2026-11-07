"""Analytic infinite-width ReLU neural tangent kernel for official MAE tasks."""
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

import joblib
import numpy as np
from scipy import sparse
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import v314_current_baseline as v
import v314_v313_experiment_runner as runner

b, r83 = v.b, v.r83
OUT = ROOT / "admet_model_outputs_v318"
TASKS = ("caco2_wang", "ppbr_az", "lipophilicity_astrazeneca", "ld50_zhu", "solubility_aqsoldb")
CONFIGS = (dict(depth=3, C=10., epsilon=.03, fingerprint_weight=.65, input_bias=.1),)
METHOD = (
    "Analytic infinite-width depth-three ReLU neural tangent kernel with epsilon-insensitive "
    "SVR for the five official MAE tasks, incremental to v313. Six archived count-fingerprint "
    "cosine blocks receive total covariance weight .65; robust fit-only median/IQR numeric "
    "and target-excluded other-assay inputs receive .35, with explicit missingness. "
    "Normalize each modality by row, add fixed .1 input bias, use exact normalized ReLU "
    "covariance and tangent recursion. Fixed C10 and epsilon .03 on fit median/IQR targets. "
    "No finite neural training, Tanimoto/RBF kernel tuning, stochastic seeds, external "
    "data, pretrained models or old predictions as fitted features. Original v148 splits, "
    "blends, development selection and scaffold confirmation via unchanged v314 adapter."
)


def sig():
    versions = "|".join(importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn"))
    return hashlib.sha256((b.sha(__file__) + v.sig() + b.sha(runner.__file__) + versions).encode()).hexdigest()


def key(c):
    assert c == CONFIGS[0]
    return "relu_ntk3_svr10"


def load():
    threadpool_limits(limits=4)
    return v.load()


def same(a, z):
    assert a.keys() == z.keys()
    for name in a:
        np.testing.assert_array_equal(a[name], z[name])


def robust(x, qx):
    x, qx = x.astype(np.float64), qx.astype(np.float64)
    median = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0. for col in x.T])
    clean = np.where(np.isfinite(x), x, median)
    scale = np.subtract(*np.quantile(clean, [.75, .25], axis=0)).clip(.1)
    keep = np.flatnonzero((np.std(clean, axis=0) > .001) | (np.isfinite(x).std(axis=0) > 0))
    assert len(keep)
    def transform(raw):
        finite = np.isfinite(raw)
        z = np.clip((np.where(finite, raw, median) - median) / scale, -8., 8.)
        # Constant prevents zero vectors and makes missingness explicit.
        z = np.c_[z[:, keep], finite[:, keep], np.ones(len(raw))]
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        return z
    return transform(x), transform(qx), dict(median=median, numeric_scale=scale, keep=keep)


def design(d, k, part, c):
    assert d["tasks"][k] in TASKS and d["metrics"][k] == "MAE"
    fit, query, raw, qraw, source = v.old.old.features(d, k, part, "augmented")
    x, qx, numeric = robust(raw, qraw)
    def fingerprints(rows):
        result = []
        for j in range(6):
            fp = sparse.csr_matrix(d["views"]["multiview"][d["mi"][rows], j * 2048:(j + 1) * 2048].astype(np.float64))
            norms = np.sqrt(np.asarray(fp.multiply(fp).sum(axis=1)).ravel())
            assert np.all(norms > 0), "Inspect zero fingerprint before training"
            fp = sparse.diags(1. / norms) @ fp
            result.append(fp.tocsr())
        return result
    y = d["y"][fit]
    center = float(np.median(y))
    scale = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05)
    assert not set(fit) & set(query) and np.all(d["tid"][source] != k)
    assert np.all(d["split"][source] < (2 if part == "final" else 1))
    assert part == "final" or not set(d["groups"][fit]) & set(d["groups"][query])
    state = dict(fit=fit, query=query, source=source, center=center, scale=scale, **numeric)
    return dict(x=x, qx=qx, fp=fingerprints(fit), qfp=fingerprints(query), state=state,
                target=(y - center) / scale)


def tangent(covariance, depth):
    covariance = np.clip(covariance, -1., 1.)
    theta = covariance.copy()
    for _ in range(depth):
        angle = np.arccos(covariance)
        derivative = (np.pi - angle) / np.pi
        covariance = np.clip((np.sqrt(np.maximum(1. - covariance * covariance, 0.)) +
                              (np.pi - angle) * covariance) / np.pi, -1., 1.)
        theta = theta * derivative + covariance
    return theta / (depth + 1.)


def kernel(x, fp, ref, rfp, c):
    out = np.empty((len(x), len(ref)), dtype=np.float64)
    for lo in range(0, len(x), 128):
        hi = min(lo + 128, len(x))
        cov = (1. - c["fingerprint_weight"]) * (x[lo:hi] @ ref.T)
        for query, reference in zip(fp, rfp):
            cov += c["fingerprint_weight"] / 6. * (query[lo:hi] @ reference.T).toarray()
        cov = (cov + c["input_bias"]) / (1. + c["input_bias"])
        out[lo:hi] = tangent(cov, c["depth"])
    assert np.isfinite(out).all()
    return out


def output(p, state, task):
    p = p * state["scale"] + state["center"]
    return np.clip(p, 0., 100.) if task == "ppbr_az" else p


def predict(d, k, part, c):
    if d["tasks"][k] not in TASKS:
        return v.baseline(d, k, part)
    folder = OUT / "models" / d["tasks"][k] / part / key(c)
    file = folder / "predictions.npz"
    if file.exists():
        z = np.load(file)
        assert str(z["signature"]) == sig()
        np.testing.assert_array_equal(z["query"], v.split(d, k, part)[1])
        return z["p"].copy()
    assert not folder.exists(), "Inspect partial artifacts before retry"
    started = time.monotonic()
    a = design(d, k, part, c)
    folder.mkdir(parents=True)
    joblib.dump(a["state"], folder / "transform.joblib", compress=3)
    xx = kernel(a["x"], a["fp"], a["x"], a["fp"], c)
    np.testing.assert_allclose(xx, xx.T, rtol=0, atol=1e-12)
    xx = (xx + xx.T) * .5
    np.fill_diagonal(xx, 1. + 1e-8)
    net = SVR(C=c["C"], epsilon=c["epsilon"], kernel="precomputed", tol=1e-5, cache_size=512)
    net.fit(xx, a["target"])
    assert net.fit_status_ == 0
    del xx
    joblib.dump(dict(signature=sig(), config=c, model=net), folder / "model.joblib", compress=3)
    qx = kernel(a["qx"], a["qfp"], a["x"], a["fp"], c)
    raw = net.predict(qx)
    loaded = joblib.load(folder / "model.joblib")
    np.testing.assert_array_equal(raw, loaded["model"].predict(qx))
    p = output(raw, a["state"], d["tasks"][k])
    assert np.isfinite(p).all()
    np.savez_compressed(file, signature=sig(), fit=a["state"]["fit"], query=a["state"]["query"], p=p)
    b.save(folder / "metadata.json", dict(signature=sig(), model_sha256=b.sha(folder / "model.joblib"),
        support_vectors=len(net.support_), numeric_features=a["x"].shape[1], seconds=time.monotonic() - started))
    print(json.dumps(dict(stage="kernel_train", task=d["tasks"][k], part=part,
                         seconds=round(time.monotonic() - started))), flush=True)
    return p


def replay(d, k, part, c):
    a = design(d, k, part, c)
    folder = OUT / "models" / d["tasks"][k] / part / key(c)
    same(a["state"], joblib.load(folder / "transform.joblib"))
    z = np.load(folder / "predictions.npz")
    model = joblib.load(folder / "model.joblib")
    assert model["signature"] == str(z["signature"]) == sig() and model["config"] == c
    qx = kernel(a["qx"], a["qfp"], a["x"], a["fp"], c)
    p = output(model["model"].predict(qx), a["state"], d["tasks"][k])
    np.testing.assert_array_equal(p, z["p"])
    return p


def scalar_reference(x, y, c):
    cov = float(np.dot(x, y))
    theta = cov
    import math
    for _ in range(c["depth"]):
        cov = max(-1., min(1., cov))
        derivative = 1. - math.acos(cov) / math.pi
        cov = math.sqrt(max(0., 1. - cov * cov)) / math.pi + derivative * cov
        theta = theta * derivative + cov
    return theta / (c["depth"] + 1.)


def explicit_inputs(a, query, c, take=12):
    numeric, fps = (a["qx"], a["qfp"]) if query else (a["x"], a["fp"])
    x = np.c_[np.sqrt(1. - c["fingerprint_weight"]) * numeric[:take],
              *[np.sqrt(c["fingerprint_weight"] / 6.) * fp[:take].toarray() for fp in fps],
              np.full(take, np.sqrt(c["input_bias"]))] / np.sqrt(1. + c["input_bias"])
    return x


def smoke(d):
    c = CONFIGS[0]
    k = d["tasks"].index("caco2_wang")
    a = design(d, k, "fold0", c)
    tx, qx = explicit_inputs(a, False, c), explicit_inputs(a, True, c)
    exact = np.array([[scalar_reference(x, y, c) for y in tx] for x in qx])
    actual = kernel(a["qx"][:12], [x[:12] for x in a["qfp"]], a["x"][:12], [x[:12] for x in a["fp"]], c)
    np.testing.assert_allclose(actual, exact, rtol=0, atol=1e-12)
    xx = kernel(a["x"][:12], [x[:12] for x in a["fp"]], a["x"][:12], [x[:12] for x in a["fp"]], c)
    np.testing.assert_allclose(np.diag(xx), 1., rtol=0, atol=1e-7)
    assert np.linalg.eigvalsh((xx + xx.T) / 2).min() > -1e-10
    saved = d["y"].copy()
    try:
        d["y"][(d["tid"] == k) & ~np.isin(np.arange(len(saved)), a["state"]["fit"])] = 999999.
        d.pop("_v96_assays", None)
        altered = design(d, k, "fold0", c)
        same(a["state"], altered["state"])
        np.testing.assert_array_equal(a["x"], altered["x"])
        np.testing.assert_array_equal(a["qx"], altered["qx"])
        np.testing.assert_array_equal(a["target"], altered["target"])
    finally:
        d["y"][:] = saved
        d.pop("_v96_assays", None)
    net = SVR(C=c["C"], epsilon=c["epsilon"], kernel="precomputed", tol=1e-5).fit(xx, a["target"][:12])
    file = OUT / "smoke.joblib"
    joblib.dump(net, file)
    np.testing.assert_array_equal(net.predict(actual), joblib.load(file).predict(actual))
    np.testing.assert_allclose(net.predict(actual[::-1])[::-1], net.predict(actual), rtol=0, atol=1e-12)
    b.save(OUT / "smoke.json", dict(status="PASS", signature=sig(), scalar_kernel_reference=True,
        kernel_sample_psd=True, fit_only_preprocessing=True, checkpoint_reload=True,
        nonfit_target_label_isolation=True, query_scores_computed=False))
    print("SMOKE_PASS", flush=True)


def finish():
    dev = json.loads((OUT / "development.json").read_text())
    conf = json.loads((OUT / "confirmation.json").read_text())
    assert dev["status"] == conf["status"] == "COMPLETE"
    best = [max((r for r in dev["records"] if r["task"] == t), key=lambda r:r["objective"]) for t in TASKS]
    b.save(ROOT / "docs/v318_final_status.json", dict(
        status="REQUIRES_INDEPENDENT_DELIVERY_VERIFICATION" if conf["accepted"] else "COMPLETE_NO_CONFIRMED_NEW_MODEL",
        at_utc=datetime.now(timezone.utc).isoformat(), signature=sig(), baseline="v313", official_best=51.205,
        target_score=52., development_best=best, accepted_tasks=conf["accepted"], confirmation=conf["records"],
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
        assert not (OUT / "development.json").exists(), "Never silently repeat a scored experiment"
        b.save(OUT / "frozen_before_scores.json", dict(signature=sig(), source_sha256=b.sha(__file__),
            baseline_signature=v.sig(), runner_sha256=b.sha(runner.__file__),
            at_utc=datetime.now(timezone.utc).isoformat(), tasks=TASKS, configs=CONFIGS,
            packages={p:importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn")},
            no_query_scores_read_for_configuration=True))
        runner.run(sys.modules[__name__], d)
        finish()


if __name__ == "__main__":
    main()
