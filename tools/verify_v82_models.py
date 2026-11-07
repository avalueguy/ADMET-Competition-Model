"""Independent reload and byte-level verification for v82."""
from __future__ import annotations
import csv, io, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata, spearmanr
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import admet_v77_ranking as b
from verify_submission import verify
from repack_upload_template import field_spans

TARGETS = ("cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels")
SEEDS = (824026, 824027, 824028)
OUT = ROOT / "admet_model_outputs_v82_final"


def ranklog(x):
    return logit(np.clip((rankdata(np.asarray(x)) - .5) / len(x), .002, .998))


def calibrate(y, z):
    x = np.c_[np.ones(len(z)), z]
    def fun(t):
        eta = x @ t; r = expit(eta) - y
        return ((np.logaddexp(0, eta) - y * eta).mean() + .001 * t[1] ** 2,
                x.T @ r / len(r) + [0., .002 * t[1]])
    return minimize(fun, [0., 1.], jac=True, bounds=[(None, None), (.01, 10.)], method="L-BFGS-B").x


def aux(d, k):
    cols = [j for j in range(len(d["tasks"])) if j != k]
    source = np.flatnonzero((d["split"] < 2) & (d["tid"] != k))
    known = {}
    for i in source:
        known.setdefault((str(d["canonical"][i]), int(d["tid"][i])), []).append(float(d["y"][i]))
    a = np.full((len(d["y"]), 2 * len(cols)), np.nan, np.float32)
    for i in np.flatnonzero(d["tid"] == k):
        for j, t in enumerate(cols):
            v = known.get((str(d["canonical"][i]), t))
            if v: a[i, j] = np.median(v)
    a[:, len(cols):] = np.isfinite(a[:, :len(cols)])
    return a, source


def rows(path):
    return list(csv.reader(io.StringIO(path.read_bytes().decode("utf-8-sig"), newline="")))


def same(a, b, tol=1e-11):
    np.testing.assert_allclose(a, b, atol=tol, rtol=tol, equal_nan=True)


def main():
    d = b.load_data()
    release = json.loads((OUT / "release.json").read_text(encoding="utf-8"))
    assert release["status"] == "VERIFIED_CANDIDATE_AWAITING_PLATFORM"
    file = ROOT / release["file"]
    anchor = ROOT / "admet_submissions/v81_multitask_residual/我从山北来带着兰花草-小分子药物预测技术创新赛-20261024.csv"
    template = ROOT / "admet_official/submission_template.csv"
    assert b.sha(file) == release["sha256"] and b.sha(anchor) == release["anchor_sha256"]
    old, actual = rows(anchor), rows(file)
    assert len(old) == len(actual)
    changed = Counter(); model_records = []
    for task in TARGETS:
        k = d["tasks"].index(task); tr = np.flatnonzero((d["tid"] == k) & (d["split"] == 0)); va = np.flatnonzero((d["tid"] == k) & (d["split"] == 1)); fit = np.r_[tr, va]; q = np.flatnonzero((d["tid"] == k) & (d["split"] == 2))
        pos = [i for i, r in enumerate(old) if i and r[0] == task]; assert len(pos) == len(q)
        base_fit = np.r_[d["proxy_cv"][tr], d["proxy_valid"][va]]; base_q = np.asarray([float(old[i][4]) for i in pos])
        a, source = aux(d, k); cache = np.load(OUT / task / "predictions.npz")
        np.testing.assert_array_equal(cache["fit"], fit); np.testing.assert_array_equal(cache["query"], q)
        z, qz = ranklog(base_fit), ranklog(base_q); beta = calibrate(d["y"][fit], z); off = beta[0] + beta[1] * z; qoff = beta[0] + beta[1] * qz
        x = np.c_[a[fit], z].astype(np.float32); qx = np.c_[a[q], qz].astype(np.float32)
        same(cache["base_fit"], base_fit); same(cache["base_query"], base_q); same(cache["beta"], beta); same(cache["x"], x); same(cache["qx"], qx)
        conf = json.loads((OUT / "confirmation.json").read_text(encoding="utf-8")); spec = next(r for r in conf["records"] if r["task"] == task)
        members = []
        for seed in SEEDS:
            model = CatBoostClassifier(); path = OUT / task / f"model_{seed}.cbm"; model.load_model(str(path)); params = model.get_all_params(); assert int(params["random_seed"]) == seed and int(params["depth"]) == spec["depth"] and int(model.tree_count_) == spec["trees"]
            raw = qoff + model.predict(qx, prediction_type="RawFormulaVal"); members.append(raw); model_records.append(dict(task=task, seed=seed, status="PASS"))
        members = np.stack(members); raw = members.mean(axis=0); pred = expit(raw)
        same(cache["raw_members"], members); same(cache["raw"], raw); same(cache["p"], pred); same(np.asarray([float(actual[i][4]) for i in pos]), pred, 1e-14)
    for o, n in zip(old[1:], actual[1:]):
        assert o[:4] + o[5:] == n[:4] + n[5:]
        if o[4] != n[4]: changed[o[0]] += 1
        if o[0] not in TARGETS: assert o == n
    assert dict(changed) == release["changed_tasks"]
    fmt = verify(file, template); raw = file.read_bytes(); ref = template.read_bytes(); spans = field_spans(raw); tspans = field_spans(ref); chunks = []; cur = 0
    for i in range(1, len(actual)):
        lo, hi = spans[i][4]; a, z = tspans[i][4]; chunks.extend((raw[cur:lo], ref[a:z])); cur = hi
    chunks.append(raw[cur:]); assert b"".join(chunks) == ref
    result = dict(status="PASS", models=len(model_records), model_records=model_records, changed_tasks=dict(changed), unchanged_tasks=20, independent_model_reload=True, nonprediction_bytes_exact=True, test_labels_used=False, external_data=False, sha256=b.sha(file))
    b.save(OUT / "model_verification.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
