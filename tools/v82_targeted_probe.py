"""Targeted gated residual probe for cyp2c9/cyp2d6 substrate tasks.

This is a development-only probe. It uses the existing official-only data
loader, train/valid labels, and exact scaffold folds. No test labels are read.
It searches low-complexity residual trees and evaluates row-wise gates based
only on the number of permitted same-canonical auxiliary assay measurements.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import admet_v77_ranking as b

OUT = ROOT / "admet_model_outputs_v82_probe"
OUT.mkdir(exist_ok=True)
FOLDS = np.load(ROOT / "recovered_history/legacy_v24_folds.npy")
TARGETS = ["cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels"]
SEED = 824026


def score(y, p):
    return float(average_precision_score(y, p))


def ranklog(p):
    return logit(np.clip((rankdata(np.asarray(p)) - .5) / len(p), .002, .998))


def calibrate(y, z):
    x = np.c_[np.ones(len(z)), z]
    def fun(t):
        eta = x @ t
        r = expit(eta) - y
        return ((np.logaddexp(0, eta) - y * eta).mean() + .001 * t[1] ** 2,
                x.T @ r / len(r) + [0., .002 * t[1]])
    return minimize(fun, [0., 1.], jac=True, bounds=[(None, None), (.01, 10.)], method="L-BFGS-B").x


def auxiliary(d, k, include_valid=False):
    cols = [j for j in range(len(d["tasks"])) if j != k]
    source = np.flatnonzero((d["split"] < (2 if include_valid else 1)) & (d["tid"] != k))
    known = {}
    for i in source:
        known.setdefault((str(d["canonical"][i]), int(d["tid"][i])), []).append(float(d["y"][i]))
    a = np.full((len(d["y"]), 2 * len(cols)), np.nan, np.float32)
    for i in np.flatnonzero(d["tid"] == k):
        for j, t in enumerate(cols):
            v = known.get((str(d["canonical"][i]), t))
            if v:
                a[i, j] = np.median(v)
    a[:, len(cols):] = np.isfinite(a[:, :len(cols)])
    return a, cols


def fit_predict(d, k, fit, q, base_fit, base_q, aux, depth, trees, structure=False):
    z, qz = ranklog(base_fit), ranklog(base_q)
    beta = calibrate(d["y"][fit], z)
    off, qoff = beta[0] + beta[1] * z, beta[0] + beta[1] * qz
    pieces = [aux[fit], z[:, None]]
    qpieces = [aux[q], qz[:, None]]
    if structure:
        pieces.insert(1, d["views"]["physchem"][d["mi"][fit]])
        qpieces.insert(1, d["views"]["physchem"][d["mi"][q]])
    x = np.concatenate(pieces, 1).astype(np.float32)
    qx = np.concatenate(qpieces, 1).astype(np.float32)
    cfg = dict(iterations=trees, depth=depth, learning_rate=.025, l2_leaf_reg=50,
               loss_function="Logloss", random_seed=SEED, thread_count=8,
               verbose=False, allow_writing_files=False, bootstrap_type="Bernoulli",
               subsample=.85, task_type="GPU", devices="0")
    m = CatBoostClassifier(**cfg)
    m.fit(Pool(x, d["y"][fit], baseline=off))
    return expit(qoff + m.predict(qx, prediction_type="RawFormulaVal"))


def evaluate(y, old, new, counts):
    out = {"base": score(y, old), "full": score(y, new), "full_gain": score(y, new) - score(y, old)}
    for threshold in range(0, 8):
        for weight in (.10, .25, .50, .75, 1.0):
            gate = counts >= threshold
            mixed = old.copy()
            mixed[gate] = (1 - weight) * old[gate] + weight * new[gate]
            out[f"n{threshold}_w{weight}"] = score(y, mixed) - score(y, old)
    return out


def main():
    d = b.load_data()
    records = []
    configs = [(2, 50), (2, 100), (2, 200), (3, 50), (3, 100), (3, 200),
               (4, 50), (4, 100), (4, 200), (5, 100), (5, 200), (5, 400)]
    for task in TARGETS:
        k = d["tasks"].index(task)
        tr = np.flatnonzero((d["tid"] == k) & (d["split"] == 0))
        va = np.flatnonzero((d["tid"] == k) & (d["split"] == 1))
        aux, cols = auxiliary(d, k, False)
        counts_all = np.isfinite(aux[:, len(cols):]).sum(1)
        base_tr, base_va = d["proxy_cv"][tr], d["proxy_valid"][va]
        ff = FOLDS[tr]
        for structure in (False, True):
            for depth, trees in configs:
                started = time.time()
                oof = np.empty(len(tr))
                for f in (0, 1, 2):
                    fit, q = tr[ff != f], tr[ff == f]
                    oof[ff == f] = fit_predict(d, k, fit, q, d["proxy_cv"][fit], d["proxy_cv"][q], aux,
                                                depth, trees, structure)
                valid = fit_predict(d, k, tr, va, base_tr, base_va, aux, depth, trees, structure)
                parts = []
                for name, q, y, old, new in [
                    ("fold0", tr[ff == 0], d["y"][tr[ff == 0]], base_tr[ff == 0], oof[ff == 0]),
                    ("fold1", tr[ff == 1], d["y"][tr[ff == 1]], base_tr[ff == 1], oof[ff == 1]),
                    ("fold2", tr[ff == 2], d["y"][tr[ff == 2]], base_tr[ff == 2], oof[ff == 2]),
                    ("valid", va, d["y"][va], base_va, valid)]:
                    parts.append(dict(part=name, rows=len(q), known_counts={str(i): int(np.sum(counts_all[q] == i)) for i in range(22)},
                                      metrics=evaluate(y, old, new, counts_all[q])))
                r = dict(task=task, structure=structure, depth=depth, trees=trees, parts=parts,
                         seconds=round(time.time() - started, 2))
                records.append(r)
                print(json.dumps(r, ensure_ascii=False), flush=True)
    b.save(OUT / "probe.json", dict(status="COMPLETE", records=records,
                                    targets=TARGETS, test_labels_used=False, external_data=False))


if __name__ == "__main__":
    main()
