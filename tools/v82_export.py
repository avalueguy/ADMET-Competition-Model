"""v82 targeted substrate residual candidate.

Adds two low-depth, low-tree conditional residual classifiers selected on
exact scaffold folds: CYP2C9 and CYP2D6 substrate AUPRC endpoints. The
existing v81 predictions remain the anchor for every other task.
"""
from __future__ import annotations
import csv, io, json, hashlib, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import admet_v77_ranking as b
from repack_upload_template import field_spans
from verify_submission import verify

ANCHOR = ROOT / "admet_submissions/v81_multitask_residual/我从山北来带着兰花草-小分子药物预测技术创新赛-20261024.csv"
TARGETS = ("cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels")
SEEDS = (824026, 824027, 824028)
CONFIG = {TARGETS[0]: (2, 100, 1.0), TARGETS[1]: (2, 50, 1.0)}
OUT = ROOT / "admet_model_outputs_v82_final"


def ranklog(p):
    return logit(np.clip((rankdata(np.asarray(p)) - .5) / len(p), .002, .998))


def calibrate(y, z):
    x = np.c_[np.ones(len(z)), z]
    def fun(t):
        eta = x @ t; r = expit(eta) - y
        return ((np.logaddexp(0, eta) - y * eta).mean() + .001 * t[1] ** 2,
                x.T @ r / len(r) + [0., .002 * t[1]])
    return minimize(fun, [0., 1.], jac=True, bounds=[(None, None), (.01, 10.)], method="L-BFGS-B").x


def auxiliary(d, k, include_valid):
    cols = [j for j in range(len(d["tasks"])) if j != k]
    source = np.flatnonzero((d["split"] < (2 if include_valid else 1)) & (d["tid"] != k))
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


def fit_one(d, k, fit, q, base_fit, base_q, aux, depth, trees, seed, save=None):
    z, qz = ranklog(base_fit), ranklog(base_q)
    beta = calibrate(d["y"][fit], z)
    off, qoff = beta[0] + beta[1] * z, beta[0] + beta[1] * qz
    x = np.c_[aux[fit], z].astype(np.float32); qx = np.c_[aux[q], qz].astype(np.float32)
    cfg = dict(iterations=trees, depth=depth, learning_rate=.025, l2_leaf_reg=50,
               loss_function="Logloss", random_seed=seed, thread_count=8,
               verbose=False, allow_writing_files=False, bootstrap_type="Bernoulli",
               subsample=.85, task_type="GPU", devices="0")
    model = CatBoostClassifier(**cfg)
    model.fit(Pool(x, d["y"][fit], baseline=off))
    raw = qoff + model.predict(qx, prediction_type="RawFormulaVal")
    if save is not None:
        save.mkdir(parents=True, exist_ok=True)
        model.save_model(str(save / f"model_{seed}.cbm"))
    return expit(raw), raw, beta, x, qx


def validate(d):
    folds = np.load(ROOT / "recovered_history/legacy_v24_folds.npy")
    records = []
    for task in TARGETS:
        k = d["tasks"].index(task); depth, trees, _ = CONFIG[task]
        tr = np.flatnonzero((d["tid"] == k) & (d["split"] == 0)); va = np.flatnonzero((d["tid"] == k) & (d["split"] == 1))
        aux, _ = auxiliary(d, k, False); ff = folds[tr]
        base_tr, base_va = d["proxy_cv"][tr], d["proxy_valid"][va]
        oof = np.empty(len(tr))
        for f in (0, 1, 2):
            fit, q = tr[ff != f], tr[ff == f]
            preds = [fit_one(d, k, fit, q, d["proxy_cv"][fit], d["proxy_cv"][q], aux, depth, trees, s)[0] for s in SEEDS]
            oof[ff == f] = np.mean(preds, axis=0)
        valid = np.mean([fit_one(d, k, tr, va, base_tr, base_va, aux, depth, trees, s)[0] for s in SEEDS], axis=0)
        parts = []
        for name, y, old, new in [("fold0", d["y"][tr[ff == 0]], base_tr[ff == 0], oof[ff == 0]),
                                  ("fold1", d["y"][tr[ff == 1]], base_tr[ff == 1], oof[ff == 1]),
                                  ("fold2", d["y"][tr[ff == 2]], base_tr[ff == 2], oof[ff == 2]),
                                  ("valid", d["y"][va], base_va, valid)]:
            parts.append(dict(part=name, baseline=float(average_precision_score(y, old)),
                              candidate=float(average_precision_score(y, new)),
                              gain=float(average_precision_score(y, new) - average_precision_score(y, old))))
        records.append(dict(task=task, depth=depth, trees=trees, seeds=list(SEEDS), parts=parts,
                            accepted=all(p["gain"] > 0 for p in parts)))
    b.save(OUT / "confirmation.json", dict(status="COMPLETE", records=records,
                                            accepted=all(r["accepted"] for r in records),
                                            test_labels_used=False, external_data=False))
    print(json.dumps(records, ensure_ascii=False), flush=True)
    assert all(r["accepted"] for r in records)


def export(d):
    conf = json.loads((OUT / "confirmation.json").read_text())
    assert conf["accepted"] and b.sha(ANCHOR) == json.loads((ROOT / "admet_model_outputs_v81_final/release.json").read_text())["sha256"]
    rows = list(csv.reader(io.StringIO(ANCHOR.read_bytes().decode("utf-8-sig"), newline="")))
    changed = {}; source_rows = {}; meta = {}
    for task in TARGETS:
        k = d["tasks"].index(task); depth, trees, blend = CONFIG[task]
        tr = np.flatnonzero((d["tid"] == k) & (d["split"] == 0)); va = np.flatnonzero((d["tid"] == k) & (d["split"] == 1)); fit = np.r_[tr, va]
        q = np.flatnonzero((d["tid"] == k) & (d["split"] == 2)); pos = [i for i, r in enumerate(rows) if i and r[0] == task]
        assert len(pos) == len(q)
        base_fit = np.r_[d["proxy_cv"][tr], d["proxy_valid"][va]]
        base_q = np.asarray([float(rows[i][4]) for i in pos])
        aux, source = auxiliary(d, k, True); folder = OUT / task; members=[]; saved=[]
        for seed in SEEDS:
            p, raw, beta, x, qx = fit_one(d, k, fit, q, base_fit, base_q, aux, depth, trees, seed, folder)
            members.append(raw); saved.append(p)
        raw = np.mean(members, axis=0); pred = expit(raw)
        np.savez_compressed(folder / "predictions.npz", fit=fit, query=q, base_fit=base_fit, base_query=base_q,
                            beta=beta, x=x, qx=qx, raw_members=np.stack(members), raw=raw, p=pred)
        source_rows[task] = int(len(source)); meta[task] = dict(depth=depth, trees=trees, seeds=list(SEEDS), model_rows=len(fit), query_rows=len(q), source_rows=len(source))
        for i, v in zip(pos, pred): rows[i][4] = repr(float(v))
        changed[task] = len(pos)
    template = ROOT / "admet_official/submission_template.csv"; raw_template = template.read_bytes(); spans = field_spans(raw_template); chunks=[]; cur=0
    for i, row in enumerate(rows[1:], 1):
        lo, hi = spans[i][4]; value=row[4].encode("ascii"); assert not any(c in value for c in b',\r\n"'); chunks.extend((raw_template[cur:lo], value)); cur=hi
    chunks.append(raw_template[cur:]); dest=ROOT / "admet_submissions/v82_targeted"; dest.mkdir(parents=True, exist_ok=True)
    file=dest / "我从山北来带着兰花草-小分子药物预测技术创新赛-20261025.csv"; file.write_bytes(b"".join(chunks))
    audit=verify(file, template); audit.update(anchor="v81", changed_tasks=changed, unchanged_tasks=20, test_labels_used=False, external_data=False)
    file.with_suffix(".verification.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    b.save(OUT / "release.json", dict(status="VERIFIED_CANDIDATE_AWAITING_PLATFORM", version="v82", file=file.relative_to(ROOT).as_posix(),
                                        sha256=b.sha(file), anchor="v81", anchor_sha256=b.sha(ANCHOR), changed_tasks=changed,
                                        unchanged_tasks=20, config=meta, source_rows=source_rows, test_labels_used=False,
                                        external_data=False, official_score=None, verification=audit))
    print(json.dumps(dict(status="EXPORTED", file=str(file), sha256=b.sha(file), changed_tasks=changed), ensure_ascii=False), flush=True)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "validate"
    d = b.load_data()
    if stage == "validate": validate(d)
    elif stage == "export": export(d)
    else: raise SystemExit(stage)


if __name__ == "__main__": main()
