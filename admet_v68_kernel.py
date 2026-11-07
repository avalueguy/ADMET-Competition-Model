"""CPU structural kernel fits to complement v67; no pretrained/external inputs."""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"

import argparse
import csv
import hashlib
import io
import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.svm import SVC, SVR

import admet_v67_data as b

ROOT, OUT = b.ROOT, b.ROOT / "admet_model_outputs_v68"
P = json.loads((ROOT / "admet_v68_protocol.json").read_text(encoding="utf-8"))


def signature():
    return hashlib.sha256((b.signature()+b.sha(__file__)+b.sha(ROOT/"admet_v68_protocol.json")).encode()).hexdigest()


def kernels(d, rows, query):
    ref, q = d["mi"][rows], d["mi"][query]
    train = np.zeros((len(ref), len(ref)), dtype=np.float32)
    test = np.zeros((len(q), len(ref)), dtype=np.float32)
    for lo, hi in ((0, 2048), (2048, 4096)):
        x = sparse.csr_matrix(d["x"][:, lo:hi] > 0, dtype=np.float32)
        r, z = x[ref], x[q]
        rn, qn = np.asarray(r.sum(1)).ravel(), np.asarray(z.sum(1)).ravel()
        for start in range(0, len(ref), 256):
            dot = (r[start:start+256] @ r.T).toarray()
            tan = dot/np.maximum(rn[start:start+256, None]+rn[None]-dot, 1e-8)
            train[start:start+256] += .5*(.7*tan+.3*tan*tan)
        for start in range(0, len(q), 256):
            dot = (z[start:start+256] @ r.T).toarray()
            tan = dot/np.maximum(qn[start:start+256, None]+rn[None]-dot, 1e-8)
            test[start:start+256] += .5*(.7*tan+.3*tan*tan)
    state = b.preprocess(d, rows)
    # Transform only the descriptor slice; fingerprints remain sparse above.
    def desc(mm):
        x = d["x"][mm, 4096:]
        x = np.where(np.isfinite(x), x, state["median"])
        return ((x-state["mean"])/state["scale"]).clip(-8, 8).astype(np.float32)
    rr, qq = desc(ref), desc(q)
    norm = (rr*rr).sum(1)
    def gaussian(x):
        result = np.empty((len(x), len(rr)), dtype=np.float32)
        for start in range(0, len(x), 256):
            z = x[start:start+256]
            distance = np.maximum(0, (z*z).sum(1)[:, None]+norm[None]-2*z@rr.T)/rr.shape[1]
            result[start:start+256] = .5*np.exp(-distance/.5)+.5*np.exp(-distance/2)
        return result
    return [(train, test), (.85*train+.15*gaussian(rr), .85*test+.15*gaussian(qq))]


def fit(d, rows, query, task, partition):
    dest = OUT/task/partition
    dest.mkdir(parents=True, exist_ok=True)
    path = dest/"predictions.npz"
    b.assert_excluded(d, rows, query)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            assert str(z["signature"]) == signature()
            np.testing.assert_array_equal(rows, z["fit"])
            np.testing.assert_array_equal(query, z["query"])
            return z["p"].copy()
    start = time.monotonic()
    k = d["tasks"].index(task); metric = d["metrics"][k]
    assert np.all(d["tid"][rows] == k)
    tf = b.target(d["y"][rows], metric)
    y = b.encode(d["y"][rows], tf)
    _, inverse, counts = np.unique(d["canonical"][d["mi"][rows]], return_inverse=True, return_counts=True)
    sample_weight = 1/counts[inverse]; sample_weight /= sample_weight.mean()
    pred = []
    for family, (tr, qe) in enumerate(kernels(d, rows, query)):
        tr = ((tr+tr.T)*.5).astype(np.float64)
        qe = qe.astype(np.float64)
        for c in P["c_values"]:
            cls = metric in ("AUROC", "AUPRC")
            model = SVC(C=c, kernel="precomputed", cache_size=512) if cls else SVR(C=c, kernel="precomputed", epsilon=P["regression_epsilon"], cache_size=512)
            model.fit(tr, y, sample_weight=sample_weight)
            def predict(m):
                p = m.decision_function(qe) if cls else m.predict(qe)
                return p*tf["scale"]+tf["center"] if metric == "MAE" else p
            p = predict(model)
            file = dest/f"family{family}_C{c:g}.joblib"
            joblib.dump(dict(model=model, target=tf, rows=rows, signature=signature(), family=family, c=c), file, compress=3)
            np.testing.assert_allclose(p, predict(joblib.load(file)["model"]), atol=1e-10, rtol=1e-10)
            pred.append(p)
    p = np.stack(pred)
    assert np.isfinite(p).all()
    np.savez_compressed(path, fit=rows, query=query, p=p, signature=signature())
    b.save(dest/"metadata.json", dict(status="PASS", task=task, signature=signature(), fit_rows=len(rows), query_rows=len(query),
           model_reload_passed=True, seconds=time.monotonic()-start, external_data=False, pretrained_weights=False))
    b.emit("v68_fit", task=task, partition=partition, seconds=round(time.monotonic()-start))
    return p


def prepare(d):
    OUT.mkdir(parents=True, exist_ok=True)
    folds = np.load(b.OUT/"folds.npy")
    allrows = np.flatnonzero(d["split"] < 2)
    groups = np.unique(d["rg"][allrows])
    held = np.random.default_rng(b.P["fresh_seed"]).permutation(groups)[:max(1,int(len(groups)*b.P["fresh_fraction"]))]
    fresh_query = allrows[np.isin(d["rg"][allrows], held)]
    fresh_fit = allrows[~np.isin(d["rg"][allrows], held)]
    # Small assays finish first; every task still receives the same protocol.
    order = sorted(range(len(d["tasks"])), key=lambda k: np.sum((d["tid"] == k)&(d["split"] == 0)))
    for k in order:
        task = d["tasks"][k]
        train = np.flatnonzero((d["tid"] == k)&(d["split"] == 0))
        for fold in range(3):
            fit(d, train[folds[train] != fold], train[folds[train] == fold], task, f"fold{fold}")
        rr = fresh_fit[d["tid"][fresh_fit] == k]
        query = fresh_query[d["tid"][fresh_query] == k]
        fit(d, rr, query, task, "fresh")
    b.save(OUT/"prepare_completed.json", dict(status="COMPLETE", signature=signature()))


def neural(d, rows, partition, record):
    families = []
    for family in b.P["families"]:
        members = []
        for seed in b.P["training_seeds"]:
            with np.load(b.OUT/partition/family/str(seed)/"predictions.npz",allow_pickle=False) as z:
                position = {int(value): i for i, value in enumerate(z["query"])}
                members.append(z["p"][[position[int(i)] for i in rows]])
        families.append(np.mean(members,0))
    return b.blend(record["metric"], *families, record["selection"]["weight"])


def finish(d):
    assert json.loads((OUT/"prepare_completed.json").read_text())["signature"] == signature()
    release67 = json.loads((b.OUT/"release.json").read_text())
    records67 = json.loads((b.OUT/"fresh.json").read_text())["tasks"]
    folds = np.load(b.OUT/"folds.npy")
    # Use v67's frozen post-gate weights in both CV and fresh comparisons.
    for record in records67:
        record["selection"] = dict(record["selection"], weight=record["final_weight"])
    selection = []
    for k, record in enumerate(records67):
        task, metric = record["task"], record["metric"]
        grid = []
        for variant in range(4):
            for weight in P["blend_weights"]:
                gains = []
                for fold in range(3):
                    with np.load(OUT/task/f"fold{fold}"/"predictions.npz") as z:
                        q, raw = z["query"], z["p"][variant]
                    old = neural(d, q, f"cv/fold{fold}", record)
                    gains.append(b.gain(metric, d["y"][q], old, b.blend(metric, old, raw, weight)))
                grid.append(dict(variant=variant, weight=weight, gains=gains, mean=float(np.mean(gains)), objective=float(np.mean(gains)-.1*np.std(gains))))
        held = [max(grid,key=lambda r:np.mean([v for j,v in enumerate(r["gains"]) if j!=fold]))["gains"][fold] for fold in range(3)]
        good = [g for g in grid if g["mean"] > .002 and sum(v>0 for v in g["gains"])>=2]
        best = max(good,key=lambda r:r["objective"]) if good and np.mean(held)>.001 else None
        fresh_gain = None
        if best:
            with np.load(OUT/task/"fresh/predictions.npz") as z:
                q, raw = z["query"], z["p"][best["variant"]]
            old = neural(d,q,"fresh",record)
            fresh_gain = b.gain(metric,d["y"][q],old,b.blend(metric,old,raw,best["weight"]))
        accepted = bool(best and np.isfinite(fresh_gain) and fresh_gain>0)
        selection.append(dict(task=task,metric=metric,selection=best,held_gains=held,fresh_gain=fresh_gain,pass_gate=accepted))
        b.emit("v68_selection",**selection[-1])
    b.save(OUT/"selection.json",dict(status="FROZEN",signature=signature(),tasks=selection))
    source = ROOT/release67["file"]
    with source.open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.reader(stream))
    replay = []
    for k,record in enumerate(selection):
        if not record["pass_gate"]:continue
        task,metric,s = record["task"],record["metric"],record["selection"]
        rr=np.flatnonzero((d["tid"]==k)&(d["split"]<2));q=np.flatnonzero((d["tid"]==k)&(d["split"]==2))
        raw=fit(d,rr,q,task,"final")[s["variant"]]
        positions=[i for i,row in enumerate(rows[1:],1) if row[0]==task];old=np.array([float(rows[i][4]) for i in positions]);p=b.blend(metric,old,raw,s["weight"])
        if task=="ppbr_az":p=p.clip(0,100)
        for i,v in zip(positions,p):rows[i][4]=repr(float(v))
        replay.append(dict(task=task,selection=s,baseline=old.tolist(),branch=raw.tolist(),result=p.tolist()))
    dest=ROOT/"admet_submissions/v68_official_only";dest.mkdir(parents=True,exist_ok=True);file=dest/source.name
    buf=io.StringIO(newline="");csv.writer(buf,lineterminator="\r\n").writerows(rows);raw=b"\xef\xbb\xbf"+buf.getvalue().encode("utf-8")
    if file.exists():assert file.read_bytes()==raw
    else:file.write_bytes(raw)
    b.save(OUT/"fusion_replay.json",replay);b.save(OUT/"release.json",dict(status="AWAITING_PLATFORM_EVALUATION",signature=signature(),file=str(file.relative_to(ROOT)),sha256=b.sha(file),source67_sha256=b.sha(source),changed_tasks=[r["task"] for r in replay],external_data=False,pretrained_weights=False,official_score=None))
    b.emit("v68_export",file=str(file),sha256=b.sha(file))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=["prepare","finish"]);args=parser.parse_args();d=b.build()
    assert os.name!="nt", "Heavy experiment fits run on the authorized server"
    prepare(d) if args.stage=="prepare" else finish(d)
