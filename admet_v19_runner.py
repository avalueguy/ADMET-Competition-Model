"""Remote v19 candidate training against the verified v16 submission."""
import argparse, json, os, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
from admet_v6_runner import remote_only, save, sha, emit, get_metric, partition
from admet_v16_models import load_task
from admet_v16_ensemble import combine, select, score
from admet_v19_models import fit, predict

OUT = Path("admet_model_outputs_v19")
ROOT = Path("admet_release_v19")
PROTOCOL = Path("admet_v19_protocol.json")

def read(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def names(): return read("admet_v13_protocol.json")["tasks"]
def protocol():
    p = read(PROTOCOL); p["tasks"] = names(); return p

def baseline(task):
    c = pd.read_csv(Path("admet_model_outputs_v16") / task / "oof_comparison.csv")
    return c

def materialize(task, d, fit_rows, query_rows, family, stage):
    dest = OUT / task / stage; dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{family.replace(':','_')}.joblib"
    meta = dest / f"{family.replace(':','_')}.json"
    cache = dest / f"{family.replace(':','_')}.npz"
    if path.exists() and meta.exists() and cache.exists():
        with np.load(cache) as z:
            assert np.array_equal(z["fit"], fit_rows) and np.array_equal(z["query"], query_rows)
            return z["prediction"].copy()
    frame = pd.concat([d["frames"]["train"], d["frames"]["valid"]], ignore_index=True)
    y = frame.Y.to_numpy(float)[fit_rows]
    started = time.time(); bundle = fit(family, d, fit_rows, y, get_metric(task))
    q = predict(bundle, d, query_rows)
    joblib.dump(bundle, path, compress=3)
    np.savez_compressed(cache, fit=fit_rows, query=query_rows, prediction=q)
    save(meta, {"name": family, "sha256": sha(path), "fit_rows": len(fit_rows), "query_rows": len(query_rows), "seconds": time.time()-started})
    emit(stage="v19_fit", task=task, family=family, partition=stage, seconds=round(time.time()-started))
    return q

def train(task):
    d = load_task(task); n = len(d["frames"]["train"]); fold = partition(task, d)
    y = d["frames"]["train"].Y.to_numpy(float)
    for family in ["v19:view_attention_v2", "v19:mol_boost_v2"]:
        dest = OUT / task; dest.mkdir(parents=True, exist_ok=True); out = dest / (family.split(":")[1] + "_oof.csv")
        if out.exists(): continue
        q = np.full(n, np.nan)
        for f in range(3):
            it = np.flatnonzero(fold != f); iv = np.flatnonzero(fold == f)
            q[iv] = materialize(task, d, it, iv, family, f"fold{f}")
        pd.DataFrame(dict(row=np.arange(n), fold=fold, Y_true=y, Y_pred=q)).to_csv(out, index=False)
        emit(stage="v19_oof", task=task, family=family)

def evaluate(task):
    dest = OUT / task; dest.mkdir(parents=True, exist_ok=True)
    if (dest / "decision.json").exists(): return
    p = protocol(); old = baseline(task); d = load_task(task); n = len(d["frames"]["train"]); nv = len(d["frames"]["valid"])
    y = old.Y_true.to_numpy(float); fold = old.fold.to_numpy(int); base = old.blend.to_numpy(float)
    valid = pd.read_csv(Path("admet_model_outputs_v16") / task / "validation_comparison.csv")
    yv = valid.Y_true.to_numpy(float); bv = valid.blend.to_numpy(float)
    candidates = []
    for family in ["v19:view_attention_v2", "v19:mol_boost_v2"]:
        c = pd.read_csv(dest / (family.split(":")[1] + "_oof.csv")); np.testing.assert_allclose(c.Y_true, y); candidates.append(c.Y_pred.to_numpy(float))
    cv = [materialize(task, d, np.arange(n), np.arange(n, n+nv), f, "valid") for f in ["v19:view_attention_v2", "v19:mol_boost_v2"]]
    best, grid = select(y, fold, base, candidates, yv, bv, cv, get_metric(task), p)
    best.update(names=["v19:view_attention_v2", "v19:mol_boost_v2"], metric=get_metric(task), protocol_sha256=sha(PROTOCOL))
    save(dest / "selection.json", best); save(dest / "selection_grid.json", grid)
    qv = combine(bv, cv, best["weights"], best["mode"], best["metric"])
    pd.DataFrame(dict(Y_true=yv, baseline=bv, blend=qv)).to_csv(dest / "validation_comparison.csv", index=False)
    emit(stage="v19_decision", task=task, weights=best["weights"], mode=best["mode"], valid_gain=score(yv, bv, qv, best["metric"]))

def publish(task):
    d = load_task(task); n = len(d["frames"]["train"]); nv = len(d["frames"]["valid"]); total = n + nv
    route = read(OUT / task / "selection.json"); old = pd.read_csv(Path("admet_submissions/v16") / next(Path("admet_submissions/v16").glob("*.csv")).name)
    base = old.loc[old.task.eq(task), "y_pred"].to_numpy(float)
    active = [i for i,w in enumerate(route["weights"][1:]) if w > 0]
    preds = [base.copy() for _ in route["names"]]
    for i in active:
        preds[i] = materialize(task, d, np.arange(total), np.arange(total, len(d["smiles"])), route["names"][i], "final")
    q = combine(base, preds, route["weights"], route["mode"], route["metric"])
    dest = ROOT / task; dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Drug_ID": old.loc[old.task.eq(task), "drug_id"].to_numpy() if "drug_id" in old else np.arange(len(q)), "Y": q}).to_csv(dest / f"submission_{task}.csv", index=False)
    route["prediction_sha256"] = sha(dest / f"submission_{task}.csv"); save(dest / "selection.json", route)

if __name__ == "__main__":
    remote_only(); ap=argparse.ArgumentParser(); ap.add_argument("stage", choices=["train","evaluate","publish"]); ap.add_argument("--tasks", nargs="*"); a=ap.parse_args()
    with threadpool_limits(limits=4):
        for t in a.tasks or names(): globals()[a.stage](t)
