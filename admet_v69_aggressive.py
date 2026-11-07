"""Aggressive official-only GPU tree and structure/neural ensemble development."""
from __future__ import annotations
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "4"
import argparse
import csv
import hashlib
import io
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.special import expit

import admet_v67_data as b
import admet_v68_kernel as kernel

ROOT = b.ROOT
OUT = ROOT / "admet_model_outputs_v69"
P = json.loads((ROOT / "admet_v69_protocol.json").read_text(encoding="utf-8"))
SOURCES = ["shared", "gated", "kernel0", "kernel1", "kernel2", "kernel3", "descriptor", "hybrid"]


def signature():
    return hashlib.sha256((kernel.signature()+b.sha(__file__)+b.sha(ROOT/"admet_v69_protocol.json")).encode()).hexdigest()


def splits(d, k, partition):
    task = d["tid"] == k
    if partition.startswith("fold"):
        folds = np.load(b.OUT/"folds.npy")
        fold = int(partition[-1])
        fit = np.flatnonzero(task & (d["split"] == 0) & (folds != fold))
        query = np.flatnonzero(task & (folds == fold))
    elif partition == "valid":
        groups = np.unique(d["rg"][d["split"] == 1])
        fit = np.flatnonzero(task & (d["split"] == 0) & ~np.isin(d["rg"], groups))
        query = np.flatnonzero(task & (d["split"] == 1))
    elif partition == "final":
        fit = np.flatnonzero(task & (d["split"] < 2))
        query = np.flatnonzero(task & (d["split"] == 2))
    else:
        raise ValueError(partition)
    b.assert_excluded(d, fit, query)
    return fit, query


def features(d, rows, query, profile, state=None):
    state = state or b.preprocess(d, rows)
    def make(ix):
        mm = d["mi"][ix]
        desc = d["x"][mm, 4096:]
        desc = np.where(np.isfinite(desc), desc, state["median"])
        desc = ((desc-state["mean"])/state["scale"]).clip(-8, 8)
        if profile["fingerprints"]:
            raw = d["x"][mm, :4096]
            # Fold each 2048-bit view to 512 columns by summing log counts.
            fp = np.concatenate([raw[:, :2048].reshape(-1, 4, 512).sum(1),
                                 raw[:, 2048:].reshape(-1, 4, 512).sum(1)], axis=1)
            return np.concatenate([fp, desc], axis=1).astype(np.float32)
        return desc.astype(np.float32)
    return make(rows), make(query), state


def predict_model(model, x, metric, target):
    if metric in ("AUROC", "AUPRC"):
        return model.predict_proba(x)[:, 1]
    p = model.predict(x)
    return p*target["scale"]+target["center"] if metric == "MAE" else p


def fit_tree(d, k, partition, profile, iterations=None, seed=None):
    rows, query = splits(d, k, partition)
    seed = P["development_seed"] if seed is None else seed
    dest = OUT/d["tasks"][k]/partition/profile["name"]/str(seed)
    dest.mkdir(parents=True, exist_ok=True)
    cache = dest/"predictions.npz"
    if cache.exists():
        metadata = json.loads((dest/"metadata.json").read_text(encoding="utf-8"))
        assert metadata["signature"] == signature()
        assert metadata["model_sha256"] == b.sha(dest/"model.cbm")
        assert (dest/"preprocessing.npz").is_file()
        with np.load(cache, allow_pickle=False) as z:
            assert str(z["signature"]) == signature()
            np.testing.assert_array_equal(z["fit"], rows)
            np.testing.assert_array_equal(z["query"], query)
            return z["p"].copy()
    started = time.monotonic()
    metric = d["metrics"][k]
    tf = b.target(d["y"][rows], metric)
    y = b.encode(d["y"][rows], tf)
    x, qx, state = features(d, rows, query, profile)
    _, inv, counts = np.unique(d["canonical"][d["mi"][rows]], return_inverse=True, return_counts=True)
    weights = 1/counts[inv]; weights /= weights.mean()
    cls = metric in ("AUROC", "AUPRC")
    loss = "Logloss" if cls else (profile["mae_loss"] if metric == "MAE" else "RMSE")
    kwargs = dict(iterations=iterations or profile["iterations"], depth=profile["depth"],
                  learning_rate=profile["learning_rate"], l2_leaf_reg=profile["l2_leaf_reg"],
                  loss_function=loss, task_type="GPU", devices="0", random_seed=seed,
                  thread_count=4, verbose=False, allow_writing_files=False, border_count=64,
                  bootstrap_type="Bernoulli", subsample=.85)
    model = (CatBoostClassifier if cls else CatBoostRegressor)(**kwargs)
    if partition == "final":
        assert iterations is not None
        model.fit(x, y, sample_weight=weights)
    else:
        model.fit(x, y, sample_weight=weights,
                  eval_set=(qx, b.encode(d["y"][query], tf)), use_best_model=True,
                  early_stopping_rounds=P["early_stopping_rounds"])
    p = predict_model(model, qx, metric, tf)
    assert np.isfinite(p).all()
    model.save_model(str(dest/"model.cbm"))
    restored = (CatBoostClassifier if cls else CatBoostRegressor)()
    restored.load_model(str(dest/"model.cbm"))
    np.testing.assert_allclose(p, predict_model(restored, qx, metric, tf), atol=1e-10, rtol=1e-10)
    np.savez_compressed(dest/"preprocessing.npz", **state)
    target_json = {key: value.tolist() if isinstance(value, np.ndarray) else value
                   for key, value in tf.items()}
    b.save(dest/"metadata.json", dict(signature=signature(), task=d["tasks"][k], metric=metric,
        target=target_json, profile=profile, seed=seed, trees=model.tree_count_, requested_iterations=iterations,
        model_sha256=b.sha(dest/"model.cbm"), reload_passed=True, fit_rows=len(rows), query_rows=len(query),
        validation_used_for_early_stopping=partition!="final", seconds=time.monotonic()-started))
    # Publish a usable cache only after its model, preprocessing and JSON metadata are complete.
    temporary = dest/"predictions.tmp.npz"
    np.savez_compressed(temporary, fit=rows, query=query, p=p, signature=signature())
    temporary.replace(cache)
    b.emit("v69_tree", task=d["tasks"][k], partition=partition, profile=profile["name"],
           seed=seed, trees=model.tree_count_, seconds=round(time.monotonic()-started))
    return p


def prepare(d):
    for k in sorted(range(len(d["tasks"])), key=lambda k:np.sum((d["tid"]==k)&(d["split"]==0))):
        for partition in P["splits"]:
            rows, query = splits(d, k, partition)
            # Reuse only matching frozen kernel partitions, never alter v68 selections.
            name = "valid_report" if partition == "valid" else partition
            kernel.fit(d, rows, query, d["tasks"][k], name)
            for profile in P["families"]:
                fit_tree(d, k, partition, profile)
    b.save(OUT/"prepare_completed.json", dict(status="COMPLETE", signature=signature()))


def neural(d, query, partition, family):
    part = "cv/"+partition if partition.startswith("fold") else partition
    values = []
    for seed in b.P["training_seeds"]:
        with np.load(b.OUT/part/family/str(seed)/"predictions.npz", allow_pickle=False) as z:
            assert str(z["signature"]) == b.signature()
            pos = {int(row):i for i,row in enumerate(z["query"])}
            values.append(z["p"][[pos[int(row)] for row in query]])
    return np.mean(values, 0)


def source_predictions(d, k, partition, final_counts=None):
    rows, query = splits(d, k, partition)
    result = {name:neural(d, query, partition, name) for name in ("shared", "gated")}
    name = "valid_report" if partition == "valid" else partition
    kernels = kernel.fit(d, rows, query, d["tasks"][k], name)
    result.update({"kernel"+str(j):p for j,p in enumerate(kernels)})
    for profile in P["families"]:
        if partition == "final":
            result[profile["name"]] = np.mean([fit_tree(d,k,partition,profile,
                final_counts[profile["name"]],seed) for seed in P["final_seeds"]],0)
        else:
            result[profile["name"]] = fit_tree(d,k,partition,profile)
    rec67 = json.loads((b.OUT/"fresh.json").read_text())["tasks"][k]
    old = b.blend(d["metrics"][k],result["shared"],result["gated"],rec67["final_weight"])
    rec68 = json.loads((kernel.OUT/"selection.json").read_text())["tasks"][k]
    if rec68["pass_gate"]:
        choice = rec68["selection"]
        old = b.blend(d["metrics"][k],old,result["kernel"+str(choice["variant"])],choice["weight"])
    if d["tasks"][k] == "ppbr_az": old = old.clip(0,100)
    return query, old, result


def apply(metric, old, sources, recipe):
    result = old.copy()
    for entry in recipe:
        result = b.blend(metric,result,sources[entry["source"]],entry["weight"])
    return result


def select(d):
    assert json.loads((OUT/"prepare_completed.json").read_text())["signature"] == signature()
    records=[]
    for k, task in enumerate(d["tasks"]):
        metric = d["metrics"][k]
        data = [source_predictions(d,k,part) for part in P["splits"]]
        def evaluate(recipe):
            gains = [b.gain(metric,d["y"][q],old,apply(metric,old,sources,recipe)) for q,old,sources in data]
            objective=P["selection"]["cv_weight"]*np.mean(gains[:3])+P["selection"]["valid_weight"]*gains[3]-P["selection"]["fold_sd_penalty"]*np.std(gains[:3])
            return dict(gains=gains,cv_mean=float(np.mean(gains[:3])),objective=float(objective))
        recipe=[]; current=evaluate(recipe); history=[]
        for _ in range(P["selection"]["rounds"]):
            trials=[]
            for source in SOURCES:
                for weight in P["selection"]["weights"]:
                    proposed=recipe+[dict(source=source,weight=weight)]
                    trials.append((evaluate(proposed),proposed))
            best, proposed=max(trials,key=lambda pair:pair[0]["objective"])
            if best["objective"] <= current["objective"]+P["selection"]["minimum_objective_step"]: break
            recipe,current=proposed,best; history.append(best)
        counts={}
        for profile in P["families"]:
            trees=[json.loads((OUT/task/part/profile["name"]/str(P["development_seed"])/"metadata.json").read_text())["trees"] for part in P["splits"]]
            counts[profile["name"]]=max(50,int(np.median(trees)))
        record=dict(task=task,metric=metric,recipe=recipe,development=current,history=history,final_tree_counts=counts)
        records.append(record); b.emit("v69_selection",**record)
    b.save(OUT/"selection.json",dict(status="FROZEN",signature=signature(),tasks=records,
        official_valid_used_for_tuning=True,independent_generalization_estimate=False))
    return records


def export(d):
    selection=json.loads((OUT/"selection.json").read_text())
    assert selection["signature"]==signature()
    release68=json.loads((kernel.OUT/"release.json").read_text())
    source=ROOT/release68["file"]
    assert b.sha(source)==release68["sha256"]
    with source.open(encoding="utf-8-sig",newline="") as stream:rows=list(csv.reader(stream))
    replay=[]
    for k,record in enumerate(selection["tasks"]):
        if not record["recipe"]:continue
        q,old,sources=source_predictions(d,k,"final",record["final_tree_counts"])
        positions=[i for i,row in enumerate(rows[1:],1) if row[0]==record["task"]]
        baseline=np.array([float(rows[i][4]) for i in positions])
        np.testing.assert_allclose(old,baseline,atol=1e-12,rtol=1e-12)
        result=apply(record["metric"],baseline,sources,record["recipe"])
        if record["task"]=="ppbr_az":result=result.clip(0,100)
        if record["metric"] in ("AUROC","AUPRC"):result=result.clip(0,1)
        for i,value in zip(positions,result):rows[i][4]=repr(float(value))
        replay.append(dict(task=record["task"],recipe=record["recipe"],baseline=baseline.tolist(),result=result.tolist()))
    date=datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")
    dest=ROOT/"admet_submissions/v69_aggressive_official_only";dest.mkdir(parents=True,exist_ok=True)
    file=dest/f"我从山北来带着兰花草-小分子药物预测技术创新赛-{date}.csv"
    buf=io.StringIO(newline="");csv.writer(buf,lineterminator="\r\n").writerows(rows)
    raw=b"\xef\xbb\xbf"+buf.getvalue().encode("utf-8")
    if file.exists():assert file.read_bytes()==raw
    else:file.write_bytes(raw)
    b.save(OUT/"fusion_replay.json",replay)
    b.save(OUT/"release.json",dict(status="AWAITING_PLATFORM_EVALUATION",signature=signature(),
        file=file.relative_to(ROOT).as_posix(),sha256=b.sha(file),baseline_sha256=b.sha(source),rows=len(rows)-1,
        tasks=len(selection["tasks"]),changed_tasks=[r["task"] for r in replay],external_data=False,
        pretrained_weights=False,official_valid_used_for_tuning=True,official_score=None))
    b.emit("v69_export",file=str(file),sha256=b.sha(file))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("stage",choices=["prepare","select","export"]);args=parser.parse_args()
    assert os.name!="nt", "Heavy fits run only on the authorized CUDA server"
    d=b.build();OUT.mkdir(parents=True,exist_ok=True)
    {"prepare":prepare,"select":select,"export":export}[args.stage](d)
