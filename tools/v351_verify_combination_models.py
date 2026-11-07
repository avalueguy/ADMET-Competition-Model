"""Independent replay and target-label isolation for frozen combination members."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import v349_diverse_model_combinations as experiment
from v161_verify_and_assemble import same


def design(model, version, d, k, part, config):
    if version in (168, 178, 181, 210, 216, 221, 224):
        return model.design(d, k, part)
    return model.design(d, k, part, config)


def reloaded(model, version, d, k, part, config):
    task, metric = d["tasks"][k], d["metrics"][k]
    folder = model.OUT / "models" / task / part / model.key(config)
    saved = np.load(folder / "predictions.npz")
    if version == 172:
        graphs = model.build_graphs(d)
        values = []
        for seed in model.SEEDS:
            file = folder / f"model_{seed}.pt"
            state = model.torch.load(file, weights_only=True)
            assert state["signature"] == model.sig()
            net = model.Net(graphs[0]["x"].shape[1], graphs[0]["e"].shape[1], config["hidden"], config["layers"]).cuda()
            net.load_state_dict(state["state"])
            values.append(model.infer(net, graphs, d["mi"][saved["query"]], "cuda"))
        raw = np.mean(values, axis=0)
        p = expit(raw) if metric in ("AUROC", "AUPRC") else raw * float(saved["scale"]) + float(saved["center"])
        return p.clip(0, 100) if task == "ppbr_az" else p
    if version == 168:
        a = model.design(d, k, part)
        stored = joblib.load(folder / "transform.joblib")
        same(stored, {key:value for key, value in a.items() if key not in ("x", "qx")})
        net = joblib.load(folder / "model.joblib")
        raw = net.decision_function(a["qx"]) if metric in ("AUROC", "AUPRC") else net.predict(a["qx"])
        p = expit(raw) if metric in ("AUROC", "AUPRC") else raw * float(saved["scale"]) + float(saved["center"])
        return p.clip(0, 100) if task == "ppbr_az" else p
    if version == 178:
        import xgboost as xgb
        a = model.design(d, k, part)
        values = []
        for seed in model.SEEDS:
            net = xgb.Booster()
            net.load_model(folder / f"model_{seed}.ubj")
            assert net.num_boosted_rounds() == config["trees"]
            values.append(net.predict(xgb.DMatrix(a["qx"], nthread=2)))
        p = np.mean(values, axis=0).astype(float) * float(saved["scale"]) + float(saved["center"])
        return p.clip(0, 100) if task == "ppbr_az" else p
    if version == 181:
        a = model.design(d, k, part)
        forest = joblib.load(folder / "forest.joblib")
        p = model.quantile(forest, a["x"], a["qx"], d["y"][a["fit"]])
        return p.clip(0, 100) if task == "ppbr_az" else p
    result = model.replay(d, k, part, config)
    return result[0] if isinstance(result, tuple) else result


def check(d, k, part, record, mods):
    model, version, config = mods[record["version"]], record["version"], record["config"]
    expected, provenance = experiment.cached(d, k, part, record, mods)
    replay = reloaded(model, version, d, k, part, config)
    np.testing.assert_allclose(replay, expected, rtol=0, atol=1e-12)
    fit, query = experiment.v.split(d, k, part)
    # Full nonfit target perturbation, including other folds and official validation.
    if version == 172:
        original = model.build_graphs(d)
        labels = d["y"].copy()
        try:
            d["y"][(d["tid"] == k) & ~np.isin(np.arange(len(labels)), fit)] = 999999.
            same(original, model.build_graphs(d))
        finally:
            d["y"][:] = labels
    elif version == 178:
        a, ids = model.embedding_design(d, part)
        assert not set(d["canonical"][query]) & set(d["canonical"][a["source"]])
        if part != "final":
            assert not set(d["groups"][query]) & set(d["groups"][a["source"]])
        labels = d["y"].copy()
        try:
            d["y"][(d["tid"] == k) & ~np.isin(np.arange(len(labels)), fit)] = 999999.
            altered, altered_ids = model.embedding_design(d, part)
            same(a, altered)
            np.testing.assert_array_equal(ids, altered_ids)
        finally:
            d["y"][:] = labels
    else:
        a = design(model, version, d, k, part, config)
        labels = d["y"].copy()
        try:
            d["y"][(d["tid"] == k) & ~np.isin(np.arange(len(labels)), fit)] = 999999.
            d.pop("_v96_assays", None)
            same(a, design(model, version, d, k, part, config))
        finally:
            d["y"][:] = labels
            d.pop("_v96_assays", None)
    folder = model.OUT / "models" / d["tasks"][k] / part / model.key(config)
    hashes = {p.relative_to(ROOT).as_posix():experiment.sha(p) for p in folder.iterdir() if p.is_file()}
    return dict(task=d["tasks"][k], part=part, version=version, rows=len(query),
        saved_prediction=provenance, saved_model_reload=True, nonfit_target_label_isolation=True,
        exact_split_alignment=True, model_files=hashes)


def load_data(required, mods):
    d = experiment.v.load()
    for version in (168, 221, 250):
        if version not in required:
            continue
        other = mods[version].load()
        for key in ("tasks", "metrics", "y", "tid", "split", "mi", "groups", "canonical"):
            np.testing.assert_array_equal(d[key], other[key])
        for key, value in other.items():
            if key not in d:
                d[key] = value
    experiment.sam.original.initialize()
    return d


if __name__ == "__main__":
    raise SystemExit("Use from the frozen delivery workflow; no automatic model selection.")
