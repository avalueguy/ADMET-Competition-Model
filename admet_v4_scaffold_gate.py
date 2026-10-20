"""Scaffold gate for the cross-task transfer challenger.

The v4 transfer run selected candidates on the public valid split.  This
script evaluates those same candidates on the fixed Bemis--Murcko scaffold
holdout used by v3 before anything is allowed into a release directory.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier, XGBRegressor

from admet_competition_train import predict_model
from admet_features_v2 import features
from admet_transfer_upgrade import REL, transfer


def kind(df):
    y = df.Y.to_numpy(float)
    return "classification" if set(np.unique(y)).issubset({0, 1}) else "regression"


def mm(k, y, p):
    p = np.asarray(p, float)
    if k == "classification":
        return {"auroc": float(roc_auc_score(y, p)),
                "auprc": float(average_precision_score(y, p))}
    r = float(spearmanr(y, p).statistic)
    return {"mae": float(mean_absolute_error(y, p)),
            "spearman": r if np.isfinite(r) else 0.0}


def dominates(k, old, new):
    if k == "classification":
        return new["auroc"] >= old["auroc"] and new["auprc"] >= old["auprc"]
    return new["mae"] <= old["mae"] and new["spearman"] >= old["spearman"]


def head_model(k, name):
    if k == "classification":
        if name == "xgb":
            return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=.035,
                                 min_child_weight=2, subsample=.9, colsample_bytree=.7,
                                 reg_lambda=4, tree_method="hist", n_jobs=6,
                                 random_state=4201)
        if name == "extra":
            return ExtraTreesClassifier(n_estimators=400, max_features=.3,
                                        class_weight="balanced", n_jobs=6,
                                        random_state=4201)
        return LogisticRegression(C=.2, class_weight="balanced", solver="liblinear",
                                  max_iter=2000)
    if name == "xgb":
        return XGBRegressor(n_estimators=550, max_depth=4, learning_rate=.03,
                            min_child_weight=2, subsample=.9, colsample_bytree=.7,
                            reg_lambda=5, tree_method="hist", n_jobs=6,
                            random_state=4201)
    if name == "extra":
        return ExtraTreesRegressor(n_estimators=400, max_features=.3, n_jobs=6,
                                   random_state=4201)
    return Ridge(alpha=40)


def predict_head(model, name, x, k):
    xx = x[:, 4096:] if name == "linear" else x
    return predict_model(model, xx, k)


def main():
    root = Path("admet_competition_data/data_benchmark")
    release = Path("admet_release")
    v4 = Path("admet_model_outputs_v4")
    out = Path("admet_v4_scaffold_gate")
    out.mkdir(exist_ok=True)
    tasks = {p.name: {s: pd.read_csv(p / f"{s}.csv") for s in ("train", "valid", "test")}
             for p in root.iterdir() if (p / "train.csv").exists()}
    transfer_metrics = pd.read_csv(v4 / "transfer_metrics.csv")
    rows = []
    cache = v4 / "scaffold_features"
    for row in transfer_metrics.itertuples(index=False):
        task = row.task
        if not bool(row.accepted):
            continue
        f = tasks[task]
        tr = f["train"]
        k = kind(tr)
        y = tr.Y.to_numpy(float)
        all_smi = tr.Drug.tolist()
        x, _, scaffolds = features(all_smi, cache)
        fit_idx, hold_idx = next(GroupShuffleSplit(n_splits=1, test_size=.2,
                                                   random_state=20260924).split(
                                                       np.arange(len(tr)), y,
                                                       groups=np.asarray(scaffolds)))
        # v3 release scaffold rows are in this exact holdout order.
        old_sc = pd.read_csv(release / task / "scaffold_predictions.csv")
        if len(old_sc) != len(hold_idx):
            raise RuntimeError(f"scaffold length mismatch for {task}")
        if not np.allclose(old_sc.Y_true.to_numpy(float), y[hold_idx]):
            raise RuntimeError(f"scaffold ordering mismatch for {task}")
        base = old_sc.Y_pred.to_numpy(float)
        base_m = mm(k, y[hold_idx], base)

        # Build transfer features using only observed training labels from the
        # public data. The target model itself is fitted on the scaffold-fit
        # subset, while source-task labels remain available as auxiliary data.
        x_all, t_all = transfer(task, all_smi, tasks, cache)
        x_all = np.nan_to_num(np.hstack([x_all, t_all]), nan=0., posinf=0., neginf=0.).astype(np.float32)
        name = row.head
        names = ["xgb", "extra", "linear"] if name == "equal" else [name]
        preds = []
        for h in names:
            model = head_model(k, h)
            model.fit(x_all[fit_idx, 4096:] if h == "linear" else x_all[fit_idx], y[fit_idx])
            preds.append(predict_head(model, h, x_all[hold_idx], k))
        cand = np.mean(preds, axis=0) if name == "equal" else preds[0]
        alpha = float(row.alpha)
        selected = (1.0 - alpha) * base + alpha * cand
        cand_m = mm(k, y[hold_idx], selected)
        accepted = dominates(k, base_m, cand_m)
        out_task = out / task
        out_task.mkdir(exist_ok=True)
        pd.DataFrame({"Y_true": y[hold_idx], "baseline": base,
                      "candidate_raw": cand, "selected": selected}).to_csv(
                          out_task / "scaffold_predictions.csv", index=False)
        rec = {"task": task, "kind": k, "head": name, "alpha": alpha,
               "accepted": bool(accepted)}
        rec.update({"baseline_" + a: b for a, b in base_m.items()})
        rec.update({"candidate_" + a: b for a, b in cand_m.items()})
        rows.append(rec)
        print(json.dumps(rec), flush=True)
    pd.DataFrame(rows).to_csv(out / "scaffold_metrics.csv", index=False)

    # Freeze a v4 release only for candidates that pass this independent gate.
    final = Path("admet_release_v4")
    if final.exists():
        shutil.rmtree(final)
    shutil.copytree(release, final)
    gate = {r["task"]: r for r in rows}
    for task, r in gate.items():
        if r["accepted"]:
            for fn in (f"submission_{task}.csv", "validation_predictions.csv"):
                shutil.copy2(v4 / task / fn, final / task / fn)
            # Preserve the gated scaffold predictions as part of the release
            # audit so the final comparison reflects the actual routing.
            shutil.copy2(out / task / "scaffold_predictions.csv",
                         final / task / "scaffold_predictions.csv")
    # Keep a machine-readable audit of the final routing decision.
    audit = []
    for task in sorted(tasks):
        if task in gate:
            audit.append(gate[task])
        else:
            audit.append({"task": task, "accepted": False,
                          "reason": "v4 valid gate rejected or unchanged"})
    pd.DataFrame(audit).to_csv(final / "v4_scaffold_audit.csv", index=False)
    print(f"final_release={final}")


if __name__ == "__main__":
    main()
