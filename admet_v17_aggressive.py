"""Build an aggressive v17 candidate from remotely verified v15/v16 predictions."""
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

ROOT = Path(__file__).resolve().parent
TASKS = pd.read_csv(ROOT / "admet_official" / "submission_template.csv", dtype=str)["task"].drop_duplicates().tolist()
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]


def metric_name(task):
    t = pd.read_csv(ROOT / "admet_official" / "submission_template.csv", dtype=str)
    return t.loc[t.task.eq(task), "metric"].iloc[0]


def utility(metric, y, p):
    if metric == "MAE":
        return -float(mean_absolute_error(y, p))
    if metric == "Spearman":
        r = spearmanr(y, p).statistic
        return float(r) if np.isfinite(r) else 0.0
    if metric == "AUROC":
        return float(roc_auc_score(y, p))
    if metric == "AUPRC":
        return float(average_precision_score(y, p))
    raise ValueError(metric)


def gain(metric, y, old, new):
    delta = utility(metric, y, new) - utility(metric, y, old)
    return delta / max(-utility(metric, y, old), 1e-8) if metric == "MAE" else delta


def extrapolate(old, new, alpha, metric):
    q = old + alpha * (new - old)
    if metric in {"AUROC", "AUPRC"}:
        q = np.clip(q, 0.0, 1.0)
    return q


def load_pair(task):
    p15 = pd.read_csv(ROOT / "admet_release_v15" / task / f"submission_{task}.csv")
    p16 = pd.read_csv(ROOT / "admet_release_v16" / task / f"submission_{task}.csv")
    assert p15.Drug_ID.tolist() == p16.Drug_ID.tolist()
    return p15, p16


def evaluate_alpha(task, alpha):
    metric = metric_name(task)
    comp = pd.read_csv(ROOT / "admet_model_outputs_v16" / task / "oof_comparison.csv")
    valid = pd.read_csv(ROOT / "admet_model_outputs_v16" / task / "validation_comparison.csv")
    q_oof = extrapolate(comp.baseline.to_numpy(float), comp.blend.to_numpy(float), alpha, metric)
    fold_gains = []
    for fold in sorted(comp.fold.unique()):
        m = comp.fold.to_numpy() == fold
        fold_gains.append(gain(metric, comp.Y_true.to_numpy(float)[m], comp.baseline.to_numpy(float)[m], q_oof[m]))
    q_valid = extrapolate(valid.baseline.to_numpy(float), valid.blend.to_numpy(float), alpha, metric)
    valid_gain = gain(metric, valid.Y_true.to_numpy(float), valid.baseline.to_numpy(float), q_valid)
    repeat_path = ROOT / "admet_model_outputs_v16" / task / "repeat_comparison.csv"
    repeat_gain = 0.0
    if repeat_path.exists():
        repeat = pd.read_csv(repeat_path)
        q_repeat = extrapolate(repeat.baseline.to_numpy(float), repeat.blend.to_numpy(float), alpha, metric)
        repeat_gain = gain(metric, repeat.Y_true.to_numpy(float), repeat.baseline.to_numpy(float), q_repeat)
    score = 0.45 * float(np.mean(fold_gains)) + 0.35 * valid_gain + 0.20 * repeat_gain
    return dict(alpha=alpha, score=score, oof_gain=float(np.mean(fold_gains)),
                valid_gain=float(valid_gain), repeat_gain=float(repeat_gain),
                fold_gains=[float(x) for x in fold_gains])


def main():
    out = ROOT / "admet_release_v17"
    out.mkdir(exist_ok=True)
    rows = []
    routes = []
    for task in TASKS:
        metric = metric_name(task)
        grid = [evaluate_alpha(task, a) for a in ALPHAS]
        best = max(grid, key=lambda x: (x["score"], x["alpha"]))
        # Do not move against the v15->v16 direction unless every observed split rejects it.
        alpha_one = next(x for x in grid if x["alpha"] == 1.0)
        if alpha_one["score"] <= 0 and best["score"] < 0:
            best = next(x for x in grid if x["alpha"] == 0.0)
        p15, p16 = load_pair(task)
        pred = extrapolate(p15.Y.to_numpy(float), p16.Y.to_numpy(float), best["alpha"], metric)
        (out / task).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Drug_ID": p16.Drug_ID, "Y": pred}).to_csv(out / task / f"submission_{task}.csv", index=False)
        route = dict(task=task, metric=metric, selected=best, alpha_one=alpha_one, grid=grid,
                     changed=bool(best["alpha"] != 0.0), source_v15=str(p15.shape), source_v16=str(p16.shape))
        (out / task / "selection.json").write_text(json.dumps(route, ensure_ascii=False, indent=2), encoding="utf-8")
        routes.append(route)
        rows.append({"task": task, "metric": metric, "alpha": best["alpha"], "score": best["score"],
                     "oof_gain": best["oof_gain"], "valid_gain": best["valid_gain"], "repeat_gain": best["repeat_gain"],
                     "alpha1_score": alpha_one["score"]})
    pd.DataFrame(rows).to_csv(out / "v17_selection.csv", index=False)
    (out / "routing.json").write_text(json.dumps(routes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
