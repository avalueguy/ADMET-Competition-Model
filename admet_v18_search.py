"""Robust v18 search along the verified v15->v16 prediction trajectory."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

ROOT = Path(__file__).resolve().parent
TASKS = pd.read_csv(ROOT / "admet_official" / "submission_template.csv", dtype=str)["task"].drop_duplicates().tolist()
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

def utility(metric, y, p):
    if metric == "MAE": return -float(mean_absolute_error(y, p))
    if metric == "Spearman":
        v = spearmanr(y, p).statistic
        return float(v) if np.isfinite(v) else 0.0
    if metric == "AUROC": return float(roc_auc_score(y, p))
    return float(average_precision_score(y, p))

def gain(metric, y, old, new):
    d = utility(metric, y, new) - utility(metric, y, old)
    return d / max(-utility(metric, y, old), 1e-8) if metric == "MAE" else d

def qline(base, v16, alpha, metric):
    q = base + alpha * (v16 - base)
    return np.clip(q, 0., 1.) if metric in {"AUROC", "AUPRC"} else q

def evaluate(task, metric, alpha):
    comp = pd.read_csv(ROOT / "admet_model_outputs_v16" / task / "oof_comparison.csv")
    y = comp.Y_true.to_numpy(float); b = comp.baseline.to_numpy(float); v = comp.blend.to_numpy(float)
    q = qline(b, v, alpha, metric)
    folds = [gain(metric, y[comp.fold.to_numpy() == f], b[comp.fold.to_numpy() == f], q[comp.fold.to_numpy() == f]) for f in sorted(comp.fold.unique())]
    valid = pd.read_csv(ROOT / "admet_model_outputs_v16" / task / "validation_comparison.csv")
    vy = valid.Y_true.to_numpy(float); vb = valid.baseline.to_numpy(float); vv = valid.blend.to_numpy(float)
    vg = gain(metric, vy, vb, qline(vb, vv, alpha, metric))
    repeat_path = ROOT / "admet_model_outputs_v16" / task / "repeat_comparison.csv"
    rg = 0.0
    if repeat_path.exists():
        rep = pd.read_csv(repeat_path); ry = rep.Y_true.to_numpy(float); rb = rep.baseline.to_numpy(float); rv = rep.blend.to_numpy(float)
        rg = gain(metric, ry, rb, qline(rb, rv, alpha, metric))
    vals = np.asarray(folds + [vg, rg], float)
    # Reward agreement across splits and penalize a single strongly negative split.
    score = float(np.mean(vals) - 0.20 * np.std(vals) - 0.25 * max(0., -float(np.min(vals))))
    return dict(alpha=alpha, score=score, oof_gain=float(np.mean(folds)), valid_gain=float(vg), repeat_gain=float(rg), fold_gains=[float(x) for x in folds], min_gain=float(np.min(vals)))

def main():
    template = pd.read_csv(ROOT / "admet_official" / "submission_template.csv", dtype=str)
    routes = []; rows = []
    out_root = ROOT / "admet_release_v18"
    for task in TASKS:
        metric = template.loc[template.task.eq(task), "metric"].iloc[0]
        grid = [evaluate(task, metric, a) for a in ALPHAS]
        anchor = next(x for x in grid if x["alpha"] == 1.0)
        best = max(grid, key=lambda x: (x["score"], -abs(x["alpha"] - 1.0)))
        # A new trajectory point must beat the exact v16 anchor by a meaningful margin.
        if best["score"] <= anchor["score"] + 1e-5:
            best = anchor
        p15 = pd.read_csv(ROOT / "admet_release_v15" / task / f"submission_{task}.csv")
        p16 = pd.read_csv(ROOT / "admet_release_v16" / task / f"submission_{task}.csv")
        assert p15.Drug_ID.tolist() == p16.Drug_ID.tolist()
        pred = qline(p15.Y.to_numpy(float), p16.Y.to_numpy(float), best["alpha"], metric)
        dest = out_root / task; dest.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Drug_ID": p16.Drug_ID, "Y": pred}).to_csv(dest / f"submission_{task}.csv", index=False)
        route = dict(task=task, metric=metric, selected=best, anchor=anchor, grid=grid)
        (dest / "selection.json").write_text(json.dumps(route, ensure_ascii=False, indent=2), encoding="utf-8")
        routes.append(route)
        rows.append(dict(task=task, metric=metric, alpha=best["alpha"], score=best["score"], anchor_score=anchor["score"], oof_gain=best["oof_gain"], valid_gain=best["valid_gain"], repeat_gain=best["repeat_gain"], min_gain=best["min_gain"]))
    pd.DataFrame(rows).to_csv(out_root / "v18_selection.csv", index=False)
    (out_root / "routing.json").write_text(json.dumps(routes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__": main()
