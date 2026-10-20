"""v20 residual stacking: scaffold-safe correction of the verified v19/v16 anchor."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.special import expit, logit
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "admet_model_outputs_v20"
RELEASE = ROOT / "admet_release_v20"
PROTOCOL = ROOT / "admet_v20_protocol.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_for(task):
    t = pd.read_csv(ROOT / "admet_official" / "submission_template.csv", dtype=str)
    return t.loc[t.task.eq(task), "metric"].iloc[0]


def kind(metric):
    return "classification" if metric in {"AUROC", "AUPRC"} else "regression"


def utility(metric, y, p):
    if metric == "MAE":
        return -float(mean_absolute_error(y, p))
    if metric == "Spearman":
        x = spearmanr(y, p).statistic
        return float(x) if np.isfinite(x) else 0.0
    if metric == "AUROC":
        return float(roc_auc_score(y, p))
    return float(average_precision_score(y, p))


def gain(metric, y, old, new):
    d = utility(metric, y, new) - utility(metric, y, old)
    return d / max(-utility(metric, y, old), 1e-8) if metric == "MAE" else d


def combine_rank(base, correction, alpha, metric):
    q = base + alpha * correction
    if metric in {"AUROC", "AUPRC"}:
        return np.clip(q, 0.0, 1.0)
    return q


def task_names():
    return read_json(ROOT / "admet_v13_protocol.json")["tasks"]


def load_task_data(task):
    from admet_v16_models import load_task
    return load_task(task)


def feature_matrix(d, base):
    # Compact descriptors/fingerprints plus pretrained embeddings. Standardization is
    # unnecessary for CatBoost and the base prediction is an explicit residual anchor.
    x = np.concatenate([
        np.asarray(d["desc"], dtype=np.float32),
        np.asarray(d["maccs"], dtype=np.float32),
        np.asarray(d["emb"], dtype=np.float32),
    ], axis=1)
    b = np.asarray(base, dtype=np.float32).reshape(-1, 1)
    return np.concatenate([x, b], axis=1)


def v16_oof(task):
    return pd.read_csv(ROOT / "admet_model_outputs_v16" / task / "oof_comparison.csv")


def v19_route(task):
    p = ROOT / "admet_model_outputs_v19" / task / "selection.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def anchor_oof(task):
    base = v16_oof(task)
    route = v19_route(task)
    if not route or route.get("candidate_mass", 0.0) <= 0:
        return base.Y_true.to_numpy(float), base.fold.to_numpy(int), base.blend.to_numpy(float)
    # Reconstruct the exact v19 OOF blend from its candidate OOF files.
    names = route["names"]
    candidates = []
    for name in names:
        if name.startswith("v19:"):
            stem = name.split(":", 1)[1]
            p = ROOT / "admet_model_outputs_v19" / task / f"{stem}_oof.csv"
            candidates.append(pd.read_csv(p).Y_pred.to_numpy(float))
        else:
            candidates.append(base.blend.to_numpy(float))
    b = base.blend.to_numpy(float)
    w = route["weights"]
    from admet_v16_ensemble import combine
    # Each fold is a separate prediction query, just like valid and blind test.
    # Ranking across folds mixes the scales of independently fitted models.
    folds = base.fold.to_numpy(int)
    q = np.empty_like(b)
    for f in np.unique(folds):
        mask = folds == f
        q[mask] = combine(b[mask], [c[mask] for c in candidates], w,
                          route["mode"], metric_for(task))
    return base.Y_true.to_numpy(float), base.fold.to_numpy(int), q


def anchor_valid(task):
    route = v19_route(task)
    if route and route.get("candidate_mass", 0.0) > 0 and (ROOT / "admet_model_outputs_v19" / task / "validation_comparison.csv").exists():
        c = pd.read_csv(ROOT / "admet_model_outputs_v19" / task / "validation_comparison.csv")
        return c.Y_true.to_numpy(float), c.blend.to_numpy(float)
    c = pd.read_csv(ROOT / "admet_model_outputs_v16" / task / "validation_comparison.csv")
    return c.Y_true.to_numpy(float), c.blend.to_numpy(float)


def anchor_test(task):
    template = pd.read_csv(ROOT / "admet_official" / "submission_template.csv", dtype=str)
    p19 = ROOT / "admet_submissions" / "v19"
    p16 = ROOT / "admet_submissions" / "v16"
    files = list(p19.glob("*.csv")) if v19_route(task) and v19_route(task).get("candidate_mass", 0.0) > 0 else list(p16.glob("*.csv"))
    frame = pd.read_csv(files[0], dtype=str, keep_default_na=False)
    return frame.loc[frame.task.eq(task), "y_pred"].to_numpy(float)


def fit_corrector(x, residual, seed, metric):
    # Residual regression is deliberately shallow and regularized. It can only win
    # through repeatable structure, while alpha selection controls its amplitude.
    m = CatBoostRegressor(
        iterations=500,
        depth=5,
        learning_rate=0.035,
        loss_function="RMSE",
        l2_leaf_reg=18.0,
        random_strength=1.2,
        random_seed=seed,
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.12,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
    )
    m.fit(x, residual)
    return m


def train_oof(task):
    dest = OUT / task
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "residual_oof.csv"
    if out.exists():
        return
    d = load_task_data(task)
    metric = metric_for(task)
    y, folds, base = anchor_oof(task)
    n = len(y)
    x = feature_matrix(d, np.concatenate([base, np.zeros(len(d["smiles"]) - n)]))[:n]
    q = np.full(n, np.nan, dtype=float)
    models = []
    for f in sorted(np.unique(folds)):
        fit_rows = np.flatnonzero(folds != f)
        hold_rows = np.flatnonzero(folds == f)
        residual = y[fit_rows] - base[fit_rows]
        model = fit_corrector(x[fit_rows], residual, 20000 + int(f), metric)
        q[hold_rows] = model.predict(x[hold_rows])
        models.append(model)
    assert np.isfinite(q).all()
    pd.DataFrame(dict(row=np.arange(n), fold=folds, Y_true=y, baseline=base, correction=q)).to_csv(out, index=False)
    emit("v20_oof", task=task)


def evaluate(task):
    dest = OUT / task
    decision = dest / "decision.json"
    if decision.exists():
        return
    metric = metric_for(task)
    d = load_task_data(task)
    y, folds, base = anchor_oof(task)
    oof = pd.read_csv(dest / "residual_oof.csv")
    correction = oof.correction.to_numpy(float)
    vy, vb = anchor_valid(task)
    n = len(y)
    base_test = anchor_test(task)
    full_anchor = np.concatenate([base, vb, base_test])
    assert len(full_anchor) == len(d["smiles"])
    xv = feature_matrix(d, full_anchor)[n: n + len(vb)]
    x = feature_matrix(d, full_anchor)[:n]
    # Refit on all OOF residuals for official valid/test predictions.
    final = fit_corrector(x, y - base, 20999, metric)
    cv = final.predict(xv)
    grid = []
    for alpha in [0.0, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0]:
        q = combine_rank(base, correction, alpha, metric)
        fold_gains = [gain(metric, y[folds == f], base[folds == f], q[folds == f]) for f in sorted(np.unique(folds))]
        qv = combine_rank(vb, cv, alpha, metric)
        vg = gain(metric, vy, vb, qv)
        values = np.asarray(fold_gains + [vg], dtype=float)
        objective = float(np.mean(values) - 0.18 * np.std(values) - 0.25 * max(0.0, -float(values.min())))
        grid.append(dict(alpha=alpha, objective=objective, oof_gain=float(np.mean(fold_gains)), valid_gain=float(vg), fold_gains=[float(z) for z in fold_gains], min_gain=float(values.min())))
    best = max(grid, key=lambda z: (z["objective"], -abs(z["alpha"] - 0.25)))
    # Require real fold evidence; otherwise keep the anchor exactly.
    if best["objective"] <= 1e-5 or sum(z >= -0.002 for z in best["fold_gains"]) < 2:
        best = next(z for z in grid if z["alpha"] == 0.0)
    save_json(decision, dict(task=task, metric=metric, selection=best, grid=grid, candidate_mass=best["alpha"], protocol_sha256=sha(PROTOCOL), valid_prediction=cv.tolist()))
    emit("v20_decision", task=task, alpha=best["alpha"], objective=best["objective"], valid_gain=best["valid_gain"])


def publish(task):
    decision = read_json(OUT / task / "decision.json")
    alpha = decision["selection"]["alpha"]
    d = load_task_data(task)
    y, folds, base = anchor_oof(task)
    oof = pd.read_csv(OUT / task / "residual_oof.csv")
    x = feature_matrix(d, np.concatenate([base, np.zeros(len(d["smiles"]) - len(base))]))[:len(base)]
    vy, vb = anchor_valid(task)
    base_test = anchor_test(task)
    full_anchor = np.concatenate([base, vb, base_test])
    assert len(full_anchor) == len(d["smiles"])
    test_rows = np.arange(len(base) + len(vb), len(d["smiles"]))
    xt = feature_matrix(d, full_anchor)[test_rows]
    final = fit_corrector(x, y - base, 20999, metric_for(task))
    correction = final.predict(xt)
    q = combine_rank(base_test, correction, alpha, metric_for(task))
    dest = RELEASE / task
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(Y=q)).to_csv(dest / f"submission_{task}.csv", index=False)
    decision["prediction_sha256"] = sha(dest / f"submission_{task}.csv")
    save_json(dest / "selection.json", decision)


def export():
    base_file = next((ROOT / "admet_submissions" / "v19").glob("*.csv"))
    template_file = ROOT / "admet_official" / "submission_template.csv"
    rows = list(csv.reader(io.StringIO(base_file.read_bytes().decode("utf-8-sig"), newline="")))
    template = list(csv.reader(io.StringIO(template_file.read_bytes().decode("utf-8-sig"), newline="")))
    changed = []
    for task_dir in RELEASE.iterdir():
        if not task_dir.is_dir():
            continue
        p = task_dir / f"submission_{task_dir.name}.csv"
        s = task_dir / "selection.json"
        if not p.exists() or not s.exists():
            continue
        route = read_json(s)
        if route["selection"]["alpha"] <= 0:
            continue
        changed.append(task_dir.name)
        pred = pd.read_csv(p).Y.to_numpy(float)
        for row in rows[1:]:
            if row[0] == task_dir.name:
                row[4] = repr(float(pred[int(row[1])]))
    assert rows[0] == template[0] and len(rows) == len(template) == 16380
    for a, b in zip(rows[1:], template[1:]):
        assert a[:4] + a[5:] == b[:4] + b[5:] and np.isfinite(float(a[4]))
    out = ROOT / "admet_submissions" / "v20" / "我从山北来带着兰花草-小分子药物预测技术创新赛-20260926-v20.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO(newline="")
    csv.writer(buf, lineterminator="\r\n").writerows(rows)
    out.write_bytes(b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8"))
    result = dict(status="EXPORTED_AWAITING_PLATFORM", submission=str(out.resolve()), sha256=sha(out), bytes=out.stat().st_size, rows=16379, tasks=22, columns=6, encoding="UTF-8 BOM", line_endings="CRLF", baseline="v19", changed=changed, automatic_submission=False)
    save_json(out.parent / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(stage, **kwargs):
    print(json.dumps(dict(stage=stage, **kwargs), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    from admet_v6_runner import remote_only
    remote_only()
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "evaluate", "publish", "export"])
    ap.add_argument("--tasks", nargs="*")
    args = ap.parse_args()
    tasks = args.tasks or task_names()
    for task in tasks:
        if args.stage == "train": train_oof(task)
        elif args.stage == "evaluate": evaluate(task)
        elif args.stage == "publish": publish(task)
    if args.stage == "export": export()
