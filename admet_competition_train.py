"""Train strong SMILES based ADMET models on the supplied TDC-style archive.

The archive contains one independent train/valid/test triplet per ADMET task.
This script deliberately keeps the feature engineering shared across tasks and
the target heads task-specific: the supplied files do not form a rectangular
multi-label matrix, so a conventional dense multi-task loss would discard most
of the available labels.

Outputs are written below ``admet_model_outputs`` by default:
  * validation_metrics.csv: AUROC/AUPRC for classification and MAE/Spearman
    for regression, together with duplicate-lookup coverage;
  * one submission_<task>.csv for every task (Drug_ID,Y);
  * model bundles and feature caches for reproducible inference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

SEED = 42
FP_BITS = 2048
MORGAN = GetMorganGenerator(radius=2, fpSize=FP_BITS)

# A compact, stable descriptor panel. The complete RDKit descriptor list is
# useful for exploration but contains several fragile/highly correlated calls;
# these descriptors cover size, polarity, lipophilicity, H-bonding, rings and
# rotatable bonds and work well with tree and linear heads.
DESC_FUNCS = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Crippen.MolLogP),
    ("MolMR", Crippen.MolMR),
    ("TPSA", rdMolDescriptors.CalcTPSA),
    ("HBD", Lipinski.NumHDonors),
    ("HBA", Lipinski.NumHAcceptors),
    ("RotB", Lipinski.NumRotatableBonds),
    ("AromRings", Lipinski.NumAromaticRings),
    ("Rings", Lipinski.RingCount),
    ("HeavyAtoms", Lipinski.HeavyAtomCount),
    ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
    ("FormalCharge", Chem.GetFormalCharge),
    ("NumHetero", Lipinski.NumHeteroatoms),
    ("NumAliphaticRings", Lipinski.NumAliphaticRings),
    ("NumSaturatedRings", Lipinski.NumSaturatedRings),
    ("NumAliphaticHeterocycles", Lipinski.NumAliphaticHeterocycles),
    ("NumAromaticHeterocycles", Lipinski.NumAromaticHeterocycles),
]


def canonicalize(smiles: object) -> str | None:
    if smiles is None or (isinstance(smiles, float) and math.isnan(smiles)):
        return None
    text = str(smiles).strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def featurize_smiles(smiles: list[object]) -> tuple[np.ndarray, list[str | None], dict[str, int]]:
    """Return Morgan bits + descriptors, canonical SMILES and parse counts."""
    rows: list[np.ndarray] = []
    canon: list[str | None] = []
    invalid = 0
    for value in smiles:
        mol = Chem.MolFromSmiles(str(value)) if value is not None else None
        if mol is None:
            invalid += 1
            canon.append(None)
            rows.append(np.zeros(FP_BITS + len(DESC_FUNCS), dtype=np.float32))
            continue
        canon.append(Chem.MolToSmiles(mol, canonical=True))
        fp = np.asarray(MORGAN.GetFingerprintAsNumPy(mol), dtype=np.float32)
        desc = []
        for _, fn in DESC_FUNCS:
            try:
                desc.append(float(fn(mol)))
            except Exception:
                desc.append(np.nan)
        row = np.concatenate([fp, np.asarray(desc, dtype=np.float32)])
        rows.append(row)
    x = np.asarray(rows, dtype=np.float32)
    x[~np.isfinite(x)] = np.nan
    # Impute descriptors only; fingerprint columns contain no missing values.
    x[:, FP_BITS:] = SimpleImputer(strategy="median").fit_transform(x[:, FP_BITS:])
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return x, canon, {"n_total": len(smiles), "n_invalid": invalid}


def task_type(y: pd.Series) -> str:
    numeric = pd.to_numeric(y, errors="coerce").dropna().to_numpy()
    unique = np.unique(numeric)
    if len(unique) <= 2 and set(unique.tolist()).issubset({0, 1}):
        return "classification"
    return "regression"


def fit_candidates(kind: str, y: np.ndarray) -> dict[str, Any]:
    if kind == "classification":
        pos = max(float((y == 1).sum()), 1.0)
        neg = max(float((y == 0).sum()), 1.0)
        scale = neg / pos
        return {
            "xgb": XGBClassifier(
                n_estimators=450,
                max_depth=5,
                learning_rate=0.035,
                min_child_weight=2,
                subsample=0.88,
                colsample_bytree=0.72,
                reg_alpha=0.05,
                reg_lambda=2.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=max(1, min(8, (os.cpu_count() or 4) // 2)),
                random_state=SEED,
                scale_pos_weight=scale,
            ),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=350,
                max_features=0.35,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            ),
            "logistic": make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.45, max_iter=1800, class_weight="balanced", solver="liblinear", random_state=SEED),
            ),
        }
    return {
        "xgb": XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.035,
            min_child_weight=2,
            subsample=0.88,
            colsample_bytree=0.72,
            reg_alpha=0.03,
            reg_lambda=3.0,
            objective="reg:squarederror",
            eval_metric="mae",
            tree_method="hist",
            n_jobs=max(1, min(8, (os.cpu_count() or 4) // 2)),
            random_state=SEED,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=350,
            max_features=0.35,
            min_samples_leaf=1,
            random_state=SEED,
            n_jobs=-1,
        ),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=8.0)),
    }


def predict_model(model: Any, x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "classification":
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    return np.asarray(model.predict(x), dtype=float)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def safe_spearman(y: np.ndarray, p: np.ndarray) -> float:
    value = spearmanr(y, p).statistic
    return float(value) if np.isfinite(value) else 0.0


def lookup_predictions(canon_train: list[str | None], y: np.ndarray, canon_target: list[str | None]) -> tuple[np.ndarray, np.ndarray]:
    table: dict[str, list[float]] = {}
    for key, value in zip(canon_train, y):
        if key is not None and np.isfinite(value):
            table.setdefault(key, []).append(float(value))
    pred = np.full(len(canon_target), np.nan, dtype=float)
    mask = np.zeros(len(canon_target), dtype=bool)
    for i, key in enumerate(canon_target):
        if key in table:
            pred[i] = float(np.mean(table[key]))
            mask[i] = True
    return pred, mask


def choose_blend(kind: str, y: np.ndarray, predictions: dict[str, np.ndarray]) -> tuple[dict[str, float], np.ndarray, dict[str, float]]:
    names = list(predictions)
    if len(names) == 1:
        weights = {names[0]: 1.0}
        p = predictions[names[0]]
    else:
        # A coarse simplex search is robust on the small official validation
        # sets and avoids fitting a high-variance meta-model.
        grid = np.arange(0.0, 1.01, 0.1)
        best_score = -np.inf
        best = None
        best_p = None
        for a in grid:
            for b in grid:
                if len(names) == 3:
                    c = 1.0 - a - b
                    if c < -1e-9:
                        continue
                    w = np.array([a, b, max(c, 0.0)])
                else:
                    w = np.array([a, 1.0 - a])
                q = sum(float(w[i]) * predictions[names[i]] for i in range(len(names)))
                if kind == "classification":
                    score = safe_auc(y, q)
                else:
                    scale = float(np.std(y)) + 1e-8
                    score = safe_spearman(y, q) - 0.20 * mean_absolute_error(y, q) / scale
                if score > best_score:
                    best_score, best, best_p = score, w, q
        weights = {name: float(best[i]) for i, name in enumerate(names)}
        p = np.asarray(best_p)
    if kind == "classification":
        metrics = {"auroc": safe_auc(y, p), "auprc": safe_ap(y, p)}
    else:
        metrics = {"mae": float(mean_absolute_error(y, p)), "spearman": safe_spearman(y, p)}
    return weights, p, metrics


def prepare_features(task_dir: Path, cache_dir: Path, train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{task_dir.name}.npz"
    all_smiles = pd.concat([train["Drug"], valid["Drug"], test["Drug"]], ignore_index=True).astype(str).tolist()
    # Cache is keyed by row order and therefore only reused when its shape and
    # source smiles checksum agree.
    import hashlib
    digest = hashlib.sha256("\n".join(all_smiles).encode("utf-8")).hexdigest()
    if cache_file.exists():
        try:
            z = np.load(cache_file, allow_pickle=True)
            if str(z["digest"]) == digest:
                n1, n2 = len(train), len(valid)
                x = z["x"]
                canon = z["canon"].tolist()
                return x[:n1], x[n1:n1+n2], x[n1+n2:], canon[:n1], canon[n1:n1+n2], canon[n1+n2:], {"n_total": len(all_smiles), "n_invalid": int(z["invalid"])}
        except Exception:
            pass
    x, canon, audit = featurize_smiles(all_smiles)
    np.savez_compressed(cache_file, x=x, canon=np.asarray(canon, dtype=object), digest=digest, invalid=audit["n_invalid"])
    n1, n2 = len(train), len(valid)
    return x[:n1], x[n1:n1+n2], x[n1+n2:], canon[:n1], canon[n1:n1+n2], canon[n1+n2:], audit


def train_task(task_dir: Path, output_dir: Path, cache_dir: Path) -> dict[str, Any]:
    train = pd.read_csv(task_dir / "train.csv")
    valid = pd.read_csv(task_dir / "valid.csv")
    test = pd.read_csv(task_dir / "test.csv")
    kind = task_type(train["Y"])
    y_train = pd.to_numeric(train["Y"], errors="coerce").to_numpy(dtype=float)
    y_valid = pd.to_numeric(valid["Y"], errors="coerce").to_numpy(dtype=float)
    xtr, xva, xte, ctr, cva, cte, audit = prepare_features(task_dir, cache_dir, train, valid, test)

    models = fit_candidates(kind, y_train)
    val_predictions: dict[str, np.ndarray] = {}
    model_metrics: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        model.fit(xtr, y_train)
        p = predict_model(model, xva, kind)
        val_predictions[name] = p
        if kind == "classification":
            model_metrics[name] = {"auroc": safe_auc(y_valid, p), "auprc": safe_ap(y_valid, p)}
        else:
            model_metrics[name] = {"mae": float(mean_absolute_error(y_valid, p)), "spearman": safe_spearman(y_valid, p)}

    weights, blend_val, blend_metrics = choose_blend(kind, y_valid, val_predictions)
    lookup_val, lookup_mask = lookup_predictions(ctr, y_train, cva)
    final_val = blend_val.copy()
    final_val[lookup_mask] = lookup_val[lookup_mask]
    if kind == "classification":
        final_metrics = {"auroc": safe_auc(y_valid, final_val), "auprc": safe_ap(y_valid, final_val)}
    else:
        final_metrics = {"mae": float(mean_absolute_error(y_valid, final_val)), "spearman": safe_spearman(y_valid, final_val)}

    # Refit each selected head on train+valid, then blend for test. The exact
    # lookup is applied after the model ensemble and is exact for repeated
    # molecules already observed in either labelled split.
    all_train = pd.concat([train, valid], ignore_index=True)
    y_all = pd.to_numeric(all_train["Y"], errors="coerce").to_numpy(dtype=float)
    x_all = np.vstack([xtr, xva])
    final_models = {}
    test_predictions = {}
    for name, weight in weights.items():
        model = fit_candidates(kind, y_all)[name]
        model.fit(x_all, y_all)
        final_models[name] = model
        test_predictions[name] = predict_model(model, xte, kind)
    blend_test = sum(weights[name] * test_predictions[name] for name in weights)
    lookup_test, lookup_test_mask = lookup_predictions(list(ctr) + list(cva), y_all, cte)
    final_test = np.asarray(blend_test, dtype=float)
    final_test[lookup_test_mask] = lookup_test[lookup_test_mask]

    task_out = output_dir / task_dir.name
    task_out.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"Drug_ID": test["Drug_ID"], "Y": final_test})
    sub.to_csv(task_out / f"submission_{task_dir.name}.csv", index=False)
    pd.DataFrame({"Drug_ID": valid["Drug_ID"], "Y_true": y_valid, "Y_pred": final_val, "exact_lookup": lookup_mask}).to_csv(task_out / "validation_predictions.csv", index=False)
    bundle = {"task": task_dir.name, "kind": kind, "weights": weights, "models": final_models, "fp_bits": FP_BITS, "descriptor_names": [x[0] for x in DESC_FUNCS], "audit": audit}
    joblib.dump(bundle, task_out / "model_bundle.joblib", compress=3)

    result: dict[str, Any] = {
        "task": task_dir.name,
        "kind": kind,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "invalid_smiles": audit["n_invalid"],
        "train_duplicate_smiles": int(train["Drug"].duplicated().sum()),
        "valid_exact_lookup_rate": float(lookup_mask.mean()),
        "test_exact_lookup_rate": float(lookup_test_mask.mean()),
        "blend_weights": json.dumps(weights, sort_keys=True),
    }
    result.update({f"model_{k}_{m}": v for k, mm in model_metrics.items() for m, v in mm.items()})
    result.update({f"blend_{m}": v for m, v in blend_metrics.items()})
    result.update({f"final_{m}": v for m, v in final_metrics.items()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("admet_competition_data/data_benchmark"))
    parser.add_argument("--output-dir", type=Path, default=Path("admet_model_outputs"))
    parser.add_argument("--tasks", nargs="*", default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "feature_cache"
    task_names = args.tasks or sorted(p.name for p in args.data_root.iterdir() if p.is_dir() and (p / "train.csv").exists())
    rows = []
    for i, name in enumerate(task_names, 1):
        print(f"[{i}/{len(task_names)}] training {name}", flush=True)
        try:
            row = train_task(args.data_root / name, args.output_dir, cache_dir)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        except Exception as exc:
            print(f"ERROR {name}: {type(exc).__name__}: {exc}", flush=True)
            rows.append({"task": name, "error": f"{type(exc).__name__}: {exc}"})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "validation_metrics.csv", index=False)
    (args.output_dir / "run_manifest.json").write_text(json.dumps({"tasks": task_names, "seed": SEED, "fp_bits": FP_BITS, "descriptor_names": [x[0] for x in DESC_FUNCS]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output_dir / 'validation_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
