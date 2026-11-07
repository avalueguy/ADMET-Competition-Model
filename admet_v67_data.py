"""Verified official-only data and fit-only chemistry preparation; no torch needed."""
from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.rdFingerprintGenerator import GetMorganFeatureAtomInvGen, GetMorganGenerator
from scipy.special import ndtri
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "admet_model_outputs_v67"
P = json.loads((ROOT / "admet_v67_protocol.json").read_text(encoding="utf-8"))
RDLogger.DisableLog("rdApp.*")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def emit(stage, **values):
    print(json.dumps(dict(stage=stage, **values), ensure_ascii=False), flush=True)


def signature():
    names = ["admet_v67_data.py", "admet_v67_train.py", "admet_v67_protocol.json"]
    inputs = sorted((ROOT / "admet_competition_data/data_benchmark").glob("*/*.csv"))
    names += [str(p.relative_to(ROOT)) for p in inputs]
    names += ["admet_official/submission_template.csv"]
    return hashlib.sha256(json.dumps({name: sha(ROOT / name) for name in names}, sort_keys=True).encode()).hexdigest()


def table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def parse_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol, "full_sanitization"
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError("Official SMILES has invalid syntax; cannot preserve its graph")
    # Keep all original atoms, bonds, charges, and hydrogens. A documented
    # valence exception must not turn into a different molecule or missing row.
    mol.UpdatePropertyCache(strict=False)
    operations = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
    flag = Chem.SanitizeMol(mol, sanitizeOps=operations, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        raise ValueError(f"Unsupported official molecule sanitization error: {flag}")
    return mol, "original_graph_preserved_valence_property_check_skipped"


class Union:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def join(self, left, right):
        a, b = self.find(left), self.find(right)
        self.parent[max(a, b)] = min(a, b)


def feature_one(sm):
    fns = [(name, fn) for name, fn in Descriptors.descList if name not in {"Ipc", "AvgIpc", "BalabanJ"}]
    mol, parse_mode = parse_molecule(sm)
    can = Chem.MolToSmiles(mol, isomericSmiles=False)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or "ACYCLIC:" + can
    gen = GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    fgen = GetMorganGenerator(radius=3, fpSize=2048, includeChirality=True,
                             atomInvariantsGenerator=GetMorganFeatureAtomInvGen())
    x = np.full(4096+len(fns), np.nan, dtype=np.float32)
    x[:2048] = np.log1p(gen.GetCountFingerprintAsNumPy(mol)).clip(0, 6)
    x[2048:4096] = np.log1p(fgen.GetCountFingerprintAsNumPy(mol)).clip(0, 6)
    for j, (_, fn) in enumerate(fns):
        try:
            value = float(fn(mol))
            if np.isfinite(value) and abs(value) < 1e10:
                x[4096+j] = value
        except Exception:
            pass
    return x, can, scaffold, parse_mode


def build():
    template_path = ROOT / "admet_official/submission_template.csv"
    assert sha(template_path) == P["official_template_sha256"]
    template = table(template_path)
    tasks = list(dict.fromkeys(row["task"] for row in template))
    metrics = [next(row["metric"] for row in template if row["task"] == task) for task in tasks]
    manifest = {v["path"]: v for v in json.loads((ROOT / P["data_manifest"]).read_text(encoding="utf-8"))}
    smiles, lookup, mi, tid, split, labels = [], {}, [], [], [], []
    for k, task in enumerate(tasks):
        expected = [row for row in template if row["task"] == task]
        for si, part in enumerate(("train", "valid", "test")):
            path = ROOT / "admet_competition_data/data_benchmark" / task / (part + ".csv")
            assert sha(path) == manifest[path.relative_to(ROOT).as_posix()]["sha256"], str(path)
            rows = table(path)
            if part == "test":
                assert len(rows) == len(expected)
            for j, row in enumerate(rows):
                sm = row["Drug"]
                if part == "test":
                    assert sm == expected[j]["smiles"] and int(expected[j]["sample_idx"]) == j
                if sm not in lookup:
                    lookup[sm] = len(smiles)
                    smiles.append(sm)
                mi.append(lookup[sm]); tid.append(k); split.append(si)
                # Deliberately never parse test Y, even if a file contains it.
                labels.append(float(row["Y"]) if si < 2 else np.nan)
    dest = OUT / "features"
    dest.mkdir(parents=True, exist_ok=True)
    key = dict(smiles_sha256=hashlib.sha256(json.dumps(smiles).encode()).hexdigest(),
               rdkit=rdBase.rdkitVersion, source_sha256=sha(__file__))
    cache, meta = dest / "official_chemistry.npz", dest / "official_chemistry.json"
    if cache.exists():
        assert json.loads(meta.read_text(encoding="utf-8")) == key, "Use a new output folder for changed features"
        with np.load(cache, allow_pickle=False) as z:
            x, canonical, groups = z["x"], z["canonical"], z["groups"]
    else:
        fns = [(name, fn) for name, fn in Descriptors.descList if name not in {"Ipc", "AvgIpc", "BalabanJ"}]
        x = np.zeros((len(smiles), 4096 + len(fns)), dtype=np.float32)
        x[:, 4096:] = np.nan
        canonical, groups, parse_exceptions = [], [], []
        union = Union()
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
            for i, (feature, can, scaffold, parse_mode) in enumerate(pool.map(feature_one, smiles, chunksize=64)):
                x[i] = feature
                if parse_mode != "full_sanitization":
                    parse_exceptions.append(dict(molecule_index=i, smiles=smiles[i], mode=parse_mode))
                canonical.append(can); groups.append(scaffold)
                union.join("C:" + can, "S:" + scaffold)
                if i % 5000 == 0:
                    emit("official_features", done=i, total=len(smiles))
        groups = np.array([union.find("C:" + can) for can in canonical])
        canonical = np.array(canonical)
        np.savez_compressed(cache, x=x, canonical=canonical, groups=groups)
        save(meta, key)
        save(dest / "parse_exceptions.json", dict(status="AUDITED", exceptions=parse_exceptions,
             original_smiles_unchanged=True, graph_replacement=False))
    d = dict(tasks=tasks, metrics=metrics, smiles=smiles, x=x, canonical=canonical, groups=groups,
             mi=np.array(mi), tid=np.array(tid), split=np.array(split), y=np.array(labels), template=template)
    d["rg"] = groups[d["mi"]]
    assert not np.isfinite(d["y"][d["split"] == 2]).any()
    return d


def assert_excluded(d, fit, query):
    assert not set(fit) & set(query) and np.all(d["split"][fit] < 2)
    if np.any(d["split"][query] < 2):
        assert not set(d["rg"][fit]) & set(d["rg"][query])
        assert not set(d["canonical"][d["mi"][fit]]) & set(d["canonical"][d["mi"][query]])


def folds(d):
    train = np.flatnonzero(d["split"] == 0)
    groups, inverse = np.unique(d["rg"][train], return_inverse=True)
    counts = np.zeros((len(groups), len(d["tasks"])))
    np.add.at(counts, (inverse, d["tid"][train]), 1)
    rel = counts / np.maximum(counts.sum(0), 1)
    rng = np.random.default_rng(P["cv_seed"])
    order = np.lexsort((rng.random(len(groups)), -rel.max(1)))
    mass = np.zeros((P["cv_folds"], len(d["tasks"])))
    assignment = np.empty(len(groups), dtype=int)
    for group in order:
        costs = [2 * np.dot(mass[f], rel[group]) + np.square(rel[group]).sum() for f in range(P["cv_folds"])]
        fold = int(np.argmin(np.array(costs) + rng.random(P["cv_folds"]) * 1e-12))
        assignment[group] = fold; mass[fold] += rel[group]
    out = np.full(len(d["y"]), -1)
    out[train] = assignment[inverse]
    for fold in range(P["cv_folds"]):
        fit, query = train[out[train] != fold], train[out[train] == fold]
        assert_excluded(d, fit, query)
        for k, metric in enumerate(d["metrics"]):
            ix = query[d["tid"][query] == k]
            assert len(ix) > 1
            if metric in ("AUROC", "AUPRC"):
                assert len(np.unique(d["y"][ix])) == 2, (d["tasks"][k], fold)
    np.save(OUT / "folds.npy", out)
    save(OUT / "split_audit.json", dict(status="PASS", seed=P["cv_seed"], fractions=mass.tolist(),
          global_scaffold_overlap=0, canonical_overlap=0, test_labels_used=False))
    return out


def preprocess(d, fit):
    unique = np.unique(d["mi"][fit]); raw = d["x"][unique, 4096:]
    med = np.nan_to_num(np.nanmedian(raw, axis=0))
    imputed = np.where(np.isfinite(raw), raw, med)
    return dict(median=med, mean=imputed.mean(0), scale=imputed.std(0).clip(1e-4),
                fit_molecules=unique)


def transform(x, state):
    desc = np.where(np.isfinite(x[:, 4096:]), x[:, 4096:], state["median"])
    return np.concatenate([x[:, :4096], ((desc-state["mean"])/state["scale"]).clip(-8, 8)], 1).astype(np.float32)


def target(y, metric):
    state = dict(metric=metric, center=float(np.median(y)),
                 scale=max(float(np.quantile(y, .75)-np.quantile(y, .25)), .001))
    if metric == "Spearman":
        values, indices = np.unique(y, return_index=True)
        state.update(knots=values, z=ndtri(np.clip((rankdata(y)-.5)/len(y), 1e-5, 1-1e-5))[indices])
    return state


def encode(y, state):
    if state["metric"] in ("AUROC", "AUPRC"):
        return y
    if state["metric"] == "Spearman":
        return np.interp(y, state["knots"], state["z"])
    return (y-state["center"])/state["scale"]


def metric_score(metric, y, p):
    if metric == "MAE":
        return float(mean_absolute_error(y, p))
    if metric == "Spearman":
        result = float(spearmanr(y, p).statistic)
        return result if np.isfinite(result) else 0.
    if len(np.unique(y)) < 2:
        return float("nan")
    return float((roc_auc_score if metric == "AUROC" else average_precision_score)(y, p))


def gain(metric, y, old, new):
    a, b = metric_score(metric, y, old), metric_score(metric, y, new)
    return (a-b)/max(a, 1e-8) if metric == "MAE" else b-a


def blend(metric, old, new, weight):
    if weight == 0:
        return old.copy()
    if metric != "MAE":
        old, new = rankdata(old)/len(old), rankdata(new)/len(new)
    return (1-weight)*old + weight*new


if __name__ == "__main__":
    data = build()
    parts = folds(data)
    save(OUT / "data_audit.json", dict(status="PASS", unique_molecules=len(data["smiles"]),
         rows=len(parts), test_rows=int(np.sum(data["split"] == 2)), tasks=len(data["tasks"]),
         signature=signature(), external_data=False, pretrained_weights=False,
         feature_dimensions=int(data["x"].shape[1])))
    emit("official_data_audit", status="PASS", rows=len(parts), unique_molecules=len(data["smiles"]))
