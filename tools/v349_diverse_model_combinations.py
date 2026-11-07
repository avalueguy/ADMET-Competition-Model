"""Frozen cross-family pairs/triples over audited historical models, versus v346."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import v314_current_baseline as v
import v329_sharpness_retrieval as sam

OUT = ROOT / "admet_model_outputs_v349"
REGISTRY = (
    (168, "v168_sparse_linear", "sparse_linear"),
    (172, "v172_graph_message_passing", "graph"),
    (178, "v178_graph_embedding_boosting", "graph_boosting"),
    (181, "v181_quantile_forest", "forest"),
    (189, "v189_leafwise_boosting", "leafwise_boosting"),
    (193, "v193_smiles_convolution", "smiles_convolution"),
    (200, "v200_exact_newton_splines", "splines"),
    (204, "v204_clamped_diffusion", "diffusion"),
    (207, "v207_denoising_encoder", "denoising_boosting"),
    (210, "v210_standardized_chemistry", "chemistry_boosting"),
    (216, "v216_ordered_scaffold", "ordered_boosting"),
    (221, "v221_fragment_factorization", "factorization"),
    (224, "v224_rotated_chemistry_forest", "rotation_forest"),
    (233, "v233_supervised_chemical_similarity", "supervised_kernel"),
    (238, "v238_molecular_view_attention", "view_attention"),
    (244, "v244_conditional_density", "density_network"),
    (250, "v250_hierarchical_atom_matching", "atom_kernel"),
    (256, "v256_exposed_polarity", "conformer_boosting"),
    (264, "v264_replicate_noise_gp", "noise_gp"),
    (271, "v271_clearance_innovations", "crossfit_context"),
    (278, "v278_v145_leaf_refit", "leaf_refit"),
    (289, "v289_stereo_schema_fix", "context_repair"),
    (297, "v297_pair_delta_regression", "pair_regression"),
    (299, "v299_distributional_trees", "distributional_boosting"),
    (301, "v301_piecewise_batch_ensemble", "shared_neural_ensemble"),
    (304, "v304_learned_label_retrieval", "label_retrieval"),
    (310, "v310_stable_differentiable_trees", "differentiable_trees"),
    (315, "v315_rank_label_retrieval", "label_retrieval"),
    (318, "v318_neural_tangent_regression", "neural_kernel"),
    (324, "v324_scaffold_manifold_mixup", "mixup_retrieval"),
    (329, "v329_sharpness_retrieval", "sam_retrieval"),
    (333, "v333_chemical_domain_dro", "domain_dro"),
    (340, "v340_smooth_ap_retrieval", "ap_retrieval"),
)
EXPERT_WEIGHT = .4
PARTS = ("fold0", "fold1", "fold2", "valid")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def now():
    return datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()


def modules():
    return {n: importlib.import_module(name) for n, name, _ in REGISTRY
            if list((ROOT / f"admet_model_outputs_v{n}").glob("models/*/fold0/*/predictions.npz"))}


def current(d, k, part):
    if part == "final":
        card = json.loads((ROOT / "docs/current_best_after_v346_20261102.json").read_text())
        path = ROOT / card["csv"]
        assert sha(path) == card["csv_sha256"]
        return np.array([float(r["y_pred"]) for r in v.b.table(path) if r["task"] == d["tasks"][k]])
    base = v.baseline(d, k, part)
    if d["tasks"][k] == "solubility_aqsoldb":
        file = sam.OUT / "models" / d["tasks"][k] / part / sam.key(sam.CONFIGS[0]) / "predictions.npz"
        z = np.load(file)
        assert str(z["signature"]) == sam.sig()
        np.testing.assert_array_equal(z["query"], v.split(d, k, part)[1])
        base = .8 * base + .2 * z["p"]
    return base


def scaled(metric, value):
    return value if metric == "MAE" else rankdata(value, method="average") / len(value)


def fused(metric, base, experts):
    assert len(experts) in (2, 3)
    mean = np.mean([scaled(metric, p) for p in experts], axis=0)
    return (1. - EXPERT_WEIGHT) * scaled(metric, base) + EXPERT_WEIGHT * mean


def cached(d, k, part, record, mods):
    model = mods[record["version"]]
    file = model.OUT / "models" / d["tasks"][k] / part / record["key"] / "predictions.npz"
    assert file.exists(), file
    z = np.load(file)
    assert str(z["signature"]) == model.sig()
    fit, query = v.split(d, k, part)
    np.testing.assert_array_equal(z["fit"], fit)
    np.testing.assert_array_equal(z["query"], query)
    assert len(z["p"]) == len(query) and np.isfinite(z["p"]).all()
    assert not set(fit) & set(query)
    if part != "final":
        assert not set(d["groups"][fit]) & set(d["groups"][query])
    return z["p"].copy(), dict(path=file.relative_to(ROOT).as_posix(), sha256=sha(file))


def freeze():
    assert not (OUT / "protocol.json").exists()
    mods = modules()
    records, unavailable = [], []
    for n, name, family in REGISTRY:
        if n not in mods:
            unavailable.append(n)
            records.append(dict(version=n, module=name, family=family, available=False))
            continue
        model = mods[n]
        records.append(dict(version=n, module=name, family=family,
            available=True, source_sha256=sha(model.__file__), signature=model.sig(), configs=model.CONFIGS))
    best = json.loads((ROOT / "docs/current_best_after_v346_20261102.json").read_text())
    assert best["version"] == "v346" and best["official_user_reported"] == 51.235
    protocol = dict(status="FROZEN_BEFORE_COMBINATION_SCORES", at_hong_kong=now(),
        source_sha256=sha(__file__), registry=records, unavailable_prediction_routes=unavailable,
        baseline=best, expert_total_weight=EXPERT_WEIGHT, baseline_weight=.6,
        hypotheses="All pairs and triples with distinct versions and distinct families, equal expert weights; no individual-model weight sweeps.",
        developer_selection="Maximum minimum fold0/1 gain, ties favor fewer members then mean gain then lexicographic ID. Freeze exactly one combination/task before confirmation, no fallback.",
        excluded="HIA already AUROC1; v315 CYP3A4 component failed officially; solubility v329 is already inside the frozen v346 baseline, never increased independently.",
        dev_thresholds=dict(MAE=.012, Spearman=.008, AUROC=.003, AUPRC=.003),
        confirmation="All4 positive points, mean practical threshold, both500 paired scaffold-bootstrap90 lower bounds>0 for strict acceptance. Any experimental alternative must be labeled and preserve rejection.",
        external_data=False, pretrained_weights=False, test_labels_used=False,
        model_training_at_search=False, manual_official_submission_only=True,
        limitations="Not every model worldwide. Historical public labels repeatedly reused, upstream OOF approximate; confirmation reuses public folds and is not a fresh holdout. Large search raises selection risk. No claim of official improvement.")
    save(OUT / "protocol.json", protocol)
    save(ROOT / "docs/v349_combination_plan.json", protocol)
    print(json.dumps(dict(status=protocol["status"], registry=len(records), unavailable=unavailable)), flush=True)


def search():
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert protocol["source_sha256"] == sha(__file__)
    assert not (OUT / "development.json").exists()
    mods = modules()
    for rec in protocol["registry"]:
        if not rec["available"]:
            continue
        m = mods[rec["version"]]
        assert rec["source_sha256"] == sha(m.__file__) and rec["signature"] == m.sig()
    d = v.load()
    dev, selected, inventory, rejected = [], [], [], []
    for task in d["tasks"]:
        if task == "hia_hou":
            continue
        k = d["tasks"].index(task)
        metric = d["metrics"][k]
        bases = [current(d, k, part) for part in PARTS[:2]]
        ys = [d["y"][v.split(d, k, part)[1]] for part in PARTS[:2]]
        members, predictions = [], []
        for n, name, family in REGISTRY:
            if n not in mods:
                continue
            if (n == 315 and task == "cyp3a4_substrate_carbonmangels") or (n == 329 and task == "solubility_aqsoldb"):
                continue
            model = mods[n]
            for config in model.CONFIGS:
                key = model.key(config)
                if not all((model.OUT / "models" / task / part / key / "predictions.npz").is_file() for part in PARTS[:2]):
                    continue
                record = dict(version=n, module=name, family=family, key=key, config=config)
                try:
                    ps, sources = zip(*(cached(d, k, part, record, mods) for part in PARTS[:2]))
                except (AssertionError, ValueError) as exc:
                    rejected.append(dict(task=task, version=n, key=key, reason=str(exc)))
                    continue
                record["development_sources"] = sources
                members.append(record)
                predictions.append([scaled(metric, p) for p in ps])
        inventory.append(dict(task=task, members=members))
        candidates = []
        for size in (2, 3):
            for ids in itertools.combinations(range(len(members)), size):
                if len({members[i]["version"] for i in ids}) != size or len({members[i]["family"] for i in ids}) != size:
                    continue
                gains, scores = [], []
                for j in range(2):
                    p = .6 * scaled(metric, bases[j]) + .4 * np.mean([predictions[i][j] for i in ids], axis=0)
                    gains.append(v.r83.gain(metric, ys[j], bases[j], p))
                    scores.append(v.r83.score(metric, ys[j], p))
                candidates.append(dict(ids=list(ids), gains=gains, scores=scores, objective=min(gains)))
        candidates.sort(key=lambda r:(-r["objective"], len(r["ids"]), -np.mean(r["gains"]), r["ids"]))
        if candidates:
            best = candidates[0]
            result = dict(task=task, metric=metric, combinations=len(candidates), pool=len(members),
                objective=best["objective"], gains=best["gains"], scores=best["scores"],
                members=[members[i] for i in best["ids"]], expert_total_weight=.4)
            dev.append(result)
            if best["objective"] > v.threshold(metric):
                selected.append(result)
            save(OUT / "search" / (task + ".json"), dict(task=task, members=members, candidates=candidates))
            print(json.dumps(dict(stage="development", task=task, pool=len(members), combinations=len(candidates),
                gains=best["gains"], versions=[members[i]["version"] for i in best["ids"]])), flush=True)
    frozen = dict(status="FROZEN_BEFORE_CONFIRMATION", at_hong_kong=now(), source_sha256=sha(__file__), selected=selected)
    save(OUT / "development.json", dict(status="COMPLETE", records=dev, inventory=inventory,
        rejected_alignment_or_signature=rejected, total_combinations=sum(r["combinations"] for r in dev)))
    save(OUT / "frozen_selection.json", frozen)
    print(json.dumps(dict(stage="selection_frozen", selected=[r["task"] for r in selected])), flush=True)


def confirm():
    assert not (OUT / "confirmation.json").exists()
    frozen = json.loads((OUT / "frozen_selection.json").read_text())
    assert frozen["source_sha256"] == sha(__file__)
    mods, d, records = modules(), v.load(), []
    for choice in frozen["selected"]:
        task, metric = choice["task"], choice["metric"]
        k = d["tasks"].index(task)
        parts, statistics, sources = [], [], []
        for part in PARTS:
            qs = v.split(d, k, part)[1]
            experts = []
            for rec in choice["members"]:
                model = mods[rec["version"]]
                # New confirmation fits, if absent, use the exact existing mechanism/config/seeds.
                model.predict(d, k, part, rec["config"])
                p, provenance = cached(d, k, part, rec, mods)
                experts.append(p)
                sources.append(provenance)
            base = current(d, k, part)
            p = fused(metric, base, experts)
            parts.append(dict(part=part, rows=len(qs), baseline=v.r83.score(metric, d["y"][qs], base),
                candidate=v.r83.score(metric, d["y"][qs], p), gain=v.r83.gain(metric, d["y"][qs], base, p)))
            folder = OUT / "predictions" / task
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / (part + ".npz"), query=qs, base=base, p=p,
                                experts=np.stack(experts), y=d["y"][qs], groups=d["groups"][qs])
            if part in ("fold2", "valid"):
                groups = np.unique(d["groups"][qs])
                ixs = {g:np.flatnonzero(d["groups"][qs] == g) for g in groups}
                rng, delta = np.random.default_rng(1484026 + k), []
                if len(groups) > 1:
                    for _ in range(500):
                        ix = np.concatenate([ixs[g] for g in rng.choice(groups, len(groups), replace=True)])
                        if metric in ("AUROC", "AUPRC") and len(np.unique(d["y"][qs][ix])) < 2:
                            continue
                        delta.append(v.r83.gain(metric, d["y"][qs][ix], base[ix], p[ix]))
                ci = np.quantile(delta, [.05, .95]).tolist() if delta else None
                statistics.append(dict(part=part, groups=len(groups), gain_ci90=ci, lower_positive=bool(ci and ci[0] > 0)))
        strict = bool(min(r["gain"] for r in parts) > 0 and np.mean([r["gain"] for r in parts]) > v.threshold(metric)
            and all(r["lower_positive"] for r in statistics))
        record = dict(**choice, confirmation=parts, bootstrap=statistics, strict_accepted=strict,
            four_positive_points=min(r["gain"] for r in parts) > 0, source_files=sources)
        records.append(record)
        save(OUT / "confirmation.progress.json", dict(records=records))
        print(json.dumps(dict(stage="confirmation", **record)), flush=True)
    result = dict(status="COMPLETE", at_hong_kong=now(), records=records,
        accepted_tasks=[r["task"] for r in records if r["strict_accepted"]],
        experimental_four_positive_tasks=[r["task"] for r in records if r["four_positive_points"] and not r["strict_accepted"]],
        csv_created=False, official_gain_established=False, platform_submission=False)
    save(OUT / "confirmation.json", result)
    save(ROOT / "docs/v349_combination_result.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("freeze", "search", "confirm"))
    globals()[p.parse_args().stage]()
