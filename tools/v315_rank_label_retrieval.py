"""Scaffold-disjoint label-context attention for rank and classification endpoints."""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "4"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import v314_current_baseline as v
import v314_v313_experiment_runner as runner
from scipy import sparse
from scipy.special import expit

b, r83 = v.b, v.r83
OUT = ROOT / "admet_model_outputs_v315"
TASKS = ("herg", "clearance_hepatocyte_az", "clearance_microsome_az", "half_life_obach",
         "vdss_lombardo", "cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels",
         "cyp3a4_substrate_carbonmangels")
CONFIGS = (dict(neighbors=24, width=128, key_width=32, bins=8, epochs=100, minimum_steps=1200, maximum_steps=2500),)
SEEDS = (3041101, 3041102)
METHOD = (
    "Train-from-scratch label-context retrieval for four Spearman endpoints and hERG/three CYP substrate "
    "classification endpoints, incremental to exact v313 two-task update. Spearman uses fit-only "
    "rank-normal targets with squared loss plus .25 pairwise rank loss; classification uses "
    "binary fit-label context, fit-prevalence intercept and BCE plus .25 positive-negative pair loss. "
    "Candidate pools of 24 "
    "nearest count-Tanimoto fit molecules always exclude identical canonical identities and "
    "scaffolds, for both fit and query. Numeric piecewise fit-quantile embeddings plus six "
    "fixed structure-only random fingerprint projections encode each molecule. Learned "
    "key-distance attention with a fixed count-similarity prior selects context observations. "
    "Context values contain a learned embedding of actual current fit labels plus a learned "
    "key-difference representation. Their weighted representation and an explicit weighted "
    "fit-label mean feed a residual property head. Context label dropout regularizes reliance "
    "on sparse support; same two predetermined v304 seeds and architecture. No pair-difference targets, "
    "precomputed label summaries, shared-layer ensemble, old predictions, external or "
    "pretrained data. All retrieval labels originate in the current fit rows. Original "
    "v148 splits, selection, blends and confirmation algorithm unchanged; its separate adapter uses v313 baseline."
)


def initialize():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def sig():
    return hashlib.sha256((b.sha(__file__) + v.sig() + b.sha(runner.__file__) +
                           importlib.metadata.version("torch") + importlib.metadata.version("numpy")).encode()).hexdigest()


def key(c):
    assert c == CONFIGS[0]
    return "learned_label_context24"


def load():
    initialize()
    return v.load()


def same(a, z):
    assert a.keys() == z.keys()
    for name in a:
        np.testing.assert_array_equal(a[name], z[name])


def fit_numeric(x, bins):
    raw = x.astype(np.float64)
    raw[~np.isfinite(raw)] = np.nan
    median, scale, columns, knot_list = [], [], [], []
    for j, col in enumerate(raw.T):
        values = col[np.isfinite(col)]
        if not len(values):
            continue
        knots = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        # A constant measured value is retained when the missingness varies.
        if len(knots) < 2 and np.isfinite(col).all():
            continue
        if len(knots) < 2:
            knots = np.array([values[0] - .5, values[0] + .5])
        columns.append(j)
        median.append(float(np.median(values)))
        scale.append(max(float(np.subtract(*np.quantile(values, [.75, .25]))), .1))
        knot_list.append(knots)
    offsets = np.r_[0, np.cumsum([len(knots) - 1 for knots in knot_list])]
    return dict(columns=np.asarray(columns, dtype=np.int64), median=np.array(median),
                numeric_scale=np.array(scale), knots=np.concatenate(knot_list),
                knot_offsets=np.r_[0, np.cumsum([len(knots) for knots in knot_list])],
                embedding_offsets=offsets)


def numeric_transform(raw, state):
    raw = raw[:, state["columns"]].astype(np.float64)
    finite = np.isfinite(raw)
    clean = np.where(finite, raw, state["median"])
    blocks = []
    for j in range(len(state["columns"])):
        lo, hi = state["knot_offsets"][j:j + 2]
        knots = state["knots"][lo:hi]
        width = np.maximum(np.diff(knots), 1e-12)
        blocks.append(np.clip((clean[:, j, None] - knots[None, :-1]) / width, 0., 1.))
    scalar = np.clip((clean - state["median"]) / state["numeric_scale"], -8., 8.)
    out = np.c_[*blocks, scalar, finite].astype(np.float32)
    assert np.isfinite(out).all()
    return out


def design(d, k, part, c):
    assert d["tasks"][k] in TASKS and d["metrics"][k] != "MAE"
    fit, query, x, qx, source = v.old.old.features(d, k, part, "augmented")
    numeric = fit_numeric(x, c["bins"])
    xn, qn = numeric_transform(x, numeric), numeric_transform(qx, numeric)
    def fp(rows):
        raw = d["views"]["multiview"][d["mi"][rows], :12288].astype(np.float64).reshape(len(rows), 6, 2048)
        assert np.all(raw >= 0) and np.isfinite(raw).all()
        raw /= np.linalg.norm(raw, axis=2, keepdims=True).clip(1e-12)
        # Fixed structure-only random projection, no fitted labels or query statistics.
        rng = np.random.default_rng(3040001)
        pieces = []
        for j in range(6):
            projection = rng.choice(np.array([-1., 1.]), size=(2048, 32)) / np.sqrt(32.)
            pieces.append(raw[:, j] @ projection)
        return np.c_[*pieces].astype(np.float32)
    train_ix, train_sim = neighbors(d, fit, fit, c["neighbors"])
    query_ix, query_sim = neighbors(d, fit, query, c["neighbors"])
    y = d["y"][fit]
    metric = d["metrics"][k]
    center = 0. if metric == "Spearman" else float(np.log(np.clip(y.mean(), .01, .99) / np.clip(1. - y.mean(), .01, .99)))
    scale = 1.
    target = r83.ranknorm(y) if metric == "Spearman" else y
    state = dict(fit=fit, query=query, source=source, center=center, scale=scale,
                 target=np.asarray(target, dtype=np.float32), **numeric)
    assert not set(fit) & set(query) and np.all(d["tid"][source] != k)
    assert np.all(d["split"][source] < (2 if part == "final" else 1))
    assert part == "final" or not set(d["groups"][fit]) & set(d["groups"][query])
    state.update(train_ix=train_ix, train_similarity=train_sim,
                 query_ix=query_ix, query_similarity=query_sim)
    return dict(x=np.c_[fp(fit), xn], qx=np.c_[fp(query), qn], state=state)


def neighbors(d, fit, rows, count):
    indices, similarities = [], []
    raw = sparse.csr_matrix(d["views"]["multiview"][:, :4096].astype(np.float64))
    ref = raw[d["mi"][fit]]
    rn = np.asarray(ref.multiply(ref).sum(axis=1)).ravel()
    for lo in range(0, len(rows), 128):
        rr = rows[lo:lo + 128]
        query = raw[d["mi"][rr]]
        dot = (query @ ref.T).toarray()
        qn = np.asarray(query.multiply(query).sum(axis=1)).ravel()
        sim = np.clip(dot / np.maximum(qn[:, None] + rn[None] - dot, 1e-12), 0., 1.)
        excluded = ((d["canonical"][rr, None] == d["canonical"][fit][None]) |
                    (d["groups"][rr, None] == d["groups"][fit][None]))
        sim[excluded] = -1.
        ix = np.argsort(-sim, axis=1, kind="stable")[:, :count]
        selected = np.take_along_axis(sim, ix, axis=1)
        assert np.all(selected >= 0), "Insufficient scaffold-disjoint fit references"
        indices.append(ix)
        similarities.append(selected)
    return np.concatenate(indices), np.concatenate(similarities).astype(np.float32)


class Net(nn.Module):
    def __init__(self, inputs, c):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(inputs, c["width"]), nn.LayerNorm(c["width"]), nn.ReLU(),
                                     nn.Linear(c["width"], c["width"]), nn.ReLU())
        self.key = nn.Linear(c["width"], c["key_width"])
        self.label = nn.Linear(1, c["width"], bias=False)
        self.difference = nn.Sequential(nn.Linear(c["key_width"], c["width"]), nn.ReLU(),
                                        nn.Linear(c["width"], c["width"]))
        self.head = nn.Sequential(nn.Linear(c["width"] * 2, c["width"]), nn.ReLU(),
                                  nn.Dropout(.15), nn.Linear(c["width"], 1))
        self.context_coefficient = nn.Parameter(torch.tensor(.5))
        with torch.no_grad():
            self.head[-1].weight.mul_(.01)
            self.head[-1].bias.zero_()

    def encoded(self, x):
        h = self.encoder(x)
        return h, self.key(h)

    def combine(self, query_h, query_key, reference_key, labels, similarity, mask=None):
        difference = query_key[:, None] - reference_key
        # No labels enter the keys, candidate selection or similarity prior.
        score = -difference.square().mean(dim=-1) + 4. * (similarity.clamp_min(1e-4)).log()
        if mask is not None:
            score = score.masked_fill(~mask, -1e9)
        weights = score.softmax(dim=1)
        values = self.label(labels[..., None]) + self.difference(difference)
        context = (weights[..., None] * values).sum(dim=1)
        mean_label = (weights * labels).sum(dim=1)
        return self.head(torch.cat([query_h, context], dim=1)).squeeze(-1) + self.context_coefficient * mean_label

    def forward(self, query, reference, labels, similarity):
        qh, qk = self.encoded(query)
        rk = self.encoded(reference.flatten(0, 1))[1].reshape(len(query), reference.shape[1], -1)
        mask = None
        if self.training:
            mask = torch.rand_like(labels) > .15
            mask[:, 0] = True
        return self.combine(qh, qk, rk, labels, similarity, mask)


def infer_cpu(net, a):
    net = net.cpu().double().eval()
    outputs = []
    with torch.inference_mode():
        reference_keys = torch.cat([net.encoded(torch.as_tensor(a["x"][lo:lo + 64], dtype=torch.float64))[1]
                                    for lo in range(0, len(a["x"]), 64)])
        labels = torch.as_tensor(a["state"]["target"], dtype=torch.float64)
        for lo in range(0, len(a["qx"]), 64):
            ix = a["state"]["query_ix"][lo:lo + 64].copy()
            similarity = torch.as_tensor(a["state"]["query_similarity"][lo:lo + 64].copy(), dtype=torch.float64)
            qh, qk = net.encoded(torch.as_tensor(a["qx"][lo:lo + 64], dtype=torch.float64))
            outputs.append(net.combine(qh, qk, reference_keys[ix], labels[ix], similarity).numpy())
    return np.concatenate(outputs)


def make_loaded(path, inputs, c):
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["signature"] == sig() and state["config"] == c
    net = Net(inputs, c)
    net.load_state_dict(state["weights"])
    return net


def output(raw, state, task):
    p = raw * state["scale"] + state["center"]
    return p if task in ("clearance_hepatocyte_az", "clearance_microsome_az", "half_life_obach", "vdss_lombardo") else expit(p)


def endpoint_loss(raw, target, metric, center):
    if metric == "Spearman":
        delta = target[:, None] - target[None]
        mask = delta.abs() > .1
        pair = nn.functional.softplus(-delta.sign() * (raw[:, None] - raw[None]))
        return .5 * (raw - target).square().mean() + (.25 * pair[mask].mean() if mask.any() else 0.)
    p = raw + center
    positive, negative = p[target > .5], p[target < .5]
    pair = nn.functional.softplus(negative[:, None] - positive[None]).mean() if len(positive) and len(negative) else 0.
    return nn.functional.binary_cross_entropy_with_logits(p, target) + .25 * pair


def replay(d, k, part, c):
    a = design(d, k, part, c)
    folder = OUT / "models" / d["tasks"][k] / part / key(c)
    same(a["state"], joblib.load(folder / "transform.joblib"))
    z = np.load(folder / "predictions.npz")
    assert str(z["signature"]) == sig()
    models = [make_loaded(folder / f"model_{seed}.pt", a["x"].shape[1], c) for seed in SEEDS]
    raw = np.mean([infer_cpu(net, a) for net in models], axis=0)
    p = output(raw, a["state"], d["tasks"][k])
    np.testing.assert_array_equal(p, z["p"])
    return p


def predict(d, k, part, c):
    if d["tasks"][k] not in TASKS:
        return v.baseline(d, k, part)
    folder = OUT / "models" / d["tasks"][k] / part / key(c)
    file = folder / "predictions.npz"
    if file.exists():
        z = np.load(file)
        assert str(z["signature"]) == sig()
        np.testing.assert_array_equal(z["query"], v.split(d, k, part)[1])
        return z["p"].copy()
    assert not folder.exists(), "Inspect partial artifacts before retry."
    started = time.monotonic()
    a = design(d, k, part, c)
    folder.mkdir(parents=True)
    joblib.dump(a["state"], folder / "transform.joblib", compress=3)
    assert torch.cuda.is_available()
    x = torch.as_tensor(a["x"], device="cuda")
    y = torch.as_tensor(a["state"]["target"], device="cuda")
    neighbors_tensor = torch.as_tensor(a["state"]["train_ix"], device="cuda")
    similarity_tensor = torch.as_tensor(a["state"]["train_similarity"], device="cuda")
    steps = max(c["minimum_steps"], min(c["maximum_steps"], math.ceil(len(y) / 128) * c["epochs"]))
    predictions, hashes = [], {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        net = Net(x.shape[1], c).cuda()
        optimizer = torch.optim.AdamW(net.parameters(), lr=.0008, weight_decay=.01)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=.00003)
        for step in range(steps):
            net.train()
            ix = rng.choice(len(y), min(128, len(y)), replace=False)
            context_ix = neighbors_tensor[ix]
            p = net(x[ix], x[context_ix], y[context_ix], similarity_tensor[ix])
            loss = endpoint_loss(p, y[ix], d["metrics"][k], a["state"]["center"])
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            if step % 500 == 0:
                print(json.dumps(dict(stage="updates", task=d["tasks"][k], part=part, seed=seed,
                                     step=step, steps=steps, fit_loss=float(loss.detach()))), flush=True)
        path = folder / f"model_{seed}.pt"
        torch.save(dict(signature=sig(), config=c, steps=steps,
                        weights={n: z.detach().cpu() for n, z in net.state_dict().items()}), path)
        expected = infer_cpu(net, a)
        actual = infer_cpu(make_loaded(path, x.shape[1], c), a)
        np.testing.assert_array_equal(expected, actual)
        predictions.append(actual)
        hashes[path.name] = b.sha(path)
        del net, optimizer
    p = output(np.mean(predictions, axis=0), a["state"], d["tasks"][k])
    assert np.isfinite(p).all()
    np.savez_compressed(file, signature=sig(), fit=a["state"]["fit"], query=a["state"]["query"], p=p)
    b.save(folder / "metadata.json", dict(signature=sig(), model_hashes=hashes,
        steps_per_seed=steps, context_labels_per_row=c["neighbors"], features=x.shape[1],
        seconds=time.monotonic() - started))
    del x, y, neighbors_tensor, similarity_tensor
    torch.cuda.empty_cache()
    print(json.dumps(dict(stage="ensemble_train", task=d["tasks"][k], part=part,
        seconds=round(time.monotonic() - started))), flush=True)
    return p


def smoke(d):
    for metric, yy in (("AUROC", [0., 1., 0., 1.]), ("AUPRC", [0., 0., 0., 0.]),
                       ("Spearman", [-1., -.2, .2, 1.]), ("Spearman", [0., 0., 0., 0.])):
        raw_loss = torch.tensor([-.2, .3, .1, -.4], dtype=torch.float64, requires_grad=True)
        yy = torch.tensor(yy, dtype=torch.float64)
        loss = endpoint_loss(raw_loss, yy, metric, -.1)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(raw_loss.grad).all()
    raw = np.array([[0., 1., np.nan], [1., 1., 1.], [2., 1., 1.], [3., 1., np.nan]])
    state = fit_numeric(raw, 3)
    encoded = numeric_transform(raw, state)
    np.testing.assert_array_equal(state["columns"], [0, 2])
    np.testing.assert_allclose(encoded[:, :3], [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]], atol=1e-6)
    k = d["tasks"].index("herg")
    a = design(d, k, "fold0", CONFIGS[0])
    saved = d["y"].copy()
    try:
        d["y"][(d["tid"] == k) & ~np.isin(np.arange(len(saved)), a["state"]["fit"])] = 999999.
        d.pop("_v96_assays", None)
        altered = design(d, k, "fold0", CONFIGS[0])
        same(a["state"], altered["state"])
        np.testing.assert_array_equal(a["x"], altered["x"])
        np.testing.assert_array_equal(a["qx"], altered["qx"])
    finally:
        d["y"][:] = saved
        d.pop("_v96_assays", None)
    torch.manual_seed(SEEDS[0])
    net = Net(a["x"].shape[1], CONFIGS[0]).cuda()
    x = torch.as_tensor(a["x"][:32], device="cuda")
    target = torch.as_tensor(a["state"]["target"][:32], device="cuda")
    opt = torch.optim.AdamW(net.parameters(), lr=.0008)
    for _ in range(3):
        ix = a["state"]["train_ix"][:32]
        reference = torch.as_tensor(a["x"][ix], device="cuda")
        labels = torch.as_tensor(a["state"]["target"][ix], device="cuda")
        similarity = torch.as_tensor(a["state"]["train_similarity"][:32], device="cuda")
        loss = endpoint_loss(net(x, reference, labels, similarity), target, d["metrics"][k], a["state"]["center"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        opt.step()
    path = OUT / "smoke.pt"
    torch.save(dict(signature=sig(), config=CONFIGS[0], weights=net.cpu().state_dict()), path)
    original = infer_cpu(net, a)
    replayed = infer_cpu(make_loaded(path, a["x"].shape[1], CONFIGS[0]), a)
    np.testing.assert_array_equal(original, replayed)
    reverse_state = dict(a["state"], query_ix=a["state"]["query_ix"][::-1],
                         query_similarity=a["state"]["query_similarity"][::-1])
    reversed_p = infer_cpu(net, dict(a, qx=a["qx"][::-1].copy(), state=reverse_state))
    np.testing.assert_allclose(original, reversed_p[::-1], rtol=0, atol=1e-12)
    fit = a["state"]["fit"]
    for rows, ids in ((fit, a["state"]["train_ix"]), (a["state"]["query"], a["state"]["query_ix"])):
        assert np.all(d["canonical"][rows, None] != d["canonical"][fit][ids])
        assert np.all(d["groups"][rows, None] != d["groups"][fit][ids])
    reverse_ix, reverse_sim = neighbors(d, fit, a["state"]["query"][::-1], CONFIGS[0]["neighbors"])
    np.testing.assert_array_equal(reverse_ix, a["state"]["query_ix"][::-1])
    np.testing.assert_array_equal(reverse_sim, a["state"]["query_similarity"][::-1])
    b.save(OUT / "smoke.json", dict(status="PASS", signature=sig(), piecewise_reference=True,
        scaffold_disjoint_context_labels=True, stable_neighbor_selection=True, finite_cuda_gradients=True,
        classification_and_ranking_loss_edge_cases=True,
        nonfit_target_label_isolation=True, independent_checkpoint_reload=True,
        reversed_query_tolerance=1e-12, query_scores_computed=False, final_predictions_computed=False))
    print("SMOKE_PASS", flush=True)


def finish():
    dev = json.loads((OUT / "development.json").read_text())
    conf = json.loads((OUT / "confirmation.json").read_text())
    assert dev["status"] == conf["status"] == "COMPLETE"
    best = [max((r for r in dev["records"] if r["task"] == t), key=lambda r: r["objective"]) for t in TASKS]
    b.save(ROOT / "docs/v315_final_status.json", dict(
        status="REQUIRES_INDEPENDENT_DELIVERY_VERIFICATION" if conf["accepted"] else "COMPLETE_NO_CONFIRMED_NEW_MODEL",
        at_utc=datetime.now(timezone.utc).isoformat(), signature=sig(), baseline="v313", official_best=51.205,
        target_score=52., development_best=best, accepted_tasks=conf["accepted"], confirmation=conf["records"],
        csv_created=False, platform_submission=False, official_gain_established=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "run", "report"))
    stage = parser.parse_args().stage
    OUT.mkdir(exist_ok=True)
    if stage == "report":
        finish()
        return
    d = load()
    if stage == "smoke":
        smoke(d)
    else:
        assert json.loads((OUT / "smoke.json").read_text())["signature"] == sig()
        assert not (OUT / "development.json").exists(), "Never silently repeat a scored experiment."
        b.save(OUT / "frozen_before_scores.json", dict(signature=sig(), source_sha256=b.sha(__file__),
            baseline_signature=v.sig(), runner_sha256=b.sha(runner.__file__),
            at_utc=datetime.now(timezone.utc).isoformat(), tasks=TASKS, configs=CONFIGS, seeds=SEEDS,
            packages={p: importlib.metadata.version(p) for p in ("numpy", "torch")},
            no_query_scores_read_for_configuration=True))
        runner.run(sys.modules[__name__], d)
        finish()


if __name__ == "__main__":
    main()
