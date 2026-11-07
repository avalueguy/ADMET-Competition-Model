"""Jointly optimized sparse feature selectors, soft thresholds and tree leaves."""
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
import v146_current_baseline as v
import v148_experiment_runner as runner

b, r83 = v.b, v.r83
OUT = ROOT / "admet_model_outputs_v307"
TASKS = ("caco2_wang", "ppbr_az", "lipophilicity_astrazeneca", "ld50_zhu", "solubility_aqsoldb")
CONFIGS = (dict(trees=64, depth=5, fingerprint_columns=512, quantiles=32,
                epochs=100, minimum_steps=1200, maximum_steps=2500),)
SEEDS = (3071101, 3071102)
METHOD = (
    "Jointly differentiable oblivious-tree ensemble for five MAE endpoints. "
    "Each of 64 depth5 trees learns sparse entmax1.5 feature selectors at each "
    "level, sigmoid thresholds/temperatures and 32 leaf values jointly with raw "
    "MAE, rather than greedy gradient boosting or a frozen topology refit. "
    "Inputs are 512 fit-variance-selected archived count fingerprint columns, "
    "official physicochemical/shape descriptors and target-excluded measured "
    "context, transformed by fit-only empirical quantiles and missing indicators. "
    "Fit-only projected-feature quantiles initialize thresholds/scales. Two fixed "
    "seeds, fixed training duration, tree dropout; no old predictions, retrieved "
    "target labels, external data or pretrained weights. Original v148 unchanged."
)


def initialize():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)


def sig():
    versions = "|".join(importlib.metadata.version(p) for p in ("numpy", "torch"))
    return hashlib.sha256((b.sha(__file__) + v.sig() + b.sha(runner.__file__) + versions).encode()).hexdigest()


def key(c):
    assert c == CONFIGS[0]
    return "joint_entmax_trees64_depth5"


def load():
    initialize()
    return v.load()


def same(a, z):
    assert a.keys() == z.keys()
    for name in a:
        np.testing.assert_array_equal(a[name], z[name])


def quantile_state(raw, quantiles):
    raw = raw.astype(np.float64)
    columns, values = [], []
    for j, col in enumerate(raw.T):
        finite = col[np.isfinite(col)]
        if not len(finite):
            continue
        knots = np.unique(np.quantile(finite, np.linspace(0, 1, quantiles + 1)))
        if len(knots) == 1 and len(finite) == len(raw):
            continue
        columns.append(j)
        values.append(knots)
    return dict(columns=np.asarray(columns, dtype=np.int64), knots=np.concatenate(values),
                offsets=np.r_[0, np.cumsum([len(k) for k in values])])


def transform(raw, state):
    values = raw[:, state["columns"]].astype(np.float64)
    out = np.zeros_like(values)
    for j in range(values.shape[1]):
        lo, hi = state["offsets"][j:j + 2]
        knots = state["knots"][lo:hi]
        finite = np.isfinite(values[:, j])
        if len(knots) > 1:
            out[finite, j] = np.interp(values[finite, j], knots, np.linspace(-1, 1, len(knots)))
    result = np.c_[out, (~np.isfinite(values)).astype(float)].astype(np.float32)
    assert np.isfinite(result).all()
    return result


def design(d, k, part, c):
    assert d["tasks"][k] in TASKS and d["metrics"][k] == "MAE"
    fit, query, x, qx, source = v.old.features(d, k, part, "augmented")
    fingerprints = d["views"]["multiview"][:, :12288]
    variance = fingerprints[d["mi"][fit]].astype(np.float64).var(axis=0)
    selected = np.sort(np.argsort(-variance, kind="stable")[:c["fingerprint_columns"]])
    raw = np.c_[x, fingerprints[d["mi"][fit]][:, selected]].astype(np.float64)
    query_raw = np.c_[qx, fingerprints[d["mi"][query]][:, selected]].astype(np.float64)
    state = quantile_state(raw, c["quantiles"])
    y = d["y"][fit]
    center = float(np.median(y))
    scale = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05)
    state.update(fit=fit, query=query, source=source, selected=selected, center=center,
                 scale=scale, target=((y - center) / scale).astype(np.float32))
    assert not set(fit) & set(query) and np.all(d["tid"][source] != k)
    assert np.all(d["split"][source] < (2 if part == "final" else 1))
    assert part == "final" or not set(d["groups"][fit]) & set(d["groups"][query])
    return dict(x=transform(raw, state), qx=transform(query_raw, state), state=state)


def entmax15(logits):
    """Exact alpha=1.5 entmax threshold; final axis is feature choice."""
    x = logits / 2.
    sorted_x = x.sort(dim=-1, descending=True).values
    rho = torch.arange(1, x.shape[-1] + 1, dtype=x.dtype, device=x.device)
    mean = sorted_x.cumsum(-1) / rho
    mean_square = sorted_x.square().cumsum(-1) / rho
    delta = ((1. - rho * (mean_square - mean.square())) / rho).clamp_min(0.)
    tau = mean - delta.sqrt()
    count = (tau <= sorted_x).sum(-1, keepdim=True).clamp_min(1)
    threshold = tau.gather(-1, count - 1)
    return (x - threshold).clamp_min(0.).square()


class Net(nn.Module):
    def __init__(self, inputs, c):
        super().__init__()
        self.feature_logits = nn.Parameter(torch.randn(c["trees"], c["depth"], inputs) * .5)
        self.threshold = nn.Parameter(torch.zeros(c["trees"], c["depth"]))
        self.log_temperature = nn.Parameter(torch.zeros(c["trees"], c["depth"]))
        self.leaves = nn.Parameter(torch.randn(c["trees"], 2 ** c["depth"]) * .1)
        self.bias = nn.Parameter(torch.zeros(()))
        bits = ((torch.arange(2 ** c["depth"])[:, None] >> torch.arange(c["depth"])) & 1).float()
        self.register_buffer("bits", bits)
        self.dropout = nn.Dropout(.1)

    def probabilities(self, x):
        selector = entmax15(self.feature_logits)
        projected = torch.einsum("bf,tdf->btd", x, selector)
        z = (projected - self.threshold) / self.log_temperature.exp().clamp(.03, 3.)
        yes, no = nn.functional.logsigmoid(z), nn.functional.logsigmoid(-z)
        logprob = torch.einsum("btd,ld->btl", yes, self.bits)
        logprob += torch.einsum("btd,ld->btl", no, 1. - self.bits)
        return logprob.exp()

    def forward(self, x):
        tree = (self.probabilities(x) * self.leaves[None]).sum(-1)
        return self.dropout(tree).mean(-1) + self.bias

    def initialize_thresholds(self, x):
        with torch.no_grad():
            values = torch.einsum("bf,tdf->btd", x, entmax15(self.feature_logits))
            # One deterministic fit quantile per tree-level; no query input.
            levels = torch.linspace(.2, .8, self.threshold.numel(), device=x.device).reshape(self.threshold.shape)
            flattened = values.reshape(len(x), -1)
            positions = torch.round(levels.flatten() * (len(x) - 1)).long()
            ordered = flattened.sort(dim=0).values
            threshold = ordered[positions, torch.arange(flattened.shape[1], device=x.device)]
            width = torch.quantile(flattened, .75, dim=0) - torch.quantile(flattened, .25, dim=0)
            self.threshold.copy_(threshold.reshape_as(self.threshold))
            self.log_temperature.copy_((width * .3).clamp_min(.05).log().reshape_as(self.threshold))


def infer_cpu(net, qx):
    net = net.cpu().double().eval()
    with torch.inference_mode():
        return np.concatenate([net(torch.as_tensor(qx[i:i + 64].copy(), dtype=torch.float64)).numpy()
                               for i in range(0, len(qx), 64)])


def make_loaded(path, inputs, c):
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["signature"] == sig() and state["config"] == c
    net = Net(inputs, c)
    net.load_state_dict(state["weights"])
    return net


def output(raw, state, task):
    p = raw * state["scale"] + state["center"]
    return p.clip(0, 100) if task == "ppbr_az" else p


def replay(d, k, part, c):
    a = design(d, k, part, c)
    folder = OUT / "models" / d["tasks"][k] / part / key(c)
    same(a["state"], joblib.load(folder / "transform.joblib"))
    z = np.load(folder / "predictions.npz")
    assert str(z["signature"]) == sig()
    models = [make_loaded(folder / f"model_{seed}.pt", a["x"].shape[1], c) for seed in SEEDS]
    raw = np.mean([infer_cpu(net, a["qx"]) for net in models], axis=0)
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
    steps = max(c["minimum_steps"], min(c["maximum_steps"], math.ceil(len(y) / 128) * c["epochs"]))
    predictions, hashes = [], {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        net = Net(x.shape[1], c).cuda()
        net.initialize_thresholds(x)
        optimizer = torch.optim.AdamW(net.parameters(), lr=.005, weight_decay=.0001)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=.0003)
        for step in range(steps):
            net.train()
            ix = rng.choice(len(y), min(128, len(y)), replace=False)
            loss = torch.mean(torch.abs(net(x[ix]) - y[ix]))
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            if step % 500 == 0:
                print(json.dumps(dict(stage="updates", task=d["tasks"][k], part=part, seed=seed,
                                     step=step, steps=steps, fit_mae=float(loss.detach()))), flush=True)
        path = folder / f"model_{seed}.pt"
        torch.save(dict(signature=sig(), config=c, steps=steps,
                        weights={n: z.detach().cpu() for n, z in net.state_dict().items()}), path)
        expected = infer_cpu(net, a["qx"])
        actual = infer_cpu(make_loaded(path, x.shape[1], c), a["qx"])
        np.testing.assert_array_equal(expected, actual)
        predictions.append(actual)
        hashes[path.name] = b.sha(path)
        del net, optimizer
    p = output(np.mean(predictions, axis=0), a["state"], d["tasks"][k])
    assert np.isfinite(p).all()
    np.savez_compressed(file, signature=sig(), fit=a["state"]["fit"], query=a["state"]["query"], p=p)
    b.save(folder / "metadata.json", dict(signature=sig(), model_hashes=hashes,
        steps_per_seed=steps, features=x.shape[1], seconds=time.monotonic() - started))
    del x, y
    torch.cuda.empty_cache()
    print(json.dumps(dict(stage="tree_train", task=d["tasks"][k], part=part,
                          seconds=round(time.monotonic() - started))), flush=True)
    return p


def entmax_reference(logits):
    result = []
    for row in np.asarray(logits).reshape(-1, logits.shape[-1]):
        z = row / 2.
        lo, hi = float(z.min() - 2.), float(z.max())
        for _ in range(100):
            mid = (lo + hi) / 2.
            if np.maximum(z - mid, 0.).dot(np.maximum(z - mid, 0.)) > 1.:
                lo = mid
            else:
                hi = mid
        result.append(np.maximum(z - (lo + hi) / 2., 0.) ** 2)
    return np.array(result).reshape(logits.shape)


def smoke(d):
    torch.manual_seed(SEEDS[0])
    logits = torch.randn(3, 4, 11, dtype=torch.float64)
    np.testing.assert_allclose(entmax15(logits).numpy(), entmax_reference(logits.numpy()), atol=1e-12, rtol=0)
    torch.autograd.gradcheck(entmax15, (logits.requires_grad_(),))
    raw = np.array([[0., 5., np.nan], [1., 5., 7.], [2., 5., 7.], [3., 5., np.nan]])
    s = quantile_state(raw, 3)
    np.testing.assert_array_equal(s["columns"], [0, 2])
    np.testing.assert_allclose(transform(raw, s)[:, 0], [-1, -1 / 3, 1 / 3, 1], atol=1e-7, rtol=0)
    k = d["tasks"].index("caco2_wang")
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
    net = Net(a["x"].shape[1], CONFIGS[0]).cuda()
    x = torch.as_tensor(a["x"][:32], device="cuda")
    net.initialize_thresholds(torch.as_tensor(a["x"], device="cuda"))
    optimizer = torch.optim.AdamW(net.parameters(), lr=.005)
    for _ in range(3):
        loss = (net(x) - torch.as_tensor(a["state"]["target"][:32], device="cuda")).abs().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        optimizer.step()
    net = net.cpu().double().eval()
    with torch.inference_mode():
        probability = net.probabilities(torch.as_tensor(a["qx"][:3], dtype=torch.float64)).numpy()
    np.testing.assert_allclose(probability.sum(-1), 1., atol=1e-12, rtol=0)
    path = OUT / "smoke.pt"
    torch.save(dict(signature=sig(), config=CONFIGS[0], weights=net.float().state_dict()), path)
    original = infer_cpu(net.float(), a["qx"])
    reloaded = infer_cpu(make_loaded(path, a["x"].shape[1], CONFIGS[0]), a["qx"])
    np.testing.assert_array_equal(original, reloaded)
    np.testing.assert_allclose(original, infer_cpu(net, a["qx"][::-1].copy())[::-1], atol=1e-12, rtol=0)
    b.save(OUT / "smoke.json", dict(status="PASS", signature=sig(), entmax_bisection_reference=True,
        entmax_gradient_check=True, quantile_reference=True, probability_conservation=True,
        finite_cuda_gradients=True, nonfit_target_label_isolation=True, checkpoint_reload=True,
        reversed_query_tolerance=1e-12, query_scores_computed=False, final_predictions_computed=False))
    print("SMOKE_PASS", flush=True)


def finish():
    dev = json.loads((OUT / "development.json").read_text())
    conf = json.loads((OUT / "confirmation.json").read_text())
    assert dev["status"] == conf["status"] == "COMPLETE"
    best = [max((r for r in dev["records"] if r["task"] == t), key=lambda r: r["objective"]) for t in TASKS]
    b.save(ROOT / "docs/v307_final_status.json", dict(
        status="REQUIRES_INDEPENDENT_DELIVERY_VERIFICATION" if conf["accepted"] else "COMPLETE_NO_CONFIRMED_NEW_MODEL",
        at_utc=datetime.now(timezone.utc).isoformat(), signature=sig(), baseline="v145", official_best=51.166,
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
