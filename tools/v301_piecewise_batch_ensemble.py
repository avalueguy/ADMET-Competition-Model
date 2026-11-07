"""From-scratch piecewise numeric embeddings with shared-weight neural ensembles."""
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
OUT = ROOT / "admet_model_outputs_v301"
TASKS = ("caco2_wang", "ppbr_az", "lipophilicity_astrazeneca", "ld50_zhu", "solubility_aqsoldb")
CONFIGS = (dict(members=8, width=192, blocks=3, bins=8, epochs=100, minimum_steps=1200, maximum_steps=3000),)
SEEDS = (3011101, 3011102)
METHOD = (
    "From-scratch shared-weight eight-member neural ensembles for five MAE tasks. "
    "Fit-quantile piecewise linear numeric embedding (8 cells), missing indicators "
    "and clipped fit-IQR scalars encode augmented physicochemistry/shape and target-excluded "
    "official context. Six archived count fingerprint blocks are independently L2-normalized. "
    "Each linear layer has a shared matrix and member-specific learned input/output scalings "
    "initialized by random signs; independent per-member biases and dropout, residual blocks. "
    "Each member is trained on raw MAE; average eight members and two fixed seeds at inference. "
    "No pretrained parameters, external observations, previous predictions as inputs, model "
    "selection on query labels or variable training duration. Original v148 gates unchanged."
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
    return "piecewise_shared8"


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
    assert d["tasks"][k] in TASKS and d["metrics"][k] == "MAE"
    fit, query, x, qx, source = v.old.features(d, k, part, "augmented")
    numeric = fit_numeric(x, c["bins"])
    xn, qn = numeric_transform(x, numeric), numeric_transform(qx, numeric)
    def fp(rows):
        raw = d["views"]["multiview"][d["mi"][rows], :12288].astype(np.float64).reshape(len(rows), 6, 2048)
        assert np.all(raw >= 0) and np.isfinite(raw).all()
        raw /= np.linalg.norm(raw, axis=2, keepdims=True).clip(1e-12)
        return raw.reshape(len(rows), -1).astype(np.float32)
    y = d["y"][fit]
    center = float(np.median(y))
    scale = max(float(np.subtract(*np.quantile(y, [.75, .25]))), .05)
    state = dict(fit=fit, query=query, source=source, center=center, scale=scale,
                 target=((y - center) / scale).astype(np.float32), **numeric)
    assert not set(fit) & set(query) and np.all(d["tid"][source] != k)
    assert np.all(d["split"][source] < (2 if part == "final" else 1))
    assert part == "final" or not set(d["groups"][fit]) & set(d["groups"][query])
    return dict(x=np.c_[fp(fit), xn], qx=np.c_[fp(query), qn], state=state)


class SharedLinear(nn.Module):
    def __init__(self, inputs, outputs, members):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(outputs, inputs))
        self.input_scale = nn.Parameter(torch.empty(members, inputs))
        self.output_scale = nn.Parameter(torch.empty(members, outputs))
        self.bias = nn.Parameter(torch.zeros(members, outputs))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        with torch.no_grad():
            self.input_scale.bernoulli_(.5).mul_(2).sub_(1)
            self.output_scale.bernoulli_(.5).mul_(2).sub_(1)

    def forward(self, x):
        return torch.nn.functional.linear(x * self.input_scale, self.weight) * self.output_scale + self.bias


class Net(nn.Module):
    def __init__(self, inputs, c):
        super().__init__()
        self.members = c["members"]
        self.first = SharedLinear(inputs, c["width"], c["members"])
        self.blocks = nn.ModuleList(SharedLinear(c["width"], c["width"], c["members"]) for _ in range(c["blocks"]))
        self.norms = nn.ModuleList(nn.LayerNorm(c["width"]) for _ in range(c["blocks"]))
        self.head = SharedLinear(c["width"], 1, c["members"])
        self.dropout = nn.Dropout(.1)
        with torch.no_grad():
            self.head.weight.mul_(.01)

    def forward(self, x):
        x = x[:, None].expand(-1, self.members, -1)
        x = torch.nn.functional.relu(self.first(x))
        for layer, norm in zip(self.blocks, self.norms):
            x = x + self.dropout(torch.nn.functional.relu(layer(norm(x)))) / math.sqrt(len(self.blocks))
        return self.head(x).squeeze(-1)


def infer_cpu(net, qx, members=False):
    net = net.cpu().double().eval()
    outputs = []
    with torch.inference_mode():
        for lo in range(0, len(qx), 64):
            p = net(torch.as_tensor(qx[lo:lo + 64], dtype=torch.float64)).numpy()
            outputs.append(p if members else p.mean(axis=1))
    return np.concatenate(outputs)


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
        optimizer = torch.optim.AdamW(net.parameters(), lr=.0008, weight_decay=.01)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=.00003)
        for step in range(steps):
            net.train()
            ix = rng.choice(len(y), min(128, len(y)), replace=False)
            p = net(x[ix])
            loss = torch.mean(torch.abs(p - y[ix, None]))
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
        steps_per_seed=steps, members_per_seed=c["members"], features=x.shape[1],
        seconds=time.monotonic() - started))
    del x, y
    torch.cuda.empty_cache()
    print(json.dumps(dict(stage="ensemble_train", task=d["tasks"][k], part=part,
        seconds=round(time.monotonic() - started))), flush=True)
    return p


def smoke(d):
    raw = np.array([[0., 1., np.nan], [1., 1., 1.], [2., 1., 1.], [3., 1., np.nan]])
    state = fit_numeric(raw, 3)
    encoded = numeric_transform(raw, state)
    np.testing.assert_array_equal(state["columns"], [0, 2])
    np.testing.assert_allclose(encoded[:, :3], [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]], atol=1e-6)
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
    torch.manual_seed(SEEDS[0])
    net = Net(a["x"].shape[1], CONFIGS[0]).cuda()
    x = torch.as_tensor(a["x"][:32], device="cuda")
    target = torch.as_tensor(a["state"]["target"][:32], device="cuda")
    opt = torch.optim.AdamW(net.parameters(), lr=.0008)
    for _ in range(3):
        loss = (net(x) - target[:, None]).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        opt.step()
    path = OUT / "smoke.pt"
    torch.save(dict(signature=sig(), config=CONFIGS[0], weights=net.cpu().state_dict()), path)
    original = infer_cpu(net, a["qx"])
    replayed = infer_cpu(make_loaded(path, a["x"].shape[1], CONFIGS[0]), a["qx"])
    np.testing.assert_array_equal(original, replayed)
    reversed_p = infer_cpu(net, a["qx"][::-1].copy())
    np.testing.assert_allclose(original, reversed_p[::-1], rtol=0, atol=1e-12)
    member = infer_cpu(net, a["qx"][:3], members=True)
    assert np.max(np.ptp(member, axis=1)) > 1e-6
    layer = SharedLinear(3, 2, 2).double()
    inputs = torch.arange(12, dtype=torch.float64).reshape(2, 2, 3)
    reference = torch.stack([torch.nn.functional.linear(inputs[:, i] * layer.input_scale[i], layer.weight)
                             * layer.output_scale[i] + layer.bias[i] for i in range(2)], dim=1)
    torch.testing.assert_close(layer(inputs), reference, rtol=0, atol=1e-12)
    b.save(OUT / "smoke.json", dict(status="PASS", signature=sig(), piecewise_reference=True,
        shared_linear_reference=True, member_diversity=True, finite_cuda_gradients=True,
        nonfit_target_label_isolation=True, independent_checkpoint_reload=True,
        reversed_query_tolerance=1e-12, query_scores_computed=False, final_predictions_computed=False))
    print("SMOKE_PASS", flush=True)


def finish():
    dev = json.loads((OUT / "development.json").read_text())
    conf = json.loads((OUT / "confirmation.json").read_text())
    assert dev["status"] == conf["status"] == "COMPLETE"
    best = [max((r for r in dev["records"] if r["task"] == t), key=lambda r: r["objective"]) for t in TASKS]
    b.save(ROOT / "docs/v301_final_status.json", dict(
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
