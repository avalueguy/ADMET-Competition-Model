"""Sharpness-aware optimization of scaffold-excluded label retrieval for MAE."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import v304_learned_label_retrieval as original
import v314_current_baseline as v
import v314_v313_experiment_runner as runner

np, torch, nn, b, r83, joblib = original.np, original.torch, original.nn, v.b, v.r83, original.joblib
OUT = ROOT / "admet_model_outputs_v329"
TASKS = original.TASKS
CONFIGS = (dict(**original.CONFIGS[0], sam_radius=.05),)
SEEDS = original.SEEDS
METHOD = (
    "Sharpness-aware minimization (SAM) for five MAE endpoints, incremental to exact v313. "
    "Same original v304 retrieval architecture, scaffold/identity-excluded fit label context, "
    "data transforms, seeds, learning rate and optimizer update count. A new two-evaluation "
    "minimax training objective: compute fit MAE gradient, add radius0.05 L2-normalized "
    "weight perturbation, recompute MAE with identical dropout/context masks, restore "
    "the exact original weights and take one AdamW step with the perturbed gradient. "
    "Weights restored by copy, never subtractive rounding. Same two fixed seeds, no "
    "old-model initialization, final checkpoint and no query-label stopping. Original "
    "v148/v314 selection, splits, blends and scaffold bootstrap unchanged."
)
same, design, infer_cpu, output = original.same, original.design, original.infer_cpu, original.output
Net, neighbors = original.Net, original.neighbors


def load():
    original.initialize()
    return v.load()


def sig():
    return hashlib.sha256((b.sha(__file__) + original.sig() + v.sig() + b.sha(runner.__file__) +
                           importlib.metadata.version("torch")).encode()).hexdigest()


def key(c):
    assert c == CONFIGS[0]
    return "sam005_label_context24"


def sam_step(net, optimizer, closure, radius):
    """The closure reads only one already selected fit minibatch."""
    optimizer.zero_grad(set_to_none=True)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state() if next(net.parameters()).is_cuda else None
    loss = closure()
    assert torch.isfinite(loss)
    loss.backward()
    parameters = [p for p in net.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    norm = torch.linalg.vector_norm(torch.stack([p.grad.norm(p=2) for p in parameters]), ord=2)
    originals = [p.detach().clone() for p in parameters]
    with torch.no_grad():
        for p in parameters:
            p.add_(p.grad * (radius / norm.clamp_min(1e-12)))
    try:
        optimizer.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng)
        perturbed = closure()
        assert torch.isfinite(perturbed)
        perturbed.backward()
        nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
    finally:
        with torch.no_grad():
            for p, old in zip(parameters, originals):
                p.copy_(old)
    optimizer.step()
    return float(loss.detach()), float(perturbed.detach()), float(norm.detach())


def make_loaded(path, inputs, c):
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["signature"] == sig() and state["config"] == c
    net = Net(inputs, c)
    net.load_state_dict(state["weights"])
    return net


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
    assert not folder.exists(), "Inspect partial artifacts before retry"
    started = time.monotonic()
    a = design(d, k, part, c)
    folder.mkdir(parents=True)
    joblib.dump(a["state"], folder / "transform.joblib", compress=3)
    assert torch.cuda.is_available()
    x = torch.as_tensor(a["x"], device="cuda")
    y = torch.as_tensor(a["state"]["target"], device="cuda")
    reference_ids = torch.as_tensor(a["state"]["train_ix"], device="cuda")
    similarities = torch.as_tensor(a["state"]["train_similarity"], device="cuda")
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
            context = reference_ids[ix]
            def closure():
                p = net(x[ix], x[context], y[context], similarities[ix])
                return (p - y[ix]).abs().mean()
            normal, perturbed, norm = sam_step(net, optimizer, closure, c["sam_radius"])
            schedule.step()
            if step % 500 == 0:
                print(json.dumps(dict(stage="sam_updates", task=d["tasks"][k], part=part, seed=seed,
                    step=step, steps=steps, fit_mae=normal, perturbed_fit_mae=perturbed, gradient_norm=norm)), flush=True)
        path = folder / f"model_{seed}.pt"
        torch.save(dict(signature=sig(), config=c, steps=steps,
                        weights={n:z.detach().cpu() for n,z in net.state_dict().items()}), path)
        raw = infer_cpu(net, a)
        np.testing.assert_array_equal(raw, infer_cpu(make_loaded(path, x.shape[1], c), a))
        predictions.append(raw)
        hashes[path.name] = b.sha(path)
        del net, optimizer
    p = output(np.mean(predictions, axis=0), a["state"], d["tasks"][k])
    assert np.isfinite(p).all()
    np.savez_compressed(file, signature=sig(), fit=a["state"]["fit"], query=a["state"]["query"], p=p)
    b.save(folder / "metadata.json", dict(signature=sig(), model_hashes=hashes,
        steps_per_seed=steps, gradient_evaluations_per_seed=2*steps, exact_weight_restore=True,
        same_stochastic_masks_per_step=True, seconds=time.monotonic() - started))
    del x, y, reference_ids, similarities
    torch.cuda.empty_cache()
    print(json.dumps(dict(stage="sam_train", task=d["tasks"][k], part=part,
                         seconds=round(time.monotonic() - started))), flush=True)
    return p


def replay(d, k, part, c):
    a = design(d, k, part, c)
    folder = OUT / "models" / d["tasks"][k] / part / key(c)
    same(a["state"], joblib.load(folder / "transform.joblib"))
    z = np.load(folder / "predictions.npz")
    assert str(z["signature"]) == sig()
    nets = [make_loaded(folder / f"model_{seed}.pt", a["x"].shape[1], c) for seed in SEEDS]
    p = output(np.mean([infer_cpu(net, a) for net in nets], axis=0), a["state"], d["tasks"][k])
    np.testing.assert_array_equal(p, z["p"])
    return p


def optimizer_reference():
    # Independent scalar-gradient reference checks both ascent and the final step.
    net = nn.Linear(2, 1, bias=True).double()
    with torch.no_grad():
        net.weight.copy_(torch.tensor([[.4, -.2]], dtype=torch.float64))
        net.bias.fill_(.1)
    x = np.array([[1., 2.], [-1., .5], [.4, -.7]])
    y = np.array([.2, -.5, 1.])
    theta = np.r_[net.weight.detach().numpy().ravel(), net.bias.detach().numpy()]
    xx = np.c_[x, np.ones(len(x))]
    grad = xx.T @ (xx @ theta-y) / len(y)
    ascent = theta + .05 * grad / np.linalg.norm(grad)
    second_grad = xx.T @ (xx @ ascent-y) / len(y)
    expected = theta - .03 * second_grad
    opt = torch.optim.SGD(net.parameters(), lr=.03)
    sam_step(net, opt, lambda:.5*(net(torch.as_tensor(x)).flatten()-torch.as_tensor(y)).square().mean(), .05)
    actual = np.r_[net.weight.detach().numpy().ravel(), net.bias.detach().numpy()]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    # Zero-step optimizer exposes exact restoration, and mask reuse is checked explicitly.
    saved = [p.detach().clone() for p in net.parameters()]
    masks = []
    zero = torch.optim.SGD(net.parameters(), lr=0.)
    def stochastic():
        mask = torch.rand(3, dtype=torch.float64)
        masks.append(mask.clone())
        return (net(torch.as_tensor(x)).flatten()*mask-torch.as_tensor(y)).square().mean()
    sam_step(net, zero, stochastic, .05)
    torch.testing.assert_close(masks[0], masks[1], rtol=0, atol=0)
    for p, old in zip(net.parameters(), saved):
        torch.testing.assert_close(p, old, rtol=0, atol=0)


def smoke(d):
    optimizer_reference()
    c = CONFIGS[0]
    k = d["tasks"].index("caco2_wang")
    a = design(d, k, "fold0", c)
    saved = d["y"].copy()
    try:
        d["y"][(d["tid"] == k) & ~np.isin(np.arange(len(saved)), a["state"]["fit"])] = 999999.
        d.pop("_v96_assays", None)
        altered = design(d, k, "fold0", c)
        same(a["state"], altered["state"])
        np.testing.assert_array_equal(a["x"], altered["x"])
        np.testing.assert_array_equal(a["qx"], altered["qx"])
    finally:
        d["y"][:] = saved
        d.pop("_v96_assays", None)
    torch.manual_seed(SEEDS[0])
    net = Net(a["x"].shape[1], c).cuda()
    ids = a["state"]["train_ix"][:32]
    xx, rr, yy, ss, target = [torch.as_tensor(z, device="cuda") for z in
        (a["x"][:32], a["x"][ids], a["state"]["target"][ids], a["state"]["train_similarity"][:32], a["state"]["target"][:32])]
    opt = torch.optim.AdamW(net.parameters(), lr=.0008)
    for _ in range(3):
        net.train()
        sam_step(net, opt, lambda:(net(xx, rr, yy, ss)-target).abs().mean(), c["sam_radius"])
    path = OUT / "smoke.pt"
    torch.save(dict(signature=sig(), config=c, weights=net.cpu().state_dict()), path)
    raw = infer_cpu(net, a)
    np.testing.assert_array_equal(raw, infer_cpu(make_loaded(path, a["x"].shape[1], c), a))
    reverse = dict(a["state"], query_ix=a["state"]["query_ix"][::-1].copy(), query_similarity=a["state"]["query_similarity"][::-1].copy())
    np.testing.assert_allclose(raw, infer_cpu(net, dict(a, qx=a["qx"][::-1].copy(), state=reverse))[::-1], rtol=0, atol=1e-12)
    b.save(OUT / "smoke.json", dict(status="PASS", signature=sig(), independent_optimizer_gradient_reference=True,
        exact_weight_restore=True, same_stochastic_mask=True, finite_cuda_gradients=True,
        checkpoint_reload=True, nonfit_target_label_isolation=True, reversed_query_tolerance=1e-12,
        query_scores_computed=False, final_predictions_computed=False))
    print("SMOKE_PASS", flush=True)


def finish():
    dev = json.loads((OUT / "development.json").read_text())
    conf = json.loads((OUT / "confirmation.json").read_text())
    assert dev["status"] == conf["status"] == "COMPLETE"
    best = [max((r for r in dev["records"] if r["task"] == t), key=lambda r:r["objective"]) for t in TASKS]
    b.save(ROOT / "docs/v329_final_status.json", dict(
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
        plan = json.loads((ROOT / "docs/v329_training_plan.json").read_text())
        assert plan["source_sha256"] == b.sha(__file__) and plan["original_source_sha256"] == b.sha(original.__file__)
        assert not (OUT / "development.json").exists(), "Never silently repeat a scored experiment"
        b.save(OUT / "frozen_before_scores.json", dict(signature=sig(), source_sha256=b.sha(__file__),
            baseline_signature=v.sig(), runner_sha256=b.sha(runner.__file__), original_source_sha256=b.sha(original.__file__),
            at_utc=datetime.now(timezone.utc).isoformat(), tasks=TASKS, configs=CONFIGS, seeds=SEEDS,
            no_query_scores_read_for_configuration=True))
        runner.run(sys.modules[__name__], d)
        finish()


if __name__ == "__main__":
    main()
