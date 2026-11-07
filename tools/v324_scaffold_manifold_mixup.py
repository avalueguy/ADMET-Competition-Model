"""Cross-scaffold hidden-representation mixup for overfitting classifiers."""
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
import v315_rank_label_retrieval as original
import v314_current_baseline as v
import v314_v313_experiment_runner as runner

np, torch, nn, b, r83 = original.np, original.torch, original.nn, v.b, v.r83
joblib = original.joblib
OUT = ROOT / "admet_model_outputs_v324"
TASKS = ("herg", "cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels")
CONFIGS = (dict(neighbors=24, width=128, key_width=32, bins=8, epochs=100,
                minimum_steps=1200, maximum_steps=2500, mixup_alpha=.4),)
SEEDS = original.SEEDS
METHOD = (
    "Cross-scaffold manifold mixup on label-context retrieval representations for three "
    "classification tasks with observed interpolation overfit. Identical v315 encoder, "
    "attention, fit-label retrieval, seeds and duration; a new training mechanism mixes "
    "the query/context hidden vector and weighted label mean with a different-scaffold "
    "fit batch partner, Beta(.4,.4) coefficients and convex soft binary targets. Equal "
    "ordinary/mixed BCE plus .25 original positive-negative rank loss. No mixing of "
    "validation/query labels or query-fitted transforms. No cyp3a4 retrial or changes to "
    "the existing two MAE improvements. Original v148/v314 selection and confirmation."
)
load, same, infer_cpu, output = original.load, original.same, original.infer_cpu, original.output
neighbors = original.neighbors


def sig():
    return hashlib.sha256((b.sha(__file__) + original.sig() + b.sha(runner.__file__) +
                           importlib.metadata.version("torch")).encode()).hexdigest()


def key(c):
    assert c == CONFIGS[0]
    return "scaffold_manifold_mixup04"


def design(d, k, part, c):
    assert d["tasks"][k] in TASKS and d["metrics"][k] in ("AUROC", "AUPRC")
    return original.design(d, k, part, c)


class Net(original.Net):
    def representation(self, query, reference, labels, similarity):
        qh, qk = self.encoded(query)
        rk = self.encoded(reference.flatten(0, 1))[1].reshape(len(query), reference.shape[1], -1)
        difference = qk[:, None] - rk
        score = -difference.square().mean(dim=-1) + 4. * similarity.clamp_min(1e-4).log()
        if self.training:
            mask = torch.rand_like(labels) > .15
            mask[:, 0] = True
            score = score.masked_fill(~mask, -1e9)
        weight = score.softmax(dim=1)
        value = self.label(labels[..., None]) + self.difference(difference)
        context = (weight[..., None] * value).sum(dim=1)
        return torch.cat([qh, context], dim=1), (weight * labels).sum(dim=1)

    def from_representation(self, hidden, mean):
        return self.head(hidden).squeeze(-1) + self.context_coefficient * mean


def mixup_partners(groups, rng):
    return np.array([rng.choice(np.flatnonzero(groups != group)) for group in groups], dtype=np.int64)


def training_loss(net, hidden, mean, target, center, partners, coefficients):
    raw = net.from_representation(hidden, mean) + center
    mixed_hidden = coefficients[:, None] * hidden + (1. - coefficients[:, None]) * hidden[partners]
    mixed_mean = coefficients * mean + (1. - coefficients) * mean[partners]
    mixed_target = coefficients * target + (1. - coefficients) * target[partners]
    mixed = net.from_representation(mixed_hidden, mixed_mean) + center
    loss = .5 * nn.functional.binary_cross_entropy_with_logits(raw, target)
    loss += .5 * nn.functional.binary_cross_entropy_with_logits(mixed, mixed_target)
    positive, negative = raw[target > .5], raw[target < .5]
    if len(positive) and len(negative):
        loss += .25 * nn.functional.softplus(negative[:, None] - positive[None]).mean()
    return loss


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
    x, y = [torch.as_tensor(a[name] if name == "x" else a["state"]["target"], device="cuda") for name in ("x", "y")]
    neighbors = torch.as_tensor(a["state"]["train_ix"], device="cuda")
    similarity = torch.as_tensor(a["state"]["train_similarity"], device="cuda")
    groups = d["groups"][a["state"]["fit"]]
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
            partners = torch.as_tensor(mixup_partners(groups[ix], rng), device="cuda")
            coefficients = torch.as_tensor(rng.beta(c["mixup_alpha"], c["mixup_alpha"], size=len(ix)), device="cuda", dtype=torch.float32)
            context = neighbors[ix]
            hidden, mean = net.representation(x[ix], x[context], y[context], similarity[ix])
            loss = training_loss(net, hidden, mean, y[ix], a["state"]["center"], partners, coefficients)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            if step % 500 == 0:
                print(json.dumps(dict(stage="updates", task=d["tasks"][k], part=part, seed=seed,
                    step=step, steps=steps, fit_mixup_loss=float(loss.detach()))), flush=True)
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
        steps_per_seed=steps, seconds=time.monotonic() - started, cross_scaffold_partners=True))
    del x, y, neighbors, similarity
    torch.cuda.empty_cache()
    print(json.dumps(dict(stage="mixup_train", task=d["tasks"][k], part=part,
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


def smoke(d):
    c = CONFIGS[0]
    k = d["tasks"].index("herg")
    a = design(d, k, "fold0", c)
    n = 32
    rng = np.random.default_rng(SEEDS[0])
    partners = mixup_partners(d["groups"][a["state"]["fit"]][:n], rng)
    assert np.all(d["groups"][a["state"]["fit"]][:n] != d["groups"][a["state"]["fit"]][partners])
    coefficients = rng.beta(c["mixup_alpha"], c["mixup_alpha"], n)
    yy = a["state"]["target"][:n]
    mixed = coefficients * yy + (1. - coefficients) * yy[partners]
    assert np.all((mixed >= 0) & (mixed <= 1))
    torch.manual_seed(SEEDS[0])
    net = Net(a["x"].shape[1], c).cuda()
    ix = a["state"]["train_ix"][:n]
    query, reference, labels, similarity = [torch.as_tensor(z, device="cuda") for z in
        (a["x"][:n], a["x"][ix], a["state"]["target"][ix], a["state"]["train_similarity"][:n])]
    net.eval()
    hidden, mean = net.representation(query, reference, labels, similarity)
    np.testing.assert_allclose(net.from_representation(hidden, mean).detach().cpu().numpy(),
        net(query, reference, labels, similarity).detach().cpu().numpy(), rtol=0, atol=1e-6)
    net.train()
    hidden, mean = net.representation(query, reference, labels, similarity)
    loss = training_loss(net, hidden, mean, torch.as_tensor(yy, device="cuda"), a["state"]["center"],
        torch.as_tensor(partners, device="cuda"), torch.as_tensor(coefficients, device="cuda", dtype=torch.float32))
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
    path = OUT / "smoke.pt"
    torch.save(dict(signature=sig(), config=c, weights=net.cpu().state_dict()), path)
    np.testing.assert_array_equal(infer_cpu(net, a), infer_cpu(make_loaded(path, a["x"].shape[1], c), a))
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
    b.save(OUT / "smoke.json", dict(status="PASS", signature=sig(), cross_scaffold_partners=True,
        convex_soft_targets=True, inference_algebra_equivalence=True, finite_cuda_gradients=True,
        checkpoint_reload=True, nonfit_target_label_isolation=True, query_scores_computed=False))
    print("SMOKE_PASS", flush=True)


def finish():
    dev = json.loads((OUT / "development.json").read_text())
    conf = json.loads((OUT / "confirmation.json").read_text())
    assert dev["status"] == conf["status"] == "COMPLETE"
    best = [max((r for r in dev["records"] if r["task"] == t), key=lambda r:r["objective"]) for t in TASKS]
    b.save(ROOT / "docs/v324_final_status.json", dict(
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
        assert not (OUT / "development.json").exists(), "Never silently repeat a scored experiment"
        b.save(OUT / "frozen_before_scores.json", dict(signature=sig(), source_sha256=b.sha(__file__),
            baseline_signature=v.sig(), runner_sha256=b.sha(runner.__file__),
            at_utc=datetime.now(timezone.utc).isoformat(), tasks=TASKS, configs=CONFIGS, seeds=SEEDS,
            original_source_sha256=b.sha(original.__file__), no_query_scores_read_for_configuration=True))
        runner.run(sys.modules[__name__], d)
        finish()


if __name__ == "__main__":
    main()
