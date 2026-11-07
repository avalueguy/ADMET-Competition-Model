"""Fit-only chemical-domain distributionally robust classification training."""
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
from sklearn.cluster import KMeans

np, torch, nn, b, r83, joblib = original.np, original.torch, original.nn, v.b, v.r83, original.joblib
OUT = ROOT / "admet_model_outputs_v333"
TASKS = ("herg", "cyp2c9_substrate_carbonmangels", "cyp2d6_substrate_carbonmangels")
CONFIGS = (dict(**original.CONFIGS[0], domains=8, dro_eta=.01, dro_fraction=.5),)
SEEDS = original.SEEDS
METHOD = (
    "Chemical-domain group DRO, trained from scratch against exact v313. Fit scaffolds "
    "are represented by their mean normalized 4096-dimensional count fingerprints and "
    "clustered into eight domains by fixed-seed KMeans. No query structures or labels "
    "enter domain fitting. Original scaffold/identity-excluded retrieval network, "
    "same fixed seeds, learning rate and optimizer update count as v315. Each batch "
    "draws 16 rows per fit domain (replacement only for domains under16). Group mean "
    "BCE updates an exponentiated-gradient adversary at eta.01. Loss is half fit-row-"
    "prior ERM plus half adversarial group BCE plus original .25 pairwise rank loss. "
    "This dynamically changes training emphasis, distinct from static scaffold weights "
    "or manifold interpolation. No CYP3A4 retrial, weight sweep, external data, old "
    "checkpoint initialization or query-label stopping. Original v148/v314 acceptance."
)
load, same, Net, infer_cpu, output = original.load, original.same, original.Net, original.infer_cpu, original.output
neighbors = original.neighbors


def sig():
    return hashlib.sha256((b.sha(__file__) + original.sig() + b.sha(runner.__file__) +
        importlib.metadata.version("torch") + importlib.metadata.version("scikit-learn")).encode()).hexdigest()


def key(c):
    assert c == CONFIGS[0]
    return "chemical_domains8_groupdro"


def chemical_domains(d, fit, count):
    groups, inverse = np.unique(d["groups"][fit], return_inverse=True)
    raw = d["views"]["multiview"][d["mi"][fit], :4096].astype(np.float64).copy()
    raw /= np.linalg.norm(raw, axis=1, keepdims=True).clip(1e-12)
    centers = np.stack([raw[inverse == j].mean(axis=0) for j in range(len(groups))])
    centers /= np.linalg.norm(centers, axis=1, keepdims=True).clip(1e-12)
    assert len(groups) >= count
    fitted = KMeans(n_clusters=count, random_state=3040001, n_init=1,
                    max_iter=100, algorithm="lloyd").fit(centers)
    domain = fitted.labels_[inverse].astype(np.int64)
    assert len(np.unique(domain)) == count
    prior = np.bincount(domain, minlength=count).astype(np.float64) / len(domain)
    return dict(domain=domain, domain_prior=prior,
                domain_centers=fitted.cluster_centers_, scaffold_domains=fitted.labels_.astype(np.int64))


def design(d, k, part, c):
    assert d["tasks"][k] in TASKS
    a = original.design(d, k, part, c)
    a["state"].update(chemical_domains(d, a["state"]["fit"], c["domains"]))
    return a


def balanced_batch(domain, rng):
    pieces = []
    for group in range(CONFIGS[0]["domains"]):
        ids = np.flatnonzero(domain == group)
        pieces.append(rng.choice(ids, 16, replace=len(ids) < 16))
    return np.concatenate(pieces)


def dro_loss(raw, target, domain, prior, log_q, c):
    losses = nn.functional.binary_cross_entropy_with_logits(raw, target, reduction="none")
    grouped = torch.stack([losses[domain == j].mean() for j in range(c["domains"])])
    assert torch.isfinite(grouped).all()
    with torch.no_grad():
        log_q = log_q + c["dro_eta"] * grouped.detach().to(log_q.dtype)
        log_q = log_q - torch.logsumexp(log_q, dim=0)
    q = log_q.exp().to(grouped.dtype)
    loss = (1. - c["dro_fraction"]) * (prior * grouped).sum()
    loss += c["dro_fraction"] * (q * grouped).sum()
    positive, negative = raw[target > .5], raw[target < .5]
    if len(positive) and len(negative):
        loss += .25 * nn.functional.softplus(negative[:, None] - positive[None]).mean()
    return loss, log_q, grouped.detach()


def objective_reference():
    c = dict(domains=3, dro_eta=.01, dro_fraction=.5)
    values = np.array([-.7, .4, .2, -.3, 1.2, -.8])
    labels = np.array([0., 1., 1., 0., 1., 0.])
    domains = np.array([0, 0, 1, 1, 2, 2])
    prior = np.array([.5, .3, .2])
    logq = np.log(np.array([.2, .5, .3]))
    raw = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    loss, updated, _ = dro_loss(raw, torch.as_tensor(labels), torch.as_tensor(domains),
        torch.as_tensor(prior), torch.as_tensor(logq), c)
    # Scalar-loop NumPy reference, independent of Torch's grouped objective.
    per_row = np.logaddexp(0., values) - labels * values
    means = np.array([per_row[domains == j].mean() for j in range(3)])
    unnormalized = np.exp(logq + c["dro_eta"] * means)
    weights = unnormalized / unnormalized.sum()
    expected = .5 * np.dot(prior, means) + .5 * np.dot(weights, means)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    expected += .25 * np.mean([np.logaddexp(0., values[n] - values[p]) for n in neg for p in pos])
    np.testing.assert_allclose(float(loss.detach()), expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(updated.exp().numpy(), weights, rtol=0, atol=1e-14)
    loss.backward()
    gradient = np.zeros(len(values))
    for i in range(len(values)):
        g = domains[i]
        gradient[i] = (.5 * prior[g] + .5 * weights[g]) / np.sum(domains == g) * (1. / (1. + np.exp(-values[i])) - labels[i])
    for n in neg:
        for p in pos:
            pair_grad = .25 / (len(neg) * len(pos)) / (1. + np.exp(-(values[n] - values[p])))
            gradient[n] += pair_grad
            gradient[p] -= pair_grad
    np.testing.assert_allclose(raw.grad.numpy(), gradient, rtol=0, atol=1e-14)
    for label in (0., 1.):
        degenerate = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        result = dro_loss(degenerate, torch.full_like(degenerate, label), torch.as_tensor(domains),
                          torch.as_tensor(prior), torch.as_tensor(logq), c)[0]
        result.backward()
        assert torch.isfinite(result) and torch.isfinite(degenerate.grad).all()


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
    ids = torch.as_tensor(a["state"]["train_ix"], device="cuda")
    similarity = torch.as_tensor(a["state"]["train_similarity"], device="cuda")
    domain = torch.as_tensor(a["state"]["domain"], device="cuda")
    prior = torch.as_tensor(a["state"]["domain_prior"], device="cuda", dtype=torch.float32)
    steps = max(c["minimum_steps"], min(c["maximum_steps"], math.ceil(len(y) / 128) * c["epochs"]))
    predictions, hashes, histories = [], {}, []
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)
        net = Net(x.shape[1], c).cuda()
        optimizer = torch.optim.AdamW(net.parameters(), lr=.0008, weight_decay=.01)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=.00003)
        logq = torch.full((c["domains"],), -math.log(c["domains"]), dtype=torch.float64, device="cuda")
        history = []
        for step in range(steps):
            net.train()
            ix = balanced_batch(a["state"]["domain"], rng)
            context = ids[ix]
            raw = net(x[ix], x[context], y[context], similarity[ix]) + a["state"]["center"]
            loss, logq, group_loss = dro_loss(raw, y[ix], domain[ix], prior, logq, c)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            if step % 500 == 0 or step == steps - 1:
                entry = dict(step=step, loss=float(loss.detach()), q=logq.exp().cpu().tolist(),
                             group_bce=group_loss.cpu().tolist())
                history.append(entry)
                print(json.dumps(dict(stage="group_dro_updates", task=d["tasks"][k], part=part,
                                      seed=seed, steps=steps, **entry)), flush=True)
        path = folder / f"model_{seed}.pt"
        torch.save(dict(signature=sig(), config=c, steps=steps, final_adversary=logq.cpu(),
                        weights={n:z.detach().cpu() for n,z in net.state_dict().items()}), path)
        raw = infer_cpu(net, a)
        np.testing.assert_array_equal(raw, infer_cpu(make_loaded(path, x.shape[1], c), a))
        predictions.append(raw)
        histories.append(dict(seed=seed, history=history))
        hashes[path.name] = b.sha(path)
        del net, optimizer
    p = output(np.mean(predictions, axis=0), a["state"], d["tasks"][k])
    assert np.isfinite(p).all()
    np.savez_compressed(file, signature=sig(), fit=a["state"]["fit"], query=a["state"]["query"], p=p)
    b.save(folder / "metadata.json", dict(signature=sig(), model_hashes=hashes,
        steps_per_seed=steps, seconds=time.monotonic() - started, histories=histories,
        domain_counts=np.bincount(a["state"]["domain"]).tolist(), fit_only_domain_construction=True))
    del x, y, ids, similarity, domain, prior
    torch.cuda.empty_cache()
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
    objective_reference()
    c, k = CONFIGS[0], d["tasks"].index("herg")
    a = design(d, k, "fold0", c)
    domain = a["state"]["domain"]
    groups = d["groups"][a["state"]["fit"]]
    assert all(len(np.unique(domain[groups == g])) == 1 for g in np.unique(groups))
    ix = balanced_batch(domain, np.random.default_rng(SEEDS[0]))
    np.testing.assert_array_equal(np.bincount(domain[ix]), np.full(c["domains"], 16))
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
    context = a["state"]["train_ix"][ix]
    xx, rr, yy, ss, target = [torch.as_tensor(z, device="cuda") for z in
        (a["x"][ix], a["x"][context], a["state"]["target"][context],
         a["state"]["train_similarity"][ix], a["state"]["target"][ix])]
    prior = torch.as_tensor(a["state"]["domain_prior"], dtype=torch.float32, device="cuda")
    logq = torch.full((c["domains"],), -math.log(c["domains"]), dtype=torch.float64, device="cuda")
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0008)
    for _ in range(3):
        loss, logq, _ = dro_loss(net(xx, rr, yy, ss) + a["state"]["center"], target,
            torch.as_tensor(domain[ix], device="cuda"), prior, logq, c)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters())
        optimizer.step()
    path = OUT / "smoke.pt"
    torch.save(dict(signature=sig(), config=c, weights=net.cpu().state_dict()), path)
    raw = infer_cpu(net, a)
    np.testing.assert_array_equal(raw, infer_cpu(make_loaded(path, a["x"].shape[1], c), a))
    b.save(OUT / "smoke.json", dict(status="PASS", signature=sig(),
        independent_loss_gradient_and_adversary_reference=True, degenerate_classes_finite=True,
        scaffold_domain_integrity=True, balanced_batch_counts=True, fit_only_domain_rebuild=True,
        nonfit_target_label_isolation=True, finite_cuda_gradients=True, checkpoint_reload=True,
        query_scores_computed=False, final_predictions_computed=False))
    print("SMOKE_PASS", flush=True)


def finish():
    dev = json.loads((OUT / "development.json").read_text())
    conf = json.loads((OUT / "confirmation.json").read_text())
    assert dev["status"] == conf["status"] == "COMPLETE"
    best = [max((r for r in dev["records"] if r["task"] == t), key=lambda r:r["objective"]) for t in TASKS]
    b.save(ROOT / "docs/v333_final_status.json", dict(
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
        plan = json.loads((ROOT / "docs/v333_training_plan.json").read_text())
        assert plan["source_sha256"] == b.sha(__file__) and plan["original_source_sha256"] == b.sha(original.__file__)
        assert not (OUT / "development.json").exists(), "Never silently repeat a scored experiment"
        b.save(OUT / "frozen_before_scores.json", dict(signature=sig(), source_sha256=b.sha(__file__),
            original_source_sha256=b.sha(original.__file__), baseline_signature=v.sig(),
            runner_sha256=b.sha(runner.__file__), at_utc=datetime.now(timezone.utc).isoformat(),
            tasks=TASKS, configs=CONFIGS, seeds=SEEDS, no_query_scores_read_for_configuration=True))
        runner.run(sys.modules[__name__], d)
        finish()


if __name__ == "__main__":
    main()
