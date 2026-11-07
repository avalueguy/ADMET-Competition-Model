"""CUDA-only, official-data-only masked multitask training and exact CSV export."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "4"

import argparse
import csv
import io
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from torch import nn

import admet_v67_data as b

ROOT, OUT, P = b.ROOT, b.OUT, b.P


class Heads(nn.Module):
    def __init__(self, dim, tasks, private):
        super().__init__()
        self.w = nn.Parameter(torch.randn(tasks, dim, private) * dim**-.5)
        self.bias = nn.Parameter(torch.zeros(tasks, private))
        self.out = nn.Parameter(torch.randn(tasks, private) * private**-.5)
        self.ob = nn.Parameter(torch.zeros(tasks))

    def forward(self, x):
        if x.ndim == 2:
            z = torch.einsum("bi,kih->bkh", x, self.w)
        else:
            z = torch.einsum("bki,kih->bkh", x, self.w)
        z = nn.functional.dropout(nn.functional.silu(z+self.bias), P["dropout"], self.training)
        return (z*self.out).sum(-1)+self.ob


class Net(nn.Module):
    def __init__(self, dim, tasks, family):
        super().__init__()
        self.family = family
        h = P["hidden"]
        def block(left, right):
            return nn.Sequential(nn.Linear(left, right), nn.LayerNorm(right), nn.SiLU(), nn.Dropout(P["dropout"]))
        self.input = block(dim, h)
        self.shared = nn.Sequential(block(h, h), nn.Linear(h, h))
        if family == "gated":
            self.experts = nn.ModuleList([nn.Sequential(block(h, h), nn.Linear(h, h)) for _ in range(3)])
            self.molecule_gate = nn.Linear(h, 3)
            self.task_gate = nn.Parameter(torch.zeros(tasks, 3))
            self.private_scale = nn.Parameter(torch.zeros(tasks, h))
        self.head = Heads(h, tasks, 64)

    def forward(self, x):
        h = self.input(nn.functional.dropout(x, .03, self.training))
        trunk = h + self.shared(h)
        if self.family == "gated":
            experts = torch.stack([layer(trunk) for layer in self.experts], 1)
            gate = (self.molecule_gate(trunk)[:, None] + self.task_gate[None]).softmax(-1)
            task = trunk[:, None] + torch.einsum("bke,beh->bkh", gate, experts)
            task = task + h[:, None] * self.private_scale[None].tanh()
            return self.head(task)
        return self.head(trunk)


def require_gpu():
    assert os.name != "nt" and torch.cuda.is_available(), "Training is authorized on the remote GPU server"
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def targets(d, fit):
    n, tasks = len(d["smiles"]), len(d["tasks"])
    labels = np.zeros((n, tasks), dtype=np.float32)
    observed = np.zeros_like(labels, dtype=bool)
    transforms, task_mols = [], []
    for k, metric in enumerate(d["metrics"]):
        ix = fit[d["tid"][fit] == k]
        assert len(ix), d["tasks"][k]
        tf = b.target(d["y"][ix], metric)
        transforms.append(tf)
        mols = d["mi"][ix]
        unique = np.unique(mols)
        task_mols.append(unique)
        for mol in unique:
            values = d["y"][ix[mols == mol]]
            value = np.mean(values) if metric in ("AUROC", "AUPRC") else np.median(values)
            labels[mol, k] = b.encode(np.array([value]), tf)[0]
            observed[mol, k] = True
    return transforms, task_mols, labels, observed


def infer(net, d, x, query, transforms):
    net.eval()
    molecules = np.unique(d["mi"][query])
    result = np.full((len(d["smiles"]), len(d["tasks"])), np.nan, dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(molecules), 1024):
            mm = molecules[start:start+1024]
            result[mm] = net(x[mm]).float().cpu().numpy()
    p = result[d["mi"][query], d["tid"][query]].astype(float)
    for k, tf in enumerate(transforms):
        ix = d["tid"][query] == k
        if tf["metric"] in ("AUROC", "AUPRC"):
            p[ix] = expit(p[ix])
        elif tf["metric"] == "MAE":
            p[ix] = p[ix]*tf["scale"] + tf["center"]
    assert np.isfinite(p).all()
    return p


def fit(d, rows, query, family, seed, dest):
    dest.mkdir(parents=True, exist_ok=True)
    sig = b.signature()
    cache = dest / "predictions.npz"
    b.assert_excluded(d, rows, query)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            assert str(z["signature"]) == sig
            np.testing.assert_array_equal(z["fit"], rows)
            np.testing.assert_array_equal(z["query"], query)
            return z["p"].copy()
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    tf, indices, labels, mask = targets(d, rows)
    pre = b.preprocess(d, rows)
    x = torch.from_numpy(b.transform(d["x"], pre)).cuda()
    y, observed = torch.from_numpy(labels).cuda(), torch.from_numpy(mask).cuda()
    cls_tasks = torch.tensor([m in ("AUROC", "AUPRC") for m in d["metrics"]], device="cuda")
    mae_tasks = torch.tensor([m == "MAE" for m in d["metrics"]], device="cuda")
    unique = np.unique(d["mi"][rows])
    net = Net(x.shape[1], len(d["tasks"]), family).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=P["learning_rate"], weight_decay=P["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, P["steps"], eta_min=2e-5)
    snapshots, started = [], time.monotonic()
    for step in range(1, P["steps"]+1):
        net.train()
        k = int(rng.integers(len(indices)))
        half = P["batch_size"] // 2
        molecules = np.r_[rng.choice(indices[k], half, replace=True), rng.choice(unique, half, replace=True)]
        pred, true, active = net(x[molecules]), y[molecules], observed[molecules]
        # Normalize each observed task independently without 22 CUDA/CPU
        # synchronizations per update. Unobserved target placeholders are masked.
        huber = nn.functional.smooth_l1_loss(pred, true, beta=.3, reduction="none")
        absolute = .9*(pred-true).abs() + .1*huber
        binary = nn.functional.binary_cross_entropy_with_logits(pred, true, reduction="none")
        element = torch.where(cls_tasks[None], binary, torch.where(mae_tasks[None], absolute, huber))
        count = active.sum(0)
        per_task = (element*active).sum(0)/count.clamp_min(1)
        loss = per_task.sum()/(count > 0).sum().clamp_min(1)
        # Ranking is applied to the uniformly sampled task, rather than allowing
        # common co-observed assays to dominate pairwise terms.
        active_cpu = mask[molecules, k]
        selected = np.flatnonzero(active_cpu)
        z = labels[molecules[selected], k]
        if d["metrics"][k] in ("AUROC", "AUPRC"):
            pos, neg = selected[z > .5], selected[z < .5]
            if len(pos) and len(neg):
                loss = loss + .1*nn.functional.softplus(pred[neg, k][None]-pred[pos, k][:, None]).mean()
        elif d["metrics"][k] == "Spearman":
            diff = z[:, None]-z[None]
            left, right = np.nonzero(np.abs(diff) > .15)
            if len(left):
                sign = torch.tensor(np.sign(diff[left, right]), device="cuda")
                loss = loss + .1*nn.functional.softplus(-sign*(pred[selected[left], k]-pred[selected[right], k])).mean()
        if step % 100 == 0:
            assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 3.)
        opt.step(); scheduler.step()
        if step in P["snapshots"]:
            p = infer(net, d, x, query, tf)
            snapshots.append(p)
            file = dest / f"checkpoint_{step}.pt"
            torch.save(dict(state={key: value.detach().cpu().clone() for key, value in net.state_dict().items()},
                            pre=pre, target=tf, family=family, dim=x.shape[1], tasks=len(d["tasks"]),
                            signature=sig, fit=rows, query=query, seed=seed, step=step), file)
            restored = torch.load(file, map_location="cpu", weights_only=False)
            net.load_state_dict(restored["state"])
            # Recreate the input with persisted preprocessing as well as reloading weights.
            xp = torch.from_numpy(b.transform(d["x"], restored["pre"])).cuda()
            np.testing.assert_allclose(p, infer(net, d, xp, query, restored["target"]), atol=2e-5, rtol=2e-5)
            del xp
        if step % 500 == 0:
            b.emit("v67_fit", partition=str(dest.relative_to(OUT)), step=step,
                   seconds=round(time.monotonic()-started), loss=float(loss.detach()))
    result = np.mean(snapshots, axis=0)
    np.savez_compressed(cache, signature=sig, fit=rows, query=query, p=result)
    b.save(dest / "metadata.json", dict(signature=sig, family=family, seed=seed,
           fit_rows=len(rows), query_rows=len(query), model_reload_passed=True,
           seconds=time.monotonic()-started, external_data=False, pretrained_weights=False))
    del net, x, y, observed, opt
    torch.cuda.empty_cache()
    return result


def ensemble(d, rows, query, family, partition):
    return np.mean([fit(d, rows, query, family, seed, OUT/partition/family/str(seed))
                    for seed in P["training_seeds"]], axis=0)


def evaluate(d):
    folds = b.folds(d)
    cv = np.full((2, len(d["y"])), np.nan)
    for fold in range(P["cv_folds"]):
        query = np.flatnonzero(folds == fold)
        rows = np.flatnonzero((folds >= 0) & (folds != fold))
        for i, family in enumerate(P["families"]):
            cv[i, query] = ensemble(d, rows, query, family, f"cv/fold{fold}")
        np.save(OUT / "cv_predictions.npy", cv)
    records = []
    for k, task in enumerate(d["tasks"]):
        metric = d["metrics"][k]
        tr = np.flatnonzero((d["tid"] == k) & (d["split"] == 0))
        grid = []
        for weight in P["weights"]:
            gains = []
            for fold in range(P["cv_folds"]):
                ix = tr[folds[tr] == fold]
                p = b.blend(metric, cv[0, ix], cv[1, ix], weight)
                gains.append(b.gain(metric, d["y"][ix], cv[0, ix], p))
            grid.append(dict(weight=weight, gains=gains, mean=float(np.mean(gains))))
        selected = max(grid, key=lambda row: row["mean"]-.1*np.std(row["gains"]))
        held = [max(grid, key=lambda row: np.mean([v for j, v in enumerate(row["gains"]) if j != fold]))["gains"][fold]
                for fold in range(P["cv_folds"])]
        if np.mean(held) <= 0 or sum(g > 0 for g in selected["gains"]) < 2:
            selected = grid[0]
        record = dict(task=task, metric=metric, selection=selected, held_gains=held,
                      shared_cv=[b.metric_score(metric, d["y"][tr[folds[tr] == f]], cv[0, tr[folds[tr] == f]]) for f in range(3)],
                      gated_cv=[b.metric_score(metric, d["y"][tr[folds[tr] == f]], cv[1, tr[folds[tr] == f]]) for f in range(3)])
        records.append(record)
        b.emit("v67_selection", **record)
    b.save(OUT / "selection.json", dict(status="FROZEN", signature=b.signature(), tasks=records,
           official_valid_used_for_selection=False, full_historical_ensemble_comparison=False))
    return records


def confirm(d, records):
    allrows = np.flatnonzero(d["split"] < 2)
    groups = np.unique(d["rg"][allrows])
    held = np.random.default_rng(P["fresh_seed"]).permutation(groups)[:max(1, int(len(groups)*P["fresh_fraction"]))]
    mask = np.isin(d["rg"][allrows], held)
    rows, query = allrows[~mask], allrows[mask]
    pred = np.stack([ensemble(d, rows, query, family, "fresh") for family in P["families"]])
    result = []
    for k, record in enumerate(records):
        ix = np.flatnonzero(d["tid"][query] == k)
        y = d["y"][query[ix]]
        weight = record["selection"]["weight"]
        merged = b.blend(record["metric"], pred[0, ix], pred[1, ix], weight)
        gain = b.gain(record["metric"], y, pred[0, ix], merged)
        passed = bool(np.isfinite(gain) and gain > 0) if weight > 0 else True
        result.append(dict(**record, fresh_gain=gain, fresh_pass=passed,
                           final_weight=weight if passed else 0.0, fresh_rows=len(ix)))
    b.save(OUT / "fresh.json", dict(status="COMPLETE", signature=b.signature(), tasks=result,
           global_scaffold_overlap=0, canonical_overlap=0))
    # Official valid is only a report, after the weight and gate are frozen.
    query = np.flatnonzero(d["split"] == 1)
    rows = np.flatnonzero((d["split"] == 0) & ~np.isin(d["rg"], np.unique(d["rg"][query])))
    pred = np.stack([ensemble(d, rows, query, family, "valid") for family in P["families"]])
    report = []
    for k, record in enumerate(result):
        ix = np.flatnonzero(d["tid"][query] == k)
        y = d["y"][query[ix]]; metric = record["metric"]
        p = b.blend(metric, pred[0, ix], pred[1, ix], record["final_weight"])
        report.append(dict(task=record["task"], metric=metric, shared=b.metric_score(metric, y, pred[0, ix]),
                           gated=b.metric_score(metric, y, pred[1, ix]), selected=b.metric_score(metric, y, p)))
    b.save(OUT / "valid_metrics.json", dict(status="REPORT_ONLY", tasks=report))
    return result


def export(d, records, team, date):
    rows, query = np.flatnonzero(d["split"] < 2), np.flatnonzero(d["split"] == 2)
    pred = np.stack([ensemble(d, rows, query, family, "final") for family in P["families"]])
    assert len(query) == len(d["template"])
    result = []
    audit = []
    for k, record in enumerate(records):
        ix = np.flatnonzero(d["tid"][query] == k)
        p = b.blend(record["metric"], pred[0, ix], pred[1, ix], record["final_weight"])
        if record["task"] == "ppbr_az":
            p = p.clip(0, 100)
        if record["metric"] in ("AUROC", "AUPRC"):
            p = p.clip(0, 1)
        assert np.isfinite(p).all()
        for position, value in zip(ix, p):
            expected = d["template"][position]
            assert expected["task"] == record["task"]
            row = expected.copy(); row["y_pred"] = repr(float(value)); result.append(row)
        audit.append(dict(task=record["task"], final_weight=record["final_weight"], rows=len(ix),
                          shared=pred[0, ix].tolist(), gated=pred[1, ix].tolist(), result=p.tolist()))
    # Assert task-loop ordering equals the template before writing.
    assert [(r["task"], r["sample_idx"]) for r in result] == [(r["task"], r["sample_idx"]) for r in d["template"]]
    dest = ROOT / "admet_submissions/v67_official_only"
    dest.mkdir(parents=True, exist_ok=True)
    file = dest / f"{team}-小分子药物预测技术创新赛-{date}.csv"
    buf = io.StringIO(newline=""); writer = csv.DictWriter(buf, fieldnames=list(d["template"][0]), lineterminator="\r\n")
    writer.writeheader(); writer.writerows(result)
    raw = b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8")
    if file.exists():
        assert file.read_bytes() == raw
    else:
        file.write_bytes(raw)
    b.save(OUT / "fusion_replay.json", audit)
    b.save(OUT / "release.json", dict(status="AWAITING_PLATFORM_EVALUATION", signature=b.signature(),
           file=str(file.relative_to(ROOT)), sha256=b.sha(file), rows=len(result), tasks=len(records),
           external_data=False, pretrained_weights=False, gpu=torch.cuda.get_device_name(),
           official_score=None, automatically_uploaded=False))
    b.emit("v67_export", file=str(file), sha256=b.sha(file), rows=len(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "evaluate", "confirm", "export", "smoke"))
    parser.add_argument("--team", default="我从山北来带着兰花草")
    parser.add_argument("--date", default=datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d"))
    args = parser.parse_args()
    require_gpu(); OUT.mkdir(parents=True, exist_ok=True)
    d = b.build()
    if args.stage == "smoke":
        state = b.preprocess(d, np.flatnonzero(d["split"] == 0))
        x = torch.from_numpy(b.transform(d["x"][:4], state)).cuda()
        for family in P["families"]:
            net = Net(x.shape[1], len(d["tasks"]), family).cuda().eval()
            with torch.inference_mode():
                pred, single = net(x), net(x[:1])
            np.testing.assert_allclose(pred[:1].cpu(), single.cpu(), atol=1e-4, rtol=1e-4)
            assert torch.isfinite(pred).all() and pred.shape == (4, len(d["tasks"]))
            net.train(); net(x).square().mean().backward()
        # A short timed training slice uses its own directory, then resets the
        # protocol in memory. Its outputs cannot enter any CV/final cache.
        steps, snaps = P["steps"], P["snapshots"]
        P["steps"], P["snapshots"] = 100, [100]
        f = b.folds(d)
        rows = np.flatnonzero((f >= 0) & (f != 0))
        query = np.flatnonzero(f == 0)[:256]
        try:
            fit(d, rows, query, "gated", 679999, OUT/"smoke_fit")
        finally:
            P["steps"], P["snapshots"] = steps, snaps
        b.save(OUT/"smoke.json", dict(status="PASS", gpu=torch.cuda.get_device_name(),
               forward_backward=True, batch_invariance=True, signature=b.signature()))
        return
    if args.stage in ("all", "evaluate"):
        records = evaluate(d)
    else:
        records = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))["tasks"]
    if args.stage in ("all", "confirm"):
        records = confirm(d, records)
    elif args.stage == "export":
        fresh = json.loads((OUT/"fresh.json").read_text(encoding="utf-8"))
        assert fresh["status"] == "COMPLETE" and fresh["signature"] == b.signature()
        records = fresh["tasks"]
    if args.stage in ("all", "export"):
        export(d, records, args.team, args.date)


if __name__ == "__main__":
    main()
