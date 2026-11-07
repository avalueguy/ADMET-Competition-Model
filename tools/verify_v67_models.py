"""Reload every final neural checkpoint and independently replay exported predictions."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import admet_v67_data as b
import admet_v67_train as model
from verify_submission import read, verify


def main():
    model.require_gpu()
    d = b.build()
    release = json.loads((b.OUT / "release.json").read_text(encoding="utf-8"))
    fresh = json.loads((b.OUT / "fresh.json").read_text(encoding="utf-8"))
    assert release["signature"] == fresh["signature"] == b.signature()
    query = np.flatnonzero(d["split"] == 2)
    fit = np.flatnonzero(d["split"] < 2)
    result, checks = [], []
    for family in b.P["families"]:
        members = []
        for seed in b.P["training_seeds"]:
            dest = b.OUT / "final" / family / str(seed)
            with np.load(dest / "predictions.npz", allow_pickle=False) as z:
                expected = z["p"].copy()
                np.testing.assert_array_equal(z["fit"], fit)
                np.testing.assert_array_equal(z["query"], query)
            snapshots = []
            for step in b.P["snapshots"]:
                file = dest / f"checkpoint_{step}.pt"
                ck = torch.load(file, map_location="cpu", weights_only=False)
                assert ck["signature"] == b.signature()
                np.testing.assert_array_equal(ck["fit"], fit)
                np.testing.assert_array_equal(ck["query"], query)
                net = model.Net(ck["dim"], ck["tasks"], family).cuda().eval()
                net.load_state_dict(ck["state"])
                x = torch.from_numpy(b.transform(d["x"], ck["pre"])).cuda()
                snapshots.append(model.infer(net, d, x, query, ck["target"]))
                checks.append(dict(family=family, seed=seed, step=step,
                                   checkpoint_sha256=b.sha(file), model_reload_passed=True))
                del net, x
            p = np.mean(snapshots, 0)
            error = float(np.max(np.abs(p-expected)))
            np.testing.assert_allclose(p, expected, atol=2e-5, rtol=2e-5)
            members.append(p)
            b.emit("v67_final_reload", family=family, seed=seed, max_error=error)
        result.append(np.mean(members, 0))
    pred = np.stack(result)
    file = ROOT / release["file"]
    exported = read(file)[1:]
    for k, record in enumerate(fresh["tasks"]):
        ix = np.flatnonzero(d["tid"][query] == k)
        expected = b.blend(record["metric"], pred[0, ix], pred[1, ix], record["final_weight"])
        if record["task"] == "ppbr_az":
            expected = expected.clip(0, 100)
        if record["metric"] in ("AUROC", "AUPRC"):
            expected = expected.clip(0, 1)
        actual = np.array([float(exported[j][4]) for j in ix])
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    audit = verify(file, ROOT / "admet_official/submission_template.csv")
    assert audit["sha256"] == release["sha256"]
    b.save(b.OUT / "model_replay.json", dict(status="PASS", checks=checks,
           independent_saved_preprocessor_reload=True, exported_predictions_replayed=True,
           sha256=audit["sha256"], external_data=False, pretrained_weights=False))
    b.emit("v67_independent_replay", status="PASS", checkpoints=len(checks), sha256=audit["sha256"])


if __name__ == "__main__":
    main()
