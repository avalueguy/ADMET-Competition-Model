"""Prefit only frozen late-task confirmation members; reads no query scores."""
import argparse
import json

import v349_diverse_model_combinations as experiment
import v351_verify_combination_models as verifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start", type=int)
    parser.add_argument("stop", type=int)
    args = parser.parse_args()
    frozen = json.loads((experiment.OUT / "frozen_selection.json").read_text())
    assert frozen["source_sha256"] == experiment.sha(experiment.__file__)
    choices = frozen["selected"][args.start:args.stop]
    assert args.start >= 6 and args.stop <= len(frozen["selected"])
    mods = experiment.modules()
    required = {r["version"] for c in choices for r in c["members"]}
    d = verifier.load_data(required, mods)
    for choice in reversed(choices):
        task = choice["task"]
        k = d["tasks"].index(task)
        # Query target values erased entirely; same fit-only predictions must result.
        for part in ("fold2", "valid"):
            fit, query = experiment.v.split(d, k, part)
            labels = d["y"].copy()
            try:
                d["y"][(d["tid"] == k) & ~experiment.np.isin(experiment.np.arange(len(labels)), fit)] = 999999.
                d.pop("_v96_assays", None)
                for rec in choice["members"]:
                    model = mods[rec["version"]]
                    folder = model.OUT / "models" / task / part / rec["key"]
                    if (folder / "predictions.npz").is_file():
                        continue
                    assert not folder.exists(), "Do not race a partially active fit"
                    print(json.dumps(dict(stage="prefit_start", task=task, part=part, version=rec["version"])), flush=True)
                    model.predict(d, k, part, rec["config"])
                    experiment.cached(d, k, part, rec, mods)
                    print(json.dumps(dict(stage="prefit_complete", task=task, part=part, version=rec["version"], query_scores_read=False)), flush=True)
            finally:
                d["y"][:] = labels
                d.pop("_v96_assays", None)


if __name__ == "__main__":
    main()
