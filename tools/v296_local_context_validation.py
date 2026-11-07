"""Run the unchanged v295 hypothesis on Windows with explicit UTF-8.

v296 identifies a local validation run, not a new trained model. Historical
sources, model files, signatures, folds, weights and acceptance gates stay exact.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import v295_v145_context_integration as model
from verify_submission import verify

OUT = ROOT / "admet_model_outputs_v296"
model.OUT = OUT


def now():
    return datetime.now(timezone.utc).isoformat()


def source_manifest():
    lookup = {p.stem: p for p in [*ROOT.glob("*.py"), *(ROOT / "tools").glob("*.py")]}
    pending = ["v295_v145_context_integration"]
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        tree = ast.parse(lookup[name].read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [x.name.split(".")[0] for x in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            pending.extend(x for x in names if x in lookup)
    files = [lookup[x] for x in sorted(seen)] + [Path(__file__).resolve()]
    return {p.relative_to(ROOT).as_posix(): model.b.sha(p) for p in files}


def freeze():
    assert sys.flags.utf8_mode, "Launch with python -X utf8; historical files are UTF-8."
    manifest = source_manifest()
    report = dict(
        status="FROZEN_BEFORE_LOCAL_SCORES", at_utc=now(),
        validation_version="v296", hypothesis_version="v295", baseline="v145",
        baseline_signature=model.v.sig(), component_signature=model.original.sig(),
        experiment_signature=model.sig(), files=manifest,
        platform=platform.platform(), python=sys.version,
        packages={p: importlib.metadata.version(p) for p in
                  ("numpy", "scipy", "catboost", "scikit-learn", "torch", "rdkit", "xgboost")},
        fixes="UTF-8 interpreter mode only; dedicated output folder; unchanged historical source bytes.",
        prior_attempt="v295 Windows default GBK decoding failed before data loading or scoring.",
        model_configs=model.CONFIGS, blend_weights=model.runner.WEIGHTS,
        tasks=model.TASKS, new_models_trained=0, external_data=False,
        automatic_platform_submission=False, official_gain_claimed=False,
    )
    path = OUT / "frozen_local_run.json"
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        for key in ("files", "experiment_signature", "baseline_signature", "packages"):
            assert prior[key] == json.loads(json.dumps(report[key])), key
    else:
        assert not (OUT / "models").exists()
        model.b.save(path, report)
    return report


def preflight(d):
    assert model.v.sig() == "5a79bb5fd6cd135690d750fa5beae79f7a5674958ef83b87689f190be5067c91"
    assert model.original.sig() == "c7af56d8f09da462fd1a9b8de80960a123e885ea84aff41a9deb43ff55577322"
    assert model.b.sha(model.runner.__file__) == "8f6d93f517d1a4acf6877f3effa63e3b2c171141b5a637b8ad51573498c0b871"
    verification = verify(model.v.ANCHOR, ROOT / "admet_official/submission_template.csv")
    assert verification["sha256"] == model.v.ANCHOR_SHA
    hashes = {}
    for task in model.TASKS:
        k = d["tasks"].index(task)
        for part in ("fold0", "fold1"):
            hashes.update(model.original_replay(d, k, part))
    model.b.save(OUT / "preflight.json", dict(
        status="PASS", at_utc=now(), signature=model.sig(),
        original_development_component_replay=True, model_hashes=hashes,
        baseline_csv_verification=verification, target_scores_computed=False,
        original_source_bytes_unchanged=True, new_models_trained=0))


def finish():
    confirmation = json.loads((OUT / "confirmation.json").read_text(encoding="utf-8"))
    assert confirmation["status"] == "COMPLETE"
    development = json.loads((OUT / "development.json").read_text(encoding="utf-8"))
    best = []
    for task in model.TASKS:
        row = max((x for x in development["records"] if x["task"] == task), key=lambda x: x["objective"])
        best.append(row)
    status = dict(
        status="CANDIDATE_REQUIRES_INDEPENDENT_DELIVERY_VERIFICATION" if confirmation["accepted"]
               else "COMPLETE_NO_CONFIRMED_NEW_MODEL",
        at_utc=now(), baseline="v145", official_best=51.166,
        validation_version="v296", hypothesis_version="v295",
        signature=model.sig(), development_best=best,
        accepted_tasks=confirmation["accepted"], confirmation=confirmation["records"],
        new_csv_created=False, new_models_trained=0, platform_submission=False,
        official_score_updated=False,
        limitations="Repeatedly used public labels and approximate historical upstream OOF; no hidden-test gain established.",
    )
    model.b.save(OUT / "final_status.json", status)
    model.b.save(ROOT / "docs/v296_final_status.json", status)
    print(json.dumps(status), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "run", "report"))
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    freeze()
    if args.stage == "report":
        finish()
        return
    d = model.load()
    if args.stage == "smoke":
        preflight(d)
        model.smoke(d)
    else:
        smoke = json.loads((OUT / "smoke.json").read_text(encoding="utf-8"))
        assert smoke["status"] == "PASS" and smoke["signature"] == model.sig()
        assert not (OUT / "failure.json").exists()
        assert not (OUT / "confirmation.json").exists(), "Do not silently rerun completed selection."
        model.runner.run(model, d)
        finish()


if __name__ == "__main__":
    main()
