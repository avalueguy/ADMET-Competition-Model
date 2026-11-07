"""Combine completed releases without importing training code or changing predictions.

This candidate inherits the historical pretrained-feature provenance. It is not
a claim of compliance with the newly published external-data restriction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return list(csv.reader(io.StringIO(path.read_bytes().decode("utf-8-sig"), newline="")))


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d"))
    args = parser.parse_args()
    # v61 is deliberately absent: its required independent chemical gate is missing.
    # Nothing here changes that gate or the incomplete status of v64.
    versions = (62, 63, 65)
    baseline_file = next((ROOT / "admet_submissions/v60").glob("*.csv"))
    baseline = read_rows(baseline_file)
    template = read_rows(ROOT / "admet_official/submission_template.csv")
    rows = [row.copy() for row in baseline]
    options = {}
    provenance = {}
    files = {"baseline": digest(baseline_file)}
    for version in versions:
        subprocess.run([sys.executable, str(ROOT / "admet_release_local_verify.py"), str(version)],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        source = ROOT / f"admet_model_outputs_v{version}"
        fresh = load(source / "fresh.json")
        release = load(source / "release.json")
        reload_report = load(source / "model_replay.json")
        assert fresh["status"] == "COMPLETE" and fresh["signature"] == release["signature"]
        assert reload_report["status"] == "PASS"
        file = next((ROOT / f"admet_submissions/v{version}").glob("*.csv"))
        assert digest(file) == release["sha256"]
        files[str(version)] = digest(file)
        provenance[version] = (read_rows(file), load(source / "fusion_replay.json"))
        for record in fresh["tasks"]:
            if not record["pass_gate"]:
                continue
            task = record["task"]
            assert any(t["task"] == task and t["model_reload_passed"] for t in reload_report["tasks"])
            selection = record["selection"]
            gains = selection["cv_gains"]
            assert len(gains) == 3 and all(map(math.isfinite, gains))
            mean = statistics.mean(gains)
            objective = mean - 0.1 * statistics.pstdev(gains)
            options.setdefault(task, []).append(dict(version=version, selection=selection,
                                                     cv_mean=mean, objective=objective))
    routing = {}
    replay = {}
    changed = {}
    for task, candidates in options.items():
        held = [max(candidates, key=lambda c: statistics.mean(g for j, g in enumerate(c["selection"]["cv_gains"]) if j != fold))["selection"]["cv_gains"][fold]
                for fold in range(3)]
        best = max(candidates, key=lambda c: c["objective"]) if statistics.mean(held) > 0.001 else min(candidates, key=lambda c: c["version"])
        source_rows, source_replay = provenance[best["version"]]
        positions = [i for i, row in enumerate(rows[1:], 1) if row[0] == task]
        record = source_replay[task]
        assert [float(baseline[i][4]) for i in positions] == record["baseline"]
        assert [float(source_rows[i][4]) for i in positions] == record["result"]
        for i in positions:
            assert source_rows[i][:4] + source_rows[i][5:] == template[i][:4] + template[i][5:]
            rows[i][4] = source_rows[i][4]
        changed[task] = sum(rows[i][4] != baseline[i][4] for i in positions)
        routing[task] = dict(**best, options=candidates, held_source_selection_gains=held,
                             held_mean=statistics.mean(held), selection_uses_official_valid=False)
        replay[task] = dict(source_version=best["version"], record=record)
    assert len(rows) == len(template) == 16380 and rows[0] == template[0]
    for row, expected in zip(rows[1:], template[1:]):
        assert row[:4] + row[5:] == expected[:4] + expected[5:]
        assert math.isfinite(float(row[4]))
        if row[5] in ("AUROC", "AUPRC"):
            assert 0 <= float(row[4]) <= 1
    dest = ROOT / "admet_submissions/v66_historical_candidate"
    dest.mkdir(parents=True, exist_ok=True)
    file = dest / f"我从山北来带着兰花草-小分子药物预测技术创新赛-{args.date}.csv"
    buf = io.StringIO(newline="")
    csv.writer(buf, lineterminator="\r\n").writerows(rows)
    raw = b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8")
    if file.exists():
        assert file.read_bytes() == raw, "Never overwrite an existing different candidate"
    else:
        file.write_bytes(raw)
    evidence = dict(status="HISTORICAL_CANDIDATE_NOT_RULE_CLEARED", version=66,
                    csv=file.name, sha256=digest(file), rows=16379, columns=6, tasks=22,
                    source_csv_hashes=files, routing=routing, changed_tasks=changed,
                    template_sha256=digest(ROOT / "admet_official/submission_template.csv"),
                    v61_excluded="Required independent_chemical_gate.json is missing",
                    v64_status="Still incomplete; this is a separate completed-source-only candidate",
                    new_training=False, source_gpu_training="Historical reports only; no server access this run",
                    official_score=None, automatically_uploaded=False,
                    external_pretrained_provenance=True,
                    limitations=["Current rules prohibit other external data; historical ChemBERTa and TabICL provenance needs organizer clarification before submission.",
                                 "CV is a repeatedly reused historical proxy, not unbiased full-ensemble OOF.",
                                 "v65 uses same-molecule other official assay covariates; not globally unseen-scaffold validation.",
                                 "CYP2D6 substrate valid AUPRC gain is -0.137501; new-split conditional-control gain is +0.054934.",
                                 "Model numbers and internal gains do not establish an official score improvement."])
    (dest / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (dest / "fusion_replay.json").write_text(json.dumps(replay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(file=str(file), sha256=digest(file), changed_tasks=changed,
                          status=evidence["status"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
