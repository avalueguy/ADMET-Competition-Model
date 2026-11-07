"""Independent standard-library verification against the exact official template."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return list(csv.reader(io.StringIO(path.read_bytes().decode("utf-8-sig"), newline="")))


def verify(file, template, expected=None):
    raw = file.read_bytes()
    rows, reference = read(file), read(template)
    assert len(rows) == len(reference) and rows[0] == reference[0]
    assert rows[0] == ["task", "sample_idx", "drug_id", "smiles", "y_pred", "metric"]
    counts = Counter()
    keys = set()
    for row, ref in zip(rows[1:], reference[1:]):
        assert len(row) == 6 and row[:4] + row[5:] == ref[:4] + ref[5:]
        key = tuple(row[:2])
        assert key not in keys
        keys.add(key)
        value = float(row[4])
        assert math.isfinite(value)
        if row[5] in ("AUROC", "AUPRC"):
            assert 0 <= value <= 1
        counts[row[0]] += 1
        if expected is not None:
            assert row == expected[len(keys)], "Source prediction changed"
    sha = hashlib.sha256(raw).hexdigest()
    result = dict(status="PASS", sha256=sha, rows=len(rows)-1, columns=6, tasks=len(counts),
                  nonprediction_fields_exact=True, finite_predictions=True,
                  unique_task_sample_keys=True, task_counts=dict(counts),
                  utf8_bom=raw.startswith(b"\xef\xbb\xbf"),
                  crlf_only=b"\n" not in raw.replace(b"\r\n", b""),
                  official_score=None, automatically_uploaded=False)
    evidence_path = file.parent / "evidence.json"
    if evidence_path.exists():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        assert sha == evidence["sha256"]
        baseline = read(next((ROOT / "admet_submissions/v60").glob("*.csv")))
        replay = json.loads((file.parent / "fusion_replay.json").read_text(encoding="utf-8"))
        for i, row in enumerate(rows[1:], 1):
            if row[0] not in replay:
                assert row == baseline[i]
                continue
            source_version = replay[row[0]]["source_version"]
            assert source_version in (62, 63, 65), "Missing v61 gate must not be bypassed"
        for task, detail in replay.items():
            source = read(next((ROOT / f"admet_submissions/v{detail['source_version']}").glob("*.csv")))
            assert [row for row in rows[1:] if row[0] == task] == [row for row in source[1:] if row[0] == task]
            old = [float(row[4]) for row in baseline[1:] if row[0] == task]
            actual = [float(row[4]) for row in rows[1:] if row[0] == task]
            assert old == detail["record"]["baseline"] and actual == detail["record"]["result"]
        result.update(source_predictions_exact=True, source_versions=[62, 63, 65],
                      rule_clearance="UNCONFIRMED_HISTORICAL_PRETRAINING")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--template", type=Path, default=ROOT / "admet_official/submission_template.csv")
    args = parser.parse_args()
    result = verify(args.file, args.template)
    args.file.with_suffix(".verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
