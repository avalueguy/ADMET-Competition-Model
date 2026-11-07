"""Insert unchanged model predictions into exact user-supplied CSV template bytes."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path

from verify_submission import verify

ROOT = Path(__file__).resolve().parents[1]


def rows(raw):
    return list(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline="")))


def field_spans(raw):
    """RFC4180 field spans including quotes, excluding separators and terminators."""
    offset = 3 if raw.startswith(b"\xef\xbb\xbf") else 0
    start = offset
    quoted = False
    record = []
    result = []
    i = offset
    while i < len(raw):
        char = raw[i]
        if char == 34:
            if quoted and i+1 < len(raw) and raw[i+1] == 34:
                i += 2
                continue
            quoted = not quoted
        elif not quoted and char in (44, 10, 13):
            record.append((start, i))
            if char != 44:
                result.append(record)
                record = []
                if char == 13 and i+1 < len(raw) and raw[i+1] == 10:
                    i += 1
            start = i+1
        i += 1
    assert not quoted
    if start < len(raw) or record:
        record.append((start, len(raw)))
        result.append(record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original, template = args.predictions.read_bytes(), args.template.read_bytes()
    predictions, expected = rows(original), rows(template)
    assert predictions[0] == expected[0] == ["task", "sample_idx", "drug_id", "smiles", "y_pred", "metric"]
    assert len(predictions) == len(expected)
    spans = field_spans(template)
    assert len(spans) == len(expected) and all(len(r) == 6 for r in spans)
    ids, changed, chunks, cursor = Counter(), Counter(), [], 0
    for i, (old, row) in enumerate(zip(predictions[1:], expected[1:]), 1):
        assert [old[j] for j in (0, 1, 3, 5)] == [row[j] for j in (0, 1, 3, 5)], f"Molecule or task mismatch at row {i}"
        assert row[4] == "", f"Template prediction not empty at row {i}"
        if old[2] != row[2]:
            ids[row[0]] += 1
        lo, hi = spans[i][4]
        assert template[lo:hi] in (b"", b'""')
        value = old[4].encode("ascii")
        assert value and not any(c in value for c in b",\r\n\"")
        chunks.extend((template[cursor:lo], value))
        cursor = hi
        changed[row[0]] += 1
    chunks.append(template[cursor:])
    output = b"".join(chunks)
    result = rows(output)
    assert [r[4] for r in result[1:]] == [r[4] for r in predictions[1:]]
    assert all(r[:4]+r[5:] == t[:4]+t[5:] for r, t in zip(result, expected))
    # Reverse the insertions to prove that every nonprediction byte is preserved.
    actual_spans = field_spans(output)
    restored, cursor = [], 0
    for i in range(1, len(result)):
        lo, hi = actual_spans[i][4]
        tlo, thi = spans[i][4]
        restored.extend((output[cursor:lo], template[tlo:thi]))
        cursor = hi
    restored.append(output[cursor:])
    assert b"".join(restored) == template
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        assert args.output.read_bytes() == output, "Refuse replacing a different release"
    else:
        args.output.write_bytes(output)
    audit = verify(args.output, args.template)
    audit.update(template_sha256=hashlib.sha256(template).hexdigest(),
                 source_prediction_sha256=hashlib.sha256(original).hexdigest(),
                 predictions_exact_strings=True, nonprediction_bytes_exact=True,
                 template_bom_preserved=True, changed_drug_ids=sum(ids.values()),
                 changed_drug_ids_by_task=dict(ids), model_retrained=False)
    args.output.with_suffix(".verification.json").write_bytes((json.dumps(audit,ensure_ascii=False,indent=2)+"\n").encode("utf-8"))
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
