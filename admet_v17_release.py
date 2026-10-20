"""Assemble and verify the official v17 submission from remote task predictions."""
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OFFICIAL = ROOT / "admet_official" / "submission_template.csv"
V16 = ROOT / "admet_submissions" / "v16" / "我从山北来带着兰花草-小分子药物预测技术创新赛-20260926.csv"
ROUTES = ROOT / "admet_release_v17" / "routing.json"
OUT = ROOT / "admet_submissions" / "v17" / "我从山北来带着兰花草-小分子药物预测技术创新赛-20260926-v17.csv"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    routes = {x["task"]: x for x in json.loads(ROUTES.read_text(encoding="utf-8"))}
    rows = list(csv.reader(io.StringIO(V16.read_bytes().decode("utf-8-sig"), newline="")))
    template = list(csv.reader(io.StringIO(OFFICIAL.read_bytes().decode("utf-8-sig"), newline="")))
    assert rows[0] == template[0] == ["task", "sample_idx", "drug_id", "smiles", "y_pred", "metric"]
    assert len(rows) == len(template) == 16380
    for row in rows[1:]:
        task = row[0]
        route = routes[task]
        pred_path = ROOT / "admet_release_v17" / task / f"submission_{task}.csv"
        pred = pd.read_csv(pred_path, dtype=str, keep_default_na=False)
        i = int(row[1])
        assert row[2] == pred.iloc[i].Drug_ID
        q = float(pred.iloc[i].Y)
        assert np.isfinite(q)
        if row[5] in {"AUROC", "AUPRC"}:
            q = float(np.clip(q, 0.0, 1.0))
        row[4] = repr(q)
    out = OUT.parent
    out.mkdir(parents=True, exist_ok=True)
    payload = io_string(rows)
    OUT.write_bytes(b"\xef\xbb\xbf" + payload.encode("utf-8"))
    # Exact non-prediction metadata and line-ending checks.
    verify = list(csv.reader(io.StringIO(OUT.read_bytes().decode("utf-8-sig"), newline="")))
    assert len(verify) == len(template)
    assert verify[0] == template[0]
    for a, b in zip(verify[1:], template[1:]):
        assert a[:4] + a[5:] == b[:4] + b[5:]
        assert np.isfinite(float(a[4]))
    result = dict(status="EXPORTED_AWAITING_PLATFORM", submission=str(OUT.resolve()), sha256=sha(OUT),
                  bytes=OUT.stat().st_size, rows=len(verify)-1, tasks=len(routes), columns=6,
                  encoding="UTF-8 BOM", line_endings="CRLF", automatic_submission=False,
                  baseline="v16", selected={t: r["selected"] for t, r in routes.items()})
    (OUT.parent / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def io_string(rows):
    out = io.StringIO(newline="")
    csv.writer(out, lineterminator="\r\n").writerows(rows)
    return out.getvalue()


if __name__ == "__main__":
    main()
