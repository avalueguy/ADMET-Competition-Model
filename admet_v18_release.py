"""Assemble v18 in the official submission format."""
import csv, hashlib, io, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V16 = next((ROOT / "admet_submissions" / "v16").glob("*.csv"))
TEMPLATE = ROOT / "admet_official" / "submission_template.csv"
ROUTES = ROOT / "admet_release_v18" / "routing.json"
OUT = ROOT / "admet_submissions" / "v18" / "我从山北来带着兰花草-小分子药物预测技术创新赛-20260926-v18.csv"

def main():
    routes = {x["task"]: x for x in json.loads(ROUTES.read_text(encoding="utf-8"))}
    rows = list(csv.reader(io.StringIO(V16.read_bytes().decode("utf-8-sig"), newline="")))
    template = list(csv.reader(io.StringIO(TEMPLATE.read_bytes().decode("utf-8-sig"), newline="")))
    assert rows[0] == template[0] == ["task", "sample_idx", "drug_id", "smiles", "y_pred", "metric"]
    assert len(rows) == len(template) == 16380
    cache = {}
    for row in rows[1:]:
        task = row[0]
        if task not in cache:
            cache[task] = pd.read_csv(ROOT / "admet_release_v18" / task / f"submission_{task}.csv", dtype=str, keep_default_na=False)
        pred = cache[task]
        i = int(row[1]); assert row[2] == pred.iloc[i].Drug_ID
        value = float(pred.iloc[i].Y); assert np.isfinite(value)
        if row[5] in {"AUROC", "AUPRC"}: value = float(np.clip(value, 0., 1.))
        row[4] = repr(value)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO(newline=""); csv.writer(buf, lineterminator="\r\n").writerows(rows)
    OUT.write_bytes(b"\xef\xbb\xbf" + buf.getvalue().encode("utf-8"))
    verify = list(csv.reader(io.StringIO(OUT.read_bytes().decode("utf-8-sig"), newline="")))
    assert verify[0] == template[0] and len(verify) == len(template)
    for a, b in zip(verify[1:], template[1:]):
        assert a[:4] + a[5:] == b[:4] + b[5:] and np.isfinite(float(a[4]))
    result = dict(status="EXPORTED_AWAITING_PLATFORM", submission=str(OUT.resolve()), sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(), bytes=OUT.stat().st_size, rows=16379, tasks=22, columns=6, encoding="UTF-8 BOM", line_endings="CRLF", baseline="v16", automatic_submission=False)
    (OUT.parent / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
