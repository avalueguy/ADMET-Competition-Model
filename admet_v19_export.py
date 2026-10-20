"""Publish selected v19 task predictions into the official v16 submission."""
import csv, hashlib, io, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = next((ROOT / "admet_submissions" / "v16").glob("*.csv"))
TEMPLATE = ROOT / "admet_official" / "submission_template.csv"
OUT = ROOT / "admet_submissions" / "v19" / "我从山北来带着兰花草-小分子药物预测技术创新赛-20260926-v19.csv"

def main():
    rows = list(csv.reader(io.StringIO(BASE.read_bytes().decode("utf-8-sig"), newline="")))
    template = list(csv.reader(io.StringIO(TEMPLATE.read_bytes().decode("utf-8-sig"), newline="")))
    routes = {}
    for p in (ROOT / "admet_model_outputs_v19").glob("*/selection.json"):
        route = json.loads(p.read_text(encoding="utf-8"))
        routes[route.get("task", p.parent.name)] = route
    # v19 task folders contain a selected prediction only after publish().
    for task, route in routes.items():
        if route.get("candidate_mass", 0) <= 0: continue
        pred_path = ROOT / "admet_release_v19" / task / f"submission_{task}.csv"
        if not pred_path.exists(): continue
        pred = pd.read_csv(pred_path, dtype=str, keep_default_na=False)
        for row in rows[1:]:
            if row[0] == task:
                q = float(pred.iloc[int(row[1])].Y); assert np.isfinite(q)
                if row[5] in {"AUROC", "AUPRC"}: q = float(np.clip(q, 0., 1.))
                row[4] = repr(q)
    assert rows[0] == template[0] and len(rows) == len(template) == 16380
    for a,b in zip(rows[1:], template[1:]): assert a[:4]+a[5:] == b[:4]+b[5:] and np.isfinite(float(a[4]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    buf=io.StringIO(newline=""); csv.writer(buf,lineterminator="\r\n").writerows(rows); OUT.write_bytes(b"\xef\xbb\xbf"+buf.getvalue().encode())
    result=dict(status="EXPORTED_AWAITING_PLATFORM",submission=str(OUT.resolve()),sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),bytes=OUT.stat().st_size,rows=16379,tasks=22,columns=6,encoding="UTF-8 BOM",line_endings="CRLF",baseline="v16",changed=[t for t,r in routes.items() if r.get("candidate_mass",0)>0],automatic_submission=False)
    (OUT.parent/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__ == "__main__": main()
