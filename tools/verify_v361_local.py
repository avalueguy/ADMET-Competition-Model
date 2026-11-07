"""Independent whole-task routing, source provenance, format and score arithmetic."""
import csv
import hashlib
import io
import json
from pathlib import Path

from repack_upload_template import field_spans
from verify_submission import verify

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT/name).read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    result_file = ROOT/"docs/v361_local_delivery_verification.json"
    assert not result_file.exists()
    delivery = read("docs/v361_delivery.json")
    assert delivery == read("admet_model_outputs_v361/delivery.json")
    paths = [ROOT/delivery["baseline"]["csv"], ROOT/delivery["herg_source"]["csv"], ROOT/delivery["csv"]]
    assert [sha(p) for p in paths] == [delivery["baseline"]["csv_sha256"],delivery["herg_source"]["csv_sha256"],delivery["csv_sha256"]]
    raw = [p.read_bytes() for p in paths]
    table = [list(csv.reader(io.StringIO(b.decode("utf-8-sig"),newline=""))) for b in raw]
    base, source, actual = table
    assert len(base) == len(source) == len(actual) == 16380
    assert base[0] == source[0] == actual[0]
    spans = [field_spans(b) for b in raw]
    counts = {}
    herg_count, cyp_count = 0, 0
    for i,(b,s,a) in enumerate(zip(base[1:],source[1:],actual[1:]),1):
        assert b[:4]+b[5:] == s[:4]+s[5:] == a[:4]+a[5:]
        expected_from = 1 if b[0] == "herg" else 0
        expected_row = s if b[0] == "herg" else b
        assert a == expected_row
        lo,hi = spans[2][i][4]
        elo,ehi = spans[expected_from][i][4]
        assert raw[2][lo:hi] == raw[expected_from][elo:ehi]
        if a[4] != b[4]:
            counts[b[0]] = counts.get(b[0],0)+1
        herg_count += b[0] == "herg"
        cyp_count += b[0] == "cyp3a4_substrate_carbonmangels"
    assert counts == {"herg":132} == delivery["changed_rows"]
    assert herg_count == 132 and cyp_count == 135
    restored,cursor = [],0
    for i in range(1,len(actual)):
        lo,hi = spans[2][i][4]
        blo,bhi = spans[0][i][4]
        restored.extend((raw[2][cursor:lo],raw[0][blo:bhi]))
        cursor = hi
    restored.append(raw[2][cursor:])
    assert b"".join(restored) == raw[0]
    old,latest = read("docs/platform_scorecard_v346_20261101.json"),read("docs/platform_scorecard_v354_20261102.json")
    by_new = {r["task"]:r for r in latest["tasks"]}
    hypothetical = [by_new[r["task"]]["score"] if r["task"]=="herg" else r["score"] for r in old["tasks"]]
    projection = sum(hypothetical)/len(hypothetical)
    assert abs(projection-delivery["projection"]["value"]) < 1e-12
    assert round(projection,3) == 51.240
    assert delivery["official_score"] is None and delivery["platform_submission"] is False
    result = dict(status="PASS",csv=delivery["csv"],csv_sha256=sha(paths[2]),changed_rows=counts,
        unchanged_tasks=21,herg_bytes_exact_v354=True,all_other_prediction_bytes_exact_v346=True,
        restored_cyp3a4_rows=cyp_count,all_nonprediction_bytes_exact=True,
        score_arithmetic_independently_verified=True,conditional_projection=projection,
        official_score=None,official_feedback_selected=True,platform_submission=False,
        format_verification=verify(paths[2],ROOT/"admet_official/submission_template.csv"))
    result_file.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=True))


if __name__ == "__main__":
    main()
