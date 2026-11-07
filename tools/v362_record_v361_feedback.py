"""Record the user's measured51.240 result, preserving all earlier registries."""
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from v184_score_objective_audit import normalized

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT/name).read_text(encoding="utf-8"))


def save(name,value):
    path=ROOT/name
    assert not path.exists(), f"Preserve existing artifact: {path}"
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def main():
    old=read("docs/platform_scorecard_v346_20261101.json")
    latest=read("docs/platform_scorecard_v354_20261102.json")
    delivery=read("docs/v361_delivery.json")
    verification=read("docs/v361_local_delivery_verification.json")
    assert verification["status"]=="PASS" and verification["csv_sha256"]==delivery["csv_sha256"]
    assert hashlib.sha256((ROOT/delivery["csv"]).read_bytes()).hexdigest()==delivery["csv_sha256"]
    tasks=copy.deepcopy(old["tasks"])
    # Full user-pasted scorecard equals v346 for21 endpoints, v354 for hERG.
    for row in tasks:
        if row["task"]=="herg":
            row.update(raw=.861119293078056,score=49.93509)
            assert row==next(r for r in latest["tasks"] if r["task"]=="herg")
        assert abs(normalized(row["metric"],row["raw"],row["reference_score"])-row["score"])<.00000051
    changed=[r["task"] for r,b in zip(tasks,old["tasks"]) if r!=b]
    assert changed==["herg"]
    mean=sum(r["score"] for r in tasks)/22
    assert round(mean,3)==51.240 and abs(mean-delivery["projection"]["value"])<1e-12
    at=datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()
    source="docs/platform_scorecard_v361_20261104.json"
    card=dict(version="v361",date="2026-11-04",submitted_at="2026/11/04 13:06",
        recorded_at_hong_kong=at,timezone="Asia/Hong_Kong",score=51.240,task_score_mean=mean,
        tasks=tasks,status="user_reported_official",platform_displayed_certification="未认证",
        source="User-pasted full22-task official scorecard and explicit latest51.240 in this conversation.",
        csv=delivery["csv"],submission_sha256=delivery["csv_sha256"],uploaded_byte_hash_confirmed=False,
        association="v361 inferred from recent delivery and exact21 v346 tasks plus v354 hERG. Platform byte hash unavailable.",
        baseline="v346",baseline_score=51.235,changed_tasks=changed,unchanged_tasks=21,
        delta_task_mean=mean-old["task_score_mean"],delta_vs_v354=mean-latest["task_score_mean"],
        original_local_strict_rejections_preserved=True)
    best=dict(version="v361",official_user_reported=51.240,exact_task_mean=mean,recorded_at_hong_kong=at,
        csv=delivery["csv"],csv_sha256=delivery["csv_sha256"],source=source,
        previous_best="v346",previous_best_official=51.235,delta_task_mean=card["delta_task_mean"],
        platform_uploaded_byte_hash_confirmed=False,
        reason="User-reported full official card confirms retained hERG improvement and restored CYP3A4. No historical strict gate rewritten.")
    save(source,card)
    save("docs/current_best_after_v361_20261104.json",best)
    report=dict(status="PASS_USER_REPORTED_OFFICIAL_IMPROVEMENT",best=best,changed_tasks=changed,
        delta_vs_v346=card["delta_task_mean"],delta_vs_v354=card["delta_vs_v354"],
        gap_to52=52-mean,required_sum_task_point_gain_for52=22*(52-mean),
        previous_conditional_arithmetic_matched=True,platform_submission=False,
        note="Scorecard is feedback evidence, not proof of statistical stability or private-test gain.")
    save("docs/v362_feedback_review.json",report)
    print(json.dumps(report,ensure_ascii=True))


if __name__=="__main__":
    main()
