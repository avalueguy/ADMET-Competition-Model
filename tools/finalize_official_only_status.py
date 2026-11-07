"""Write completion metadata from actual verified release evidence."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def save(path, value):
    (ROOT / path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2)+"\n").encode("utf-8"))


def main():
    status = read("CURRENT_STATUS.json")
    for version in (67, 68):
        out = f"admet_model_outputs_v{version}"
        release, replay, pipeline = [read(f"{out}/{name}.json") for name in ("release", "model_replay", "pipeline_status")]
        assert replay["status"] == "PASS" and pipeline["status"] == "COMPLETE"
        file = ROOT / release["file"]
        assert hashlib.sha256(file.read_bytes()).hexdigest() == release["sha256"] == replay["sha256"]
        with file.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        status["releases"][str(version)] = dict(csv=release["file"], sha256=release["sha256"],
            rows=len(rows), columns=6, tasks=len({r["task"] for r in rows}), official_score=None,
            status="LOCALLY_VERIFIED_AWAITING_PLATFORM_EVALUATION", independent_model_replay="PASS",
            external_data=False, pretrained_weights=False)
    status["latest_completed_model_branch"] = "v68"
    status["latest_completed_model_branch_is_proven_best"] = False
    status["official_score"] = None
    status["latest_user_reported_official_score"] = dict(version="v63",score=50.603,
                                                        uploaded_byte_hash_confirmed=False)
    status["target_achieved"] = False
    status["running_new_iteration_processes_at_completion"] = 0
    status["priority_next_step"] = "Evaluate the exact v68 CSV on the platform, record file hash and actual score, then compare against user-reported v63 50.603 and tenth-place 51.265."
    status["new_iteration"].update(official_only_kernel_version=68,
        status="COMPLETE_LOCALLY_VERIFIED_AWAITING_PLATFORM_EVALUATION", recommended_candidate=68,
        comparator_candidate=67, archive=read("releases/official_only_v67_v68_archive.json"),
        local_archive_restore=read("releases/official_only_v67_v68_20261020.tar.restore.json"))
    valid = read("admet_model_outputs_v68/valid_metrics.json")
    summary = dict(status="REPORT_ONLY", improved=sum(r["gain"]>0 for r in valid["tasks"]),
                   worsened=sum(r["gain"]<0 for r in valid["tasks"]),
                   unchanged=sum(r["gain"]==0 for r in valid["tasks"]),
                   comparison="v68 versus v67 on official valid after frozen selection",
                   official_score_gain=None,
                   not_an_unbiased_full_pipeline_estimate=True)
    status["new_iteration"]["valid_diagnostic"] = summary
    status["known_issues"].extend([
        "v67/v68 have no platform score yet; local diagnostics cannot establish improvement over v63.",
        "v67/v68 development folds and fresh gate are reused for selection; not fully nested independent estimates.",
        "v68 official-valid diagnostic decreases on bioavailability, hepatocyte clearance and DILI; selection was frozen beforehand."
    ])
    save("CURRENT_STATUS.json",status)
    save("docs/v68_validation_summary.json",summary)
    save("docs/official_only_delivery.json",dict(status="COMPLETE_LOCALLY_VERIFIED",candidate=status["releases"]["68"],
         comparator=status["releases"]["67"],archive=status["new_iteration"]["archive"],
         platform_baseline=status["latest_user_reported_official_score"],tenth_place=51.265,
         automatic_platform_submission=False,automatic_github_push=False,valid=summary))
    print(json.dumps(dict(status="PASS", candidate_sha256=status["releases"]["68"]["sha256"], valid=summary),indent=2))


if __name__ == "__main__":
    main()
