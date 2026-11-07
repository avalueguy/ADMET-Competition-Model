"""Record actual platform ID feedback and withdraw the incompatible integer-ID copy."""
import hashlib
import json
from pathlib import Path

from verify_submission import read, verify

ROOT = Path(__file__).resolve().parents[1]
NAME = "我从山北来带着兰花草-小分子药物预测技术创新赛-20261020.csv"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    source = ROOT / "admet_submissions/v68_official_only" / NAME
    copy = Path("D:/bili_download/v68_平台要求保留小数ID") / NAME
    template = ROOT / "admet_official/submission_template.csv"
    audit = verify(copy, template)
    assert source.read_bytes() == copy.read_bytes()
    assert audit["sha256"] == "0aa975b4e2dfce0890296854fbe621de0beb4955bfd5ed0a0131e88cd4625ccf"
    example = next(row for row in read(copy)[1:] if row[:2] == ["cyp2c9_veith", "0"])
    assert example[2] == "5348064.0"
    correction = dict(
        status="CORRECTED_TO_ACTUAL_PLATFORM_EXPECTED_IDS",
        platform_error="cyp2c9_veith: Drug_ID mismatch at sample_idx 0: expected '5348064.0', received '5348064'",
        template="admet_official/submission_template.csv",
        template_sha256=hashlib.sha256(template.read_bytes()).hexdigest(),
        preferred_v68_csv=source.relative_to(ROOT).as_posix(),
        local_upload_file=copy.as_posix(), sha256=audit["sha256"],
        user_template_sha256="9604b78fd59cb2e7a803df4ed731e054f152af401c4ec6ec79d11af70cfd292b",
        user_template_drug_id_mismatch_rows=7611,
        revoked_integer_csv_sha256="bc7575519ff011dc0a69cfe31af9e84c5d1bcd23376a4f94bb9fe39f133ef26c",
        predictions_unchanged=True, nonprediction_bytes_exact=True,
        upload_success_confirmed=False, official_score=None)
    for folder in (ROOT / "admet_submissions/v68_upload_template_fixed",
                   Path("D:/bili_download/v68_格式修复")):
        wrong = folder / NAME
        revoked = wrong.with_suffix(".csv.DO_NOT_UPLOAD")
        if wrong.exists():
            assert hashlib.sha256(wrong.read_bytes()).hexdigest() == correction["revoked_integer_csv_sha256"]
            if revoked.exists():
                assert revoked.read_bytes() == wrong.read_bytes()
                raise RuntimeError("Both revoked and uploadable integer copies exist")
            wrong.rename(revoked)
        if folder.exists():
            (folder / "勿上传.txt").write_text(
                "此整数ID版本与平台实际要求不符，已撤回。请使用 D:/bili_download/v68_平台要求保留小数ID/ 内的 CSV。\n",
                encoding="utf-8")
            write_json(folder / "REVOKED.json", correction)
    status_path = ROOT / "CURRENT_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["upload_template_correction"] = correction
    status["new_iteration"]["aggressive_status"] = "REPAIRING_SPEARMAN_METADATA_SERIALIZATION_FAILURE"
    status["priority_next_step"] = "Upload the exact decimal-ID v68 CSV; continue repaired aggressive v69 search on the authorized A100."
    write_json(status_path, status)
    path = ROOT / "docs/official_only_delivery.json"
    delivery = json.loads(path.read_text(encoding="utf-8"))
    delivery["upload_format_correction"] = correction
    write_json(path, delivery)
    write_json(ROOT / "docs/platform_id_feedback_20261020.json", correction)
    print(json.dumps(dict(status="PASS", expected_drug_id=example[2],
                          rows=audit["rows"], sha256=audit["sha256"],
                          corrected_copy=copy.as_posix()), ensure_ascii=False))


if __name__ == "__main__":
    main()
