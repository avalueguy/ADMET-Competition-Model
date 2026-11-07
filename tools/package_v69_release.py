"""Archive the verified aggressive release without credentials or unrelated files."""
import hashlib
import importlib.metadata
import json
import platform
import sys
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import admet_v67_data as b


def main():
    out=ROOT/"admet_model_outputs_v69"
    status=json.loads((out/"pipeline_status.json").read_text())
    replay=json.loads((out/"model_replay.json").read_text())
    upload=json.loads((out/"upload_release.json").read_text())
    assert status["status"]=="COMPLETE" and replay["status"]=="PASS"
    assert b.sha(ROOT/upload["file"])==upload["sha256"]==replay["upload_sha256"]
    paths=set(p for p in out.rglob("*") if p.is_file())
    for name in ("admet_v69_aggressive.py","admet_v69_protocol.json","v69_pipeline.log",
                 "admet_official/submission_template.csv","docs/REPRODUCE_V69_V70.md"):
        paths.add(ROOT/name)
    for name in ("verify_v69_models.py","run_v69_pipeline.py","repack_upload_template.py",
                 "test_upload_template_repack.py","test_v69_serialization.py",
                 "report_v69_development.py","restore_aggressive_bundle.py","package_v69_release.py"):
        paths.add(ROOT/"tools"/name)
    for directory in ("v69_aggressive_official_only","v69_aggressive_platform_ids"):
        paths.update((ROOT/"admet_submissions"/directory).glob("*"))
    # v69 may have trained additional final/valid kernel partitions.
    for task in (ROOT/"admet_model_outputs_v68").iterdir():
        if not task.is_dir():continue
        for part in ("final","valid_report"):
            folder=task/part
            if folder.exists():paths.update(p for p in folder.rglob("*") if p.is_file())
    environment=dict(python=sys.version,platform=platform.platform(),
        packages={name:importlib.metadata.version(name) for name in
                  ("catboost","torch","rdkit","scikit-learn","numpy","scipy","joblib")},
        required_base_archive_sha256="ae4dac28be3a399410cae70c1ab7421aceadaaf03eba9b5d6300eda063244e0b")
    b.save(out/"environment.json",environment);paths.add(out/"environment.json")
    selected=sorted(p for p in paths if p.is_file())
    records=[dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,sha256=b.sha(p)) for p in selected]
    dest=ROOT/"releases";dest.mkdir(exist_ok=True)
    manifest=dest/"aggressive_v69_manifest.json"
    b.save(manifest,dict(status="PACKAGED",files=records,total_uncompressed_bytes=sum(r["size"] for r in records)))
    archive=dest/"aggressive_v69_20261020.tar.gz"
    with tarfile.open(archive,"w:gz",compresslevel=3) as stream:
        for p in selected:stream.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
        stream.add(manifest,arcname="releases/"+manifest.name,recursive=False)
    b.save(dest/"aggressive_v69_archive.json",dict(file=archive.name,bytes=archive.stat().st_size,
        sha256=b.sha(archive),manifest_sha256=b.sha(manifest),files=len(records),
        required_base_archive_sha256=environment["required_base_archive_sha256"]))
    b.emit("v69_bundle",file=str(archive),bytes=archive.stat().st_size,sha256=b.sha(archive))


if __name__=="__main__":main()
