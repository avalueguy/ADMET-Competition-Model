"""Package verified sparse kernel release and reproducible official-only sources."""
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
    out=ROOT/"admet_model_outputs_v70"
    status=json.loads((out/"pipeline_status.json").read_text())
    replay=json.loads((out/"model_replay.json").read_text())
    upload=json.loads((out/"upload_release.json").read_text())
    assert status["status"]=="COMPLETE" and replay["status"]=="PASS"
    assert b.sha(ROOT/upload["file"])==upload["sha256"]==replay["upload_sha256"]
    parent=json.loads((ROOT/"releases/aggressive_v69_archive.json").read_text())
    environment=dict(python=sys.version,platform=platform.platform(),
        packages={name:importlib.metadata.version(name) for name in
            ("catboost","rdkit","scikit-learn","numpy","scipy","joblib")},
        required_v69_archive_sha256=parent["sha256"],
        required_base_archive_sha256="ae4dac28be3a399410cae70c1ab7421aceadaaf03eba9b5d6300eda063244e0b")
    b.save(out/"environment.json",environment)
    paths=set(p for p in out.rglob("*") if p.is_file())
    paths={p for p in paths if p.name!="pre_smoke_manifest.json"}
    for name in ("admet_v70_sparse.py","admet_v70_protocol.json","v70_pipeline.log","v70_smoke.log",
                 "admet_official/submission_template.csv","docs/REPRODUCE_V69_V70.md"):
        paths.add(ROOT/name)
    for name in ("verify_v70_models.py","run_v70_pipeline.py","test_v70_kernels.py",
                 "package_v70_release.py","restore_aggressive_bundle.py","repack_upload_template.py",
                 "report_v70_development.py"):
        paths.add(ROOT/"tools"/name)
    for directory in ("v70_sparse_aggressive","v70_platform_ids"):
        paths.update((ROOT/"admet_submissions"/directory).glob("*"))
    selected=sorted(p for p in paths if p.is_file())
    records=[dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,sha256=b.sha(p)) for p in selected]
    dest=ROOT/"releases";dest.mkdir(exist_ok=True)
    manifest=dest/"aggressive_v70_manifest.json"
    b.save(manifest,dict(status="PACKAGED",files=records,total_uncompressed_bytes=sum(r["size"] for r in records)))
    archive=dest/"aggressive_v70_20261020.tar.gz"
    with tarfile.open(archive,"w:gz",compresslevel=3) as stream:
        for p in selected:stream.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
        stream.add(manifest,arcname="releases/"+manifest.name,recursive=False)
    b.save(dest/"aggressive_v70_archive.json",dict(file=archive.name,bytes=archive.stat().st_size,
        sha256=b.sha(archive),manifest_sha256=b.sha(manifest),manifest=manifest.name,files=len(records),
        required_v69_archive_sha256=parent["sha256"],
        required_base_archive_sha256=environment["required_base_archive_sha256"]))
    b.emit("v70_bundle",file=str(archive),bytes=archive.stat().st_size,sha256=b.sha(archive))


if __name__=="__main__":main()
