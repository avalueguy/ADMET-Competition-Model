"""Package verified v67/v68 releases using an explicit credential-free allowlist."""
import hashlib
import importlib.metadata
import json
import platform
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import admet_v67_data as b
import torch


def main():
    paths = set()
    for version in (67, 68):
        out = ROOT / f"admet_model_outputs_v{version}"
        assert json.loads((out / "model_replay.json").read_text())["status"] == "PASS"
        assert json.loads((out / "pipeline_status.json").read_text())["status"] == "COMPLETE"
        # Preserve all split indices/predictions/metadata and final persisted models.
        paths.update(out.rglob("*.json"))
        paths.update(out.rglob("*.npz"))
        paths.update(out.glob("*.npy"))
        for path in out.rglob("*"):
            if "final" in path.parts and path.suffix in (".pt", ".joblib"):
                paths.add(path)
        paths.update((ROOT / f"admet_submissions/v{version}_official_only").glob("*"))
    paths.update((ROOT / "admet_competition_data/data_benchmark").glob("*/*.csv"))
    paths.add(ROOT / "admet_official/submission_template.csv")
    paths.update(ROOT.glob("admet_v6[78]*.py"))
    paths.update(ROOT.glob("admet_v6[78]*protocol.json"))
    paths.add(ROOT / "docs/artifact-manifests/runtime-inputs.manifest.json")
    paths.add(ROOT / "docs/official_rules_20261020.json")
    paths.add(ROOT / "docs/REPRODUCE_V67_V68.md")
    paths.add(ROOT / "admet_model_outputs_v67/integrity_tests.log")
    for name in ("verify_submission.py", "verify_v67_models.py", "verify_v68_models.py",
                 "run_v67_pipeline.py", "run_v68_pipeline.py", "test_v67_integrity.py",
                 "package_official_only_release.py", "report_v68_valid.py"):
        paths.add(ROOT / "tools" / name)
    for name in ("pipeline.log", "v68_prepare.log", "v68_pipeline.log"):
        paths.add(ROOT / name)
    environment = dict(python=sys.version, platform=platform.platform(),
                       gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
                       packages={name: importlib.metadata.version(name) for name in
                                 ("torch", "numpy", "scipy", "scikit-learn", "rdkit", "joblib")},
                       v67_signature=b.signature(), external_data=False, pretrained_weights=False)
    b.save(ROOT / "admet_model_outputs_v67/environment.json", environment)
    paths.add(ROOT / "admet_model_outputs_v67/environment.json")
    selected = sorted(p for p in paths if p.is_file())
    records = [dict(path=p.relative_to(ROOT).as_posix(), size=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in selected]
    dest = ROOT / "releases"
    dest.mkdir(exist_ok=True)
    manifest = dest / "official_only_v67_v68_manifest.json"
    b.save(manifest, dict(status="PACKAGED", files=records,
                         total_uncompressed_bytes=sum(r["size"] for r in records)))
    archive = dest / "official_only_v67_v68_20261020.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=3) as stream:
        for p in selected:
            stream.add(p, arcname=p.relative_to(ROOT).as_posix(), recursive=False)
        stream.add(manifest, arcname="releases/" + manifest.name, recursive=False)
    b.save(dest / "official_only_v67_v68_archive.json", dict(file=archive.name,
           sha256=b.sha(archive), bytes=archive.stat().st_size, manifest_sha256=b.sha(manifest),
           files=len(records)))
    b.emit("release_bundle", file=str(archive), sha256=b.sha(archive), bytes=archive.stat().st_size)


if __name__ == "__main__":
    main()
