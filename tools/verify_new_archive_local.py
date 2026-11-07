"""Verify a recent manifest-in-archive release; never replace existing files."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def sha(file):
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("--restore-absent", action="store_true")
    args = parser.parse_args()
    assert args.version.startswith("v") and args.version[1:].isdigit()
    version = args.version
    info = json.loads((ROOT / f"docs/{version}_archive.json").read_text(encoding="utf-8-sig"))
    archive = ROOT / info["path"]
    assert archive.resolve().is_relative_to((ROOT / "releases").resolve())
    assert sha(archive) == info["sha256"]
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        seen = set()
        for member in members:
            relative = PurePosixPath(member.name)
            assert member.isfile() and not relative.is_absolute() and ".." not in relative.parts
            assert member.name not in seen
            seen.add(member.name)
        manifest_name = f"admet_model_outputs_{version}/manifest.json"
        manifest = json.loads(tf.extractfile(manifest_name).read())
        expected = {record["path"]: record for record in manifest["files"]}
        assert len(expected) == len(manifest["files"])
        assert seen == set(expected) | {manifest_name}
        for name, record in expected.items():
            data = tf.extractfile(name).read()
            assert len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"]
    dependencies = manifest.get("dependencies", {})
    assert isinstance(dependencies, dict)
    for name, digest in dependencies.items():
        relative = PurePosixPath(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        dependency = ROOT / name
        assert dependency.resolve().is_relative_to(ROOT.resolve()) and dependency.is_file()
        assert sha(dependency) == digest, name
    restored, preserved = [], []
    if args.restore_absent:
        with tarfile.open(archive) as tf:
            for member in tf:
                dest = ROOT / member.name
                assert dest.resolve().is_relative_to(ROOT.resolve())
                if dest.exists():
                    if sha(dest) != hashlib.sha256(tf.extractfile(member).read()).hexdigest():
                        preserved.append(member.name)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(tf.extractfile(member).read())
                restored.append(member.name)
    result = dict(status="PASS", archive=info["path"], sha256=info["sha256"],
        bytes=archive.stat().st_size, members=len(members), manifest_files=len(expected),
        all_member_hashes_verified=True, csv_members=[p for p in seen if p.endswith(".csv")],
        restored_absent=len(restored), existing_different_preserved=preserved,
        dependency_contents_reverified=True, dependency_hashes_checked=len(dependencies))
    (ROOT / f"docs/{version}_local_archive_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
