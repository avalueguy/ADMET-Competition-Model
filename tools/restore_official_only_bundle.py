"""Validate the exact release archive, then restore outputs without replacing source."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    assert args.archive.stat().st_size == record["bytes"]
    assert hashlib.sha256(args.archive.read_bytes()).hexdigest() == record["sha256"]
    restored, retained = [], []
    with tarfile.open(args.archive, "r:gz") as archive:
        members = archive.getmembers()
        assert all(member.isfile() for member in members)
        names = [member.name for member in members]
        assert len(names) == len(set(names))
        for name in names:
            path = PurePosixPath(name)
            assert not path.is_absolute() and ".." not in path.parts and "\\" not in name and ":" not in name
        manifest_name = "releases/official_only_v67_v68_manifest.json"
        raw = archive.extractfile(manifest_name).read()
        assert hashlib.sha256(raw).hexdigest() == record["manifest_sha256"]
        manifest = json.loads(raw)
        entries = {row["path"]: row for row in manifest["files"]}
        assert set(names) == set(entries) | {manifest_name}
        assert len(entries) == record["files"]
        for name in names:
            raw = archive.extractfile(name).read()
            if name != manifest_name:
                entry = entries[name]
                assert len(raw) == entry["size"]
                assert hashlib.sha256(raw).hexdigest() == entry["sha256"], name
            destination = ROOT.joinpath(*PurePosixPath(name).parts)
            assert destination.resolve().is_relative_to(ROOT.resolve())
            if destination.exists():
                assert destination.read_bytes() == raw, f"Existing local file differs: {name}"
                retained.append(name)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(raw)
                restored.append(name)
    result = dict(status="PASS", archive_sha256=record["sha256"],
                  manifest_files=record["files"], restored=len(restored), existing_exact=len(retained))
    target = args.archive.with_suffix(".restore.json")
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
