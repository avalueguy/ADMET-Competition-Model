"""Validate release and every manifest entry; restore matching or absent files only."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path,PurePosixPath

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("--record",type=Path,required=True)
    args=parser.parse_args()
    record=json.loads(args.record.read_text(encoding="utf-8"))
    assert args.archive.stat().st_size==record["bytes"]
    assert hashlib.sha256(args.archive.read_bytes()).hexdigest()==record["sha256"]
    restored=retained=0
    with tarfile.open(args.archive,"r:gz") as archive:
        members=archive.getmembers()
        assert all(m.isfile() for m in members)
        names=[m.name for m in members]
        assert len(names)==len(set(names))
        manifest_name="releases/"+record.get("manifest",args.record.name.replace("_archive.json","_manifest.json"))
        raw=archive.extractfile(manifest_name).read()
        assert hashlib.sha256(raw).hexdigest()==record["manifest_sha256"]
        entries={r["path"]:r for r in json.loads(raw)["files"]}
        assert len(entries)==record["files"] and set(names)==set(entries)|{manifest_name}
        for name in names:
            path=PurePosixPath(name)
            assert not path.is_absolute() and ".." not in path.parts and "\\" not in name and ":" not in name
            raw=archive.extractfile(name).read()
            if name!=manifest_name:
                assert len(raw)==entries[name]["size"]
                assert hashlib.sha256(raw).hexdigest()==entries[name]["sha256"],name
            target=ROOT.joinpath(*path.parts)
            assert target.resolve().is_relative_to(ROOT.resolve())
            if target.exists():
                assert target.read_bytes()==raw,f"Existing local file differs: {name}"
                retained+=1
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(raw);restored+=1
    result=dict(status="PASS",archive_sha256=record["sha256"],manifest_files=record["files"],
        restored=restored,existing_exact=retained)
    args.archive.with_suffix(".restore.json").write_bytes((json.dumps(result,indent=2)+"\n").encode("utf-8"))
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
