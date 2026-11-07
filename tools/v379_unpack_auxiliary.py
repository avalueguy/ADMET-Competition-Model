"""Verify and safely unpack returned scheduling/loss logs locally; no fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--version',choices=['378','379','380','381'],required=True);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    exp=root/('experiments/regression9_v'+args.version)
    archive=root/('releases/v'+args.version+'_auxiliary_logs.tar.gz')
    receipt=json.loads((exp/'auxiliary_archive.json').read_text())
    assert receipt['status']=='PASS' and sha(archive)==receipt['sha256']
    with tarfile.open(archive) as tf:
        seen=set()
        for member in tf.getmembers():
            dest=(exp/member.name).resolve()
            assert member.isfile() and dest.is_relative_to(exp.resolve()) and member.name not in seen
            seen.add(member.name)
            payload=tf.extractfile(member).read()
            if dest.exists():assert dest.read_bytes()==payload,member.name
            else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
    manifest=json.loads((exp/'auxiliary_manifest.json').read_text())
    for row in manifest['files']:assert sha(exp/row['path'])==row['sha256']
    print(json.dumps(dict(status='PASS',files=len(manifest['files']),all_hashes_verified=True)))


if __name__=='__main__':main()
