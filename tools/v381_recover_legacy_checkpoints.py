"""Extract only hash-verified v67 final checkpoints to isolated research folder."""
from pathlib import Path
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=ROOT/'releases/official_only_v67_v68_manifest.json'
    rows=json.loads(manifest.read_text())['files']
    wanted={r['path']:r for r in rows if r['path'].startswith('admet_model_outputs_v67/final/') and r['path'].endswith('.pt')}
    destroot=ROOT/'experiments/v381_legacy_checkpoint_recovery'
    records=[]
    with tarfile.open(ROOT/'releases/official_only_v67_v68_20261020.tar.gz') as tf:
        members={m.name.removeprefix('./'):m for m in tf.getmembers() if m.isfile()}
        for name,row in wanted.items():
            assert name in members,name
            payload=tf.extractfile(members[name]).read()
            assert len(payload)==row['size'] and hashlib.sha256(payload).hexdigest()==row['sha256']
            target=destroot/name;assert target.resolve().is_relative_to(destroot.resolve())
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert target.read_bytes()==payload
            else:target.write_bytes(payload)
            records.append(dict(path=target.relative_to(ROOT).as_posix(),sha256=row['sha256'],bytes=len(payload)))
    result=dict(status='HASH_VERIFIED_RECOVERY_ONLY',files=records,no_model_execution=True,no_fitting=True,
                source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                limitation='Eight v67 final shared/gated checkpoints only. Not complete historic ensemble OOF or recovered continuous-task upstream weights. No gain inferred.')
    (ROOT/'docs/v381_legacy_recovery.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],checkpoints=len(records),all_member_hashes_verified=True)))


if __name__=='__main__':main()
