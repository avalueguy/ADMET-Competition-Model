"""Validate immutable delta, complete smoke on remote host, launch once."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tarfile

ROOT=Path('/root/autodl-tmp/regression9_v380')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    receipt=json.loads(sys.argv[1])
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT/'launch.json').exists(),'Duplicate launch prohibited'
    previous=ROOT.parent/'regression9_v377'
    assert json.loads((ROOT.parent/'regression9_v379/completion_receipt.json').read_text())['status']=='PASS'
    archive=ROOT.parent/Path(receipt['archive']).name
    assert sha(archive)==receipt['sha256']
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            dest=(ROOT/member.name).resolve()
            assert member.isfile() and dest.is_relative_to(ROOT.resolve())
            data=tf.extractfile(member).read()
            if dest.exists():assert dest.read_bytes()==data
            else:dest.write_bytes(data)
    manifest=json.loads((previous/'data_manifest.json').read_text())
    names=['official_rows.npz','expert_features.npz','stereo_keys.npz','submission_template.csv','v374_baseline.csv','data_manifest.json']
    names += [row[k] for row in manifest['controls'] for k in ('control','design')]
    for name in names:
        src,dest=previous/name,ROOT/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha(src)==sha(dest)
        else:os.link(src,dest)
    for name,key in (('official_rows.npz','official_rows_sha256'),('expert_features.npz','expert_features_sha256'),('stereo_keys.npz','stereo_keys_sha256')):
        assert sha(ROOT/name)==manifest[key]
    for row in manifest['controls']:
        for key in ('control','design'):assert sha(ROOT/row[key])==row[key+'_sha256']
    subprocess.run([sys.executable,'-u','v380_smoke.py'],cwd=ROOT,check=True)
    with (ROOT/'training.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,'-u','v380_run_experts.py'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value=dict(status='STARTED',pid=proc.pid,root=str(ROOT),workers=3,input_receipt=receipt)
    (ROOT/'launch.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':main()
