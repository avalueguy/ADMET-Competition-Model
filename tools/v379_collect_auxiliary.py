"""Collect completed job scheduling and loss logs without modifying models."""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--version',choices=['378','379','380','381'],required=True);args=ap.parse_args()
    root=Path('/root/autodl-tmp/regression9_v'+args.version)
    assert json.loads((root/'run_status.json').read_text())['stage']=='EVIDENCE_READY'
    files=list((root/'prefit_logs').glob('*.log'))
    files += [root/n for n in ('training.log','launch.json','run_status.json','smoke_receipt.json',
              'parallel_prefit_launch.json','parallel_prefit_results.json','parallel_prefit_complete.json') if (root/n).is_file()]
    entries=[dict(path=p.relative_to(root).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in files]
    manifest=root/'auxiliary_manifest.json';manifest.write_text(json.dumps(dict(files=entries),indent=2)+'\n')
    archive=root/'auxiliary_logs.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for p in files+[manifest]:tf.add(p,arcname=p.relative_to(root).as_posix())
    value=dict(status='PASS',files=len(files),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (root/'auxiliary_archive.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()
