"""Archive the six adopted/experimental final checkpoints without credentials."""
from pathlib import Path
import hashlib
import json
import platform
import tarfile
import torch
import numpy
import scipy
import sklearn
import rdkit

BASE = Path('/root/autodl-tmp')
DEST = BASE / 'admet_handoff_v366_20261104'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DEST.mkdir(exist_ok=True)
    files = []
    a = BASE / 'transformer_stage1_v364'
    b = BASE / 'transformer_stage2_v366'
    first = json.loads((a/'result_manifest.json').read_text())
    second = json.loads((b/'result_manifest.json').read_text())
    for origin, root, key in ((first, a, 'adopted_final_checkpoints'), (second, b, 'final_checkpoints')):
        for entry in origin[key]:
            path = Path(entry['remote_path'])
            assert path.is_relative_to(root) and sha(path) == entry['sha256']
            stage = root.name
            target = f'experiments/{stage}/{entry["relative_path"]}'
            files.append(dict(source=path, path=target, sha256=entry['sha256'], bytes=path.stat().st_size))
    assert len(files) == 6
    environment = dict(python=platform.python_version(), torch=torch.__version__, numpy=numpy.__version__,
                       scipy=scipy.__version__, sklearn=sklearn.__version__, rdkit=rdkit.__version__,
                       gpu='RTX 3090', training_gpu_remote_only=True)
    manifest = DEST / 'remote_weight_manifest.json'
    manifest.write_text(json.dumps(dict(status='PASS', files=[{k:v for k,v in r.items() if k!='source'} for r in files],
                                       environment=environment, passwords_included=False), indent=2)+'\n')
    archive = DEST / 'v366_final_weights.tar.gz'
    with tarfile.open(archive, 'w:gz') as tf:
        for r in files:
            tf.add(r['source'], arcname=r['path'])
        tf.add(manifest, arcname='docs/v367_remote_weight_manifest.json')
    with tarfile.open(archive) as tf:
        for r in files:
            assert hashlib.sha256(tf.extractfile(r['path']).read()).hexdigest() == r['sha256']
    result = dict(status='PASS', path=str(archive), sha256=sha(archive), bytes=archive.stat().st_size,
                  files=6, environment=environment)
    (DEST/'weight_archive.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
