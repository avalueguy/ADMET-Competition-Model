"""Collect lightweight evidence; keep graph weights and caches on the server."""
import json,hashlib,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
assert str(ROOT)=='/home/czp123/admet_v23_20260928'
out=ROOT/'admet_model_outputs_v27'
names=['admet_v27_protocol.json','admet_v27_models.py','admet_v27_run.py']
(out/'source_manifest.json').write_text(json.dumps({n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},indent=2),encoding='utf-8')
files=list(out.glob('*.json'))+list(out.glob('*.log'))+list(out.glob('*/*/metadata.json'))+list(out.glob('*/*/*/metadata.json'))
files+=list((ROOT/'admet_submissions/v27').glob('*'))
dest=ROOT/'admet_v27_evidence.tar.gz'
with tarfile.open(dest,'w:gz') as tar:
    for file in files:
        if file.is_file():tar.add(file,arcname=file.relative_to(ROOT).as_posix(),recursive=False)
print(json.dumps(dict(files=len(files),bytes=dest.stat().st_size)))
