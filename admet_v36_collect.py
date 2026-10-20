import tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'admet_model_outputs_v36'
files=list(OUT.glob('*.json'))+list((ROOT/'admet_submissions/v36').glob('*'))
for v in [33,34,35]:
    out=ROOT/f'admet_model_outputs_v{v}'
    files+=list(out.glob('*.json'))+list(out.glob('*/valid/metadata.json'))+list(out.glob('*/final/metadata.json'))+list(out.glob('*/fold*/metadata.json'))
with tarfile.open(ROOT/'admet_v36_evidence.tar.gz','w:gz') as tar:
    for p in files:
        if p.is_file():tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
print(len(files),(ROOT/'admet_v36_evidence.tar.gz').stat().st_size)
