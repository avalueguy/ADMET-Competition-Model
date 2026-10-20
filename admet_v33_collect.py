"""Download-ready archive of predictions, audits and submission; no checkpoints."""
import tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'admet_model_outputs_v33'
files=list(OUT.glob('*.json'))+list(OUT.glob('*/valid/metadata.json'))+list(OUT.glob('*/final/metadata.json'))
files+=list(OUT.glob('*/fold*/metadata.json'))+list((ROOT/'admet_submissions/v33').glob('*'))
with tarfile.open(ROOT/'admet_v33_evidence.tar.gz','w:gz') as tar:
    for file in files:
        if file.is_file():tar.add(file,arcname=file.relative_to(ROOT).as_posix(),recursive=False)
print(len(files),(ROOT/'admet_v33_evidence.tar.gz').stat().st_size)
