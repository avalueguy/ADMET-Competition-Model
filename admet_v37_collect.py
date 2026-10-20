"""Collect predictions and small metadata, keeping model weights and feature caches remote."""
import tarfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'admet_model_outputs_v37'
files = list(OUT.glob('*.json')) + list((ROOT / 'admet_submissions/v37').glob('*'))
files += list(OUT.glob('*/valid/metadata.json')) + list(OUT.glob('*/final/metadata.json'))
files += list(OUT.glob('*/fold*/metadata.json')) + list((OUT / 'features').glob('*.json'))
with tarfile.open(ROOT / 'admet_v37_evidence.tar.gz', 'w:gz') as archive:
    for path in files:
        if path.is_file():
            archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
print(len(files), (ROOT / 'admet_v37_evidence.tar.gz').stat().st_size)
