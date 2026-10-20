"""Collect only lightweight evidence, never remote checkpoints or feature matrices."""
import tarfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'admet_model_outputs_v43'
files = list((ROOT / 'admet_submissions/v43').glob('*'))
files += list(OUT.glob('*.json')) + list(OUT.glob('*/valid/metadata.json'))
files += list(OUT.glob('*/final/metadata.json')) + list(OUT.glob('*/fold*/metadata.json'))
files += list(OUT.glob('features/*/*.json'))
with tarfile.open(ROOT / 'admet_v43_evidence.tar.gz', 'w:gz') as archive:
    for path in files:
        if path.is_file():
            archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
print('Collected', len(files), 'lightweight artifacts', flush=True)
