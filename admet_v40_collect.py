import tarfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'admet_model_outputs_v40'
files = list(OUT.glob('*.json')) + list((ROOT / 'admet_submissions/v40').glob('*'))
files += list(OUT.glob('*/valid/metadata.json')) + list(OUT.glob('*/fold*/metadata.json')) + list(OUT.glob('*/final/metadata.json'))
with tarfile.open(ROOT / 'admet_v40_evidence.tar.gz', 'w:gz') as archive:
    for path in files:
        if path.is_file():
            archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
print(len(files), (ROOT / 'admet_v40_evidence.tar.gz').stat().st_size)
