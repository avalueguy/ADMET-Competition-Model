import tarfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent
files = list((ROOT / 'admet_submissions/v42').glob('*'))
for version in [40, 41, 42]:
    out = ROOT / f'admet_model_outputs_v{version}'
    files += list(out.glob('*.json')) + list(out.glob('*/valid/metadata.json'))
    files += list(out.glob('*/final/metadata.json')) + list(out.glob('*/fold*/metadata.json'))
    files += list(out.glob('features/*.json'))
with tarfile.open(ROOT / 'admet_v42_evidence.tar.gz', 'w:gz') as archive:
    for path in files:
        if path.is_file():
            archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
print(len(files), (ROOT / 'admet_v42_evidence.tar.gz').stat().st_size)
