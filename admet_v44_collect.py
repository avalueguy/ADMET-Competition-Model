import tarfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT/'admet_model_outputs_v44'
files = list((ROOT/'admet_submissions/v44').glob('*'))+list(OUT.glob('*.json'))+list(OUT.glob('*/*/metadata.json'))
with tarfile.open(ROOT/'admet_v44_evidence.tar.gz','w:gz') as archive:
    for path in files:
        if path.is_file():archive.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
print('Collected',len(files),'lightweight evidence files',flush=True)
