"""Verify complete gzip archive and constrain extraction to intended evidence paths."""
import tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
archive=ROOT/'admet_v36_evidence_retry.tar.gz'
allowed={f'admet_model_outputs_v{v}' for v in [33,34,35,36]}|{'admet_submissions'}
with tarfile.open(archive,'r:gz') as tar:
    members=tar.getmembers()
    for member in members:
        rel=Path(member.name);target=(ROOT/rel).resolve()
        assert rel.parts[0] in allowed and not rel.is_absolute() and '..' not in rel.parts
        assert target.is_relative_to(ROOT) and member.isfile() and not member.issym() and not member.islnk()
        if rel.parts[0]=='admet_submissions':assert rel.parts[1]=='v36'
    tar.extractall(ROOT,members=members,filter='data')
print('Extracted',len(members),'lightweight evidence files')
