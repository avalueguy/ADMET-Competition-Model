"""Constrain downloaded evidence extraction to the new version's artifact paths."""
import tarfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent
with tarfile.open(ROOT / 'admet_v40_evidence.tar.gz', 'r:gz') as archive:
    members = archive.getmembers()
    for member in members:
        rel = Path(member.name)
        target = (ROOT / rel).resolve()
        assert not rel.is_absolute() and '..' not in rel.parts and target.is_relative_to(ROOT)
        assert member.isfile() and not member.issym() and not member.islnk()
        assert rel.parts[0] == 'admet_model_outputs_v40' or (
            rel.parts[0] == 'admet_submissions' and rel.parts[1] == 'v40')
    archive.extractall(ROOT, members=members, filter='data')
print('Extracted', len(members), 'v40 evidence files')
