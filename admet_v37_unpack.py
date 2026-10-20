"""Restrict archive extraction to this version's lightweight evidence artifacts."""
import tarfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent
with tarfile.open(ROOT / 'admet_v37_evidence.tar.gz', 'r:gz') as archive:
    members = archive.getmembers()
    for member in members:
        rel = Path(member.name)
        target = (ROOT / rel).resolve()
        assert not rel.is_absolute() and '..' not in rel.parts and target.is_relative_to(ROOT)
        assert member.isfile() and not member.issym() and not member.islnk()
        assert rel.parts[0] == 'admet_model_outputs_v37' or (rel.parts[0] == 'admet_submissions' and rel.parts[1] == 'v37')
    archive.extractall(ROOT, members=members, filter='data')
print('Extracted', len(members), 'v37 evidence files')
