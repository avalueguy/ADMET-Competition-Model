import tarfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent
with tarfile.open(ROOT/'admet_v44_evidence_remote.tar.gz','r:gz') as archive:
    members = archive.getmembers()
    for member in members:
        rel = Path(member.name)
        target = (ROOT/rel).resolve()
        assert not rel.is_absolute() and '..' not in rel.parts and target.is_relative_to(ROOT)
        assert member.isfile() and not member.issym() and not member.islnk()
        assert rel.parts[0]=='admet_model_outputs_v44' or (rel.parts[0]=='admet_submissions' and rel.parts[1]=='v44')
    archive.extractall(ROOT,members=members,filter='data')
print('Extracted',len(members),'lightweight files')
