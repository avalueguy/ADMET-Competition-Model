"""Freeze transitive local Python dependencies before running v16."""
import ast,hashlib,json
from pathlib import Path

pending=[Path(f'admet_v16_{part}.py') for part in ['coordinate','runner','release','models','ensemble']]
pending+=[Path('test_admet_v16.py')]+[Path(f'admet_v{v}_models.py') for v in range(6,14)]
seen=set()
while pending:
    p=pending.pop()
    if p in seen:continue
    assert p.exists(),p
    seen.add(p)
    for n in ast.walk(ast.parse(p.read_text(encoding='utf-8'))):
        modules=[x.name for x in n.names] if isinstance(n,ast.Import) else [n.module] if isinstance(n,ast.ImportFrom) else []
        for module in modules:
            if module:
                dep=Path(module.split('.')[0]+'.py')
                if dep.is_file():pending.append(dep)
seen.update(Path(f'admet_v{v}_protocol.json') for v in [*range(6,14),15,16])
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(seen)}
out=Path('admet_v16_source_freeze.json')
if out.exists() and json.loads(out.read_text())!=hashes:
    out.write_text(json.dumps(hashes,indent=2),encoding='utf-8')
elif not out.exists():out.write_text(json.dumps(hashes,indent=2),encoding='utf-8')
print(json.dumps({'frozen_files':len(hashes),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
