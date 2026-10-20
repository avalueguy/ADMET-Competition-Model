"""Archive failed adaptive-inference caches, then restart both bounded sources."""
import json,os,subprocess
from pathlib import Path

root=Path('/home/czp123/admet_v23_20260928')
assert os.name!='nt' and Path.cwd()==root
archive=root/'admet_experiment_archive'/'adaptive_inference_20261017'
archive.mkdir(parents=True,exist_ok=True)
for version in (63,65):
    name=f'admet_v{version}_run.py'.encode()
    for file in Path('/proc').glob('[0-9]*/cmdline'):
        try: cmd=file.read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        assert not any(Path(arg.decode(errors='replace')).name.encode()==name for arg in cmd),str(file)
    source=root/f'admet_model_outputs_v{version}'
    dest=archive/source.name
    if not dest.exists():
        assert source.resolve().parent==root and dest.resolve().parent==archive
        source.rename(dest)
    else:
        assert not source.exists() or (source/'run.pid').exists()
    subprocess.run([str(root/'.venv/bin/python'),f'admet_v{version}_launch.py'],check=True)
print(json.dumps(dict(status='RESUMED',archived=str(archive),sources=[63,65])),flush=True)
