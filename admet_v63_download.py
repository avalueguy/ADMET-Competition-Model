"""Fetch pinned public synthetic-pretraining checkpoints; never fetch benchmark labels."""
import json,hashlib,urllib.request,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent;dest=ROOT/'admet_pretrained/tabicl_v2';dest.mkdir(parents=True,exist_ok=True)
files={'tabicl-classifier-v2-20260212.ckpt':'bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0','tabicl-regressor-v2-20260212.ckpt':'0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a'}
for name,sha in files.items():
    file=dest/name
    if not file.exists():
        tmp=file.with_suffix('.download');url='https://huggingface.co/jingang/TabICL/resolve/main/'+name
        with urllib.request.urlopen(url,timeout=30) as r,tmp.open('wb') as out:
            while True:
                block=r.read(1048576)
                if not block:break
                out.write(block)
        assert hashlib.sha256(tmp.read_bytes()).hexdigest()==sha;tmp.replace(file)
    assert hashlib.sha256(file.read_bytes()).hexdigest()==sha
    subprocess.run(['ssh','-o','BatchMode=yes','czp123@100.109.143.42','mkdir -p /home/czp123/admet_v23_20260928/admet_pretrained/tabicl_v2'],check=True)
    subprocess.run(['scp','-q','-o','BatchMode=yes',str(file),'czp123@100.109.143.42:/home/czp123/admet_v23_20260928/admet_pretrained/tabicl_v2/'+name],check=True)
    print(json.dumps(dict(file=name,sha256=sha,bytes=file.stat().st_size,remote_uploaded=True)),flush=True)
