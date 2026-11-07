"""Parallelize disjoint frozen parts; score only after every worker finishes."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    workers=[];logs=[]
    for part in ('fold2','valid','random','scaffold'):
        log=(ROOT/f'v108_confirm_{part}.log').open('ab');logs.append(log)
        workers.append(subprocess.Popen([sys.executable,'-u','tools/v108_confirm_part.py',part],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in workers]
    for log in logs:log.close()
    assert codes==[0]*4,codes
    for file,stage in [('v108_supervised_transfer.py','confirm'),('v108_delivery.py','assemble'),('v108_delivery.py','package')]:
        subprocess.run([sys.executable,'-u','tools/'+file,stage],cwd=ROOT,check=True)
