"""Run disjoint training workers, frozen validation and conditional delivery."""
import os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v108_supervised_transfer as v

def main():
    v.torch.set_num_threads(2);v.torch.set_num_interop_threads(1);v.torch.backends.cuda.matmul.allow_tf32=False
    v.protocol();d=v.load()
    for part in ('fold0','fold1'):
        a=v.design(d,part)
        for seed in v.SEEDS:v.pretrain(d,part,seed,a)
        del a
        workers=[];logs=[]
        for i in range(3):
            log=(ROOT/f'v108_{part}_worker{i}.log').open('ab');logs.append(log)
            workers.append(subprocess.Popen([sys.executable,'-u','tools/v108_part_worker.py','--part',part,'--worker',str(i),'--workers','3'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT))
        codes=[p.wait() for p in workers]
        for log in logs:log.close()
        assert codes==[0,0,0],(part,codes)
        v.emit(stage='part_complete',part=part)
    del d
    for file,stage in [('v108_supervised_transfer.py','develop'),('v108_supervised_transfer.py','confirm'),('v108_delivery.py','assemble'),('v108_delivery.py','package')]:
        subprocess.run([sys.executable,'-u',str(ROOT/'tools'/file),stage],cwd=ROOT,check=True)
    v.emit(stage='pipeline_complete')

if __name__=='__main__':main()
