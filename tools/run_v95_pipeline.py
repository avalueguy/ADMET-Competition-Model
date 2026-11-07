"""Schedule disjoint endpoint jobs; concurrency does not change model settings."""
from pathlib import Path
import argparse, json, subprocess, sys, time

ROOT=Path(__file__).resolve().parents[1]


def worker(index, workers):
    sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
    import v95_graph_specialists as v
    v.torch.set_num_threads(2); v.torch.set_num_interop_threads(1)
    v.torch.backends.cuda.matmul.allow_tf32=False
    v.torch.set_float32_matmul_precision('highest')
    d=v.load()
    order=sorted(range(len(d['tasks'])),key=lambda k:-int(((d['tid']==k)&(d['split']==0)).sum()))
    for k in order[index::workers]:
        for part in ('fold0','fold1'):
            for seed in v.SEEDS:v.train(d,k,part,seed)


def run(workers):
    out=ROOT/'admet_model_outputs_v95';out.mkdir(exist_ok=True)
    processes=[]
    for i in range(workers):
        f=(ROOT/f'v95_worker_{i}.log').open('ab')
        p=subprocess.Popen([sys.executable,'-u',__file__,'worker','--index',str(i),'--workers',str(workers)],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
        f.close();processes.append(p)
    status=dict(status='DEVELOPMENT_TRAINING',workers=workers,pids=[p.pid for p in processes])
    (out/'pipeline_status.json').write_text(json.dumps(status,indent=2)+'\n')
    codes=[p.wait() for p in processes]
    if any(codes):
        status.update(status='WORKER_FAILED',returncodes=codes)
        (out/'pipeline_status.json').write_text(json.dumps(status,indent=2)+'\n')
        raise RuntimeError(status)
    for stage in ('develop','confirm'):
        status.update(status=stage.upper());(out/'pipeline_status.json').write_text(json.dumps(status,indent=2)+'\n')
        subprocess.run([sys.executable,'-u',str(ROOT/'tools/v95_graph_specialists.py'),stage],cwd=ROOT,check=True)
    status.update(status='CONFIRMATION_COMPLETE');(out/'pipeline_status.json').write_text(json.dumps(status,indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['run','worker'])
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--index',type=int,default=0);args=ap.parse_args()
    if args.stage=='run':run(args.workers)
    else:worker(args.index,args.workers)
