"""Reuse disjoint development workers, then parallelize only frozen task fits."""
from pathlib import Path
import argparse,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'admet_model_outputs_v98'
WORKERS=(11231,11232,11233,11234,11235,11236)

def save(x):(OUT/'pipeline_status.json').write_text(json.dumps(x))
def worker(index,workers):
    sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
    import v98_shape_experts as v
    d=v.load();f=json.loads((OUT/'frozen_selection.json').read_text());assert f['signature']==v.sig()
    for r in f['selected'][index::workers]:
        k=d['tasks'].index(r['task']);spec=next(s for s in v.SPECS if s['name']==r['config']['spec'])
        for part in ('fold2','valid'):
            for seed in v.SEEDS:v.train(d,k,part,spec,r['config']['mode'],seed)

def main():
    for _ in range(720):
        n=len(list((OUT/'models').glob('*/fold[01]/*/*/*/metadata.json')))
        if n==352:break
        alive=[pid for pid in WORKERS if Path(f'/proc/{pid}').exists()]
        if not alive:raise RuntimeError(dict(message='Development workers ended before all caches',models=n))
        save(dict(status='DEVELOPMENT_TRAINING',pids=alive,models=n,resumed_scheduler=True));time.sleep(10)
    else:raise TimeoutError('Development cache not complete')
    # No worker mutates a completed cache; waiting for 352 avoids any duplication.
    save(dict(status='DEVELOP',models=352,resumed_scheduler=True))
    subprocess.run([sys.executable,'-u',str(ROOT/'tools/v98_shape_experts.py'),'develop'],cwd=ROOT,check=True)
    f=json.loads((OUT/'frozen_selection.json').read_text());workers=min(6,len(f['selected']));ps=[]
    save(dict(status='PARALLEL_FROZEN_CONFIRMATION_TRAINING',selected=[r['task'] for r in f['selected']],workers=workers))
    for index in range(workers):
        log=(ROOT/f'v98_confirm_worker_{index}.log').open('ab')
        ps.append(subprocess.Popen([sys.executable,'-u',__file__,'worker','--index',str(index),'--workers',str(workers)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT));log.close()
    codes=[p.wait() for p in ps]
    if any(codes):save(dict(status='WORKER_FAILED',returncodes=codes));raise RuntimeError(codes)
    save(dict(status='CONFIRM',models=352,resumed_scheduler=True))
    subprocess.run([sys.executable,'-u',str(ROOT/'tools/v98_shape_experts.py'),'confirm'],cwd=ROOT,check=True)
    save(dict(status='CONFIRMATION_COMPLETE',resumed_scheduler=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['run','worker']);ap.add_argument('--index',type=int,default=0);ap.add_argument('--workers',type=int,default=6);args=ap.parse_args()
    if args.stage=='run':main()
    else:worker(args.index,args.workers)
