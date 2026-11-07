"""Disjoint CPU endpoint workers; freeze selection before confirmation."""
from pathlib import Path
import argparse,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]

def worker(index,workers):
    sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
    import v98_shape_experts as v
    d=v.load()
    order=sorted(range(len(d['tasks'])),key=lambda k:-int(((d['tid']==k)&(d['split']==0)).sum()))
    for k in order[index::workers]:
        for part in ('fold0','fold1'):
            for spec in v.SPECS:
                for mode in v.MODES:
                    for seed in v.SEEDS:v.train(d,k,part,spec,mode,seed)

def run(workers):
    out=ROOT/'admet_model_outputs_v98';out.mkdir(exist_ok=True);ps=[]
    features=out/'features/audit.json'
    while not features.exists():
        if not (ROOT/'v98_features.log').exists():raise RuntimeError('Feature generation was not launched.')
        time.sleep(10)
    subprocess.run([sys.executable,'-u',str(ROOT/'tools/v98_shape_experts.py'),'smoke'],cwd=ROOT,check=True)
    for i in range(workers):
        f=(ROOT/f'v98_worker_{i}.log').open('ab')
        ps.append(subprocess.Popen([sys.executable,'-u',__file__,'worker','--index',str(i),'--workers',str(workers)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT));f.close()
    status=dict(status='DEVELOPMENT_TRAINING',pids=[p.pid for p in ps]);(out/'pipeline_status.json').write_text(json.dumps(status))
    codes=[p.wait() for p in ps]
    if any(codes):
        status.update(status='WORKER_FAILED',returncodes=codes);(out/'pipeline_status.json').write_text(json.dumps(status));raise RuntimeError(status)
    for stage in ('develop','confirm'):
        status.update(status=stage.upper());(out/'pipeline_status.json').write_text(json.dumps(status))
        subprocess.run([sys.executable,'-u',str(ROOT/'tools/v98_shape_experts.py'),stage],cwd=ROOT,check=True)
    status.update(status='CONFIRMATION_COMPLETE');(out/'pipeline_status.json').write_text(json.dumps(status))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['run','worker']);ap.add_argument('--index',type=int,default=0);ap.add_argument('--workers',type=int,default=6);args=ap.parse_args()
    if args.stage=='run':run(args.workers)
    else:worker(args.index,args.workers)
