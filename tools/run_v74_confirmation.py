"""Run frozen confirmation; never tune choices using confirmation labels."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import admet_v67_data as b
import admet_v74_specialists as v
from select_v74 import select,confirm

def main():
    v.torch.set_num_threads(4);v.torch.set_num_interop_threads(2);v.torch.backends.cuda.matmul.allow_tf32=False
    d=b.build();s=select(d);tasks=[r['task'] for r in s['tasks'] if r['selection']]
    assert tasks,'No selection-fold winner; inspect results before further experiments'
    g=v.graphs(d)
    v.develop(d,g,tasks,['fold2','valid'])
    for task in tasks:
        k=d['tasks'].index(task)
        for part in ('fold2','valid'):
            fit,q=v.rows(d,k,part)
            meta=json.loads((v.OUT/task/part/'inner/metadata.json').read_text(encoding='utf-8'))
            for seed in v.P['final_seeds'][1:]:
                v.train_net(d,g,k,fit,q,seed,v.OUT/task/part/f'outer_{seed}',meta['epochs'])
    confirm(d)

if __name__=='__main__':main()
