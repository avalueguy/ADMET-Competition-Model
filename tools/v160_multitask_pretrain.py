"""Disjoint pretraining shards for the new structure-only transfer route."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v159_multitask_transfer as model
ap=argparse.ArgumentParser();ap.add_argument('start',type=int);ap.add_argument('stop',type=int);args=ap.parse_args()
model.torch.set_num_threads(2);model.torch.backends.cuda.matmul.allow_tf32=False;model.torch.set_float32_matmul_precision('highest')
d=model.v.load();tasks=model.v.tasks(d)
for task in tasks[args.start:args.stop]:
    k=d['tasks'].index(task)
    for part in ('fold0','fold1'):
        for c in model.CONFIGS:model.predict(d,k,part,c)
    print(json.dumps(dict(stage='pretrained',task=task,selection_effect='NONE',signature=model.sig())),flush=True)
