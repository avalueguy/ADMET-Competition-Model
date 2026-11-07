"""Disjoint deterministic pretraining for late v153 tasks; no model selection."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v153_path_latent_trees as model
ap=argparse.ArgumentParser();ap.add_argument('start',type=int);ap.add_argument('stop',type=int);args=ap.parse_args()
d=model.features.load(model.v.load());tasks=model.v.tasks(d)
for task in tasks[args.start:args.stop]:
    k=d['tasks'].index(task)
    for part in ('fold0','fold1'):
        for c in model.CONFIGS:model.predict(d,k,part,c)
    print(json.dumps(dict(stage='pretrained',task=task,selection_effect='NONE',signature=model.sig())),flush=True)
