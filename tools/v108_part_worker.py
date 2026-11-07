"""Disjoint endpoint workers after the part's shared checkpoint is complete.

Use only on a part not currently being trained by the sequential pipeline.
"""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v108_supervised_transfer as v

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--part',required=True);ap.add_argument('--worker',type=int,required=True);ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    v.torch.set_num_threads(2);v.torch.set_num_interop_threads(1);v.torch.backends.cuda.matmul.allow_tf32=False
    v.protocol();d=v.load();a=v.design(d,args.part)
    for seed in v.SEEDS:
        assert v.cached(v.OUT/'pretrain'/args.part/str(seed)) is not None,'Run shared training before parallel workers.'
    for k in range(len(d['tasks'])):
        if k%args.workers!=args.worker:continue
        for kind in ('transfer','scratch'):
            v.predictions(d,k,args.part,kind,a)
