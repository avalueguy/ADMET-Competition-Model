"""Train one disjoint frozen-confirmation partition, without reading its scores."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import v108_supervised_transfer as v
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('part',choices=['fold2','valid','random','scaffold']);args=ap.parse_args()
    v.torch.set_num_threads(2);v.torch.set_num_interop_threads(1);v.torch.backends.cuda.matmul.allow_tf32=False
    f=json.loads((v.OUT/'frozen_selection.json').read_text());assert f['signature']==v.sig()
    tasks=[r['task'] for r in f['selected']]
    if tasks:v.run_part(v.load(),args.part,tasks)
    v.emit(stage='confirmation_part_complete',part=args.part,tasks=tasks)
