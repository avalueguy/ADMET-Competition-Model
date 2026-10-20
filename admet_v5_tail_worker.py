"""Run independent final tasks in a separate directory and publish atomically."""
import argparse
import json
import os
from pathlib import Path
import torch
from threadpoolctl import threadpool_limits
from admet_v5_remote_train import run_task


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tasks',nargs='+',default=['vdss_lombardo','solubility_aqsoldb','ppbr_az','pgp_broccatelli'])
    ap.add_argument('--tag',default='tail')
    args=ap.parse_args()
    assert args.tag.isalnum()
    if os.name=='nt' or not Path('/root/autodl-tmp/admet_20260925').is_dir():
        raise RuntimeError('Remote workspace required')
    assert torch.cuda.is_available()
    torch.set_num_threads(6)
    protocol=json.loads(Path('admet_v5_protocol.json').read_text())
    scratch=Path('admet_model_outputs_v5_'+args.tag);scratch.mkdir(exist_ok=True)
    main_out=Path('admet_model_outputs_v5')
    for task in args.tasks:
        dest=main_out/task
        if dest.exists():
            print('SKIP main runner already reached '+task,flush=True)
            continue
        if not (scratch/task/'result.json').exists():
            run_task(task,protocol,scratch)
        # POSIX directory rename fails if the main task directory is nonempty.
        # A duplicate experiment is preserved separately, never overwrites it.
        if not dest.exists():
            try:
                (scratch/task).rename(dest)
                print('PUBLISHED completed task '+task,flush=True)
            except OSError:
                print('KEEP separate completed task '+task,flush=True)
        else:
            print('KEEP separate completed task '+task,flush=True)
    print('TAIL_COMPLETE',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=6):main()
