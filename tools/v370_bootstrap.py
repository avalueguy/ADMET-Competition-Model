"""Prepare structure cache, check CUDA numerics, then run the bounded experiment."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def status(stage, **extra):
    value = dict(stage=stage, pid=os.getpid(), at_unix=time.time(), **extra)
    (ROOT / 'bootstrap_status.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value), flush=True)


def main():
    try:
        for stage, arguments, log in (
            ('GRAPHS', ['v370_graph_transformer.py', 'graphs'], 'graphs.log'),
            ('SMOKE', ['v370_graph_transformer.py', 'smoke'], 'smoke.log'),
            ('TRAINING', ['v370_run_stage5.py'], 'training.log'),
        ):
            status(stage)
            with (ROOT / log).open('a') as stream:
                subprocess.run([sys.executable, '-u', *arguments], cwd=ROOT,
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
        status('COMPLETE')
    except Exception as exc:
        status('FAILED', error=str(exc))
        raise


if __name__ == '__main__':
    main()
