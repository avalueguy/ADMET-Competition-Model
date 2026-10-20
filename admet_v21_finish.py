"""Finish the current remote run after all fixed-protocol training completes."""
import json
import os
import time
import traceback
from pathlib import Path
from admet_v21_finetune import OUT, PROTOCOL, final, read_json, save_json, remote_only, pretrained
from admet_v21_release import repeat, export


def main():
    remote_only()
    tasks=read_json(PROTOCOL)['tasks']
    while not all((OUT/t/'selection.json').exists() for t in tasks):
        try:
            os.kill(20114,0)
        except ProcessLookupError:
            raise RuntimeError('Training stopped before all task selections existed')
        time.sleep(10)
    # Initialize the frozen base before per-fit seeds are set, including when
    # all label-free prefix caches already exist in this fresh process.
    pretrained()
    for task in tasks:
        route=read_json(OUT/task/'selection.json')
        if route['selection']['weight']>0:
            save_json(OUT/'finish_status.json',dict(stage='final',task=task))
            final(task)
    for task in tasks:
        if read_json(OUT/task/'selection.json')['selection']['weight']>0:
            save_json(OUT/'finish_status.json',dict(stage='repeat',task=task))
            repeat(task)
    save_json(OUT/'finish_status.json',dict(stage='export'))
    export()
    save_json(OUT/'finish_status.json',dict(stage='COMPLETE',automatic_submission=False))


if __name__=='__main__':
    try:
        main()
    except Exception as e:
        save_json(OUT/'finish_status.json',dict(stage='FAILED',error=str(e),traceback=traceback.format_exc()))
        raise
