"""Resume task-local gates once their fixed candidate pools finish."""
import json
import subprocess
import time
from pathlib import Path
from admet_v6_runner import remote_only
from admet_v7_runner import protocol,evaluate,OUT


def main():
    remote_only();p=protocol();tasks=sorted(x.name for x in Path('admet_competition_data/data_benchmark').iterdir() if x.is_dir())
    while True:
        pending=[]
        for task in tasks:
            if (OUT/task/'decision.json').exists():continue
            names=p['families_all']+(['chemberta_ft'] if task in p['finetune_tasks'] else [])
            if all((OUT/task/(name+'_cv.json')).exists() for name in names):evaluate(task)
            else:pending.append(task)
        if not pending:break
        print(json.dumps(dict(stage='waiting_for_fixed_pools',tasks=pending)),flush=True);time.sleep(30)
    for stage in ['publish','verify','report']:
        subprocess.run(['.venv/bin/python','-u','admet_v7_release.py',stage],check=True)


if __name__=='__main__':main()
