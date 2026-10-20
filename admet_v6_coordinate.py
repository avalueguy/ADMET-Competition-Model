"""Finish ready tasks as candidate workers complete; logs persist remotely."""
import json
import subprocess
import time
from pathlib import Path
from admet_v6_runner import OUT, DATA

def ready(task):
    big={'solubility_aqsoldb','ld50_zhu','ames','cyp2c9_veith','cyp2d6_veith','cyp3a4_veith'}
    expected={'kernel_1','kernel_10','xgb','dense'}
    if task not in big:expected.add('extra')
    if task in {'caco2_wang','lipophilicity_astrazeneca','ppbr_az','solubility_aqsoldb'}:expected.add('graph')
    return (OUT/task/'baseline_oof.csv').exists() and all((OUT/task/(f+'_oof.csv')).exists() for f in expected)

def main():
    tasks=sorted(p.name for p in DATA.iterdir() if p.is_dir());python='.venv/bin/python'
    done=set();failures={};start=time.time()
    while len(done)<len(tasks):
        changed=False
        for task in tasks:
            if task in done or not ready(task):continue
            if task in failures:continue
            commands=[[python,'-u','admet_v6_runner.py','evaluate','--tasks',task],
                      [python,'-u','admet_v6_finalize.py','confirm','--tasks',task]]
            for command in commands:
                result=subprocess.run(command)
                if result.returncode:
                    failures[task]=result.returncode;break
            else:done.add(task)
            changed=True
            Path('admet_model_outputs_v6/coordinator_status.json').write_text(json.dumps(dict(done=sorted(done),
               failures=failures,pending=sorted(set(tasks)-done),elapsed_seconds=round(time.time()-start)),indent=2))
        if failures or time.time()-start>14400:
            raise RuntimeError(f'Coordinator requires attention: {failures}')
        if not changed:time.sleep(15)
    print('ALL_TASK_DECISIONS_COMPLETE',flush=True)


if __name__=='__main__':main()
