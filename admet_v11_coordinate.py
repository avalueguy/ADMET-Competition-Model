"""Supervise remote v11 sparse-kernel workers and per-task gates."""
import json, os, subprocess, sys, time
from pathlib import Path
from admet_v6_runner import remote_only, save, emit
from admet_v11_runner import protocol, OUT


def launch(args, log):
    handle=open(log,'a',buffering=1)
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    proc=subprocess.Popen([sys.executable,'-u',*args],stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    handle.close();return proc


def main():
    remote_only();p=protocol();OUT.mkdir(exist_ok=True)
    anchor=json.loads(Path('admet_release_v10_guard/verification.json').read_text())
    assert anchor['status']=='VERIFIED_GUARDED_RELEASE'
    tasks=p['tasks'];families=p['families'];workers={}
    for shard in range(3):
        subset=tasks[shard::3]
        # Three workers plus one evaluator: at most 16 numerical-library threads.
        workers[str(shard)]=launch(['admet_v11_runner.py','train','--family','both','--tasks',*subset],f'logs/v11_worker_{shard}.log')
    gate=None;gate_task=None
    while True:
        failed=[name for name,proc in workers.items() if proc.poll() not in [None,0]]
        if failed:
            save(OUT/'coordinator_status.json',dict(status='FAILED',workers=failed));raise RuntimeError(failed)
        if gate is not None and gate.poll() is not None:
            if gate.returncode:
                save(OUT/'coordinator_status.json',dict(status='FAILED',stage='evaluate',task=gate_task));raise RuntimeError(gate_task)
            gate=None;gate_task=None
        pending=[t for t in tasks if not (OUT/t/'decision.json').exists()]
        if gate is None:
            for task in pending:
                if all((OUT/task/(f+'_cv.json')).exists() for f in families):
                    gate_task=task;gate=launch(['admet_v11_runner.py','evaluate','--tasks',task],'logs/v11_evaluate.log');break
        done={name:proc.poll() for name,proc in workers.items()}
        save(OUT/'coordinator_status.json',dict(status='RUNNING',worker_exit_codes=done,active_evaluation=gate_task,pending_tasks=pending,
            completed_uniform=len(list(OUT.glob('*/sparse_uniform_cv.json'))),completed_scaffold=len(list(OUT.glob('*/sparse_scaffold_cv.json')))))
        if not pending and gate is None:break
        if pending and gate is None and all(v==0 for v in done.values()):raise RuntimeError('workers finished with incomplete pools')
        time.sleep(20)
    emit(stage='v11_training_complete',tasks=len(tasks))


if __name__=='__main__':main()
