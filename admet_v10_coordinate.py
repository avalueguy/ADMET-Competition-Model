"""Run v10 graph candidates on the authorized remote GPU and publish only gated changes."""
import json, os, subprocess, sys, time
from pathlib import Path
from admet_v6_runner import remote_only, save, emit, sha
from admet_v10_runner import protocol, OUT


def launch(args, log):
    h=open(log,'a',buffering=1); env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    p=subprocess.Popen([sys.executable,'-u',*args],stdout=h,stderr=subprocess.STDOUT,env=env,start_new_session=True); h.close(); return p


def main():
    remote_only(); p=protocol(); OUT.mkdir(exist_ok=True)
    assert json.loads(Path('admet_release_v9/independent_verification.json').read_text())['status']=='VERIFIED'
    tasks=p['tasks']; workers={}
    for shard in range(3):
        subset=tasks[shard::3]
        workers[str(shard)]=launch(['admet_v10_runner.py','train','--family','graph_transfer','--tasks',*subset],f'logs/v10_graph_{shard}.log')
    gate=None; gate_task=None
    while True:
        failed=[k for k,v in workers.items() if v.poll() not in [None,0]]
        if failed: save(OUT/'coordinator_status.json',dict(status='FAILED',workers=failed)); raise RuntimeError(failed)
        if gate is not None and gate.poll() is not None:
            if gate.returncode: save(OUT/'coordinator_status.json',dict(status='FAILED',stage='evaluate',task=gate_task)); raise RuntimeError(gate_task)
            gate=None; gate_task=None
        pending=[t for t in tasks if not (OUT/t/'decision.json').exists()]
        if gate is None:
            for task in pending:
                if (OUT/task/'graph_transfer_cv.json').exists():
                    gate_task=task; gate=launch(['admet_v10_runner.py','evaluate','--tasks',task],'logs/v10_evaluate.log'); break
        done={k:v.poll() for k,v in workers.items()}
        save(OUT/'coordinator_status.json',dict(status='RUNNING',worker_exit_codes=done,active_evaluation=gate_task,pending_tasks=pending,
            completed_candidates=len(list(OUT.glob('*/graph_transfer_cv.json')))))
        if not pending and gate is None: break
        if pending and gate is None and all(v==0 for v in done.values()): raise RuntimeError('incomplete candidate pool')
        time.sleep(20)
    emit(stage='v10_training_complete',tasks=len(tasks))


if __name__=='__main__': main()
