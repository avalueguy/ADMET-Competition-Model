"""Independent fixed MoLFormer experiment versus scored v11."""
import json,os,subprocess,sys,time
from pathlib import Path
from admet_v6_runner import remote_only,save,emit
from admet_v13_runner import protocol,OUT


def launch(args,log):
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    with open(log,'a') as handle:
        return subprocess.Popen([sys.executable,'-u',*args],stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True)


def main():
    remote_only();p=protocol();OUT.mkdir(exist_ok=True)
    workers={f:launch(['admet_v13_runner.py','train','--family',f],f'logs/v13_{f}.log') for f in p['families']}
    gate=None;gate_task=None
    while True:
        failed=[n for n,q in workers.items() if q.poll() not in [None,0]]
        if failed:save(OUT/'coordinator_status.json',dict(status='FAILED',workers=failed));raise RuntimeError(failed)
        if gate is not None and gate.poll() is not None:
            if gate.returncode:save(OUT/'coordinator_status.json',dict(status='FAILED',task=gate_task));raise RuntimeError(gate_task)
            gate=None;gate_task=None
        pending=[t for t in p['tasks'] if not (OUT/t/'decision.json').exists()]
        if gate is None:
            for t in pending:
                if all((OUT/t/(f+'_cv.json')).exists() for f in p['families']):
                    gate_task=t;gate=launch(['admet_v13_runner.py','evaluate','--tasks',t],'logs/v13_evaluate.log');break
        counts={f:len(list(OUT.glob('*/'+f+'_cv.json'))) for f in p['families']}
        save(OUT/'coordinator_status.json',dict(status='RUNNING',counts=counts,active_evaluation=gate_task,pending_tasks=pending,worker_exit_codes={k:v.poll() for k,v in workers.items()}))
        if not pending and gate is None:break
        time.sleep(10)
    emit(stage='v13_training_complete',tasks=len(p['tasks']))


if __name__=='__main__':main()
