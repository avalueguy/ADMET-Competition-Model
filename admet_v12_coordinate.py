"""Resume cached fits using disjoint task shards, then serialized gates."""
import json, os, subprocess, sys, time
from pathlib import Path
from admet_v6_runner import remote_only, save, emit
from admet_v12_runner import protocol, OUT


def launch(args,log):
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    with open(log,'a') as handle:
        return subprocess.Popen([sys.executable,'-u',*args],stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,env=env,start_new_session=True)


def main():
    remote_only();p=protocol();OUT.mkdir(exist_ok=True)
    tasks=p['tasks'];workers={}
    for family in p['families']:
        remaining=[t for t in tasks if not (OUT/t/(family+'_cv.json')).exists()]
        if not remaining:continue
        shards=3 if family=='boost_fragments' else 1
        for shard in range(shards):
            subset=remaining[shard::shards]
            if not subset:continue
            name=family+f'_shard{shard}'
            workers[name]=launch(['admet_v12_runner.py','train','--family',family,'--tasks',*subset],f'logs/v12_{name}.log')
    while True:
        for name,proc in workers.items():
            if proc.poll() not in [None,0]:
                save(OUT/'coordinator_status.json',dict(status='FAILED',stage='train',worker=name));raise RuntimeError(name)
        counts={f:len(list(OUT.glob('*/'+f+'_cv.json'))) for f in p['families']}
        save(OUT/'coordinator_status.json',dict(status='TRAINING',counts=counts,worker_exit_codes={k:v.poll() for k,v in workers.items()}))
        if all(v.poll()==0 for v in workers.values()):break
        time.sleep(10)
    for task in tasks:
        save(OUT/'coordinator_status.json',dict(status='EVALUATING',task=task,completed_decisions=len(list(OUT.glob('*/decision.json')))))
        proc=launch(['admet_v12_runner.py','evaluate','--tasks',task],'logs/v12_evaluate.log')
        if proc.wait()!=0:raise RuntimeError('Evaluation failed: '+task)
    save(OUT/'coordinator_status.json',dict(status='EVALUATED',tasks=len(tasks)))
    emit(stage='v12_training_complete',tasks=len(tasks))


if __name__=='__main__':main()
