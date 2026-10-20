"""Disjoint task workers followed by verified publication, without upload."""
import json,os,subprocess,sys,time
from pathlib import Path
from admet_v6_runner import remote_only,save,sha
from admet_v16_runner import OUT,names


def launch(args,log):
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    return subprocess.Popen([sys.executable,'-u',*args],stdin=subprocess.DEVNULL,stdout=open(log,'a'),stderr=subprocess.STDOUT,env=env,start_new_session=True)


def run_workers(stage,script,partitions):
    workers=[launch([script,stage,'--tasks',*ts],f'logs/v16_{stage}_{i}.log') for i,ts in enumerate(partitions)]
    while any(p.poll() is None for p in workers):
        bad=[p.returncode for p in workers if p.poll() not in [None,0]]
        if bad:
            for worker in workers:
                if worker.poll() is None:worker.terminate()
            raise RuntimeError((stage,bad))
        save(OUT/'status.json',dict(status=stage.upper(),decisions=len(list(OUT.glob('*/decision.json'))),
            worker_pids=[p.pid for p in workers],worker_exit_codes=[p.poll() for p in workers]))
        time.sleep(10)
    assert all(p.returncode==0 for p in workers)


def main():
    remote_only();tasks=names();assert len(list(OUT.glob('*/shortlist.json')))==22
    for f,h in json.loads(Path('admet_v16_source_freeze.json').read_text()).items():assert sha(f)==h,f
    parts=[tasks[i::3] for i in range(3)]
    run_workers('train','admet_v16_runner.py',parts)
    run_workers('evaluate','admet_v16_runner.py',parts)
    run_workers('publish','admet_v16_release.py',parts)
    for stage in ['export','verify','report']:
        save(OUT/'status.json',dict(status=stage.upper()))
        p=launch(['admet_v16_release.py',stage],f'logs/v16_{stage}.log')
        assert p.wait()==0,stage
    save(OUT/'status.json',dict(status='COMPLETE_VERIFIED',platform_score=None,automatic_submission=False))


if __name__=='__main__':
    try:main()
    except Exception as e:
        save(OUT/'status.json',dict(status='FAILED',error=repr(e)));raise
