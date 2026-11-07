"""Run unchanged frozen curve jobs concurrently, then resume the original driver.

The driver is suspended only for scheduling and its in-progress curve is excluded.
No scoring, early termination, configuration changes, or concurrent cache writes.
"""
from pathlib import Path
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/regression9_v378')


def run_one(job):
    part, task, name = job
    log = ROOT/'prefit_logs'/f'{part}_{task}_{name}.log'
    script = "import v378_retrieval_transformer as m; m.setup(); d=m.load(); m.train_curve(d,%r,%r,next(b for b in m.BASES if b['name']==%r))" % (task, part, name)
    start = time.monotonic()
    with log.open('w') as out:
        code = subprocess.call([sys.executable, '-u', '-c', script], cwd=ROOT,
                               stdout=out, stderr=subprocess.STDOUT)
    return dict(job=job, exit_code=code, seconds=time.monotonic()-start, log=str(log))


def main():
    assert not (ROOT/'parallel_prefit_launch.json').exists(), 'Already scheduled'
    import v378_retrieval_transformer as m
    current = json.loads((ROOT/'run_status.json').read_text())
    assert current['stage'] == 'DEVELOPMENT'
    pid = current['pid']
    # Freeze the main driver, inspect progress only afterwards, then reserve the
    # current tuple. Signal does not cancel or shorten its training horizon.
    os.kill(pid, signal.SIGSTOP)
    try:
        events = []
        for line in (ROOT/'training.log').read_text().splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if value.get('stage') == 'neural_horizon_done':
                events.append(value)
        assert events, 'No checkpoint tuple available'
        last = events[-1]
        ordered = [(part,t,b['name']) for part in ('fold0','fold1') for t in m.TASKS for b in m.BASES]
        active = (last['part'], last['task'], last['base'])
        # If that state is already saved the driver may be in setup for the next
        # tuple. Reserve both the last event tuple and its immediate successor.
        index = ordered.index(active)
        reserved = set(ordered[max(0,index):index+2])
        jobs = [j for j in ordered if j not in reserved and
                not (ROOT/'outputs/curves'/j[1]/j[0]/j[2]/'state.joblib').exists()]
        (ROOT/'prefit_logs').mkdir(exist_ok=True)
        receipt = dict(status='RUNNING', pid=os.getpid(), original_driver_pid=pid,
                       signature=m.signature(), workers=6, jobs=jobs, reserved=list(reserved),
                       unchanged_model_source=True, no_scoring=True, no_shortened_training=True)
        (ROOT/'parallel_prefit_launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({k:v for k,v in receipt.items() if k!='jobs'}),flush=True)
        results=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(run_one,j) for j in jobs]
            for future in concurrent.futures.as_completed(futures):
                result=future.result();results.append(result)
                (ROOT/'parallel_prefit_results.json').write_text(json.dumps(dict(results=results,total=len(jobs)),indent=2)+'\n')
                print(json.dumps(dict(stage='prefit_done',completed=len(results),total=len(jobs),**result)),flush=True)
        assert all(r['exit_code']==0 for r in results), results
        receipt.update(status='COMPLETE',results=results)
        (ROOT/'parallel_prefit_complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
    finally:
        os.kill(pid,signal.SIGCONT)
        print('ORIGINAL_DRIVER_RESUMED',flush=True)


if __name__=='__main__':main()
