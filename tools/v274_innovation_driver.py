"""Disjoint development precomputation, followed by the unchanged frozen selector."""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v271_clearance_innovations as model


def work(tasks):
    d = model.load()
    for task in tasks:
        k = d['tasks'].index(task)
        assert task != 'hia_hou'
        for c in model.CONFIGS:
            for part in ('fold0', 'fold1'):
                model.predict(d, k, part, c)
    print(json.dumps(dict(status='PRECOMPUTE_COMPLETE', tasks=tasks)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+')
    args = parser.parse_args()
    if args.tasks:
        work(args.tasks)
        return
    status = model.OUT / 'parallel_scheduling.json'
    assert not status.exists(), 'Existing driver must be inspected, never retried blindly'
    assert not (model.OUT / 'models').exists()
    frozen = json.loads((ROOT / 'docs/v271_frozen_sources.json').read_text())
    for row in frozen['files']:
        assert model.b.sha(ROOT / row['path']) == row['sha256']
    assert json.loads((model.OUT / 'smoke.json').read_text())['signature'] == model.sig()
    quota = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    assert quota[0] == 'max' or int(quota[0]) / int(quota[1]) >= 4
    tasks = list(model.TASKS)
    workers, buckets = [], [tasks]
    for i, bucket in enumerate(buckets):
        with (ROOT / ('v274_worker_' + str(i) + '.log')).open('ab') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__)), '--tasks', *bucket],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        workers.append(child)
    report = dict(status='PARALLEL_DEVELOPMENT', driver_pid=os.getpid(),
        source_signature=model.sig(), workers=[dict(pid=p.pid, tasks=t,
        start_ticks=Path('/proc', str(p.pid), 'stat').read_text().split()[21])
        for p, t in zip(workers, buckets)], threads_per_worker=4,
        max_cpu_threads=4, selection_performed=False, confirmation_observed=False)
    model.b.save(status, report)
    for child in workers:
        child.wait()
    assert all(child.returncode == 0 for child in workers), [p.returncode for p in workers]
    assert not (model.OUT / 'failure.json').exists()
    report.update(status='PRECOMPUTE_COMPLETE_SELECTOR_RUNNING', worker_exit_codes=[p.returncode for p in workers])
    model.b.save(status, report)
    child = subprocess.Popen([sys.executable, str(ROOT / 'tools/v271_clearance_innovations.py')],
                             cwd=ROOT, stdin=subprocess.DEVNULL)
    report['selector_pid'] = child.pid
    report['selector_start_ticks'] = Path('/proc', str(child.pid), 'stat').read_text().split()[21]
    model.b.save(status, report)
    child.wait()
    assert child.returncode == 0
    report.update(status='COMPLETE', selector_exit_code=child.returncode)
    model.b.save(status, report)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        model.b.save(model.OUT / 'failure.json', dict(status='FAILED',
                     traceback=traceback.format_exc(), retry_permitted=False))
        raise
