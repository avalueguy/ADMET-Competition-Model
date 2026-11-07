"""Disjoint development-task scheduling, without changing frozen v181 models."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v181_quantile_forest as model


def work(task):
    d = model.v.load()
    k = d['tasks'].index(task)
    assert d['metrics'][k] == 'MAE' and task != 'solubility_aqsoldb'
    for config in model.CONFIGS:
        for part in ('fold0', 'fold1'):
            model.predict(d, k, part, config)
    model.b.save(model.OUT / ('precompute_' + task + '.json'),
                 dict(status='COMPLETE', signature=model.sig(), task=task,
                      selection_changed=False, confirmation_observed=False))


def schedule():
    pid = json.loads((ROOT / 'v181_training_pid.json').read_text())['pid']
    tasks = ('caco2_wang', 'ppbr_az', 'ld50_zhu', 'lipophilicity_astrazeneca')
    status = model.OUT / 'scheduling.json'
    assert not status.exists()
    cmd = Path('/proc') / str(pid) / 'cmdline'
    assert b'v181_quantile_forest.py' in cmd.read_bytes()
    development = model.OUT / 'development.json'
    # The main process must still be within its first solubility task.
    assert not development.exists()
    os.kill(pid, signal.SIGSTOP)
    workers = []
    try:
        for task in tasks:
            log = open(ROOT / ('v185_' + task + '.log'), 'ab')
            child = subprocess.Popen([sys.executable, str(Path(__file__)), '--task', task],
                                     cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT)
            workers.append((task, child, log))
        model.b.save(status, dict(status='PRECOMPUTING_DISJOINT_TASKS', main_pid=pid,
                                 workers={t: p.pid for t, p, _ in workers}, frozen_signature=model.sig(),
                                 data_or_model_choices_changed=False))
        started = time.monotonic()
        while any(p.poll() is None for _, p, _ in workers):
            if time.monotonic() - started > 7200:
                raise TimeoutError('Task scheduling exceeded2h; main process will resume')
            time.sleep(10)
        result = {t: p.returncode for t, p, _ in workers}
        model.b.save(status, dict(status='PRECOMPUTE_COMPLETE' if not any(result.values()) else 'PRECOMPUTE_FAILED',
                                 worker_exit_codes=result, main_pid=pid, frozen_signature=model.sig(),
                                 data_or_model_choices_changed=False))
    finally:
        # Never strand the already authorized original training in a stopped state.
        for _, p, log in workers:
            if p.poll() is None:
                p.wait()
            log.close()
        os.kill(pid, signal.SIGCONT)
    print('ORIGINAL_TRAINING_RESUMED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task')
    args = parser.parse_args()
    work(args.task) if args.task else schedule()
