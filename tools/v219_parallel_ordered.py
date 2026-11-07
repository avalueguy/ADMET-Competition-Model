"""Schedule disjoint unfinished v216 development models, without retuning."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v216_ordered_scaffold as model


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def work(tasks):
    d = model.v.load()
    for task in tasks:
        k = d['tasks'].index(task)
        assert task != 'hia_hou'
        status = model.OUT / ('precompute_' + task + '.json')
        save(status, dict(status='TRAINING', task=task, pid=os.getpid(), started_at_utc=now(),
                          signature=model.sig(), confirmation_observed=False))
        for config in model.CONFIGS:
            for part in ('fold0', 'fold1'):
                model.predict(d, k, part, config)
        save(status, dict(status='COMPLETE', task=task, pid=os.getpid(), finished_at_utc=now(),
                          signature=model.sig(), confirmation_observed=False,
                          configuration_changed=False, selection_performed=False))


def schedule():
    start = json.loads((ROOT / 'docs/v216_remote_start.json').read_text())
    expected = start['processes']['training']
    process = Path('/proc') / str(expected['pid'])
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == start['boot_id']
    assert (process / 'stat').read_text().split()[21] == expected['start_ticks']
    assert expected['script'] in (process / 'cmdline').read_bytes().decode()
    assert not (model.OUT / 'failure.json').exists()
    assert not (model.OUT / 'frozen_selection.json').exists(), 'Development already ended'
    plan = json.loads((ROOT / 'docs/v216_training_plan.json').read_text())
    assert model.b.sha(model.__file__) == plan['source_sha256']
    signature = model.sig()
    status = model.OUT / 'parallel_scheduling.json'
    assert not status.exists(), 'Inspect existing scheduler; do not launch twice'
    cpu = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    assert cpu[0] == 'max' or int(cpu[0]) / int(cpu[1]) >= 12
    assert int(Path('/sys/fs/cgroup/memory.max').read_text()) >= 48 * 1024 ** 3
    os.kill(expected['pid'], signal.SIGSTOP)
    for _ in range(100):
        if (process / 'stat').read_text().split()[2] in ('T', 't'):
            break
        time.sleep(.05)
    else:
        os.kill(expected['pid'], signal.SIGCONT)
        raise RuntimeError('Main process did not stop for exclusive scheduling')
    children = []
    tasks = []
    success = False
    try:
        protocol = json.loads((model.OUT / 'protocol.json').read_text())
        # Every task with ANY existing artifact belongs exclusively to the main
        # process, including a currently incomplete or partially saved seed.
        tasks = [t for t in protocol['tasks'] if not (model.OUT / 'models' / t).exists()]
        assert tasks and len(tasks) == len(set(tasks))
        buckets = [tasks[i::3] for i in range(3)]
        for i, bucket in enumerate(buckets):
            if not bucket:
                continue
            log = ROOT / ('v219_worker_' + str(i) + '.log')
            with log.open('ab') as stream:
                child = subprocess.Popen([sys.executable, str(Path(__file__)), '--tasks', *bucket],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
            children.append((child, bucket, log.name))
        save(status, dict(status='PARALLEL_DEVELOPMENT', started_at_utc=now(),
            coordinator_pid=os.getpid(), main_pid=expected['pid'], main_temporarily_stopped=True,
            signature=signature, source_sha256=plan['source_sha256'], tasks=tasks,
            workers=[dict(pid=c.pid, tasks=bucket, log=log) for c, bucket, log in children],
            concurrent_model_fits=len(children), threads_per_model=4,
            max_cpu_threads=4 * len(children), configuration_or_validation_changed=False,
            confirmation_observed=False))
        for child, _, _ in children:
            child.wait()
        exits = {str(c.pid): c.returncode for c, _, _ in children}
        assert not any(exits.values()), exits
        assert model.sig() == signature
        for task in tasks:
            marker = json.loads((model.OUT / ('precompute_' + task + '.json')).read_text())
            assert marker['status'] == 'COMPLETE' and marker['signature'] == signature
        success = True
    except Exception:
        save(model.OUT / 'parallel_scheduling_failure.json', dict(status='FAILED', at_utc=now(),
             traceback=traceback.format_exc(), main_pid=expected['pid'],
             main_stopped_to_prevent_uncontrolled_retry=True))
        raise
    finally:
        # Drain all task owners before the original selector can read their cache.
        for child, _, _ in children:
            child.wait()
        if success:
            os.kill(expected['pid'], signal.SIGCONT)
            old = json.loads(status.read_text())
            old.update(status='PRECOMPUTE_COMPLETE_ORIGINAL_RESUMED', finished_at_utc=now(),
                       main_temporarily_stopped=False,
                       worker_exit_codes={str(c.pid): c.returncode for c, _, _ in children})
            save(status, old)
            print('ORIGINAL_FROZEN_SELECTOR_RESUMED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', nargs='+')
    args = parser.parse_args()
    work(args.tasks) if args.tasks else schedule()
