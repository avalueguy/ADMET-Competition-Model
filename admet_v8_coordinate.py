"""Supervise two remote training workers and one gated evaluator, then verify release."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from admet_v6_runner import remote_only, save, emit, sha
from admet_v8_runner import protocol, OUT, expected


def launch(arguments, log):
    handle = open(log, 'a', buffering=1)
    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    child = subprocess.Popen([sys.executable, '-u', *arguments], stdout=handle, stderr=subprocess.STDOUT, env=env)
    handle.close()
    return child


def main():
    remote_only(); p = protocol()
    tasks = sorted(x.name for x in Path('admet_competition_data/data_benchmark').iterdir() if x.is_dir())
    OUT.mkdir(exist_ok=True)
    save(OUT/'source_manifest.json', [dict(path=str(f), sha256=sha(f)) for f in sorted(Path('.').glob('admet_v8*.py'))])
    workers = {'multiscale_kernel': launch(['admet_v8_runner.py', 'train', '--family', 'multiscale_kernel'], 'logs/v8_kernel.log')}
    for shard in range(3):
        tasks_in_shard = p['mtl_tasks'][shard::3]
        workers['gated_transfer_'+str(shard)] = launch(
            ['admet_v8_runner.py', 'train', '--family', 'gated_transfer', '--tasks', *tasks_in_shard],
            f'logs/v8_mtl_{shard}.log')
    gate = None; gate_task = None
    while True:
        failed = [name for name, proc in workers.items() if proc.poll() not in [None, 0]]
        if failed:
            save(OUT/'coordinator_status.json', dict(status='FAILED', workers=failed))
            raise RuntimeError(f'Training worker failed: {failed}; inspect logs before resume')
        if gate is not None and gate.poll() is not None:
            if gate.returncode:
                save(OUT/'coordinator_status.json', dict(status='FAILED', stage='evaluate', task=gate_task))
                raise RuntimeError(f'Evaluation failed: {gate_task}')
            gate = None; gate_task = None
        pending = [task for task in tasks if not (OUT/task/'decision.json').exists()]
        if gate is None:
            for task in pending:
                if all((OUT/task/(name+'_cv.json')).exists() for name in expected(task, p)):
                    gate_task = task
                    gate = launch(['admet_v8_runner.py', 'evaluate', '--tasks', task], 'logs/v8_evaluate.log')
                    break
        done = {name: proc.poll() for name, proc in workers.items()}
        state = dict(status='RUNNING', worker_exit_codes=done, active_evaluation=gate_task,
                     pending_tasks=pending, completed_candidates={name:len(list(OUT.glob('*/'+name+'_cv.json'))) for name in ['multiscale_kernel','gated_transfer']})
        save(OUT/'coordinator_status.json', state)
        if not pending and gate is None:
            break
        if pending and gate is None and all(code == 0 for code in done.values()):
            raise RuntimeError(f'Workers finished but incomplete pools remain: {pending}')
        time.sleep(20)
    for stage in ['publish', 'verify', 'report']:
        emit(stage='v8_release_stage', operation=stage)
        subprocess.run([sys.executable, '-u', 'admet_v8_release.py', stage], check=True)
    save(OUT/'coordinator_status.json', dict(status='COMPLETE_VERIFIED', automatic_platform_submission=False))


if __name__ == '__main__':
    main()
