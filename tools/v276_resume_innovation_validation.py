"""Resume only validation of frozen, intact v271 checkpoints after container loss.

No model fitting is permitted in this recovery preflight. Original selector and
packager run only after all four development checkpoints independently replay.
"""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v271_clearance_innovations as model


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    threadpool_limits(limits=4)
    print(json.dumps(dict(stage='thread_limits', pools=threadpool_info())), flush=True)
    record = ROOT / 'docs/v276_recovery.json'
    assert not record.exists(), 'Inspect prior recovery; never launch twice'
    assert not (model.OUT / 'failure.json').exists()
    assert not (model.OUT / 'confirmation.json').exists()
    scripts = {'v271_clearance_innovations.py', 'v273_package_innovations.py',
               'v274_innovation_driver.py', 'v275_start_innovations.py'}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError):
            continue
        assert not any(Path(x.decode(errors='replace')).name in scripts
                       for x in args if x), proc.name
    old = json.loads((ROOT / 'docs/v271_remote_start.json').read_text())
    for item in old['processes'].values():
        assert not Path('/proc', str(item['pid'])).exists()
    scheduler = json.loads((model.OUT / 'parallel_scheduling.json').read_text())
    assert scheduler['worker_exit_codes'] == [0]
    assert not Path('/proc', str(scheduler['selector_pid'])).exists()
    frozen = json.loads((ROOT / 'docs/v271_frozen_sources.json').read_text())
    for row in frozen['files']:
        assert model.b.sha(ROOT / row['path']) == row['sha256']
    assert model.sig() == old['signature']
    smoke = json.loads((model.OUT / 'smoke.json').read_text())
    assert smoke['status'] == 'PASS' and smoke['signature'] == model.sig()
    state = dict(status='VALIDATING_SAVED_CHECKPOINTS', at_utc=now(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        container_pid1_start_ticks=Path('/proc/1/stat').read_text().split()[21],
        prior_selector_pid=scheduler['selector_pid'],
        prior_selector_start_ticks=scheduler['selector_start_ticks'],
        source_sha256=model.b.sha(model.__file__), signature=model.sig(),
        model_training_repeated=False, gates_changed=False, credentials_stored=False)
    model.b.save(record, state)
    d = model.load()
    count = 0
    for task in model.v.tasks(d):
        k = d['tasks'].index(task)
        for part in model.v.PARTS:
            path = model.v.OUT / 'controls' / task / (part + '.npz')
            assert path.exists()
            model.v.baseline(d, k, part)
            count += 1
    assert count == 105
    print('BASELINE_105_CACHE_PASS', flush=True)
    dependency = ROOT / 'releases/v176_shared_graph_batch_20261027.tar.gz'
    depinfo = json.loads(dependency.with_suffix('').with_suffix('.archive.json').read_text())
    assert model.b.sha(dependency) == depinfo['sha256']
    print('DEPENDENCY_HASH_PASS', flush=True)
    checks = []
    for task in model.TASKS:
        k = d['tasks'].index(task)
        for part in ('fold0', 'fold1'):
            folder = model.OUT / 'models' / task / part / model.key(model.CONFIGS[0])
            expected = {'transform.joblib', 'model.joblib', 'predictions.npz', 'metadata.json'}
            assert {p.name for p in folder.iterdir()} == expected
            cached = np.load(folder / 'predictions.npz')
            assert str(cached['signature']) == model.sig()
            fit, query = model.v.split(d, k, part)
            np.testing.assert_array_equal(cached['fit'], fit)
            np.testing.assert_array_equal(cached['query'], query)
            replay, hashes = model.replay(d, k, part, model.CONFIGS[0])
            np.testing.assert_allclose(replay, cached['p'], atol=1e-12, rtol=0)
            checks.append(dict(task=task, part=part, independent_replay='PASS',
                               hashes=hashes, predictions_sha256=model.b.sha(folder / 'predictions.npz')))
            print(json.dumps(dict(stage='checkpoint_replay_pass', task=task, part=part)), flush=True)
    state.update(status='SAVED_CHECKPOINTS_PASS_RESUMING_ORIGINAL_VALIDATION',
                 baseline_cache_verified=count, dependency_verified=True,
                 checks=checks, processes={}, at_utc=now())
    model.b.save(record, state)
    children = []
    for role, script in [('selector', 'v271_clearance_innovations.py'),
                         ('pipeline', 'v273_package_innovations.py')]:
        logpath = ROOT / ('v276_recovery_' + role + '.log')
        with logpath.open('xb') as log:
            child = subprocess.Popen([sys.executable, str(ROOT / 'tools' / script)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        state['processes'][role] = dict(pid=child.pid, script=script,
            start_ticks=Path('/proc', str(child.pid), 'stat').read_text().split()[21],
            log=logpath.name)
        model.b.save(record, state)
    scheduler.update(status='INTERRUPTED_SELECTOR_RECOVERED_CHECKPOINTS_VERIFIED',
                     recovery_evidence='docs/v276_recovery.json',
                     recovered_selector=state['processes']['selector'])
    model.b.save(model.OUT / 'parallel_scheduling.json', scheduler)
    for child in children:
        child.wait()
    state.update(status='COMPLETE' if all(p.returncode == 0 for p in children) else 'FAILED',
                 exit_codes=[p.returncode for p in children], completed_at_utc=now())
    model.b.save(record, state)
    assert state['status'] == 'COMPLETE', state['exit_codes']
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        model.b.save(ROOT / 'docs/v276_recovery_failure.json',
                     dict(status='FAILED', traceback=traceback.format_exc(), retry_permitted=False))
        raise
