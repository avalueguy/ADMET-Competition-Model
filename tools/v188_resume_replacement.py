"""Resume only the unfinished frozen v181 route on an empty replacement host."""
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')


def spawn(script, log, args=()):
    with (ROOT / log).open('ab') as f:
        return subprocess.Popen([sys.executable, str(ROOT / 'tools' / script), *args],
                                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=f,
                                stderr=subprocess.STDOUT, start_new_session=True)


def main():
    marker = ROOT / 'v188_resume_status.json'
    assert not marker.exists(), 'Inspect existing recovery process; never launch twice'
    save(marker, dict(status='VERIFYING_RESTORED_BASELINE', pid=os.getpid()))
    import v181_quantile_forest as m
    assert m.b.sha(m.__file__) == 'b7d1828a0e5802f61edea3e6e23eeb4c8f8cf2bbf3bfc105383a5eba59e4218d'
    assert not m.OUT.exists(), 'Unexpected existing experiment output; inspect first'
    assert m.v.sig() == '5a79bb5fd6cd135690d750fa5beae79f7a5674958ef83b87689f190be5067c91'
    d = m.v.load()
    tasks = [t for t in m.v.tasks(d) if d['metrics'][d['tasks'].index(t)] == 'MAE']
    checks = []
    for task in tasks:
        k = d['tasks'].index(task)
        for part in m.v.PARTS:
            cache = m.v.OUT / 'controls' / task / (part + '.npz')
            assert cache.is_file(), 'Do not regenerate historical controls'
            fit, q = m.v.split(d, k, part)
            p = m.v.baseline(d, k, part)
            assert len(p) == len(q)
            checks.append(dict(task=task, part=part, rows=len(q), sha256=m.b.sha(cache)))
    m.smoke(d)
    env = {n: importlib.metadata.version(n) for n in
           ('numpy', 'scipy', 'scikit-learn', 'pandas', 'rdkit', 'joblib', 'catboost', 'xgboost', 'torch')}
    save(ROOT / 'docs/v188_remote_baseline_verification.json',
         dict(status='PASS', baseline='v145', official_best=51.166,
              signature=m.v.sig(), cached_controls=checks, environment=env,
              smoke='PASS', frozen_gates_unchanged=True,
              previous_unfinished_results_available=False, rejected_routes_restarted=False))
    del d
    # Precompute disjoint task outputs before starting the serial selector.
    # This changes scheduling only; each worker uses the original frozen model.
    workers = {t: spawn('v185_schedule_quantile_tasks.py', 'v185_' + t + '.log', ('--task', t))
               for t in tasks if t != 'solubility_aqsoldb'}
    scheduling = m.OUT / 'scheduling.json'
    save(scheduling, dict(status='PRECOMPUTING_DISJOINT_TASKS', recovery_version='v188',
                          workers={t: p.pid for t, p in workers.items()},
                          data_or_model_choices_changed=False, frozen_signature=m.sig()))
    while any(p.poll() is None for p in workers.values()):
        time.sleep(10)
    exits = {t: p.returncode for t, p in workers.items()}
    assert not any(exits.values()), exits
    save(scheduling, dict(status='PRECOMPUTE_COMPLETE', recovery_version='v188',
                          worker_exit_codes=exits, frozen_signature=m.sig(),
                          data_or_model_choices_changed=False))
    training = spawn('v181_quantile_forest.py', 'v181_quantile_forest.log')
    save(ROOT / 'v181_training_pid.json', dict(pid=training.pid, recovery='v188'))
    pipeline = spawn('v183_package_quantile_batch.py', 'v183_pipeline.log')
    save(marker, dict(status='TRAINING_AND_PIPELINE_STARTED', training_pid=training.pid,
                      pipeline_pid=pipeline.pid, precompute_exit_codes=exits))
    print(json.dumps(dict(status='TRAINING_AND_PIPELINE_STARTED', training_pid=training.pid,
                          pipeline_pid=pipeline.pid)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        save(ROOT / 'v188_resume_failure.json', dict(status='FAILED', traceback=traceback.format_exc()))
        raise
