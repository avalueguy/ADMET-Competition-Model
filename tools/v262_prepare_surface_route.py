"""One-shot label-free features, smoke and preflight; launch only after all pass."""
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v256_exposed_polarity as model


def main():
    record = ROOT / 'docs/v256_preparation.json'
    assert not record.exists(), 'Inspect existing preparation, never duplicate'
    assert not (ROOT / 'docs/v256_remote_start.json').exists()
    assert not (model.OUT / 'models').exists()
    assert not (model.OUT / 'feature_build_failure.json').exists()
    frozen = json.loads((ROOT / 'docs/v256_frozen_sources.json').read_text())
    for row in frozen['files']:
        assert model.b.sha(ROOT / row['path']) == row['sha256']
    report = dict(status='PREPARING_FEATURES', at_utc=datetime.now(timezone.utc).isoformat(),
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  pid=os.getpid(), start_ticks=Path('/proc', str(os.getpid()), 'stat').read_text().split()[21],
                  signature=model.sig(), source_sha256=model.b.sha(model.__file__),
                  model_training_started=False, credentials_stored=False, children=[])
    model.b.save(record, report)
    for stage, script, arguments, logname in (
            ('PREPARING_FEATURES', 'tools/v258_build_surface.py', [], 'v256_feature_build.log'),
            ('RUNNING_SMOKE', 'tools/v256_exposed_polarity.py', ['--smoke'], 'v256_smoke.log'),
            ('PREFLIGHT_AND_START', 'tools/v261_start_surface.py', [], 'v261_start.log')):
        report['status'] = stage
        with (ROOT / logname).open('xb') as log:
            child = subprocess.Popen([sys.executable, str(ROOT / script), *arguments],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
        row = dict(stage=stage, pid=child.pid, script=script,
                   start_ticks=Path('/proc', str(child.pid), 'stat').read_text().split()[21], log=logname)
        report['children'].append(row)
        model.b.save(record, report)
        child.wait()
        row['exit_code'] = child.returncode
        model.b.save(record, report)
        assert child.returncode == 0, row
        assert not (model.OUT / 'feature_build_failure.json').exists()
        assert not (model.OUT / 'failure.json').exists()
    report.update(status='TRAINING_STARTED', model_training_started=True,
                  training_start_record='docs/v256_remote_start.json')
    model.b.save(record, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        model.b.save(model.OUT / 'preparation_failure.json', dict(status='FAILED',
                     traceback=traceback.format_exc(), retry_permitted=False))
        raise
