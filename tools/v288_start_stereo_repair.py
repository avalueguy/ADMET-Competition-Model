"""One-shot preflight and explicit process identity record for frozen v284."""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import v284_stereo_context_repair as model


def main():
    path = ROOT / 'docs/v284_remote_start.json'
    assert not path.exists(), 'Inspect existing start record, never launch twice'
    assert not (model.OUT / 'models').exists()
    assert not (model.OUT / 'failure.json').exists()
    frozen = json.loads((ROOT / 'docs/v284_frozen_sources.json').read_text())
    for row in frozen['files']:
        assert model.b.sha(ROOT / row['path']) == row['sha256']
    smoke = json.loads((model.OUT / 'smoke.json').read_text())
    assert smoke['status'] == 'PASS' and smoke['signature'] == model.sig()
    assert model.b.sha(ROOT / 'tools/v148_experiment_runner.py') == '8f6d93f517d1a4acf6877f3effa63e3b2c171141b5a637b8ad51573498c0b871'
    assert model.v.sig() == '5a79bb5fd6cd135690d750fa5beae79f7a5674958ef83b87689f190be5067c91'
    d = model.load()
    originals = json.loads((ROOT / 'docs/v284_original_models.json').read_text())
    for item in originals['files']:
        assert model.b.sha(ROOT / item['path']) == item['sha256']
    for task in model.TASKS:
        k = d['tasks'].index(task)
        for part in model.v.PARTS:
            model.design(d, k, part)
            model.original_replay(d, k, part)
    count = 0
    for task in model.v.tasks(d):
        k = d['tasks'].index(task)
        for part in model.v.PARTS:
            model.v.baseline(d, k, part)
            count += 1
    assert count == 105
    prefix = 'v176_shared_graph_batch_20261027'
    info = json.loads((ROOT / 'releases' / (prefix + '.archive.json')).read_text())
    assert model.b.sha(ROOT / 'releases' / (prefix + '.tar.gz')) == info['sha256']
    # A second launch, including through another transport, is never implicit.
    scripts = ('v287_stereo_driver.py', 'v286_package_stereo_repair.py',
               'v284_stereo_context_repair.py')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError):
            continue
        assert not any(Path(x.decode(errors='replace')).name in scripts for x in args if x), proc.name
    report = dict(status='PREFLIGHT_PASS_STARTING', at_utc=datetime.now(timezone.utc).isoformat(),
                  host='connect.nmb2.seetacloud.com', port=24365,
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                  source_sha256=model.b.sha(model.__file__), signature=model.sig(),
                  baseline_cache_verified=count, smoke='PASS', frozen_sources_verified=True,
                  dependency_sha256_verified=True, processes={}, credentials_stored=False)
    report['original_v145_component_and_all_partitions_replay'] = 'PASS'
    model.b.save(path, report)
    for role, script, log in (
        ('driver', 'tools/v287_stereo_driver.py', 'v284_stereo_context_repair.log'),
        ('pipeline', 'tools/v286_package_stereo_repair.py', 'v286_pipeline.log')):
        with (ROOT / log).open('xb') as stream:
            child = subprocess.Popen([sys.executable, str(ROOT / script)], cwd=ROOT,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True)
        report['processes'][role] = dict(pid=child.pid, script=script,
            start_ticks=Path('/proc', str(child.pid), 'stat').read_text().split()[21], log=log)
        model.b.save(path, report)
    report['status'] = 'STARTED'
    model.b.save(path, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
