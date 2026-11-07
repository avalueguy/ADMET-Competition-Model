"""Remote terminal receipt, after the frozen runner and verifier have exited."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT = Path('/root/autodl-tmp/regression9_v387')


def main():
    status = json.loads((ROOT / 'run_status.json').read_text())
    verified = json.loads((ROOT / 'outputs/independent_verification.json').read_text())
    assert status['stage'] == 'EVIDENCE_READY' and verified['status'] == 'PASS'
    assert status['signature'] == verified['signature']
    freeze = json.loads((ROOT / 'source_freeze.json').read_text())
    for row in freeze['files']:
        assert hashlib.sha256((ROOT / row['name']).read_bytes()).hexdigest() == row['sha256']
    matching = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            state = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if state != 'Z' and (int(proc.name) == status['pid'] or
                            any(name in command for name in ('v387_run_experts.py', 'v387_verify_experts.py',
                                                             'import v387_oof_pareto'))):
            matching.append(dict(pid=int(proc.name), command=command))
    assert not matching, matching
    compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                       '--format=csv,noheader'], text=True).strip()
    assert not compute, compute
    value = dict(status='PASS', at_unix=time.time(), signature=status['signature'],
                 no_training_processes_running=True, experiment_pid=status['pid'],
                 scope='v387 runner, workers, verifier exited; no GPU compute process at receipt time.',
                 gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu',
                                              '--format=csv,noheader'], text=True).strip(),
                 gpu_compute_processes=compute, source_hashes_verified=True)
    (ROOT / 'completion_receipt.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()

