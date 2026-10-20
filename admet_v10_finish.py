"""Wait for the existing coordinator process, then replay and package its release."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from admet_v6_runner import remote_only, save, emit
from admet_v10_runner import OUT, protocol


def main():
    remote_only(); launch=json.loads((OUT/'launch.json').read_text()); pid=launch['pid']
    while Path(f'/proc/{pid}').exists():
        stat=Path(f'/proc/{pid}/stat').read_text().split()
        if stat[2]=='Z': break
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        if b'admet_v10_coordinate.py' not in cmd: raise RuntimeError('Coordinator PID reused')
        time.sleep(20)
    p=protocol()
    missing=[t for t in p['tasks'] if not (OUT/t/'decision.json').exists()]
    if missing:
        save(OUT/'finish_status.json',dict(status='FAILED',reason='coordinator exited with missing decisions',missing=missing))
        raise RuntimeError(missing)
    for stage in ['publish','verify','report']:
        save(OUT/'finish_status.json',dict(status='RUNNING',stage=stage))
        subprocess.run([sys.executable,'-u','admet_v10_release.py',stage],check=True)
    save(OUT/'coordinator_status.json',dict(status='COMPLETE_VERIFIED',automatic_platform_submission=False))
    save(OUT/'finish_status.json',dict(status='COMPLETE_VERIFIED',automatic_platform_submission=False))


if __name__=='__main__': main()
