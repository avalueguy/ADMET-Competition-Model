"""Wait for the existing v12 coordinator, then publish/verify/report."""
import json, os, subprocess, sys, time
from pathlib import Path
from admet_v6_runner import remote_only, save
from admet_v12_runner import protocol, OUT


def main():
    remote_only();launch=json.loads((OUT/'launch.json').read_text());pid=launch['pid']
    while Path(f'/proc/{pid}').exists():
        stat=Path(f'/proc/{pid}/stat').read_text().split()
        if stat[2]=='Z':break
        if b'admet_v12_coordinate.py' not in Path(f'/proc/{pid}/cmdline').read_bytes():raise RuntimeError('coordinator PID reused')
        time.sleep(20)
    missing=[t for t in protocol()['tasks'] if not (OUT/t/'decision.json').exists()]
    if missing:
        save(OUT/'finish_status.json',dict(status='FAILED',missing=missing));raise RuntimeError(missing)
    for stage in ['publish','verify','report']:
        save(OUT/'finish_status.json',dict(status='RUNNING',stage=stage))
        subprocess.run([sys.executable,'-u','admet_v12_release.py',stage],check=True)
    save(OUT/'coordinator_status.json',dict(status='COMPLETE_VERIFIED',automatic_platform_submission=False))
    save(OUT/'finish_status.json',dict(status='COMPLETE_VERIFIED',automatic_platform_submission=False))


if __name__=='__main__':main()
