"""Small SSH helper; copies only explicitly selected experiment artifacts."""
import argparse
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOST = 'czp123@100.109.143.42'
REMOTE = '/home/czp123/admet_v23_20260928'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['script', 'put', 'get', 'launch', 'status'])
    p.add_argument('files', nargs='*')
    a = p.parse_args()
    ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', HOST]
    if a.action == 'script':
        command = 'cd ' + shlex.quote(REMOTE) + ' && ' + shlex.join([REMOTE+'/.venv/bin/python', '-'])
        result = subprocess.run(ssh + [command],
                                input=(ROOT/a.files[0]).read_bytes())
        raise SystemExit(result.returncode)
    if a.action == 'put':
        for name in a.files:
            subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', str(ROOT/name),
                            HOST+':'+REMOTE+'/'+name], check=True)
    if a.action == 'get':
        for name in a.files:
            path = ROOT/name
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['scp', '-q', '-o', 'BatchMode=yes', HOST+':'+REMOTE+'/'+name,
                            str(path)], check=True)
    if a.action == 'launch':
        subprocess.run(ssh + [shlex.join([REMOTE+'/.venv/bin/python', REMOTE+'/admet_v46_launch.py'])], check=True)
    if a.action == 'status':
        subprocess.run(ssh + [shlex.join(['tail', '-n', '14', REMOTE+'/admet_model_outputs_v46/run.log'])], check=True)


if __name__ == '__main__':
    main()
