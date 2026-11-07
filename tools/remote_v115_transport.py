"""Small password-prompted SSH transport; credentials stay in process memory."""
import argparse
import getpass
import json
import sys
from pathlib import Path
import paramiko

ap = argparse.ArgumentParser()
ap.add_argument('op', choices=['exec', 'put', 'get', 'session'])
ap.add_argument('paths', nargs='*')
args = ap.parse_args()
cli = paramiko.SSHClient()
cli.load_system_host_keys()
# This is the user-provided training host, previously connected in this task.
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect('connect.nmb2.seetacloud.com', port=24365, username='root',
            password=getpass.getpass('SSH password: '), timeout=20,
            look_for_keys=False, allow_agent=False)
if args.op == 'session':
    print('CONNECTED', flush=True)
    for line in sys.stdin:
        op=json.loads(line)
        if op['op']=='close':break
        if op['op']=='exec':
            _,out,err=cli.exec_command(op['command'],timeout=180)
            print(json.dumps(dict(output=out.read().decode('utf-8',errors='replace'),
                                  error=err.read().decode('utf-8',errors='replace'),
                                  exit_code=out.channel.recv_exit_status()),ensure_ascii=False),flush=True)
        else:
            with cli.open_sftp() as s:
                if op['op']=='put':s.put(op['local'],op['remote'])
                elif op['op']=='get':
                    Path(op['local']).parent.mkdir(parents=True,exist_ok=True)
                    s.get(op['remote'],op['local'])
            print('TRANSFERRED',flush=True)
elif args.op == 'exec':
    _, out, err = cli.exec_command(args.paths[0], timeout=180)
    print(out.read().decode('utf-8', errors='replace'))
    print(err.read().decode('utf-8', errors='replace'))
    print('EXIT', out.channel.recv_exit_status())
else:
    with cli.open_sftp() as s:
        if args.op == 'put':
            s.put(args.paths[0], args.paths[1])
        else:
            Path(args.paths[1]).parent.mkdir(parents=True, exist_ok=True)
            s.get(args.paths[0], args.paths[1])
cli.close()
