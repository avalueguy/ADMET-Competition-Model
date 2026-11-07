"""Password-prompted read-only status sync; credentials remain process memory."""
import argparse
import getpass
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    temp = path.with_name(path.name + '.watch.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--port', required=True, type=int)
    args = parser.parse_args()
    marker = ROOT / 'docs/v216_local_watcher.json'
    password = getpass.getpass('SSH password: ')
    client = paramiko.SSHClient()
    client.load_host_keys(str(Path.home() / '.ssh/known_hosts'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    remote = '/root/autodl-tmp/v76/'
    try:
        for step in range(1440):
            try:
                transport = client.get_transport()
                if transport is None or not transport.is_active():
                    client.connect(args.host, port=args.port, username='root', password=password,
                        timeout=15, banner_timeout=15, auth_timeout=15, look_for_keys=False, allow_agent=False)
                    client.get_transport().set_keepalive(10)
                pause = json.loads((ROOT / 'docs/USER_PAUSE.json').read_text(encoding='utf-8-sig'))
                if not pause.get('training_permitted', False):
                    save(marker, dict(status='STOPPED_USER_PAUSE', credentials_stored=False))
                    return
                command = 'cd /root/autodl-tmp/v76 && /root/miniconda3/bin/python tools/v220_progress_report.py'
                _, stdout, stderr = client.exec_command(command, timeout=20)
                output, error = stdout.read().decode(), stderr.read().decode()
                assert stdout.channel.recv_exit_status() == 0, error
                with client.open_sftp() as sftp:
                    for file in ('docs/v216_live_progress.json', 'docs/TRAINING_LIVE_PROGRESS.md'):
                        target = ROOT / file
                        temp = target.with_name(target.name + '.watch.tmp')
                        sftp.get(remote + file, str(temp))
                        temp.replace(target)
                report = json.loads((ROOT / 'docs/v216_live_progress.json').read_text(encoding='utf-8'))
                save(marker, dict(status='SYNCED', checked_at_utc=report['checked_at_utc'],
                    interval_seconds=30, credentials_stored=False, read_only_remote_operations=True,
                    evaluated_tasks=report['evaluated_development_tasks'], saved_models=report['saved_development_models']))
                print(output.strip(), flush=True)
                if report['failure'] or report['scheduler_failure'] or report['pipeline'].get('status') == 'FAILED':
                    save(marker, dict(status='STOPPED_ON_FAILURE', checked_at_utc=report['checked_at_utc'], credentials_stored=False))
                    return
                if report['pipeline'].get('status') == 'COMPLETE':
                    save(marker, dict(status='COMPLETE', checked_at_utc=report['checked_at_utc'], credentials_stored=False))
                    return
            except Exception as exc:
                save(marker, dict(status='CONNECTION_OR_SYNC_ERROR', checked_at_utc=datetime.now(timezone.utc).isoformat(),
                                   error=str(exc), credentials_stored=False))
                print(json.dumps(dict(status='SYNC_ERROR', error=str(exc))), flush=True)
            time.sleep(30)
    finally:
        del password
        client.close()


if __name__ == '__main__':
    main()
