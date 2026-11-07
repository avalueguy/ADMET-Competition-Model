"""Password-prompted SSH transport. Never writes credentials to project files.

Input after authentication is one JSON operation per line: exec, put, get, close.
Server identity must already be trusted in the user's OpenSSH known_hosts.
"""
import argparse
import getpass
import json
import sys
from pathlib import Path

import paramiko


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", required=True)
    args = parser.parse_args()
    password = getpass.getpass("SSH password: ")
    client = paramiko.SSHClient()
    client.load_host_keys(str(Path.home()/".ssh/known_hosts"))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(args.host, port=args.port, username=args.user, password=password,
                   timeout=20, banner_timeout=20, auth_timeout=20,
                   look_for_keys=False, allow_agent=False)
    # Keep the password only in process memory so transport drops can reconnect.
    # Never replay an exec automatically after dispatch: it may already have run.
    client.get_transport().set_keepalive(10)
    print(json.dumps(dict(status="CONNECTED", host=args.host, port=args.port)), flush=True)
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            op = json.loads(line)
            try:
                transport = client.get_transport()
                if transport is None or not transport.is_active():
                    client.close()
                    client.connect(args.host, port=args.port, username=args.user, password=password,
                                   timeout=20, banner_timeout=20, auth_timeout=20,
                                   look_for_keys=False, allow_agent=False)
                    client.get_transport().set_keepalive(10)
                    print(json.dumps(dict(status="RECONNECTED")), flush=True)
                if op["op"] == "exec":
                    _, stdout, stderr = client.exec_command(op["command"], timeout=op.get("timeout", 60))
                    output = stdout.read().decode("utf-8", errors="replace")
                    error = stderr.read().decode("utf-8", errors="replace")
                    print(json.dumps(dict(status="DONE", exit_code=stdout.channel.recv_exit_status(),
                                          output=output, error=error), ensure_ascii=False), flush=True)
                elif op["op"] in ("put", "get"):
                    with client.open_sftp() as sftp:
                        if op["op"] == "put":
                            sftp.put(op["local"], op["remote"])
                        else:
                            Path(op["local"]).parent.mkdir(parents=True, exist_ok=True)
                            sftp.get(op["remote"], op["local"])
                    print(json.dumps(dict(status="TRANSFERRED", op=op["op"], remote=op["remote"])), flush=True)
                elif op["op"] == "close":
                    break
                else:
                    raise ValueError("Unknown operation")
            except Exception as exc:
                print(json.dumps(dict(status="ERROR", error=str(exc))), flush=True)
    finally:
        del password
        client.close()


if __name__ == "__main__":
    main()

