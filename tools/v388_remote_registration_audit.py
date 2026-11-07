"""Record remote file order before development is complete; no model access."""
from pathlib import Path
from datetime import datetime
import hashlib,json,time

def main():
    root=Path('/root/autodl-tmp');protocol=root/'v388_union_protocol.json';launch=root/'regression9_v387/launch.json'
    r=json.loads(protocol.read_text());launch_time=launch.stat().st_mtime
    assert datetime.fromisoformat(r['registered_at_utc']).timestamp()<launch_time
    assert protocol.stat().st_mtime<launch_time
    value=dict(status='PASS',at_unix=time.time(),launch_mtime_unix=launch_time,remote_protocol_mtime_unix=protocol.stat().st_mtime,
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),scope='Remote transferred protocol predates remote launch file; no claim of tamperproof registry.')
    (root/'v388_remote_preregistration_audit.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))

if __name__=='__main__':main()
