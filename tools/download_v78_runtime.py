"""Download one PyPI wheel with bounded range requests and verify its SHA256."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    meta = json.load(urllib.request.urlopen('https://pypi.org/pypi/xgboost/3.4.1/json', timeout=30))
    info = next(f for f in meta['urls'] if f['filename'].endswith('manylinux_2_28_x86_64.whl'))
    root = Path('/root/autodl-tmp/v76/runtime_wheels')
    root.mkdir(exist_ok=True)
    output = root / info['filename']
    if output.exists() and hashlib.sha256(output.read_bytes()).hexdigest() == info['digests']['sha256']:
        print(str(output), flush=True)
        return
    chunk = 4 * 1024 * 1024
    ranges = [(i, min(i + chunk, info['size']) - 1) for i in range(0, info['size'], chunk)]
    def fetch(pair):
        start, end = pair
        request = urllib.request.Request(info['url'], headers={'Range': f'bytes={start}-{end}'})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    assert response.status == 206
                    raw = response.read()
                assert len(raw) == end - start + 1
                print(json.dumps({'start': start, 'bytes': len(raw)}), flush=True)
                return raw
            except Exception:
                if attempt == 2:
                    raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        raw = b''.join(pool.map(fetch, ranges))
    assert len(raw) == info['size']
    assert hashlib.sha256(raw).hexdigest() == info['digests']['sha256']
    output.write_bytes(raw)
    (root / 'xgboost_download.json').write_text(json.dumps(info, indent=2))
    print(str(output), flush=True)


if __name__ == '__main__':
    main()
