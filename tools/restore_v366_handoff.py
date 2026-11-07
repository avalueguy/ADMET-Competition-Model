"""Restore and verify the v366 GitHub handoff using only Python's standard library."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import tarfile
import zipfile


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();base=args.bundle_dir.resolve();dest=args.destination.resolve()
    assert dest!=base,'Choose a fresh destination separate from the downloaded handoff'
    manifest=json.loads((base/'handoff-manifest-v366.json').read_text(encoding='utf-8'))
    assert manifest['format_version']==1
    dest.mkdir(parents=True,exist_ok=True)
    source=base/manifest['source_archive']['name'];assert sha(source)==manifest['source_archive']['sha256']
    runtime=base/manifest['runtime_archive']['name']
    if not runtime.exists():
        temporary=runtime.with_suffix(runtime.suffix+'.joining')
        with temporary.open('wb') as output:
            for entry in manifest['runtime_archive']['parts']:
                p=base/entry['name'];assert sha(p)==entry['sha256'],entry['name']
                with p.open('rb') as stream:shutil.copyfileobj(stream,output)
        assert sha(temporary)==manifest['runtime_archive']['sha256']
        temporary.replace(runtime)
    assert sha(runtime)==manifest['runtime_archive']['sha256']
    expected={r['path']:r for r in manifest['files']}
    seen=set()

    def put(name,content):
        assert name in expected and name not in seen,name
        seen.add(name);p=(dest/name).resolve()
        assert p.is_relative_to(dest),name
        assert hashlib.sha256(content).hexdigest()==expected[name]['sha256'],name
        p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():assert sha(p)==expected[name]['sha256'],'Refusing overwrite: '+name
        else:p.write_bytes(content)

    with zipfile.ZipFile(source) as zf:
        for entry in zf.infolist():
            assert not entry.is_dir()
            put(entry.filename,zf.read(entry))
    with tarfile.open(runtime) as tf:
        for entry in tf:
            assert entry.isfile(),entry.name
            put(entry.name,tf.extractfile(entry).read())
    assert seen==set(expected)
    for r in manifest['files']:assert sha(dest/r['path'])==r['sha256'],r['path']
    print(json.dumps(dict(status='PASS',files=len(seen),destination=str(dest),
                         known_official_best=manifest['known_official_best'],candidate=manifest['candidate']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
