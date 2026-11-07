"""Verify a versioned incremental archive before restoring absent local files."""
import argparse,hashlib,json,tarfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('prefix');args=ap.parse_args();prefix=args.prefix
    assert '/' not in prefix and '\\' not in prefix and '..' not in prefix
    archive=ROOT/'releases'/(prefix+'.tar.gz');info=load(archive.with_suffix('').with_suffix('.archive.json'))
    assert sha(archive)==info['sha256'] and archive.stat().st_size==info['bytes']
    mp=ROOT/'releases'/(prefix+'.manifest.json');manifest=load(mp);expected={r['path']:r for r in manifest['files']};seen=set()
    for dep in manifest.get('dependencies',[]):assert sha(ROOT/dep['path'])==dep['sha256']
    with tarfile.open(archive,'r|gz') as tf:
        for m in tf:
            rel=PurePosixPath(m.name);assert m.isfile() and not rel.is_absolute() and '..' not in rel.parts and m.name not in seen
            seen.add(m.name);data=tf.extractfile(m).read()
            if m.name==mp.relative_to(ROOT).as_posix():assert data==mp.read_bytes();continue
            row=expected[m.name];assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    assert seen==set(expected)|{mp.relative_to(ROOT).as_posix()}
    restored=0;exact=0;local_different=[]
    with tarfile.open(archive,'r|gz') as tf:
        for m in tf:
            if m.name not in expected:continue
            p=ROOT/m.name;row=expected[m.name]
            if p.exists():
                if sha(p)==row['sha256']:exact+=1
                else:local_different.append(m.name)
                continue
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tf.extractfile(m).read());assert sha(p)==row['sha256'];restored+=1
    assert all(p.startswith('docs/') for p in local_different),local_different
    result=dict(status='PASS',prefix=prefix,archive_sha256=sha(archive),manifest_files=len(expected),all_member_hashes_verified=True,dependency_hashes_verified=True,
        restored=restored,existing_exact=exact,existing_updated_docs_preserved=local_different)
    out=ROOT/'docs'/(prefix.split('_')[0]+'_local_archive_verification.json');out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
