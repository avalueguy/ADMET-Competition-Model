"""Verify and restore the five-archive v389 handoff without external packages."""
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, shutil, tarfile, zipfile
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def safe(root,name):
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:raise ValueError('Unsafe archive path: '+name)
 dest=(root/name).resolve()
 if not dest.is_relative_to(root.resolve()):raise ValueError('Path escapes destination')
 return dest
def main():
 a=argparse.ArgumentParser();a.add_argument('--bundle-dir',type=Path,default=Path('.'));a.add_argument('--destination',type=Path,required=True);o=a.parse_args()
 manifest=json.loads((o.bundle_dir/'handoff-manifest-v389.json').read_text(encoding='utf-8'))
 if o.destination.exists() and any(o.destination.iterdir()):raise SystemExit('Destination must be absent or empty')
 items=manifest['runtime_archives']+[manifest['source_archive']]
 for item in items:
  src=o.bundle_dir/item['name']
  if src.stat().st_size!=item['bytes'] or sha(src)!=item['sha256']:raise SystemExit('Checksum mismatch: '+item['name'])
 o.destination.mkdir(parents=True,exist_ok=True)
 for item in manifest['runtime_archives']:
  src=o.bundle_dir/item['name'];copy=safe(o.destination,item['destination']);copy.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,copy)
  with tarfile.open(src,'r:gz') as tf:
   for m in tf:
    dest=safe(o.destination,m.name)
    if m.isdir():dest.mkdir(parents=True,exist_ok=True);continue
    if not m.isfile():raise ValueError('Only regular files supported: '+m.name)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with tf.extractfile(m) as r,dest.open('wb') as w:shutil.copyfileobj(r,w)
 with zipfile.ZipFile(o.bundle_dir/manifest['source_archive']['name']) as zf:
  for m in zf.infolist():
   dest=safe(o.destination,m.filename)
   if m.is_dir():dest.mkdir(parents=True,exist_ok=True);continue
   dest.parent.mkdir(parents=True,exist_ok=True)
   with zf.open(m) as r,dest.open('wb') as w:shutil.copyfileobj(r,w)
 for item in manifest['source_files']:
  if sha(o.destination/item['path'])!=item['sha256']:raise SystemExit('Restored source mismatch: '+item['path'])
 for item in manifest['submissions']:
  src=o.bundle_dir/item['name']
  if sha(src)!=item['sha256']:raise SystemExit('CSV checksum mismatch')
  dest=safe(o.destination,item['original_path']);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
 print('PASS: restored source, inputs, model artifacts and submissions; read docs/V389_RESULTS_AND_HANDOFF.md')
if __name__=='__main__':main()
