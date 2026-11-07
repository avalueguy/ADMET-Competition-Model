"""Preserve original archive, repair one post-manifest diagnostic snapshot only."""
import copy,hashlib,io,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OLD='v161_new_structure_models_20261027';NEW=OLD+'_repaired'
DIAGNOSTIC='admet_model_outputs_v161/route_diagnostics.json'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
class HashReader:
    def __init__(self,file):self.file=file;self.h=hashlib.sha256();self.count=0
    def read(self,size=-1):
        value=self.file.read(size);self.h.update(value);self.count+=len(value);return value
def main():
    oldmp=ROOT/'releases'/(OLD+'.manifest.json');oldarchive=ROOT/'releases'/(OLD+'.tar.gz');oldinfo=json.loads((ROOT/'releases'/(OLD+'.archive.json')).read_text(encoding='utf-8'));assert sha(oldarchive)==oldinfo['sha256']
    doc=json.loads(oldmp.read_text(encoding='utf-8'));expected={r['path']:r for r in doc['files']};updated=copy.deepcopy(doc);changed=[];seen=set()
    # The diagnostics writer ran while packaging; all model files stay unchanged.
    with tarfile.open(oldarchive,'r|gz') as tf:
        for member in tf:
            if member.name not in expected:continue
            seen.add(member.name);reader=HashReader(tf.extractfile(member))
            for _ in iter(lambda:reader.read(4*1024*1024),b''):pass
            r=expected[member.name]
            if reader.h.hexdigest()!=r['sha256'] or reader.count!=r['bytes']:
                assert member.name==DIAGNOSTIC;changed.append(member.name)
                replacement=next(q for q in updated['files'] if q['path']==member.name);replacement.update(sha256=reader.h.hexdigest(),bytes=reader.count)
    assert changed==[DIAGNOSTIC] and seen==set(expected)
    updated['repair']=dict(original_archive_sha256=oldinfo['sha256'],reason='Read-only diagnostics regenerated after original manifest creation, during packaging.',changed_members=changed,training_or_prediction_files_changed=False,original_archive_preserved=True)
    mp=ROOT/'releases'/(NEW+'.manifest.json');save(mp,updated);manifestbytes=mp.read_bytes();out=ROOT/'releases'/(NEW+'.tar.gz');assert not out.exists()
    with tarfile.open(oldarchive,'r|gz') as source,tarfile.open(out,'w|gz',compresslevel=4) as target:
        for member in source:
            if member.name==oldmp.relative_to(ROOT).as_posix():continue
            assert member.name in expected and member.isfile();target.addfile(member,source.extractfile(member))
        info=tarfile.TarInfo(mp.relative_to(ROOT).as_posix());info.size=len(manifestbytes);target.addfile(info,io.BytesIO(manifestbytes))
    result=dict(path=out.relative_to(ROOT).as_posix(),sha256=sha(out),bytes=out.stat().st_size,files=len(expected),manifest_sha256=sha(mp),official_score=None,submission_created=False,original_archive_preserved=True)
    save(ROOT/'releases'/(NEW+'.archive.json'),result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
