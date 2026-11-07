"""Read-only check of completed v161 archive members against its manifest."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PREFIX='v161_new_structure_models_20261027'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    mp=ROOT/'releases'/(PREFIX+'.manifest.json');manifest=json.loads(mp.read_text());expected={r['path']:r for r in manifest['files']};changed=[]
    for name,r in expected.items():
        if sha(ROOT/name)!=r['sha256']:changed.append(name)
    invalid=[];seen=set();archive=ROOT/'releases'/(PREFIX+'.tar.gz')
    with tarfile.open(archive,'r|gz') as tf:
        for member in tf:
            seen.add(member.name)
            if member.name not in expected:continue
            r=expected[member.name];h=hashlib.sha256();f=tf.extractfile(member)
            for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
            if h.hexdigest()!=r['sha256'] or member.size!=r['bytes']:invalid.append(member.name)
    result=dict(status='PASS' if not invalid and set(expected)<=seen else 'FAILED',changed_after_manifest=changed,invalid_archive_members=invalid,missing_members=sorted(set(expected)-seen),manifest_files=len(expected))
    out=ROOT/'admet_model_outputs_v170';out.mkdir(exist_ok=True);(out/'v161_archive_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
