"""Local standard-library-only verification of retrieved final combination."""
import csv,io,json,hashlib,math,tarfile
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.reader(f))


def main():
    workspace=Path.cwd().resolve()
    with tarfile.open('admet_v14_review.tar.gz') as a:
        for member in a.getmembers():
            p=Path(member.name);dest=(workspace/p).resolve()
            assert member.isfile() and dest.is_relative_to(workspace)
            assert p.parts[0]=='admet_release_v14' or (p.parts[0]=='admet_submissions' and p.parts[1]=='v14') or (p.parts[0]=='admet_diagnostics' and p.name.startswith('v14_'))
            assert p.suffix in ['.csv','.json','.md']
            blob=a.extractfile(member).read()
            if dest.exists() and p.parts[0]=='admet_submissions':assert dest.read_bytes()==blob
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(blob)
    root=Path('admet_release_v14');v=json.loads((root/'verification.json').read_text(encoding='utf-8'));assert v['status']=='VERIFIED_COMBINED'
    upload=Path(v['submission']);assert sha(upload)==v['sha256']
    prior=next(Path('admet_submissions/v11').glob('*.csv'));assert sha(prior)=='082183f0c462081ea8735a41ec2e47326ef4cdcaf9d1188ae86a006f8057122d'
    b=upload.read_bytes();assert b.startswith(b'\xef\xbb\xbf') and b.count(b'\r\n')==16380
    assert b.count(b'task,sample_idx,drug_id,smiles,y_pred,metric')==1
    new=rows(upload);old=rows(prior);template=rows('admet_official/submission_template.csv')
    assert len(new)==len(old)==len(template)==16380 and new[0]==old[0]==template[0]
    changed=set();unchanged=set()
    for n,o,t in zip(new[1:],old[1:],template[1:]):
        assert len(n)==6 and n[:4]+n[5:]==o[:4]+o[5:]==t[:4]+t[5:]
        value=float(n[4]);assert math.isfinite(value)
        if n[5] in ['AUROC','AUPRC']:assert 0<=value<=1
        if n[0] not in v['changed']:assert n==o;unchanged.add(n[0])
        elif n[4]!=o[4]:changed.add(n[0])
    assert changed==set(v['changed']) and unchanged==set(v['unchanged']) and len(changed|unchanged)==22
    for route in json.loads((root/'routing.json').read_text(encoding='utf-8')):
        task=route['task'];f=root/task/f'submission_{task}.csv';assert sha(f)==route['prediction_sha256']
        part=rows(f)[1:];combined=[r for r in new[1:] if r[0]==task];assert len(part)==len(combined)
        for p,r in zip(part,combined):assert math.isclose(float(p[1]),float(r[4]),rel_tol=1e-12,abs_tol=1e-12)
    v.update(status='VERIFIED_LOCAL_AND_REMOTE',local_submission=str(upload.resolve()),bytes=len(b))
    (root/'local_verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(v,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
