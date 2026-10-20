"""Retrieve-only lightweight verification. Never train or load model weights locally."""
import csv,hashlib,io,json,math,tarfile
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):return list(csv.reader(io.StringIO(Path(p).read_bytes().decode('utf-8-sig'),newline='')))


def main():
    workspace=Path.cwd().resolve()
    with tarfile.open('admet_v15_review.tar.gz') as archive:
        members=archive.getmembers();assert sum(m.size for m in members)<150_000_000
        for m in members:
            p=Path(m.name);dest=(workspace/p).resolve()
            assert m.isfile() and dest.is_relative_to(workspace) and len(p.parts)>1,m.name
            allowed=(p.parts[0] in ['admet_release_v15','admet_model_outputs_v15'] or
                (p.parts[0]=='admet_submissions' and p.parts[1]=='v15') or
                (p.parts[0] in ['admet_diagnostics','logs'] and p.name.startswith('v15')))
            assert allowed and p.suffix in ['.csv','.json','.log'],m.name
            blob=archive.extractfile(m).read()
            if p.parts[0]=='admet_submissions' and dest.exists():assert dest.read_bytes()==blob
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(blob)
    root=Path('admet_release_v15');v=read(root/'independent_verification.json')
    assert v['status']=='VERIFIED' and v['independent_replay']
    assert len(v['task_checks'])==22 and all(t['passed'] for t in v['task_checks'])
    upload=Path(v['submission']);assert sha(upload)==v['sha256']
    old=next(Path('admet_submissions/v14').glob('*.csv'))
    assert sha(old)=='97c3eb59a65585cd897e843d990940bd2a7ac67ef8ea825f7a70598101a1fd94'
    b=upload.read_bytes();assert b.startswith(b'\xef\xbb\xbf') and b.count(b'\r\n')==16380
    assert b.count(b'task,sample_idx,drug_id,smiles,y_pred,metric')==1
    new=rows(upload);prior=rows(old);template=rows('admet_official/submission_template.csv')
    assert len(new)==len(prior)==len(template)==16380
    assert new[0]==prior[0]==template[0]==['task','sample_idx','drug_id','smiles','y_pred','metric']
    changed=set();actual_changed=set();unchanged=set();identities=set()
    for n,o,t in zip(new[1:],prior[1:],template[1:]):
        assert len(n)==6 and n[:4]+n[5:]==o[:4]+o[5:]==t[:4]+t[5:]
        key=(n[0],n[1]);assert key not in identities;identities.add(key)
        value=float(n[4]);assert math.isfinite(value)
        if n[5] in ['AUROC','AUPRC']:assert 0<=value<=1
        if n[0] in v['changed']:
            changed.add(n[0])
            if n[4]!=o[4]:actual_changed.add(n[0])
        else:assert n==o;unchanged.add(n[0])
    assert changed==actual_changed==set(v['changed']) and unchanged==set(v['unchanged'])
    assert len(changed|unchanged)==22
    for route in read(root/'routing.json'):
        task=route['task'];part=root/task/f'submission_{task}.csv'
        assert sha(part)==route['prediction_sha256']
        assert route['protocol_sha256']==sha('admet_v15_protocol.json')
        a=rows(part)[1:];c=[r for r in new[1:] if r[0]==task]
        assert len(a)==len(c)
        for aa,cc in zip(a,c):
            assert aa[0]==cc[2] and math.isclose(float(aa[1]),float(cc[4]),abs_tol=1e-12,rel_tol=1e-12)
    for f,h in read('admet_v15_source_freeze.json').items():assert sha(f)==h,f
    # The remote exporter preserves the old filename; delivery uses this run's date.
    dated=upload.with_name(upload.stem.rsplit('-',1)[0]+'-20260926.csv')
    if dated.exists():assert dated.read_bytes()==b
    else:dated.write_bytes(b)
    assert sha(dated)==v['sha256']
    v.update(status='VERIFIED_LOCAL_AND_REMOTE',local_submission=str(dated.resolve()),bytes=len(b),
        remote_export_name_preserved=str(upload),delivery_filename_only_changed=True,
        local_training=False,local_model_weights_loaded=False,template_metadata_exact=True)
    (root/'local_verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v[k] for k in ['status','local_submission','sha256','bytes','changed','unchanged']},ensure_ascii=True,indent=2))


if __name__=='__main__':main()
