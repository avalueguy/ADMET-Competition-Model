"""Package only lightweight v25 evidence on the training host."""
import hashlib,json,tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def main():
    assert str(ROOT)=='/home/czp123/admet_v23_20260928'
    out=ROOT/'admet_model_outputs_v25'
    names=['admet_v25_models.py','admet_v25_run.py','admet_v25_protocol.json','admet_v26_run.py','admet_v26_protocol.json']
    manifest={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    (out/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    files=list(out.glob('*.json'))+list(out.glob('*.log'))+list(out.glob('*/*/*/metadata.json'))+list(out.glob('*/*/metadata.json'))
    files+=list((ROOT/'admet_model_outputs_v26').glob('*.json'))+list((ROOT/'admet_model_outputs_v26').glob('*.log'))
    files+=list((ROOT/'admet_submissions/v26').glob('*'))
    dest=ROOT/'admet_v25_evidence.tar.gz'
    with tarfile.open(dest,'w:gz') as tar:
        for p in sorted(set(files)):
            if p.is_file():tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    print(json.dumps(dict(files=len(set(files)),bytes=dest.stat().st_size)))
if __name__=='__main__':main()
