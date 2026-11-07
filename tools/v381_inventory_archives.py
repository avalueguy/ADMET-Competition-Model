"""Read-only legacy checkpoint inventory from archive manifests, not model fits."""
from pathlib import Path
import json,hashlib,re
ROOT=Path(__file__).resolve().parents[1]
PATTERN=re.compile(r'admet_model_outputs_v(23|44|46|52|56|67)(?:_|/)')


def main():
    records=[]
    sources=list((ROOT/'releases').glob('*manifest*.json'))+list((ROOT/'docs/artifact-manifests').glob('*.json'))
    for path in sources:
        data=json.loads(path.read_text(encoding='utf-8'))
        rows=data if isinstance(data,list) else data.get('files',[])
        matching=[]
        for row in rows:
            if not isinstance(row,dict):continue
            name=row.get('path','')
            if PATTERN.search(name) and name.endswith(('.pt','.pth','.joblib','.pkl','.cbm','.ubj')):
                matching.append(row)
        if matching:
            archive=path.with_name(path.name.replace('.manifest.json','.tar.gz'))
            if path.name=='official_only_v67_v68_manifest.json':archive=path.parent/'official_only_v67_v68_20261020.tar.gz'
            records.append(dict(manifest=path.relative_to(ROOT).as_posix(),manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 archive=archive.relative_to(ROOT).as_posix(),archive_exists=archive.is_file(),checkpoints=len(matching),
                 checkpoint_examples=matching[:8]))
    value=dict(read_only=True,no_fitting=True,scope='Available release/runtime manifests, matching historical model checkpoint entries only.',
               records=records,limitation='Manifest entries do not prove archived bytes, usable dependencies, exact source/environment or complete ensemble OOF. Archives require selective hash-verified recovery and remote-only replay.')
    (ROOT/'docs/v381_history_archive_inventory.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(manifests_with_checkpoints=len(records),records=[{k:v for k,v in r.items() if k!='checkpoint_examples'} for r in records])))


if __name__=='__main__':main()
