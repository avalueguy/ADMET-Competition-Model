"""Preserve completed v168 training and v169 verification with fixed dependencies."""
import ast,json,sys,time,tarfile
from pathlib import Path
from v161_package import sha,save
ROOT=Path(__file__).resolve().parents[1];PREFIX='v169_sparse_linear_20261027'
def main():
    for _ in range(720):
        out=ROOT/'admet_model_outputs_v169'
        if (out/'evidence.json').exists() or (out/'no_candidate.json').exists():break
        time.sleep(15)
    else:raise TimeoutError('v169 verification incomplete')
    confirmed=json.loads((ROOT/'admet_model_outputs_v168/confirmation.json').read_text());assert confirmed['status']=='COMPLETE'
    files={}
    for folder in ('admet_model_outputs_v168','admet_model_outputs_v169','admet_submissions/v169_sparse_linear'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and p.suffix not in ('.pyc','.tmp'):files[p.relative_to(ROOT).as_posix()]=p
    paths=[ROOT/'tools/v168_sparse_linear.py',ROOT/'tools/v169_verify_sparse_linear.py',Path(__file__),ROOT/'docs/v168_training_plan.json',ROOT/'v168_sparse_linear.log',ROOT/'v169_verify.log'];queue=[p for p in paths if p.suffix=='.py'];seen=set()
    while queue:
        p=queue.pop()
        if p in seen:continue
        seen.add(p)
        for n in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            names=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module] if isinstance(n,ast.ImportFrom) and n.module else []
            for name in names:
                stem=name.split('.')[0]
                for candidate in (ROOT/(stem+'.py'),ROOT/'tools'/(stem+'.py')):
                    if candidate.is_file() and candidate not in seen:queue.append(candidate)
    for p in paths+list(seen):
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=p
    dep=ROOT/'releases/v161_new_structure_models_20261027.tar.gz';info=json.loads(dep.with_name('v161_new_structure_models_20261027.archive.json').read_text());assert sha(dep)==info['sha256']
    mp=ROOT/'releases'/(PREFIX+'.manifest.json');save(mp,dict(version='v169',files=[dict(path=name,bytes=p.stat().st_size,sha256=sha(p)) for name,p in sorted(files.items())],dependencies=[dict(path=dep.relative_to(ROOT).as_posix(),sha256=info['sha256'])],dependency_note='Original v161 has one post-manifest read-only diagnostic inconsistency. Locally repaired manifest/archive preserves all original model bytes. Original archive retained and hash-pinned.',credentials_included=False,official_score=None,submission_created=(out/'evidence.json').exists()))
    archive=ROOT/'releases'/(PREFIX+'.tar.gz');assert not archive.exists()
    with tarfile.open(archive,'w:gz',compresslevel=4) as tf:
        for name,p in sorted(files.items()):tf.add(p,arcname=name)
        tf.add(mp,arcname=mp.relative_to(ROOT).as_posix())
    record=dict(path=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,files=len(files),manifest_sha256=sha(mp),submission_created=(out/'evidence.json').exists(),official_score=None);save(mp.with_name(PREFIX+'.archive.json'),record);print(json.dumps(record),flush=True)
if __name__=='__main__':main()
