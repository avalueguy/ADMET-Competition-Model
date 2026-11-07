"""Package completed v172/v173/v175 experiments and independent verification."""
import ast,json,sys,tarfile,time,traceback
from pathlib import Path
from v161_package import sha,save
ROOT=Path(__file__).resolve().parents[1]
PREFIX='v176_shared_graph_batch_20261027'

def package():
    out=ROOT/'admet_model_outputs_v176';ep=out/'evidence.json';no=out/'no_candidate.json'
    assert ep.exists() or no.exists()
    if ep.exists():
        report=json.loads(ep.read_text());assert sha(ROOT/report['file'])==report['sha256'];assert json.loads((out/'independent_verification.json').read_text())['status']=='PASS'
    files={};paths=[]
    for n in (172,173,175,176):
        folder=ROOT/f'admet_model_outputs_v{n}'
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix not in ('.tmp','.pyc') and p.name!='pipeline_status.json':paths.append(p)
        paths.extend((ROOT/'tools').glob(f'v{n}_*.py'))
        paths.extend(ROOT.glob(f'v{n}*.log'))
    for n in (172,173,175):assert json.loads((ROOT/f'admet_model_outputs_v{n}/confirmation.json').read_text())['status']=='COMPLETE'
    for folder in ('admet_submissions/v176_shared_graph_hybrid','admet_submissions/v145_feedback_repair'):
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
    paths.extend([Path(__file__),ROOT/'docs/v175_training_plan.json',ROOT/'docs/platform_scorecard_v145_20261027.json'])
    queue=[p for p in paths if p.suffix=='.py'];seen=set()
    while queue:
        p=queue.pop()
        if p in seen:continue
        seen.add(p)
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                stem=name.split('.')[0]
                for candidate in (ROOT/(stem+'.py'),ROOT/'tools'/(stem+'.py')):
                    if candidate.is_file() and candidate not in seen:queue.append(candidate)
    for p in paths+list(seen):
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=p
    deps=[]
    for prefix in ('v161_new_structure_models_20261027','v169_sparse_linear_20261027'):
        p=ROOT/'releases'/(prefix+'.tar.gz');info=json.loads(p.with_name(prefix+'.archive.json').read_text());assert sha(p)==info['sha256'];deps.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=info['sha256']))
    mp=ROOT/'releases'/(PREFIX+'.manifest.json');save(mp,dict(version='v176',files=[dict(path=name,bytes=p.stat().st_size,sha256=sha(p)) for name,p in sorted(files.items())],dependencies=deps,credentials_included=False,test_label_files_included=False,submission_created=ep.exists(),official_score=None))
    archive=ROOT/'releases'/(PREFIX+'.tar.gz');assert not archive.exists()
    with tarfile.open(archive,'w:gz',compresslevel=4) as tf:
        for name,p in sorted(files.items()):tf.add(p,arcname=name)
        tf.add(mp,arcname=mp.relative_to(ROOT).as_posix())
    info=dict(path=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,files=len(files),manifest_sha256=sha(mp),submission_created=ep.exists(),official_score=None)
    save(mp.with_name(PREFIX+'.archive.json'),info);print(json.dumps(info),flush=True)

def main():
    import subprocess
    out=ROOT/'admet_model_outputs_v176';out.mkdir(exist_ok=True);status=out/'pipeline_status.json'
    for _ in range(720):
        cp=ROOT/'admet_model_outputs_v175/confirmation.json'
        if cp.exists() and json.loads(cp.read_text()).get('status')=='COMPLETE':break
        save(status,dict(status='WAITING_TRAINING',route='v175'));time.sleep(30)
    else:raise TimeoutError('Training incomplete within 6h; preserve partial results')
    save(status,dict(status='INDEPENDENT_VERIFICATION'))
    with (ROOT/'v176_verify.log').open('a',encoding='utf-8') as f:
        subprocess.run([sys.executable,str(ROOT/'tools/v176_verify_shared_graph.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    save(status,dict(status='PACKAGING'));package()
    save(status,dict(status='COMPLETE',submission_created=(out/'evidence.json').exists(),official_score=None,automatically_uploaded=False));print('COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        save(ROOT/'admet_model_outputs_v176/pipeline_status.json',dict(status='FAILED',traceback=traceback.format_exc()));raise
