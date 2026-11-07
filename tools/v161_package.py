"""Preserve the complete trained batch and pinned baseline dependencies."""
import ast,hashlib,importlib.metadata,json,platform,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PREFIX='v161_new_structure_models_20261027'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    out=ROOT/'admet_model_outputs_v161';ep=out/'evidence.json';no=out/'no_candidate.json';assert ep.exists() or no.exists()
    if ep.exists():
        report=json.loads(ep.read_text());assert sha(ROOT/report['file'])==report['sha256'];assert json.loads((out/'independent_verification.json').read_text())['status']=='PASS'
    paths=[]
    for number in range(146,167):
        paths.extend(p for p in (ROOT/f'admet_model_outputs_v{number}').rglob('*') if p.is_file() and p.suffix not in ('.tmp','.pyc') and p.name!='pipeline_status.json')
        paths.extend((ROOT/'tools').glob(f'v{number}_*.py'));paths.extend(p for p in ROOT.glob(f'v{number}*.log') if p.name not in ('v161_package.log','v166_pipeline.log'))
    for number in (149,150,151,153,157,159):assert json.loads((ROOT/f'admet_model_outputs_v{number}/confirmation.json').read_text())['status']=='COMPLETE'
    for folder in ('admet_submissions/v161_new_structure_models','admet_submissions/v145_feedback_repair'):
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
    # v104 sparse features were referenced but absent from the v144 archive.
    paths.extend(p for p in (ROOT/'admet_model_outputs_v104/features').rglob('*') if p.is_file())
    paths.extend([ROOT/'docs/v155_training_plan.json',ROOT/'docs/v161_training_plan.json',ROOT/'docs/platform_scorecard_v145_20261027.json'])
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
    paths.extend(seen);env=out/'environment.json';save(env,dict(python=platform.python_version(),packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','catboost','xgboost','torch','rdkit']}));paths.append(env)
    files={p.relative_to(ROOT).as_posix():p for p in paths if p.is_file()};dependencies=[]
    for name in ('v124_models_history_20261027','v125_baseline_inputs','v130_low_task_experts_20261027','v144_broad_task_models_20261027','v145_feedback_repair_20261027'):
        p=ROOT/'releases'/(name+'.tar.gz');assert p.is_file();dependencies.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
    mp=ROOT/'releases'/(PREFIX+'.manifest.json');save(mp,dict(version='v161',files=[dict(path=name,bytes=p.stat().st_size,sha256=sha(p)) for name,p in sorted(files.items())],dependencies=dependencies,credentials_included=False,test_label_files_included=False,official_score=None,submission_created=ep.exists()))
    archive=ROOT/'releases'/(PREFIX+'.tar.gz');assert not archive.exists()
    with tarfile.open(archive,'w:gz',compresslevel=4) as tf:
        for name,p in sorted(files.items()):tf.add(p,arcname=name)
        tf.add(mp,arcname=mp.relative_to(ROOT).as_posix())
    info=dict(path=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,files=len(files),manifest_sha256=sha(mp),official_score=None,submission_created=ep.exists());save(mp.with_name(PREFIX+'.archive.json'),info);print(json.dumps(info),flush=True)
if __name__=='__main__':main()
