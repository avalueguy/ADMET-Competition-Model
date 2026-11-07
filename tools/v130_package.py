"""Hash-pinned incremental archive of trained low-task models and evidence."""
import ast,hashlib,importlib.metadata,json,platform,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PREFIX='v130_low_task_experts_20261027'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    report=json.loads((ROOT/'admet_model_outputs_v130/evidence.json').read_text());assert sha(ROOT/report['file'])==report['sha256']
    assert json.loads((ROOT/'admet_model_outputs_v129/independent_verification.json').read_text())['status']=='PASS'
    paths=[]
    for number in range(125,131):
        paths.extend(p for p in (ROOT/f'admet_model_outputs_v{number}').rglob('*') if p.is_file() and p.suffix not in ('.tmp','.pyc'))
        paths.extend((ROOT/'tools').glob(f'v{number}_*.py'));paths.extend(ROOT.glob(f'v{number}*.log'))
    for number in (126,127,128):assert json.loads((ROOT/f'admet_model_outputs_v{number}/confirmation.json').read_text())['status']=='COMPLETE'
    for folder in ['admet_submissions/v130_low_task_experts','admet_submissions/v124_matched_models_history']:
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
    paths.extend([ROOT/'docs/platform_scorecard_v124_20261027.json',ROOT/'releases/v125_baseline_inputs.manifest.json'])
    # Preserve exact source-import chain without importing executable code.
    queue=[p for p in paths if p.suffix=='.py'];seen=set()
    while queue:
        p=queue.pop()
        if p in seen:continue
        seen.add(p)
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                stem=name.split('.')[0]
                for candidate in [ROOT/(stem+'.py'),ROOT/'tools'/(stem+'.py')]:
                    if candidate.is_file() and candidate not in seen:queue.append(candidate)
    paths.extend(seen)
    ep=ROOT/'admet_model_outputs_v130/environment.json';save(ep,dict(python=platform.python_version(),packages={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','catboost','xgboost','torch','rdkit']}));paths.append(ep)
    files={p.relative_to(ROOT).as_posix():p for p in paths if p.is_file()}
    dependencies=[]
    for name in ['v124_models_history_20261027','v125_baseline_inputs']:
        p=ROOT/'releases'/(name+'.tar.gz');assert p.is_file();dependencies.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
    mp=ROOT/'releases'/(PREFIX+'.manifest.json')
    save(mp,dict(version='v130',files=[dict(path=name,bytes=p.stat().st_size,sha256=sha(p)) for name,p in sorted(files.items())],dependencies=dependencies,
        credentials_included=False,test_label_files_included=False,official_score=None,dependencies_note='Incremental archive; v124 manifest pins earlier features/history. v125 archive adds exact v67 matched-control prediction arrays.'))
    archive=ROOT/'releases'/(PREFIX+'.tar.gz');assert not archive.exists()
    with tarfile.open(archive,'w:gz') as tf:
        for name,p in sorted(files.items()):tf.add(p,arcname=name)
        tf.add(mp,arcname=mp.relative_to(ROOT).as_posix())
    info=dict(path=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,files=len(files),manifest_sha256=sha(mp),official_score=None)
    save(mp.with_name(PREFIX+'.archive.json'),info);print(json.dumps(info),flush=True)
if __name__=='__main__':main()
