"""Manifested incremental model archive with immutable dependency hashes."""
from pathlib import Path
import hashlib,json,platform,tarfile,sys
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    rel=read(ROOT/'admet_model_outputs_v107/release.json');assert rel['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM' and sha(ROOT/rel['file'])==rel['sha256']
    for n in (102,103,104,106,107):assert read(ROOT/f'admet_model_outputs_v{n}/model_verification.json')['status']=='PASS'
    assert read(ROOT/'admet_model_outputs_v104/features/replay.json')['status']=='PASS'
    import numpy,scipy,torch,sklearn,xgboost,rdkit,catboost,joblib
    env=dict(python=sys.version,platform=platform.platform(),gpu=torch.cuda.get_device_name(),versions={m.__name__:m.__version__ for m in (numpy,scipy,torch,sklearn,xgboost,rdkit,catboost,joblib)})
    save(ROOT/'admet_model_outputs_v107/environment.json',env)
    paths=[ROOT/f'admet_model_outputs_v{n}' for n in (102,103,104,106,107)]
    for n in (102,103,104,105,106,107):
        for pattern in (f'v{n}_*.py',f'run_v{n}_*.py',f'finish_v{n}.py',f'package_v{n}.py',f'finalize_v{n}_local.py'):paths.extend((ROOT/'tools').glob(pattern))
        paths.extend((ROOT/'docs').glob(f'ITERATION_V{n}_*.md'));paths.extend((ROOT/'docs').glob(f'V{n}_*PROTOCOL*.json'))
    paths.extend([ROOT/'tools/record_v101_feedback.py',ROOT/'tools/restore_iteration_archive.py',ROOT/'tools/verify_submission.py',ROOT/'tools/repack_upload_template.py',ROOT/'docs/platform_feedback_v101_20261026.json',ROOT/rel['file'],(ROOT/rel['file']).with_suffix('.verification.json')])
    for pattern in ('v102*.log','v103*.log','v104*.log','v106*.log','v107_finish.log'):paths.extend(ROOT.glob(pattern))
    files={}
    for path in paths:
        for p in ([path] if path.is_file() else path.rglob('*')):
            if p.is_file() and p.suffix!='.pyc':files[p.relative_to(ROOT).as_posix()]=p
    folder=ROOT/'releases';prefix='v107_count_difference_boost_20261026';manifest=folder/(prefix+'.manifest.json');archive=folder/(prefix+'.tar.gz')
    assert not archive.exists(),'Do not overwrite completed archive.'
    names=['v86_metric_ranking_20261025','v88_context_kernel_20261025','v92_full_kernel_20261025','v94_isolated_assay_20261025','v95_graph_specialists_20261026','v96_normalized_assays_20261026','v101_multi_route_20261026']
    deps=[dict(path=f'releases/{n}.tar.gz',sha256=sha(folder/f'{n}.tar.gz')) for n in names]
    save(manifest,dict(version='v107',files=[dict(path=n,sha256=sha(p),bytes=p.stat().st_size) for n,p in sorted(files.items())],dependencies=deps,credentials_included=False,test_labels_included=False))
    with tarfile.open(archive,'w:gz') as tf:
        for name,p in sorted(files.items()):tf.add(p,arcname=name)
        tf.add(manifest,arcname=manifest.relative_to(ROOT).as_posix())
    record=dict(path=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,files=len(files),version='v107',official_score=None)
    save(folder/(prefix+'.archive.json'),record);print(json.dumps(record),flush=True)
if __name__=='__main__':main()
