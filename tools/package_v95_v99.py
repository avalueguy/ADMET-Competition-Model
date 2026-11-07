"""Hash-manifest incremental archive of this multi-route experiment and release."""
import argparse,hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def package(version=99):
    assert version in (97,99,101);out=ROOT/f'admet_model_outputs_v{version}';rel=read(out/'release.json')
    for n in (95,96,98,version):assert read(ROOT/f'admet_model_outputs_v{n}/model_verification.json')['status']=='PASS'
    assert read(ROOT/'admet_model_outputs_v98/confirmation.json')['status']=='COMPLETE'
    assert read(ROOT/'admet_model_outputs_v98/features/replay.json')['status']=='PASS'
    if version==101:assert read(ROOT/'admet_model_outputs_v100/model_verification.json')['status']=='PASS'
    assert sha(ROOT/rel['file'])==rel['sha256'] and rel['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM'
    paths=[ROOT/f'admet_model_outputs_v{n}' for n in (97,98)]
    if version!=97:paths.append(out)
    if version==101:paths.append(ROOT/'admet_model_outputs_v100')
    paths += [ROOT/'admet_model_outputs_v79/features/graphs.pt',ROOT/'admet_v79_joint_graph.py',
        ROOT/'docs/platform_feedback_v94_20261026.json',Path(__file__)]
    for n in (95,96,97,98,99,100,101):
        paths.extend((ROOT/'tools').glob(f'v{n}_*.py'));paths.extend((ROOT/'tools').glob(f'run_v{n}_*.py'))
        paths.extend((ROOT/'tools').glob(f'finalize_v{n}_*.py'));paths.extend((ROOT/'docs').glob(f'ITERATION_V{n}_*.md'))
        paths.extend((ROOT/'docs').glob(f'V{n}_*PROTOCOL*.json'))
        paths.extend((ROOT/'admet_submissions').glob(f'v{n}_*'))
    paths.extend((ROOT/'tools/finalize_v97_v99_local.py',ROOT/'tools/restore_iteration_archive.py',ROOT/'tools/verify_submission.py',ROOT/'tools/repack_upload_template.py',ROOT/'tools/finish_v101.py',ROOT/'tools/audit_v101_identity_coverage.py',ROOT/'tools/resume_v98_parallel_confirmation.py'))
    for pattern in ('v95*.log','v96*.log','v97*.log','v98*.log','v99*.log','v100*.log','v101*.log'):paths.extend(ROOT.glob(pattern))
    files={}
    for path in paths:
        for p in ([path] if path.is_file() else path.rglob('*')):
            if p.is_file() and p.suffix!='.pyc' and p.name not in ('smoke.pt','smoke.cbm') and 'chunks' not in p.parts:
                files[p.relative_to(ROOT).as_posix()]=p
    folder=ROOT/'releases';prefix=f'v{version}_multi_route_20261026';mp=folder/(prefix+'.manifest.json')
    deps=[dict(path='releases/'+n,sha256=sha(folder/n)) for n in ('v86_metric_ranking_20261025.tar.gz','v88_context_kernel_20261025.tar.gz','v92_full_kernel_20261025.tar.gz','v94_isolated_assay_20261025.tar.gz','v95_graph_specialists_20261026.tar.gz','v96_normalized_assays_20261026.tar.gz')]
    save(mp,dict(version=f'v{version}',files=[dict(path=n,sha256=sha(p),bytes=p.stat().st_size) for n,p in sorted(files.items())],dependencies=deps,test_labels_included=False,credentials_included=False))
    archive=folder/(prefix+'.tar.gz')
    assert not archive.exists(),'Archive already exists; do not overwrite a completed snapshot.'
    with tarfile.open(archive,'w:gz') as tf:
        for name,p in sorted(files.items()):tf.add(p,arcname=name)
        tf.add(mp,arcname=mp.relative_to(ROOT).as_posix())
    report=dict(path=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,files=len(files),version=f'v{version}',official_score=None)
    save(folder/(prefix+'.archive.json'),report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--version',type=int,choices=[97,99,101],default=101);args=ap.parse_args();package(args.version)
