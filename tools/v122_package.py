"""Snapshot this iteration, models and rejection evidence without credentials."""
import hashlib,json,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PREFIX='v122_models_history_20261027'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run():
    report=json.loads((ROOT/'admet_model_outputs_v122/evidence.json').read_text())
    assert report['status']=='VERIFIED_CANDIDATE_AWAITING_PLATFORM' and sha(ROOT/report['file'])==report['sha256']
    assert json.loads((ROOT/'admet_model_outputs_v119/independent_verification.json').read_text())['status']=='PASS'
    folders=[ROOT/f'admet_model_outputs_v{n}' for n in (115,116,117,118,119,121,122)]
    folders += [ROOT/'admet_submissions/v122_matched_models_history',ROOT/'admet_submissions/v114_herg_history52',ROOT/'admet_submissions/v112_history_expansion']
    paths=[]
    for folder in folders:paths.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix not in ('.tmp','.pyc'))
    for n in (115,116,117,118,119,120,121,122):
        paths.extend((ROOT/'tools').glob(f'v{n}_*.py'));paths.extend(ROOT.glob(f'v{n}*.log'))
    paths.extend([ROOT/'tools/verify_submission.py',ROOT/'tools/repack_upload_template.py',ROOT/'v118_original_manifest.json',
        ROOT/'docs/platform_scorecard_v114_20261027.json',ROOT/'docs/v115_optimization_gap.json'])
    # Exact incremental baseline arrays required to replay the carried branch.
    paths.extend([ROOT/'admet_model_outputs_v67'/part/'shared'/str(seed)/'predictions.npz'
        for part in ('valid','final','cv/fold0','cv/fold1','cv/fold2') for seed in (672031,672032)])
    paths.extend([ROOT/'admet_model_outputs_v68/half_life_obach'/part/'predictions.npz'
        for part in ('valid_report','final','fold0','fold1','fold2')])
    files={p.relative_to(ROOT).as_posix():p for p in paths if p.is_file()}
    for n in (115,116,117,119,121):
        c=json.loads((ROOT/f'admet_model_outputs_v{n}/confirmation.json').read_text());assert c['status']=='COMPLETE'
    mp=ROOT/'releases'/(PREFIX+'.manifest.json');mp.parent.mkdir(exist_ok=True)
    save(mp,dict(version='v122',files=[dict(path=n,bytes=p.stat().st_size,sha256=sha(p)) for n,p in sorted(files.items())],
        dependencies_note='Historical baseline/features and source import chain use the existing v101 and earlier verified archives. This snapshot adds all current models and matched half-life baseline arrays.',
        credentials_included=False,test_label_files_included=False,official_score=None))
    archive=mp.with_name(PREFIX+'.tar.gz');assert not archive.exists()
    with tarfile.open(archive,'w:gz') as tf:
        for name,p in sorted(files.items()):tf.add(p,arcname=name)
        tf.add(mp,arcname=mp.relative_to(ROOT).as_posix())
    info=dict(path=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,files=len(files),manifest_sha256=sha(mp),official_score=None)
    save(mp.with_name(PREFIX+'.archive.json'),info);print(json.dumps(info),flush=True)
if __name__=='__main__':run()
