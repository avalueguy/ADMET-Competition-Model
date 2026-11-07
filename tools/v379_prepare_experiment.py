"""Local byte packaging only; freeze v379 sources before any model fit."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/regression9_v379'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (EXP/'source_freeze.json').exists(),'Preserve frozen experiment'
    EXP.mkdir(parents=True,exist_ok=True)
    runner=(ROOT/'tools/v378_run_experts.py').read_text(encoding='utf-8')
    runner=runner.replace('v378_retrieval_transformer as m','v379_distribution_transformer as m').replace('378000','379000').replace('v378_verify_experts.py','v379_verify_experts.py')
    # Use exactly the native-unit output that will be submitted. A different
    # rank-fusion output must not inherit MAE measurements from this blend.
    runner=runner.replace("official=(1-w)*rankdata(a['score_p'])/len(y)+w*rankdata(expert)/len(y)","official=new")
    start=runner.index("  for part in ('fold0','fold1'):")
    end=runner.index('  records=[];selected=[]',start)
    runner=runner[:start]+'''  # Independent fixed curves fit concurrently on the same authorized GPU.
  # The parent evaluates nothing until all complete; no cache path is shared.
  from concurrent.futures import ThreadPoolExecutor,as_completed
  jobs=[(p,t,b['name']) for p in ('fold0','fold1') for t in m.TASKS for b in m.BASES]
  logs=ROOT/'prefit_logs';logs.mkdir(exist_ok=True)
  def fit(job):
   p,t,name=job
   code="import v379_distribution_transformer as m; m.setup(); d=m.load(); m.train_curve(d,%r,%r,next(b for b in m.BASES if b['name']==%r))"%(t,p,name)
   with (logs/(p+'_'+t+'_'+name+'.log')).open('w') as log:
    subprocess.run([sys.executable,'-u','-c',code],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
   return job
  with ThreadPoolExecutor(max_workers=6) as pool:
   for future in as_completed([pool.submit(fit,j) for j in jobs]):
    job=future.result();done+=1;m.emit('curve_job_done',completed=done,total=len(jobs),job=job)
''' +runner[end:]
    (ROOT/'tools/v379_run_experts.py').write_text(runner,encoding='utf-8')
    verifier=(ROOT/'tools/v378_verify_experts.py').read_text(encoding='utf-8')
    verifier=verifier.replace('v378_retrieval_transformer','v379_distribution_transformer').replace('v378_run_experts','v379_run_experts').replace('v378_verify_experts','v379_verify_experts')
    (ROOT/'tools/v379_verify_experts.py').write_text(verifier,encoding='utf-8')
    previous=ROOT/'experiments/regression9_v377'
    manifest=json.loads((previous/'data_manifest.json').read_text(encoding='utf-8'))
    names=['official_rows.npz','expert_features.npz','stereo_keys.npz','submission_template.csv','v374_baseline.csv','data_manifest.json']
    names += [row[k] for row in manifest['controls'] for k in ('control','design')]
    for name in names:
        src,dest=previous/name,EXP/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha(src)==sha(dest)
        else:os.link(src,dest)
    sources=('v371_mae_experts.py','v377_regression_experts.py','v379_distribution_transformer.py',
             'v379_run_experts.py','v379_verify_experts.py','v379_smoke.py')
    records=[]
    for name in sources:
        shutil.copyfile(ROOT/'tools'/name,EXP/name)
        records.append(dict(name=name,sha256=sha(EXP/name)))
    freeze=dict(files=records,official_baseline=51.271,all_fitting_remote=True,
                inputs_reused_immutable='regression9_v377',workers=6)
    (EXP/'source_freeze.json').write_text(json.dumps(freeze,indent=2)+'\n',encoding='utf-8')
    archive=ROOT/'releases/v379_distribution_delta.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for name in sources+('source_freeze.json',):tf.add(EXP/name,arcname=name)
    receipt=dict(status='PASS',archive=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),
                 bytes=archive.stat().st_size,local_input_files_verified=len(names),source_files=records,all_fitting_remote=True)
    (ROOT/'docs/v379_input_package.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
