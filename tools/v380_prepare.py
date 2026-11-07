"""Freeze new kernel comparison and reuse immutable official inputs, no fitting."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/regression9_v380'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (EXP/'source_freeze.json').exists(),'Preserve frozen experiment'
    EXP.mkdir(parents=True,exist_ok=True)
    source=(ROOT/'tools/v379_run_experts.py').read_text(encoding='utf-8')
    source=source.replace('v379_distribution_transformer','v380_histogram_median').replace('379000','380000').replace('v379_verify_experts','v380_verify_experts')
    start=source.index('  # Independent fixed curves')
    stop=source.index('  records=[];selected=[]',start)
    source=source[:start]+'''  from concurrent.futures import ThreadPoolExecutor,as_completed
  jobs=[(p,t) for p in ('fold0','fold1') for t in m.TASKS]
  logs=ROOT/'prefit_logs';logs.mkdir(exist_ok=True)
  def fit(job):
   p,t=job
   code="import v380_histogram_median as m; m.setup(); d=m.load(); [m.predict(d,%r,%r,c) for c in m.configs(%r)]"%(t,p,t)
   with (logs/(p+'_'+t+'.log')).open('w') as log:
    subprocess.run([sys.executable,'-u','-c',code],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
   return job
  with ThreadPoolExecutor(max_workers=3) as pool:
   for future in as_completed([pool.submit(fit,j) for j in jobs]):
    job=future.result();done+=1;m.emit('task_job_done',completed=done,total=len(jobs),job=job)
''' +source[stop:]
    source=source.replace('for c in m.CONFIGS:', 'for c in m.configs(t):')
    source=source.replace("-r['config']['rounds'],-list(m.CONFIGS).index(r['config'])", "-list(m.configs(t)).index(r['config'])")
    (ROOT/'tools/v380_run_experts.py').write_text(source,encoding='utf-8')
    source=(ROOT/'tools/v379_verify_experts.py').read_text(encoding='utf-8')
    source=source.replace('v379_distribution_transformer','v380_histogram_median').replace('v379_run_experts','v380_run_experts').replace('v379_verify_experts','v380_verify_experts')
    source=source.replace('for c in m.CONFIGS:',"for c in m.configs(r['task']):")
    source=source.replace("-x['config']['rounds'],-list(m.CONFIGS).index(x['config'])", "-list(m.configs(r['task'])).index(x['config'])")
    source=source.replace("meta['model_path']","str(path.with_name('state.joblib').relative_to(ROOT))")
    source=source.replace("p=m.OUT/'curves'/r['task']/'final'/r['config']['base']/'state.joblib'", "p=m.OUT/'tasks'/r['task']/'final'/r['config']['name']/'state.joblib'")
    # Regenerate small kernel blocks with query labels perturbed, independent of cache.
    source=source.replace(" records=[];maximum=0.",''' kernel_checks=[]
 for task in m.TASKS:
  for part in ('fold0','fold1'):
   a=m.design(d,task,part)
   small={**a,'fit':a['fit'][:40],'query':a['query'][:8],'x':a['x'][:40],'qx':a['qx'][:8]}
   k,q=m.matrices(d,small)
   changed={**d,'y':d['y'].copy()};changed['y'][a['query']]=987654.
   kk,qq=m.matrices(changed,small)
   for name in k:
    np.testing.assert_array_equal(k[name],kk[name]);np.testing.assert_array_equal(q[name],qq[name])
    assert np.linalg.eigvalsh(k[name].astype(float)).min()>-1e-4
   kernel_checks.append(dict(task=task,part=part,psd_small_block=True,query_labels_not_in_kernels=True))
 records=[];maximum=0.''')
    source=source.replace('no_fitting_performed=True)', 'no_fitting_performed=True,kernel_checks=kernel_checks)')
    (ROOT/'tools/v380_verify_experts.py').write_text(source,encoding='utf-8')
    previous=ROOT/'experiments/regression9_v377'
    manifest=json.loads((previous/'data_manifest.json').read_text(encoding='utf-8'))
    names=['official_rows.npz','expert_features.npz','stereo_keys.npz','submission_template.csv','v374_baseline.csv','data_manifest.json']
    names += [row[k] for row in manifest['controls'] for k in ('control','design')]
    for name in names:
        src,dest=previous/name,EXP/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert sha(src)==sha(dest)
        else:os.link(src,dest)
    sources=('v371_mae_experts.py','v377_regression_experts.py','v380_histogram_median.py',
             'v380_run_experts.py','v380_verify_experts.py','v380_smoke.py')
    entries=[]
    for name in sources:
        shutil.copyfile(ROOT/'tools'/name,EXP/name)
        entries.append(dict(name=name,sha256=sha(EXP/name)))
    freeze=dict(files=entries,official_best=51.271,all_fitting_remote=True,workers=3,
                immutable_inherited_inputs='regression9_v377',no_external_data=True)
    (EXP/'source_freeze.json').write_text(json.dumps(freeze,indent=2)+'\n',encoding='utf-8')
    archive=ROOT/'releases/v380_histogram_delta.tar.gz'
    with tarfile.open(archive,'w:gz') as tf:
        for name in sources+('source_freeze.json',):tf.add(EXP/name,arcname=name)
    receipt=dict(status='PASS',archive=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),
                 bytes=archive.stat().st_size,source_files=entries,local_input_files_verified=len(names))
    (ROOT/'docs/v380_input_package.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
