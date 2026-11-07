"""Audit finished process and archive all selected source states for replay."""
from pathlib import Path
import json,hashlib,os,subprocess,tarfile,time
ROOT=Path('/root/autodl-tmp/regression9_v389')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    status=read(ROOT/'run_status.json');verification=read(ROOT/'verification.json');assert status['stage']=='EVIDENCE_READY'and verification['status']=='PASS'
    plan=read(ROOT/'plan.json');assert sha(ROOT/'v389_combination.py')==plan['source_sha256']
    for path in Path('/proc').iterdir():
        if not path.name.isdigit()or int(path.name)==os.getpid():continue
        try:state=(path/'stat').read_text().rsplit(')',1)[1].split()[0];cmd=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except(FileNotFoundError,PermissionError,ProcessLookupError):continue
        assert state=='Z'or not(int(path.name)==status['pid']or'v389_combination.py'in cmd),(path.name,cmd)
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not compute
    files=[];model_counts={}
    for r in read(ROOT/'confirmation.json')['records']:
        if not r['experimental_eligible']:continue
        t=r['task']
        for v in('v386','v387'):
            source=ROOT.parent/f'regression9_{v}';config=plan['fixed_experts'][t][v]
            files+=[source/'data_manifest.json',source/'source_freeze.json']
            files += [source/x['name']for x in read(source/'source_freeze.json')['files']]
            if v=='v386':
                files += list((source/'outputs/base'/t/'final').glob('*/state.joblib'))
                files += [source/'outputs/oof'/t/'final/state.joblib',source/'outputs/tasks'/t/'final'/config['name']/'state.joblib']
            else:files += [source/'outputs/curves'/t/'final'/config['base']/'state.joblib']
    files=sorted(set(files));entries=[dict(path=p.relative_to(ROOT.parent).as_posix(),sha256=sha(p),bytes=p.stat().st_size)for p in files]
    for v in('v386','v387'):
        source=ROOT.parent/f'regression9_{v}';model_counts[v]=dict(base_packages=len(list((source/'outputs/base').glob('*/*/*/state.joblib'))),
            curve_packages=len(list((source/'outputs/curves').glob('*/*/*/state.joblib'))),prediction_packages=len(list((source/'outputs/tasks').glob('*/*/*/predictions.npz'))))
    manifest=ROOT/'selected_models_manifest.json';manifest.write_text(json.dumps(dict(files=entries,model_package_inventory=model_counts),indent=2)+'\n')
    archive=ROOT/'selected_models.tar.gz'
    with tarfile.open(archive,'w:gz',dereference=True)as tf:
        for p in files:tf.add(p,arcname=p.relative_to(ROOT.parent).as_posix())
    receipt=dict(status='PASS',at_unix=time.time(),no_training_processes_running=True,no_gpu_compute_processes=True,
        source_sha256=plan['source_sha256'],selected_models_archive_sha256=sha(archive),selected_model_files=len(entries),
        source_model_inventory=model_counts,
        verification_correction='Frozen runner new_training_fits=False refers to coefficient arithmetic only. Source model generation can perform additional original frozen fits; inventory provided separately, not claimed zero new fits.')
    (ROOT/'completion_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
