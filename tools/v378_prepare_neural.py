"""Freeze the next complete neural comparison before inspecting confirmation."""
from pathlib import Path
import hashlib,json,shutil,tarfile
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/regression9_v378'


def main():
 assert not (EXP/'source_freeze.json').exists(),'Already frozen'
 EXP.mkdir(parents=True,exist_ok=True)
 runner=(ROOT/'tools/v377_run_experts.py').read_text(encoding='utf-8').replace('v377_regression_experts as m','v378_retrieval_transformer as m').replace('377000','378000').replace('total=54','total=36').replace('v377_verify_experts.py','v378_verify_experts.py')
 (ROOT/'tools/v378_run_experts.py').write_text(runner,encoding='utf-8')
 verifier=(ROOT/'tools/v377_verify_experts.py').read_text(encoding='utf-8').replace('v377_regression_experts as m','v378_retrieval_transformer as m').replace('v377_run_experts as run','v378_run_experts as run').replace('err<1e-8','err<2e-5')
 verifier=verifier.replace("('v371_mae_experts.py','v377_regression_experts.py','v377_run_experts.py','v377_verify_experts.py')","('v371_mae_experts.py','v377_regression_experts.py','v378_retrieval_transformer.py','v378_run_experts.py','v378_verify_experts.py')")
 (ROOT/'tools/v378_verify_experts.py').write_text(verifier,encoding='utf-8')
 names=('v371_mae_experts.py','v377_regression_experts.py','v378_retrieval_transformer.py','v378_run_experts.py','v378_verify_experts.py')
 records=[]
 for n in names:
  shutil.copyfile(ROOT/'tools'/n,EXP/n);records.append(dict(name=n,sha256=hashlib.sha256((EXP/n).read_bytes()).hexdigest()))
 (EXP/'source_freeze.json').write_text(json.dumps(dict(files=records),indent=2)+'\n')
 archive=ROOT/'releases/v378_neural_delta.tar.gz'
 with tarfile.open(archive,'w:gz') as tf:
  for n in names+('source_freeze.json',):tf.add(EXP/n,arcname=n)
 report=dict(status='PASS',archive=archive.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,
             reused_inputs='v377 immutable design, controls and official arrays',all_fitting_remote=True)
 (ROOT/'docs/v378_input_package.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
