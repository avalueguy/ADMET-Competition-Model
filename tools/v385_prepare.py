"""Reuse v382 inputs unchanged; freeze a distinct target-warp experiment."""
from pathlib import Path
import hashlib,json,os,shutil,tarfile
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/regression9_v385'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 assert not(EXP/'source_freeze.json').exists();EXP.mkdir(exist_ok=True);prev=ROOT/'experiments/regression9_v384'
 manifest=json.loads((prev/'data_manifest.json').read_text());names=['official_rows.npz','expert_features.npz','stereo_keys.npz','submission_template.csv','v382_baseline.csv','data_manifest.json']
 names += [r[k]for r in manifest['controls']for k in('control','design')]
 for n in names:
  src,dest=prev/n,EXP/n;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert sha(src)==sha(dest)
  else:os.link(src,dest)
 runner=(ROOT/'tools/v381_run_experts.py').read_text().replace('v381_oof_pareto','v385_warped_median').replace('v381_verify_experts','v385_verify_experts').replace('381000','385000')
 (ROOT/'tools/v385_run_experts.py').write_text(runner,encoding='utf-8')
 completion=(ROOT/'tools/v381_completion.py').read_text().replace('v381','v385');(ROOT/'tools/v385_completion.py').write_text(completion,encoding='utf-8')
 sources=('v371_mae_experts.py','v377_regression_experts.py','v380_histogram_median.py','v385_warped_median.py','v385_run_experts.py','v385_verify_experts.py','v385_smoke.py','v385_completion.py')
 entries=[]
 for n in sources:shutil.copyfile(ROOT/'tools'/n,EXP/n);entries.append(dict(name=n,sha256=sha(EXP/n)))
 save(EXP/'source_freeze.json',dict(files=entries,baseline=manifest['baseline'],all_fitting_remote=True,workers=2))
 archive=ROOT/'releases/v385_input_delta.tar.gz'
 with tarfile.open(archive,'w:gz')as tf:
  for n in sources+('source_freeze.json',):tf.add(EXP/n,arcname=n)
 receipt=dict(archive=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,inherited_inputs='regression9_v384, unchanged complete v382 controls')
 save(ROOT/'docs/v385_input_package.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
