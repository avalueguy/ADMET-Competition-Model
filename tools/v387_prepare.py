"""Freeze full correlated feature metric curves, preserving diagonal predecessor."""
from pathlib import Path
import hashlib,json,os,shutil,tarfile
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/regression9_v387'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 assert not(EXP/'source_freeze.json').exists();EXP.mkdir(exist_ok=True);prev=ROOT/'experiments/regression9_v386';manifest=json.loads((prev/'data_manifest.json').read_text())
 names=['official_rows.npz','expert_features.npz','stereo_keys.npz','submission_template.csv','v385_baseline.csv','data_manifest.json']+[r[k]for r in manifest['controls']for k in('control','design')]
 for n in names:
  src,dest=prev/n,EXP/n;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert sha(src)==sha(dest)
  else:os.link(src,dest)
 source=(ROOT/'tools/v384_recursive_median.py').read_text(encoding='utf-8')
 source=source.replace('diagonal recursive feature metrics','correlated recursive feature metrics').replace("f'laplace_C{int(c)}'","f'gaussian_full_C{int(c)}'")
 source=source.replace('uniform_shrink=.2','uniform_shrink=.5').replace('gradient_rows=512','gradient_rows=512,metric_rank=32')
 source=source.replace('metric=np.ones(x.shape[1])','metric=np.eye(x.shape[1])')
 start=source.index('def kernel(');end=source.index('def native(',start)
 source=source[:start]+'''def kernel(x,ref,metric,bandwidth):
 ref=torch.as_tensor(ref,device='cuda');metric=torch.as_tensor(metric,dtype=torch.float32,device='cuda')
 weighted=ref@metric;norm=(weighted*ref).sum(1);out=[]
 for lo in range(0,len(x),256):
  z=torch.as_tensor(x[lo:lo+256],device='cuda');dist=((z@metric*z).sum(1)[:,None]+norm[None]-2*z@weighted.T).clamp_min(0)
  out.append(torch.exp(-dist/(2*bandwidth**2)).cpu().numpy())
 return np.concatenate(out)

def gradient_moment(x,model,metric,bw):
 ix=np.linspace(0,len(x)-1,min(len(x),SETTINGS['gradient_rows']),dtype=int)
 ref=torch.as_tensor(x[model.support_],device='cuda');alpha=torch.as_tensor(model.dual_coef_[0],dtype=torch.float32,device='cuda')
 m=torch.as_tensor(metric,dtype=torch.float32,device='cuda');weighted=ref@m;norm=(ref*weighted).sum(1);gradients=[]
 for lo in range(0,len(ix),128):
  z=torch.as_tensor(x[ix[lo:lo+128]],device='cuda');dist=((z@m*z).sum(1)[:,None]+norm[None]-2*z@weighted.T).clamp_min(0)
  ratio=torch.exp(-dist/(2*bw*bw))*alpha[None]
  gradients.append(-(z*ratio.sum(1)[:,None]-ratio@ref)@m/(bw*bw))
 grad=torch.cat(gradients);cov=grad.T@grad/len(ix);cov=(cov+cov.T)*.5
 # Correlated PSD directions, bounded capacity; every direction retains identity.
 eig,vec=torch.linalg.eigh(cov.double());rank=min(SETTINGS['metric_rank'],len(eig));eig=eig[-rank:].clamp_min(0);vec=vec[:,-rank:]
 learned=(vec*eig[None])@vec.T;learned=learned/learned.trace().clamp_min(1e-12)*x.shape[1]
 identity=torch.eye(x.shape[1],dtype=torch.float64,device='cuda')
 proposal=SETTINGS['uniform_shrink']*identity+(1-SETTINGS['uniform_shrink'])*learned
 new=SETTINGS['metric_damping']*m.double()+(1-SETTINGS['metric_damping'])*proposal
 new=(new+new.T)*.5;new=new/new.trace()*x.shape[1]
 assert torch.isfinite(new).all();return new.cpu().numpy()

'''+source[end:]
 start=source.index('def freeze():');source=source[:start]+'''def freeze():
 save(OUT/'protocol.json',dict(signature=signature(),baseline='complete v385 associated official51.308',tasks=TASKS,bases=BASES,horizons=HORIZONS,settings=SETTINGS,weights=WEIGHTS,
  all_fitting_remote=True,external_data=False,external_weights=False,
  method='Gaussian kernel, full input-gradient covariance top32 directions plus50percent identity, half damped updates. Per-task correlated feature metric.',
  difference_from_v384='v384 diagonal Laplace geometry failed. Here smooth Gaussian gradient, correlated directions and stronger identity shrinkage are tested as one package; not isolated attribution of each change.',
  budget='9tasks x2 development partitions x2C x9 converged SVRs=324 development fits; every curve completes0..8, saved0/2/5/8; no early abandonment.',
  selection='Unique maxmin fold0/1 config/weight frozen before fold2/valid; no fallback. Rank tasks need four positive MAE and Spearman gains.',
  limitations='Not a published RFM benchmark reproduction; full covariance can overfit. Public data reused, historical proxy controls and calibrated rank references. No guaranteed score improvement.'))
'''
 (ROOT/'tools/v387_correlated_metric.py').write_text(source,encoding='utf-8')
 run=(ROOT/'tools/v384_run_experts.py').read_text(encoding='utf-8').replace('v384_recursive_median','v387_correlated_metric').replace('v384','v387').replace('384000','387000')
 (ROOT/'tools/v387_run_experts.py').write_text(run,encoding='utf-8')
 verify=(ROOT/'tools/v384_verify_experts.py').read_text(encoding='utf-8').replace('v384_recursive_median','v387_correlated_metric').replace('v384','v387')
 verify=verify.replace("('v382_baseline.csv','v382_baseline_sha256')","('v385_baseline.csv','baseline_csv_sha256')")
 verify=verify.replace("r['metric'].min()>0","np.linalg.eigvalsh(r['metric']).min()>0")
 (ROOT/'tools/v387_verify_experts.py').write_text(verify,encoding='utf-8')
 smoke=(ROOT/'tools/v384_smoke.py').read_text(encoding='utf-8').replace('v384_recursive_median','v387_correlated_metric');(ROOT/'tools/v387_smoke.py').write_text(smoke,encoding='utf-8')
 completion=(ROOT/'tools/v384_completion.py').read_text(encoding='utf-8').replace('v384','v387');(ROOT/'tools/v387_completion.py').write_text(completion,encoding='utf-8')
 sources=('v371_mae_experts.py','v377_regression_experts.py','v387_correlated_metric.py','v387_run_experts.py','v387_verify_experts.py','v387_smoke.py','v387_completion.py')
 entries=[]
 for n in sources:shutil.copyfile(ROOT/'tools'/n,EXP/n);entries.append(dict(name=n,sha256=sha(EXP/n)))
 save(EXP/'source_freeze.json',dict(files=entries,all_fitting_remote=True,baseline=manifest['baseline'],workers=2))
 archive=ROOT/'releases/v387_input_delta.tar.gz'
 with tarfile.open(archive,'w:gz',dereference=True)as tf:
  for n in sources+('source_freeze.json',):tf.add(EXP/n,arcname=n)
 receipt=dict(archive=archive.relative_to(ROOT).as_posix(),sha256=sha(archive),bytes=archive.stat().st_size,inputs='Unchanged full v385 baseline frozen in v386')
 save(ROOT/'docs/v387_input_package.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
