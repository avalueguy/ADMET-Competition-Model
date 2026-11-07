"""Independent algebra checks and source freeze audit, without model fitting."""
from pathlib import Path
import hashlib,json,ast
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 checked=[]
 for version in('v384','v385'):
  exp=ROOT/f'experiments/regression9_{version}';freeze=json.loads((exp/'source_freeze.json').read_text())
  for r in freeze['files']:
   assert sha(exp/r['name'])==r['sha256']==sha(ROOT/'tools'/r['name'])
   ast.parse((exp/r['name']).read_text(encoding='utf-8'));checked.append(dict(version=version,source=r['name'],sha256=r['sha256']))
 rng=np.random.default_rng(385);x=rng.normal(size=7);refs=rng.normal(size=(11,7));m=rng.uniform(.2,3,size=7);alpha=rng.normal(size=11);bw=1.3
 def f(z):
  dist=np.sqrt(((z-refs)**2*m).sum(1));return float(np.exp(-dist/bw)@alpha)
 delta=x-refs;dist=np.sqrt((delta**2*m).sum(1));analytic=-(np.exp(-dist/bw)*alpha/dist)@(delta*m)/bw
 eps=1e-5;numeric=np.array([(f(x+np.eye(7)[j]*eps)-f(x-np.eye(7)[j]*eps))/(2*eps)for j in range(7)])
 error=float(np.max(np.abs(analytic-numeric)));assert error<1e-8
 # PPBR pseudocount logit and inverse are analytic inverses, including endpoints.
 y=np.array([0.,.1,10.,50.,90.,99.95,100.]);z=np.log((y+.5)/(100-y+.5));inverse=101/(1+np.exp(-z))-.5
 np.testing.assert_allclose(inverse,y,rtol=0,atol=1e-12)
 score=json.loads((ROOT/'docs/platform_scorecard_v382_20261105.json').read_text());score_errors=[]
 for r in score['tasks']:
  if r['metric']=='MAE':score_errors.append(abs(100*r['reference_score']/(r['reference_score']+r['raw'])-r['score']))
 assert max(score_errors)<.00000051
 receipt=dict(status='PASS',no_fitting_performed=True,source_records=checked,laplace_gradient_finite_difference_max_error=error,
  bounded_logit_exact_roundtrip=True,MAE_score_formula_max_rounding_error=max(score_errors),
  scope='Algebra verification on synthetic fixed coefficients, no hyperparameter selection or model fit.')
 (ROOT/'docs/v385_math_audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
 failure=dict(status='RECORDED_TEST_FAILURE',phase='Predevelopment smoke',
  original_smoke_sha256=sha(ROOT/'tools/v385_smoke.py'),
  observed_failure='AssertionError on fixed abs(rankmap_inverse(rankmap(y))-y)<1.5, first sqrt_gaussian_rank_C3 smoke.',
  cause='Finite midranks mapped through the linear empirical quantile function do not round-trip exactly; discrepancy depends on label gaps.',
  correction='Separate smoke amendment checks monotonicity and bounds for empirical rank transform, exact inverse for analytic warps.',
  training_source_changed=False,outer_scores_computed_before_amendment=False,
  record_scope='Compact reconstruction from observed bootstrap error; not a saved verbatim original stderr stream.')
 (ROOT/'docs/v385_smoke_failure.json').write_text(json.dumps(failure,indent=2)+'\n')
 print(json.dumps({k:v for k,v in receipt.items()if k!='source_records'}))
if __name__=='__main__':main()
