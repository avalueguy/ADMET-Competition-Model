"""Check fixed-coefficient calibration/kernel algebra locally, without fitting."""
from pathlib import Path
import ast, hashlib, json
import numpy as np
import v386_calibration_core as calibration

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    checked=[]
    for v in('v386','v387'):
        exp=ROOT/f'experiments/regression9_{v}'
        for r in json.loads((exp/'source_freeze.json').read_text(encoding='utf-8'))['files']:
            assert sha(exp/r['name'])==r['sha256']==sha(ROOT/'tools'/r['name'])
            ast.parse((exp/r['name']).read_text(encoding='utf-8'));checked.append(dict(version=v,**r))
    rng=np.random.default_rng(387);x=rng.normal(size=7);refs=rng.normal(size=(11,7));a=rng.normal(size=(7,7));metric=a.T@a+.5*np.eye(7);alpha=rng.normal(size=11);bw=1.3
    delta=x-refs;dist=np.einsum('ni,ij,nj->n',delta,metric,delta)
    weights=np.exp(-dist/(2*bw*bw))*alpha
    analytic=-(x*weights.sum()-weights@refs)@metric/(bw*bw)
    def f(z):
        diff=z-refs;d=np.einsum('ni,ij,nj->n',diff,metric,diff)
        return np.exp(-d/(2*bw*bw))@alpha
    eps=1e-5;numeric=np.array([(f(x+np.eye(7)[j]*eps)-f(x-np.eye(7)[j]*eps))/(2*eps)for j in range(7)])
    error=float(abs(analytic-numeric).max());assert error<1e-8
    knots=np.array([-3.,-1.,0.,2.,4.]);values=knots+np.array([-.1,.1,.12,.1,-.1])
    s=dict(knots=knots,values=values,scale=2.3,center=10.)
    grid=np.linspace(-20,40,801);z=calibration.apply(grid,s);assert np.diff(z).min()>0
    identity=calibration.apply(grid,dict(s,values=knots));np.testing.assert_allclose(identity,grid,rtol=0,atol=1e-14)
    np.testing.assert_allclose(calibration.apply(knots*s['scale']+s['center'],s),values*s['scale']+s['center'],rtol=0,atol=1e-14)
    outer=np.array([-100.,100.]);mapped=calibration.apply(outer,s)
    np.testing.assert_allclose(calibration.apply(outer+1e-3,s)-mapped,[1e-3,1e-3],rtol=0,atol=1e-12)
    official=json.loads((ROOT/'docs/platform_scorecard_v385_20261106.json').read_text(encoding='utf-8'))
    errors=[abs(100*r['reference_score']/(r['reference_score']+r['raw'])-r['score'])for r in official['tasks']if r['metric']=='MAE']
    assert max(errors)<5.1e-7
    result=dict(status='PASS',no_fitting_performed=True,source_records=checked,
        gaussian_full_metric_gradient_finite_difference_max_error=error,calibration_identity_and_knots_and_monotonicity_pass=True,
        calibration_outer_fixed_slope_one_pass=True,MAE_score_formula_max_rounding_error=max(errors))
    (ROOT/'docs/v387_math_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items()if k!='source_records'}))


if __name__=='__main__':main()
