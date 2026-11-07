"""Convex monotone piecewise-linear native-MAE calibration; remote fit only."""
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
def interpolation(x,knots):
 # Linear extrapolation with fixed slope1 beyond boundary, monotone inside.
 k=np.searchsorted(knots,x,side='right')-1;k=np.clip(k,0,len(knots)-2)
 z=np.clip(x,knots[0],knots[-1]);w=(z-knots[k])/(knots[k+1]-knots[k])
 a=np.zeros((len(x),len(knots)));a[np.arange(len(x)),k]=1-w;a[np.arange(len(x)),k+1]=w
 offset=x-z
 return a,offset
def problem(pred,y,penalty):
 center=float(np.median(y));scale=max(float(np.subtract(*np.quantile(y,[.75,.25]))),.05)
 x=(pred-center)/scale;yy=(y-center)/scale
 knots=np.unique(np.quantile(x,np.linspace(0,1,9)))
 if len(knots)<2:knots=np.array([x[0]-.05,x[0]+.05])
 a,offset=interpolation(x,knots);n,j=a.shape;total=j+n+j
 cost=np.r_[np.zeros(j),np.full(n,1/n),np.full(j,penalty/j)]
 A=sparse.lil_matrix((2*n+2*j+2*(j-1),total));rhs=np.zeros(A.shape[0])
 A[:n,:j]=a;A[:n,j:j+n]=-sparse.eye(n);rhs[:n]=yy-offset
 A[n:2*n,:j]=-a;A[n:2*n,j:j+n]=-sparse.eye(n);rhs[n:2*n]=-yy+offset
 for i in range(j):
  A[2*n+i,i]=1;A[2*n+i,j+n+i]=-1;rhs[2*n+i]=knots[i]
  A[2*n+j+i,i]=-1;A[2*n+j+i,j+n+i]=-1;rhs[2*n+j+i]=-knots[i]
 for i,gap in enumerate(np.diff(knots)):
  row=2*n+2*j+2*i
  A[row,i]=1;A[row,i+1]=-1;rhs[row]=-.05*gap
  A[row+1,i]=-1;A[row+1,i+1]=1;rhs[row+1]=5*gap
 bounds=[(v-.5,v+.5)for v in knots]+[(0,None)]*(n+j)
 return dict(cost=cost,A=A.tocsr(),rhs=rhs,bounds=bounds,knots=knots,center=center,scale=scale,n=n,j=j)
def fit(pred,y,penalty):
 import platform,torch
 assert platform.system()=='Linux'and torch.cuda.is_available(),'Fit only on designated remote CUDA host'
 z=problem(pred,y,penalty);r=linprog(z['cost'],A_ub=z['A'],b_ub=z['rhs'],bounds=z['bounds'],method='highs');assert r.success,r.message
 s=dict(knots=z['knots'],values=r.x[:z['j']],center=z['center'],scale=z['scale'],solution=r.x,objective=float(r.fun),
  dual=r.ineqlin.marginals,lower_dual=r.lower.marginals,upper_dual=r.upper.marginals,penalty=penalty)
 certificate(pred,y,s);return s
def apply(pred,s):
 a,offset=interpolation((pred-s['center'])/s['scale'],s['knots']);return(a@s['values']+offset)*s['scale']+s['center']
def certificate(pred,y,s):
 z=problem(pred,y,s['penalty']);v=s['solution'];res=z['rhs']-z['A']@v
 np.testing.assert_array_equal(z['knots'],s['knots']);assert res.min()>-2e-6 and abs(z['cost']@v-s['objective'])<1e-7
 lower=np.array([b[0]for b in z['bounds']]);upper=np.array([b[1]if b[1]is not None else np.inf for b in z['bounds']]);finite=np.isfinite(upper)
 assert np.all(v>=lower-1e-7)and np.all(v[finite]<=upper[finite]+1e-7)
 stationarity=z['cost']-z['A'].T@s['dual']-s['lower_dual']-s['upper_dual']
 assert abs(stationarity).max()<2e-6 and abs(res*s['dual']).max()<2e-6
 assert abs((v-lower)*s['lower_dual']).max()<2e-6 and np.max(abs((upper[finite]-v[finite])*s['upper_dual'][finite]),initial=0)<2e-6
 assert s['dual'].max(initial=0)<1e-7 and s['lower_dual'].min(initial=0)>-1e-7 and s['upper_dual'].max(initial=0)<1e-7
 return dict(primal_feasible=True,dual_stationarity=True,complementarity=True,no_fitting_performed=True)
