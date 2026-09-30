"""Checks both adaptive IE variants and their failure paths."""
import numpy as np
from models import TF
from solvers import adaptive_implicit_euler, adaptive_implicit_euler_kernel, _implicit_euler_step
from experiments import reference
from diagnostics import e_inf

def main():
    ref=reference('Radau',1e-12,1e-14)
    results={}
    for extrapolate in (False,True):
        errors=[]
        for tol in (1e-4,1e-6,1e-8):
            t,Y,ratio,rejected,updates=adaptive_implicit_euler(tol,extrapolate)
            assert t[-1]==TF and np.all(np.diff(t)>0)
            assert np.all(np.isfinite(Y)) and np.min(Y)>=-1e-12
            assert ratio.max()<=1 and np.max(np.abs(Y.sum(axis=1)-1))<1e-10
            errors.append(e_inf(t,Y,ref))
            for k in np.unique(np.linspace(0,len(t)-2,12).astype(int)):
                h=t[k+1]-t[k]
                full,ok1,_=_implicit_euler_step(Y[k],h,1e-13,50)
                mid,ok2,_=_implicit_euler_step(Y[k],h/2,1e-13,50)
                two,ok3,_=_implicit_euler_step(mid,h/2,1e-13,50)
                assert ok1 and ok2 and ok3
                expected=2*two-full if extrapolate else two
                np.testing.assert_allclose(expected,Y[k+1],rtol=0,atol=2e-13)
                np.testing.assert_allclose(np.max(np.abs(two-full))/tol,ratio[k],rtol=1e-4,atol=2e-5)
        assert errors[2]<errors[1]<errors[0]
        results[extrapolate]=errors
    assert np.all(np.array(results[True])<np.array(results[False]))
    out=adaptive_implicit_euler_kernel(1e-14,True,1e-13,1.,1.,1.)
    assert out[-1]==2 and len(out[0])==1
    out=adaptive_implicit_euler_kernel(1e-6,max_accepted=1)
    assert out[-1]==1
    _,ok,_=_implicit_euler_step(np.array([1.,0.,0.]),1.,1e-13,0)
    assert not ok
    for tol in (0.,-1.,np.nan):
        try: adaptive_implicit_euler(tol)
        except ValueError: pass
        else: raise AssertionError('invalid tolerance accepted')
    # Smooth-region fixed-step refinement checks extrapolation's formal order.
    errors=[]
    for h in (.1,.05,.025):
        y=ref.sol(1.).copy()
        for _ in range(round(1/h)):
            full,ok1,_=_implicit_euler_step(y,h,1e-13,50)
            mid,ok2,_=_implicit_euler_step(y,h/2,1e-13,50)
            two,ok3,_=_implicit_euler_step(mid,h/2,1e-13,50)
            assert ok1 and ok2 and ok3
            y=2*two-full
        errors.append(np.linalg.norm(y-ref.sol(2.)))
    orders=np.log2(np.array(errors[:-1])/errors[1:])
    assert np.all((orders>1.8)&(orders<2.2)),orders
    print('ADAPTIVE IMPLICIT TESTS PASSED',results,'smooth-region orders',orders)

if __name__=='__main__': main()
