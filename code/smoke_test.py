#!/usr/bin/env python3
"""Fast sanity test: imports, four solvers, reference, and Newton residual criterion."""
from pathlib import Path
import sys, numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from experiments import reference
from solvers import warm_up, explicit_euler, rk4, implicit_euler, trapezoidal, adaptive_euler, adaptive_implicit_euler
from diagnostics import e_inf

warm_up()
ref=reference('Radau',1e-12,1e-14)
checks=[]
t,Y=explicit_euler(5e-4); checks.append(('Explicit Euler',e_inf(t,Y,ref),len(t)-1))
t,Y=rk4(8e-4); checks.append(('RK4',e_inf(t,Y,ref),len(t)-1))
t,Y,nit,res=implicit_euler(1e-2,1e-13); checks.append(('Implicit Euler',e_inf(t,Y,ref),len(t)-1))
assert np.max(res)<=1.1e-13
t,Y,nit,res=trapezoidal(1e-3,1e-13); checks.append(('Trapezoidal/CN',e_inf(t,Y,ref),len(t)-1))
assert np.max(res)<=1.1e-13
t,Y,rej=adaptive_euler(1e-6); checks.append(('Adaptive Euler',e_inf(t,Y,ref),len(t)-1))
for extrapolate in (False, True):
    t,Y,ratio,rej,upd=adaptive_implicit_euler(1e-6,extrapolate)
    assert ratio.max()<=1
    checks.append((f'Adaptive IE ({extrapolate=})',e_inf(t,Y,ref),len(t)-1))
for row in checks: print(f'{row[0]:17s} E_inf={row[1]:.6e}, accepted/fixed steps={row[2]}')
print('SMOKE TEST PASSED')


from solvers import adaptive_implicit_euler
t, Y, ratio, rej, nupd = adaptive_implicit_euler(1e-6, True)
E = e_inf(t, Y, ref)
assert E < 1e-6 and ratio.max() <= 1.0 and np.abs(Y.sum(1) - 1).max() < 1e-12 and Y.min() >= -1e-12