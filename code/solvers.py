#!/usr/bin/env python3
from __future__ import annotations
import math
import numpy as np
from numba import njit
from models import T0, TF, Y0, rhs_numba, jac_numba

@njit(cache=True)
def explicit_euler_kernel(h: float, blowup_limit: float = 1.0e12):
    n = int(math.ceil((TF - T0) / h))
    t = np.empty(n + 1, dtype=np.float64)
    Y = np.empty((n + 1, 3), dtype=np.float64)
    t[0] = T0
    Y[0, 0], Y[0, 1], Y[0, 2] = Y0[0], Y0[1], Y0[2]
    used = n + 1
    for k in range(n):
        step = h
        if t[k] + step > TF:
            step = TF - t[k]
        t[k + 1] = t[k] + step
        Y[k + 1] = Y[k] + step * rhs_numba(Y[k])
        if (not np.isfinite(Y[k + 1]).all()) or np.max(np.abs(Y[k + 1])) > blowup_limit:
            used = k + 2
            break
    return t[:used], Y[:used]

@njit(cache=True)
def rk4_kernel(h: float, blowup_limit: float = 1.0e12):
    n = int(math.ceil((TF - T0) / h))
    t = np.empty(n + 1, dtype=np.float64)
    Y = np.empty((n + 1, 3), dtype=np.float64)
    t[0] = T0
    Y[0, 0], Y[0, 1], Y[0, 2] = Y0[0], Y0[1], Y0[2]
    used = n + 1
    for k in range(n):
        step = h
        if t[k] + step > TF:
            step = TF - t[k]
        y = Y[k]
        k1 = rhs_numba(y)
        k2 = rhs_numba(y + 0.5 * step * k1)
        k3 = rhs_numba(y + 0.5 * step * k2)
        k4 = rhs_numba(y + step * k3)
        t[k + 1] = t[k] + step
        Y[k + 1] = y + (step / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
        if (not np.isfinite(Y[k + 1]).all()) or np.max(np.abs(Y[k + 1])) > blowup_limit:
            used = k + 2
            break
    return t[:used], Y[:used]

@njit(cache=True)
def implicit_euler_kernel(h: float, newton_tol: float = 1.0e-10, newton_maxiter: int = 50, guess_mode: int = 0):
    n = int(math.ceil((TF - T0) / h))
    t = np.empty(n + 1, dtype=np.float64)
    Y = np.empty((n + 1, 3), dtype=np.float64)
    nit = np.zeros(n, dtype=np.int64)
    residuals = np.zeros(n, dtype=np.float64)
    I = np.eye(3)
    t[0] = T0
    Y[0, 0], Y[0, 1], Y[0, 2] = Y0[0], Y0[1], Y0[2]

    for k in range(n):
        step = h
        if t[k] + step > TF:
            step = TF - t[k]
        yn = Y[k]
        # Initial-guess modes used by the validation study:
        # 0 = explicit-Euler predictor (production default)
        # 1 = previous solution
        # 2 = deliberately perturbed predictor (edge-case test)
        if guess_mode == 0:
            z = yn + step * rhs_numba(yn)
        elif guess_mode == 1:
            z = yn.copy()
        else:
            z = yn + step * rhs_numba(yn)
            z[0] += 0.1
            z[1] -= 0.1
        converged = False
        last_res = 1.0e300
        for it in range(1, newton_maxiter + 1):
            F = z - yn - step * rhs_numba(z)
            last_res = np.max(np.abs(F))
            if last_res <= newton_tol:
                nit[k] = it
                residuals[k] = last_res
                converged = True
                break
            A = I - step * jac_numba(z)
            z = z + np.linalg.solve(A, -F)
        if not converged:
            # status=1 and fail index are returned to Python; do not silently accept.
            nit[k] = newton_maxiter
            residuals[k] = last_res
            return t[:k+1], Y[:k+1], nit[:k+1], residuals[:k+1], 1, k, last_res
        t[k + 1] = t[k] + step
        Y[k + 1] = z
    return t, Y, nit, residuals, 0, -1, 0.0

@njit(cache=True)
def trapezoidal_kernel(h: float, newton_tol: float = 1.0e-10, newton_maxiter: int = 50):
    n = int(math.ceil((TF - T0) / h))
    t = np.empty(n + 1, dtype=np.float64)
    Y = np.empty((n + 1, 3), dtype=np.float64)
    nit = np.zeros(n, dtype=np.int64)
    residuals = np.zeros(n, dtype=np.float64)
    I = np.eye(3)
    t[0] = T0
    Y[0, 0], Y[0, 1], Y[0, 2] = Y0[0], Y0[1], Y0[2]

    for k in range(n):
        step = h
        if t[k] + step > TF:
            step = TF - t[k]
        yn = Y[k]
        fn = rhs_numba(yn)
        z = yn + step * fn
        converged = False
        last_res = 1.0e300
        for it in range(1, newton_maxiter + 1):
            F = z - yn - 0.5 * step * (fn + rhs_numba(z))
            last_res = np.max(np.abs(F))
            if last_res <= newton_tol:
                nit[k] = it
                residuals[k] = last_res
                converged = True
                break
            A = I - 0.5 * step * jac_numba(z)
            z = z + np.linalg.solve(A, -F)
        if not converged:
            nit[k] = newton_maxiter
            residuals[k] = last_res
            return t[:k+1], Y[:k+1], nit[:k+1], residuals[:k+1], 1, k, last_res
        t[k + 1] = t[k] + step
        Y[k + 1] = z
    return t, Y, nit, residuals, 0, -1, 0.0

@njit(cache=True)
def adaptive_euler_kernel(
    tol: float,
    h0: float = 1.0e-4,
    h_min: float = 1.0e-10,
    h_max: float = 1.0e-2,
    max_accepted: int = 200000,
):
    """Step-doubling adaptive Explicit Euler.

    The accepted solution is the two-half-step value.  The controller is left
    uncapped by the Section-4 linear-stability estimate on purpose: Section 5
    is meant to reveal the residual stiffness limitation empirically.
    """
    t = np.empty(max_accepted + 1, dtype=np.float64)
    Y = np.empty((max_accepted + 1, 3), dtype=np.float64)
    t[0] = T0
    Y[0, 0], Y[0, 1], Y[0, 2] = Y0[0], Y0[1], Y0[2]
    nacc = 0
    rejected = 0
    h = h0

    while t[nacc] < TF:
        if nacc >= max_accepted:
            return t[:nacc+1], Y[:nacc+1], rejected, 1
        tn = t[nacc]
        y = Y[nacc]
        if h > TF - tn:
            h = TF - tn
        y_full = y + h * rhs_numba(y)
        h2 = 0.5 * h
        y_half = y + h2 * rhs_numba(y)
        y_half2 = y_half + h2 * rhs_numba(y_half)
        if not np.isfinite(y_half2).all():
            return t[:nacc+1], Y[:nacc+1], rejected, 2
        err = np.max(np.abs(y_half2 - y_full))

        if err <= tol or h <= 1.01 * h_min:
            nacc += 1
            t[nacc] = tn + h
            Y[nacc] = y_half2
            if err == 0.0:
                factor = 2.0
            else:
                factor = 0.9 * math.sqrt(tol / err)
                if factor < 0.5:
                    factor = 0.5
                if factor > 2.0:
                    factor = 2.0
            h = h * factor
            if h < h_min:
                h = h_min
            if h > h_max:
                h = h_max
        else:
            rejected += 1
            factor = 0.9 * math.sqrt(tol / err)
            if factor < 0.1:
                factor = 0.1
            if factor > 0.5:
                factor = 0.5
            h = h * factor
            if h < h_min:
                h = h_min
    return t[:nacc+1], Y[:nacc+1], rejected, 0

# Python wrappers -----------------------------------------------------------

def explicit_euler(h: float, blowup_limit: float = 1.0e12):
    return explicit_euler_kernel(float(h), float(blowup_limit))

def rk4(h: float, blowup_limit: float = 1.0e12):
    return rk4_kernel(float(h), float(blowup_limit))

def implicit_euler(h: float, newton_tol: float = 1.0e-10, newton_maxiter: int = 50, guess_mode: int = 0):
    out = implicit_euler_kernel(float(h), float(newton_tol), int(newton_maxiter), int(guess_mode))
    t, Y, nit, res, status, fail, last = out
    if status != 0:
        fail_time = (fail + 1) * h
        raise RuntimeError(
            f"Implicit-Euler Newton failed near t={fail_time:.8g}; "
            f"tol={newton_tol:.1e}; residual={last:.3e}"
        )
    return t, Y, nit, res

def trapezoidal(h: float, newton_tol: float = 1.0e-10, newton_maxiter: int = 50):
    out = trapezoidal_kernel(float(h), float(newton_tol), int(newton_maxiter))
    t, Y, nit, res, status, fail, last = out
    if status != 0:
        fail_time = (fail + 1) * h
        raise RuntimeError(
            f"Trapezoidal/CN Newton failed near t={fail_time:.8g}; "
            f"tol={newton_tol:.1e}; residual={last:.3e}"
        )
    return t, Y, nit, res

def adaptive_euler(tol: float, h0: float = 1.0e-4, h_min: float = 1.0e-10, h_max: float = 1.0e-2):
    t, Y, rejected, status = adaptive_euler_kernel(float(tol), float(h0), float(h_min), float(h_max))
    if status != 0:
        raise RuntimeError(f"Adaptive Euler stopped with status {status}")
    return t, Y, int(rejected)

def warm_up() -> None:
    """Compile each Numba kernel once so JIT overhead is excluded from timings."""
    explicit_euler_kernel(1.0)
    rk4_kernel(1.0)
    implicit_euler_kernel(1.0, 1.0e-8, 5)
    trapezoidal_kernel(1.0, 1.0e-8, 5)
    adaptive_euler_kernel(1.0e-3, 1.0, 1.0e-6, 1.0)
