#!/usr/bin/env python3
"""Robertson model and Jacobian.

The ODE convention exactly follows the project problem pack:
  y1' = -0.04 y1 + 1e4 y2 y3
  y2' =  0.04 y1 - 1e4 y2 y3 - 3e7 y2^2
  y3' =  3e7 y2^2
No extra factor of two is introduced in the y2^2 term.
"""
from __future__ import annotations
import numpy as np
from numba import njit

T0 = 0.0
TF = 40.0
Y0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)

@njit(cache=True)
def rhs_numba(y: np.ndarray) -> np.ndarray:
    y1, y2, y3 = y[0], y[1], y[2]
    out = np.empty(3, dtype=np.float64)
    out[0] = -0.04 * y1 + 1.0e4 * y2 * y3
    out[1] =  0.04 * y1 - 1.0e4 * y2 * y3 - 3.0e7 * y2 * y2
    out[2] =  3.0e7 * y2 * y2
    return out

@njit(cache=True)
def jac_numba(y: np.ndarray) -> np.ndarray:
    y1, y2, y3 = y[0], y[1], y[2]
    J = np.empty((3, 3), dtype=np.float64)
    J[0, 0] = -0.04
    J[0, 1] = 1.0e4 * y3
    J[0, 2] = 1.0e4 * y2
    J[1, 0] = 0.04
    J[1, 1] = -1.0e4 * y3 - 6.0e7 * y2
    J[1, 2] = -1.0e4 * y2
    J[2, 0] = 0.0
    J[2, 1] = 6.0e7 * y2
    J[2, 2] = 0.0
    return J

def rhs(t: float, y: np.ndarray) -> np.ndarray:
    """SciPy-compatible right-hand side."""
    return np.asarray(rhs_numba(np.asarray(y, dtype=np.float64)))

def jacobian(t: float, y: np.ndarray) -> np.ndarray:
    """SciPy-compatible analytic Jacobian."""
    return np.asarray(jac_numba(np.asarray(y, dtype=np.float64)))
