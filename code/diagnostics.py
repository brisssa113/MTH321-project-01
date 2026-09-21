#!/usr/bin/env python3
from __future__ import annotations
import numpy as np

POSITIVITY_TOL = 1.0e-12
# A stored state is called "physical-scale" while max_j |y_j| <= ADMISSIBLE_CAP.
# The exact Robertson state satisfies 0 <= y_j <= 1, so a cap of 2 is generous.
ADMISSIBLE_CAP = 2.0


def trajectory_error(t, Y, reference):
    """Euclidean full-state error against a dense-output reference."""
    refY = reference.sol(np.asarray(t)).T
    return np.linalg.norm(np.asarray(Y) - refY, axis=1)


def e_inf(t, Y, reference) -> float:
    return float(np.max(trajectory_error(t, Y, reference)))


def final_error(t, Y, reference) -> float:
    refY = reference.sol(np.asarray([t[-1]])).T[0]
    return float(np.linalg.norm(np.asarray(Y[-1]) - refY))


def conservation_error(Y) -> float:
    return float(np.max(np.abs(np.sum(np.asarray(Y), axis=1) - 1.0)))


def conservation_error_admissible(Y, cap: float = ADMISSIBLE_CAP) -> float:
    """Invariant defect restricted to steps with max|y| <= cap.

    Once a diverging run reaches |y| ~ 1e16 the quantity |sum(y)-1| is pure
    round-off of huge numbers and carries no information, so it is not used.
    """
    Y = np.asarray(Y)
    keep = np.max(np.abs(Y), axis=1) <= cap
    return conservation_error(Y[keep]) if keep.any() else float("nan")


def divergence_time(t, Y, cap: float = ADMISSIBLE_CAP) -> float:
    """First stored time with max|y| > cap (NaN if it never happens)."""
    Y = np.asarray(Y)
    bad = np.max(np.abs(Y), axis=1) > cap
    return float(np.asarray(t)[np.argmax(bad)]) if bad.any() else float("nan")


def first_negative_time(t, Y, tol: float = POSITIVITY_TOL) -> float:
    """First stored time with a component below -tol (NaN if none)."""
    Y = np.asarray(Y)
    bad = np.min(Y, axis=1) < -tol
    return float(np.asarray(t)[np.argmax(bad)]) if bad.any() else float("nan")


def min_component(Y) -> float:
    return float(np.min(np.asarray(Y)))


def is_nonnegative(Y, tol: float = POSITIVITY_TOL) -> bool:
    return min_component(Y) >= -tol