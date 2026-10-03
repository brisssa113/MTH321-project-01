r"""Stand-alone check for Section 5.5 (Newton-tolerance rule, adaptive extrapolated IE).
Not part of run_all.py -- run as `python code/verify_newton_rule_adaptive.py` from the repo root.

What it checks
---------------
1. (Rule violation, no solver call needed) Using the committed data/adaptive_implicit_tolerance_refinement.csv,
   recomputes N*tau_N / E_inf for tol=1e-9 and tol=1e-10, with N = accepted_steps and the fixed
   tau_N = 1e-13 used throughout the adaptive-implicit study (NEWTON_TOL_FIXED in experiments.py).
   This is exactly what the new macros \NewtonRuleRatio9 / \NewtonRuleRatio10 report.
2. (Does it matter in practice?) Re-runs the extrapolated adaptive IE solver at tol=1e-9 and tol=1e-10
   with a much tighter Newton tolerance, tau_N=1e-15, and compares the resulting E_inf and accepted-step
   count against the committed CSV row. If accuracy and step count are unchanged, the rule's "violation"
   does not actually contaminate the reported numbers at tau_N=1e-13, which is the claim made in the text.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from solvers import adaptive_implicit_euler, warm_up          # noqa: E402
from experiments import reference                              # noqa: E402
from diagnostics import e_inf                                  # noqa: E402

NEWTON_TOL_FIXED = 1e-13   # must match experiments.py


def part1_rule_check():
    ai = pd.read_csv(ROOT / "data" / "adaptive_implicit_tolerance_refinement.csv")
    ai = ai[(ai.status == "completed") & (ai.variant == "extrapolated")]
    print("Part 1: tau_N * N_accepted / E_inf, with the fixed tau_N =", NEWTON_TOL_FIXED)
    for tol_target in (1e-9, 1e-10):
        row = ai[np.isclose(ai.tol, tol_target, rtol=1e-6, atol=0.0)].iloc[0]
        N = float(row.accepted_steps)
        ratio = N * NEWTON_TOL_FIXED / float(row.E_inf)
        print(f"  tol={tol_target:.0e}: N={int(N)}  E_inf={row.E_inf:.4e}  "
              f"N*tau_N={N*NEWTON_TOL_FIXED:.3e}  ratio={ratio:.2f}"
              f"  {'(rule holds)' if ratio <= 1 else '(rule VIOLATED)'}")


def part2_rerun_with_tighter_newton_tol():
    warm_up()
    tight = reference("Radau", 1e-12, 1e-14)
    ai = pd.read_csv(ROOT / "data" / "adaptive_implicit_tolerance_refinement.csv")
    ai = ai[(ai.status == "completed") & (ai.variant == "extrapolated")]
    print("\nPart 2: re-running with tau_N = 1e-15 instead of 1e-13")
    for tol_target in (1e-9, 1e-10):
        committed = ai[np.isclose(ai.tol, tol_target, rtol=1e-6, atol=0.0)].iloc[0]
        t, Y, ratio, rejected, updates = adaptive_implicit_euler(tol_target, True, newton_tol=1e-15)
        E = e_inf(t, Y, tight)
        N = len(t) - 1
        dE = abs(E - committed.E_inf) / committed.E_inf
        dN = abs(N - committed.accepted_steps) / committed.accepted_steps
        print(f"  tol={tol_target:.0e}: committed (tau_N=1e-13) E_inf={committed.E_inf:.6e}, "
              f"N={int(committed.accepted_steps)}")
        print(f"              rerun     (tau_N=1e-15) E_inf={E:.6e}, N={N}  "
              f"[rel. diff: E {dE:.2e}, N {dN:.2e}]")


if __name__ == "__main__":
    part1_rule_check()
    part2_rerun_with_tighter_newton_tol()