#!/usr/bin/env python3
from __future__ import annotations
import csv
import math
import statistics
import time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

from models import T0, TF, Y0, rhs, rhs_numba, jacobian
from solvers import (warm_up, explicit_euler, rk4, implicit_euler, trapezoidal,
                     implicit_euler_kernel, adaptive_euler_kernel, adaptive_implicit_euler)
from diagnostics import (e_inf, final_error, conservation_error,
                         conservation_error_admissible, divergence_time,
                         first_negative_time, min_component, is_nonnegative)

NEWTON_TOL_FIXED = 1.0e-13      # absolute residual tolerance of all fixed-step implicit runs
POSITIVITY_TOL = 1.0e-12
TIMING_REPEATS = 7              # median of this many warmed calls
RK4_REAL_AXIS_LIMIT = 2.785293563405  # |z| bound of RK4's stability region on the negative real axis
REACHED_TOL = 1.0e-8           # 'reached T': |t_end - T| below this (round-off accumulates over 6.4e5 steps)
IMPLICIT = {"Implicit Euler", "Trapezoidal / CN"}
AIE_TOLERANCES = [1e-4, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8, 1e-9, 1e-10]   # first 8 = list of the adaptive-Euler study; 1e-9, 1e-10 locate the crossover with RK4 (do not go below ~1e-10: reference certifies ~2e-11)
AIE_DEMO_TOL = 1e-6                                                # demonstration run (same as adaptive Euler)


def common_output_grid():
    return np.concatenate(([0.0], np.geomspace(1.0e-8, TF, 200)))


def fine_check_grid():
    """60 000 points used to check the reference's dense output."""
    return np.concatenate((np.geomspace(1.0e-8, TF, 20000), np.linspace(1.0e-3, TF, 40000)))


def reference(method: str, rtol: float, atol: float, t_eval=None):
    if t_eval is None:
        t_eval = common_output_grid()
    sol = solve_ivp(
        rhs, (T0, TF), Y0, method=method,
        jac=jacobian if method in {"Radau", "BDF"} else None,
        t_eval=t_eval, dense_output=True, rtol=rtol, atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"{method} reference failed: {sol.message}")
    return sol


def save_rows(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def timed_solver(callable_, repeats=TIMING_REPEATS):
    """Return one numerical result and median runtime over repeated warmed calls."""
    times = []
    result = None
    for j in range(repeats):
        tic = time.perf_counter()
        out = callable_()
        dt = time.perf_counter() - tic
        if j == 0:
            result = out
        times.append(dt)
    return result, float(statistics.median(times))


def run_fixed_method(method, h):
    if method == "Explicit Euler":
        return explicit_euler(h)
    if method == "RK4":
        return rk4(h)
    if method == "Implicit Euler":
        return implicit_euler(h, newton_tol=NEWTON_TOL_FIXED)
    if method == "Trapezoidal / CN":
        return trapezoidal(h, newton_tol=NEWTON_TOL_FIXED)
    raise ValueError(method)


def measure(method, h, tight, repeats=TIMING_REPEATS):
    """Run one fixed-step configuration, return (csv row, t, Y)."""
    result, elapsed = timed_solver(lambda: run_fixed_method(method, h), repeats=repeats)
    if method in IMPLICIT:
        t, Y, nit, _ = result
        n_res = int(np.sum(nit))                 # residual evaluations (incl. the final check of every step)
        n_upd = n_res - len(nit)                 # actual Newton updates = linear solves
        max_nit = int(np.max(nit))
        work, work_type = n_upd, "Newton updates"
    else:
        t, Y = result
        n_res = n_upd = max_nit = 0
        work = (len(t) - 1) if method == "Explicit Euler" else 4 * (len(t) - 1)
        work_type = "RHS evaluations"
    row = {
        "method": method, "h": h, "E_inf": e_inf(t, Y, tight),
        "steps": len(t) - 1, "work_count": work, "work_type": work_type,
        "newton_iterations": n_res, "newton_updates": n_upd, "max_newton_iterations": max_nit,
        "wall_clock_time": elapsed,
        "newton_tolerance": NEWTON_TOL_FIXED if method in IMPLICIT else "",
        "min_component": min_component(Y), "nonnegative": is_nonnegative(Y),
        "reached_T": bool(abs(t[-1] - TF) < REACHED_TOL), "finite": bool(np.all(np.isfinite(Y))),
    }
    return row, t, Y


def fit_slope(hs, errs):
    return float(np.polyfit(np.log(np.asarray(hs, float)), np.log(np.asarray(errs, float)), 1)[0])


def lam_max_along(tight, times):
    """max |Re lambda| over the negative-real nonzero eigenvalues of J(y_ref(t))."""
    Ys = tight.sol(np.asarray(times)).T
    out = np.empty(len(Ys))
    for i, y in enumerate(Ys):
        eig = np.linalg.eigvals(jacobian(0.0, y))
        vals = [abs(ev.real) for ev in eig if abs(ev) > 1.0e-10 and ev.real < 0.0]
        out[i] = max(vals) if vals else np.nan
    return out


def theoretical_euler_boundary(reference_solution):
    """Return (2/max|Re lambda|, max|Re lambda|, time of the maximum) on the reference path."""
    t = reference_solution.t[1:]
    lam = lam_max_along(reference_solution, t)
    i = int(np.nanargmax(lam))
    return float(2.0 / lam[i]), float(lam[i]), float(t[i])


def empirical_euler_boundary(lo, hi, iters=40):
    """Bisection for the largest h with 'reaches T and stays non-negative' (assumes a monotone transition)."""
    def admissible(h):
        t, Y = explicit_euler(h)
        return abs(t[-1] - TF) < REACHED_TOL and is_nonnegative(Y)
    if not admissible(lo) or admissible(hi):
        return float("nan")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if admissible(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def reduced_jacobian(y):
    """2x2 Jacobian of (y1', y2') after eliminating y3 = 1 - y1 - y2:  J_r[i,j] = J[i,j] - J[i,2]."""
    J = jacobian(0.0, np.asarray(y, dtype=float))
    return J[:2, :2] - J[:2, 2:3]


def eig_diagnostics(y):
    """Eigen-analysis at one state.

    Returns (lambda_slow, lambda_fast, kappa(V), |structural zero of the full J|, relative mismatch between the
    reduced-Jacobian eigenvalues and the two nonzero eigenvalues of the full 3x3 Jacobian).
    lambda_slow/lambda_fast are the real parts, ordered by modulus.
    """
    w, V = np.linalg.eig(reduced_jacobian(y))
    order = np.argsort(np.abs(w))
    w, V = w[order], V[:, order]
    kappa = float(np.linalg.cond(V))
    wf = np.linalg.eigvals(jacobian(0.0, np.asarray(y, dtype=float)))
    wf = wf[np.argsort(np.abs(wf))]
    mismatch = float(np.max(np.abs(np.abs(w) - np.abs(wf[1:]))) / np.max(np.abs(w)))
    return float(w[0].real), float(w[1].real), kappa, float(abs(wf[0])), mismatch


def adaptive_local_errors(t, Y):
    """Step-doubling error estimate ||y_two_halves - y_full||_inf recomputed for every ACCEPTED step of a stored run
    (the kernel does not store it). Also checks that the recomputed two-half-step value equals the stored y_{n+1}."""
    h = np.diff(t)
    errs = np.empty(len(h))
    for k in range(len(h)):
        y = Y[k]
        f0 = rhs_numba(y)
        y_full = y + h[k] * f0
        y_mid = y + 0.5 * h[k] * f0
        y_two = y_mid + 0.5 * h[k] * rhs_numba(y_mid)
        if not np.allclose(y_two, Y[k + 1], rtol=0.0, atol=1e-13):
            raise RuntimeError("stored adaptive state is not the two-half-step value")
        errs[k] = np.max(np.abs(y_two - y_full))
    return errs


def main(root: Path):
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)

    print("[1/8] Warming up Numba kernels (JIT time will not enter solver timings)...")
    warm_up()

    # ------------------------------------------------------------------ reference
    print("[2/8] Building and checking the Radau/BDF reference...")
    loose = reference("Radau", 1e-10, 1e-12)
    tight = reference("Radau", 1e-12, 1e-14)
    bdf = reference("BDF", 1e-12, 1e-14)
    d_ref = float(np.max(np.linalg.norm((loose.y - tight.y).T, axis=1)))
    d_bdf = float(np.max(np.linalg.norm((bdf.y - tight.y).T, axis=1)))
    d_mass = float(np.max(np.abs(np.sum(tight.y, axis=0) - 1.0)))
    tf = fine_check_grid()
    d_ref_dense = float(np.max(np.linalg.norm(loose.sol(tf) - tight.sol(tf), axis=0)))
    d_bdf_dense = float(np.max(np.linalg.norm(bdf.sol(tf) - tight.sol(tf), axis=0)))
    np.savetxt(data / "reference_trajectory.csv",
               np.column_stack((tight.t, tight.y.T)), delimiter=",",
               header="t,y1,y2,y3", comments="")
    save_rows(data / "reference_validation.csv", ["check", "value"], [
        {"check": "Radau_100x_refinement_max_L2", "value": d_ref},
        {"check": "Radau_vs_BDF_max_L2", "value": d_bdf},
        {"check": "Radau_100x_refinement_dense_max_L2", "value": d_ref_dense},
        {"check": "Radau_vs_BDF_dense_max_L2", "value": d_bdf_dense},
        {"check": "dense_check_grid_points", "value": len(tf)},
        {"check": "reference_max_mass_defect", "value": d_mass},
        {"check": "reference_y1_T40", "value": float(tight.y[0, -1])},
        {"check": "reference_y2_T40", "value": float(tight.y[1, -1])},
        {"check": "reference_y3_T40", "value": float(tight.y[2, -1])},
    ])

    # ------------------------------------------------------------------ convergence
    print("[3/8] Running four-method convergence study...")
    cases = {
        "Explicit Euler": [5e-4, 2.5e-4, 1.25e-4, 6.25e-5],
        "RK4": [8e-4, 4e-4, 2e-4, 1e-4],
        "Implicit Euler": [1e-2, 5e-3, 2.5e-3, 1.25e-3],
        "Trapezoidal / CN": [1e-3, 5e-4, 2.5e-4, 1.25e-4],
    }
    conv_rows = []
    for method, hs in cases.items():
        prev_h = prev_e = None
        for h in hs:
            print(f"    {method:17s} h={h:.3g}")
            row, _, _ = measure(method, h, tight)
            row["observed_order"] = "" if prev_h is None else math.log(prev_e / row["E_inf"]) / math.log(prev_h / h)
            conv_rows.append(row)
            prev_h, prev_e = h, row["E_inf"]
    conv_fields = ["method", "h", "E_inf", "steps", "work_count", "work_type", "newton_iterations",
                   "newton_updates", "max_newton_iterations", "wall_clock_time", "observed_order",
                   "newton_tolerance", "min_component", "nonnegative", "reached_T", "finite"]
    save_rows(data / "four_method_results.csv", conv_fields, conv_rows)

    fit_rows = []
    for method in cases:
        rows = sorted([r for r in conv_rows if r["method"] == method], key=lambda r: r["h"])
        h_all = [r["h"] for r in rows]
        e_all = [r["E_inf"] for r in rows]
        fit_rows.append({"method": method, "fit_all_points": fit_slope(h_all, e_all),
                         "fit_three_finest": fit_slope(h_all[:3], e_all[:3])})
    save_rows(data / "order_fits.csv", ["method", "fit_all_points", "fit_three_finest"], fit_rows)

    # ------------------------------------------------------------------ work-precision
    print("[4/8] Extending the h-range for the work-precision / matched-error comparison...")
    extras = [("Implicit Euler", 6.5e-4),           # matched in accuracy to Explicit Euler h=5e-4
              ("Trapezoidal / CN", 1e-2), ("Trapezoidal / CN", 5e-3), ("Trapezoidal / CN", 2e-3)]
    wp_rows = [dict(r) for r in conv_rows]
    for method, h in extras:
        print(f"    {method:17s} h={h:.3g}")
        row, _, _ = measure(method, h, tight)
        row["observed_order"] = ""
        wp_rows.append(row)
    save_rows(data / "work_precision.csv", conv_fields, wp_rows)

    # library baseline: adaptive implicit solver from SciPy (not one of the self-implemented methods)
    def run_radau():
        return solve_ivp(rhs, (T0, TF), Y0, method="Radau", jac=jacobian, rtol=1e-6, atol=1e-9, dense_output=True)
    sol_lib, t_lib = timed_solver(run_radau, repeats=TIMING_REPEATS)
    E_lib = float(np.max(np.linalg.norm(sol_lib.sol(tf) - tight.sol(tf), axis=0)))
    save_rows(data / "library_baseline.csv",
              ["method", "rtol", "atol", "E_inf", "steps", "nfev", "njev", "nlu", "wall_clock_time", "min_component"], [{
                  "method": "SciPy Radau (adaptive)", "rtol": 1e-6, "atol": 1e-9, "E_inf": E_lib,
                  "steps": len(sol_lib.t) - 1, "nfev": int(sol_lib.nfev), "njev": int(sol_lib.njev),
                  "nlu": int(sol_lib.nlu), "wall_clock_time": t_lib, "min_component": float(np.min(sol_lib.y)),
              }])

    # ------------------------------------------------------------------ explicit-Euler boundary
    print("[5/8] Sweeping the Explicit-Euler stability/positivity boundary...")
    boundary_h = [4e-4, 5e-4, 5.5e-4, 5.9e-4, 5.95e-4, 6e-4, 7e-4, 8e-4]
    b_rows = []
    for h in boundary_h:
        t, Y = explicit_euler(h)
        reached = abs(t[-1] - TF) < REACHED_TOL
        finite = bool(np.all(np.isfinite(Y)))
        nonneg = is_nonnegative(Y)
        status = "diverged" if not reached else ("admissible" if nonneg else "negative")
        ferr = final_error(t, Y, tight) if reached and finite else ""
        b_rows.append({
            "h": h, "status": status, "reached_T": reached, "finite": finite, "nonnegative": nonneg,
            "min_component": min_component(Y), "t_end": float(t[-1]), "t_diverge": divergence_time(t, Y),
            "conservation_error": conservation_error_admissible(Y),
            "conservation_error_all": conservation_error(Y), "final_error": ferr,
        })
    save_rows(data / "euler_stability_boundary.csv",
              ["h", "status", "reached_T", "finite", "nonnegative", "min_component", "t_end", "t_diverge",
               "conservation_error", "conservation_error_all", "final_error"], b_rows)

    hcrit, lam_fast, t_lam = theoretical_euler_boundary(tight)
    h_emp = empirical_euler_boundary(5.9e-4, 6.0e-4)
    ts_fine = np.linspace(0.5, TF, 4000)
    lam_fine = lam_max_along(tight, ts_fine)
    exceeded = ts_fine[2.0 / lam_fine < h_emp] if np.isfinite(h_emp) else np.array([])
    save_rows(data / "euler_stability_metadata.csv", ["quantity", "value"], [
        {"quantity": "theoretical_h_critical", "value": hcrit},
        {"quantity": "max_fast_negative_real_eigenvalue", "value": lam_fast},
        {"quantity": "time_of_max_eigenvalue", "value": t_lam},
        {"quantity": "empirical_h_boundary_bisection", "value": h_emp},
        {"quantity": "time_local_bound_exceeded_at_h_emp", "value": float(exceeded[0]) if len(exceeded) else float("nan")},
        {"quantity": "rk4_h_critical_real_axis", "value": RK4_REAL_AXIS_LIMIT / lam_fast},
    ])

    # ------------------------------------------------------------------ Newton tolerance
    print("[6/8] Running strict residual-based Newton-tolerance sweep and initial-guess test...")
    y2_ref = float(tight.y[1, -1])
    n_rows = []
    h_n = 1e-2
    for tol in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]:
        t, Y, nit, res, status, fail, last = implicit_euler_kernel(h_n, float(tol), 50, 0)
        base = {"newton_tolerance": tol, "h": h_n,
                "conservation_error": conservation_error(Y), "min_component": min_component(Y),
                "first_negative_time": first_negative_time(t, Y)}
        if status == 0:
            n_res = int(np.sum(nit))
            base.update({
                "status": "completed" if is_nonnegative(Y) else "completed_nonphysical", "fail_time": "",
                "final_y2": float(Y[-1, 1]), "abs_y2_error": abs(float(Y[-1, 1]) - y2_ref),
                "final_reference_error": final_error(t, Y, tight),
                "total_newton_iterations": n_res, "total_newton_updates": n_res - len(nit),
                "max_newton_iterations": int(np.max(nit)), "max_newton_residual": float(np.max(res)),
            })
        else:
            base.update({
                "status": "failed", "fail_time": float(t[-1] + h_n),
                "final_y2": "", "abs_y2_error": "", "final_reference_error": "",
                "total_newton_iterations": "", "total_newton_updates": "",
                "max_newton_iterations": "", "max_newton_residual": "",
            })
        n_rows.append(base)
    save_rows(data / "newton_tolerance.csv",
              ["newton_tolerance", "h", "status", "fail_time", "final_y2", "abs_y2_error", "final_reference_error",
               "total_newton_iterations", "total_newton_updates", "max_newton_iterations", "max_newton_residual",
               "conservation_error", "min_component", "first_negative_time"], n_rows)

    guess_rows = []
    for mode, label in [(0, "Euler predictor"), (1, "Previous solution"), (2, "Perturbed predictor")]:
        try:
            t, Y, nit, res = implicit_euler(h_n, newton_tol=1e-10, guess_mode=mode)
            guess_rows.append({
                "initial_guess": label, "status": "completed" if is_nonnegative(Y) else "completed_nonphysical",
                "E_inf": e_inf(t, Y, tight), "final_reference_error": final_error(t, Y, tight),
                "minimum_component": min_component(Y), "negative_count": int(np.sum(Y < -POSITIVITY_TOL)),
                "total_newton_iterations": int(np.sum(nit)), "max_newton_iterations": int(np.max(nit)),
                "max_newton_residual": float(np.max(res)),
            })
        except RuntimeError as exc:
            guess_rows.append({
                "initial_guess": label, "status": f"failed: {exc}", "E_inf": "", "final_reference_error": "",
                "minimum_component": "", "negative_count": "", "total_newton_iterations": "",
                "max_newton_iterations": "", "max_newton_residual": "",
            })
    save_rows(data / "newton_initial_guess_study.csv",
              ["initial_guess", "status", "E_inf", "final_reference_error", "minimum_component", "negative_count",
               "total_newton_iterations", "max_newton_iterations", "max_newton_residual"], guess_rows)

    # ------------------------------------------------------------------ adaptive Euler
    print("[7/8] Running adaptive-Euler tolerance study and step-history diagnostics...")
    a_rows = []
    for tol in [1e-4, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8]:
        t, Y, rejected, status = adaptive_euler_kernel(float(tol), 1e-4, 1e-10, 1e-2)
        accepted = len(t) - 1
        if status == 0:
            hs = np.diff(t)
            E = e_inf(t, Y, tight)
            a_rows.append({
                "tol": tol, "status": "completed", "t_reached": float(t[-1]),
                "accepted_steps": accepted, "rejected_steps": int(rejected),
                "rejection_fraction": rejected / (accepted + rejected),
                "max_L2_error": E, "E_over_tol": E / tol,
                "mean_accepted_h": float(np.mean(hs)), "max_accepted_h": float(np.max(hs)),
                "min_accepted_h": float(np.min(hs)), "min_component": min_component(Y),
            })
        else:
            a_rows.append({
                "tol": tol, "status": "diverged" if status == 2 else "step_limit", "t_reached": float(t[-1]),
                "accepted_steps": accepted, "rejected_steps": int(rejected),
                "rejection_fraction": rejected / max(accepted + rejected, 1),
                "max_L2_error": "", "E_over_tol": "", "mean_accepted_h": "", "max_accepted_h": "",
                "min_accepted_h": "", "min_component": min_component(Y),
            })
    save_rows(data / "adaptive_tolerance_refinement.csv",
              ["tol", "status", "t_reached", "accepted_steps", "rejected_steps", "rejection_fraction",
               "max_L2_error", "E_over_tol", "mean_accepted_h", "max_accepted_h", "min_accepted_h",
               "min_component"], a_rows)

    # ---- demonstration run of Section 3.5 (tol=1e-6): full accepted-step history and summary statistics
    tol_demo = 1e-6
    t_a, Y_a, rej_a, st_a = adaptive_euler_kernel(tol_demo, 1e-4, 1e-10, 1e-2)
    if st_a != 0:
        raise RuntimeError("adaptive demonstration run did not complete")
    h_a = np.diff(t_a)
    tm = t_a[1:]
    err_loc = adaptive_local_errors(t_a, Y_a)
    lam_a = lam_max_along(tight, tm)
    y_ref_a = tight.sol(tm).T
    err_glob = np.linalg.norm(Y_a[1:] - y_ref_a, axis=1)
    save_rows(data / "adaptive_demo_history.csv",
              ["t", "h", "local_error_estimate", "y2", "y2_ref", "global_error_L2", "lam_max"],
              [{"t": float(tm[i]), "h": float(h_a[i]), "local_error_estimate": float(err_loc[i]),
                "y2": float(Y_a[i + 1, 1]), "y2_ref": float(y_ref_a[i, 1]),
                "global_error_L2": float(err_glob[i]), "lam_max": float(lam_a[i])} for i in range(len(h_a))])
    ratio = err_loc / tol_demo
    late = tm > 0.1
    r_stab = h_a[late] * lam_a[late] / 2.0                 # accepted h in units of the Explicit-Euler limit 2/|lambda|
    i3 = int(np.argmax(h_a > 3e-3))
    save_rows(data / "adaptive_demo_summary.csv", ["quantity", "value"], [
        {"quantity": "tol", "value": tol_demo},
        {"quantity": "accepted_steps", "value": len(h_a)}, {"quantity": "rejected_steps", "value": int(rej_a)},
        {"quantity": "h_min", "value": float(h_a.min())}, {"quantity": "h_max", "value": float(h_a.max())},
        {"quantity": "h_mean", "value": float(h_a.mean())}, {"quantity": "h_median", "value": float(np.median(h_a))},
        {"quantity": "h_median_1_to_10", "value": float(np.median(h_a[(tm >= 1) & (tm < 10)]))},
        {"quantity": "h_median_after_10", "value": float(np.median(h_a[tm >= 10]))},
        {"quantity": "h_min_after_10", "value": float(h_a[tm >= 10].min())},
        {"quantity": "steps_before_h_exceeds_3e-3", "value": i3}, {"quantity": "time_h_exceeds_3e-3", "value": float(t_a[i3])},
        {"quantity": "final_y1", "value": float(Y_a[-1, 0])}, {"quantity": "final_y2", "value": float(Y_a[-1, 1])},
        {"quantity": "final_y3", "value": float(Y_a[-1, 2])},
        {"quantity": "final_error_L2", "value": final_error(t_a, Y_a, tight)},
        {"quantity": "E_inf", "value": e_inf(t_a, Y_a, tight)},
        {"quantity": "conservation_defect", "value": conservation_error(Y_a)},
        {"quantity": "min_component", "value": min_component(Y_a)},
        {"quantity": "local_err_over_tol_p05", "value": float(np.quantile(ratio, 0.05))},
        {"quantity": "local_err_over_tol_median", "value": float(np.median(ratio))},
        {"quantity": "local_err_over_tol_p95", "value": float(np.quantile(ratio, 0.95))},
        {"quantity": "local_err_over_tol_max", "value": float(ratio.max())},
        {"quantity": "sum_local_error_estimates", "value": float(err_loc.sum())},
        {"quantity": "accepted_steps_times_tol", "value": len(h_a) * tol_demo},
        {"quantity": "h_over_EE_limit_p05_t_gt_0.1", "value": float(np.quantile(r_stab, 0.05))},
        {"quantity": "h_over_EE_limit_p95_t_gt_0.1", "value": float(np.quantile(r_stab, 0.95))},
    ])

    # ------------------------------------------------------------------ adaptive implicit Euler (Sections 3.5, 5.5.2, 5.6)
    print("[7b/8] Adaptive implicit Euler: tolerance study, timing and demonstration run...")
    ee_timed = []
    for r in a_rows:
        if r['status'] != 'completed':
            continue
        (te, Ye, rejected, status), elapsed = timed_solver(
            lambda: adaptive_euler_kernel(float(r['tol']), 1e-4, 1e-10, 1e-2))
        ee_timed.append(dict(tol=r['tol'], E_inf=r['max_L2_error'], accepted_steps=len(te)-1,
                             rejected_steps=int(rejected), rhs_evaluations=2*(len(te)-1+rejected),
                             nonnegative=is_nonnegative(Ye), min_component=min_component(Ye),
                             wall_clock_time=elapsed))
    save_rows(data / "adaptive_explicit_work_precision.csv", list(ee_timed[0]), ee_timed)
    aie_fields = ["variant", "tol", "status", "accepted_steps", "rejected_steps", "newton_updates", "E_inf", "E_over_tol",
                  "final_y2_abs_error", "min_component", "nonnegative", "conservation_error", "median_accepted_h",
                  "max_accepted_h", "h_over_EE_limit_p05", "h_over_EE_limit_median", "h_over_EE_limit_p95",
                  "local_ratio_max", "wall_clock_time"]
    aie_rows = []
    y2_ref_T = float(tight.y[1, -1])
    for variant, extrap in [("extrapolated", True), ("first-order", False)]:
        for tol in AIE_TOLERANCES:
            print(f"    adaptive IE ({variant}) tol={tol:.0e}")
            try:
                (t, Y, ratio, rejected, n_upd), elapsed = timed_solver(
                    lambda: adaptive_implicit_euler(tol, extrap, newton_tol=NEWTON_TOL_FIXED))
            except RuntimeError as exc:
                aie_rows.append({"variant": variant, "tol": tol, "status": f"failed: {exc}"})
                continue
            hs, tm = np.diff(t), t[1:]
            late = tm > 0.1
            r_stab = hs[late] * lam_max_along(tight, tm[late]) / 2.0     # accepted h in units of the Explicit-Euler limit
            E = e_inf(t, Y, tight)
            aie_rows.append({
                "variant": variant, "tol": tol, "status": "completed", "accepted_steps": len(hs),
                "rejected_steps": rejected, "newton_updates": n_upd, "E_inf": E, "E_over_tol": E / tol,
                "final_y2_abs_error": abs(float(Y[-1, 1]) - y2_ref_T), "min_component": min_component(Y),
                "nonnegative": is_nonnegative(Y), "conservation_error": conservation_error(Y),
                "median_accepted_h": float(np.median(hs)), "max_accepted_h": float(np.max(hs)),
                "h_over_EE_limit_p05": float(np.quantile(r_stab, 0.05)),
                "h_over_EE_limit_median": float(np.median(r_stab)),
                "h_over_EE_limit_p95": float(np.quantile(r_stab, 0.95)),
                "local_ratio_max": float(np.max(ratio)), "wall_clock_time": elapsed,
            })
    save_rows(data / "adaptive_implicit_tolerance_refinement.csv", aie_fields, aie_rows)

    # ---- demonstration run (tol = AIE_DEMO_TOL, extrapolated variant): step history + summary
    t_i, Y_i, ratio_i, rej_i, upd_i = adaptive_implicit_euler(AIE_DEMO_TOL, True, newton_tol=NEWTON_TOL_FIXED)
    h_i, tm_i = np.diff(t_i), t_i[1:]
    lam_i = lam_max_along(tight, tm_i)
    y_ref_i = tight.sol(tm_i).T
    err_glob_i = np.linalg.norm(Y_i[1:] - y_ref_i, axis=1)
    save_rows(data / "adaptive_implicit_demo_history.csv",
              ["t", "h", "local_error_over_tol", "y2", "y2_ref", "global_error_L2", "lam_max"],
              [{"t": float(tm_i[i]), "h": float(h_i[i]), "local_error_over_tol": float(ratio_i[i]),
                "y2": float(Y_i[i + 1, 1]), "y2_ref": float(y_ref_i[i, 1]),
                "global_error_L2": float(err_glob_i[i]), "lam_max": float(lam_i[i])} for i in range(len(h_i))])
    late_i = tm_i > 0.1
    r_i = h_i[late_i] * lam_i[late_i] / 2.0
    save_rows(data / "adaptive_implicit_demo_summary.csv", ["quantity", "value"], [
        {"quantity": "tol", "value": AIE_DEMO_TOL}, {"quantity": "newton_tolerance", "value": NEWTON_TOL_FIXED},
        {"quantity": "accepted_steps", "value": len(h_i)}, {"quantity": "rejected_steps", "value": int(rej_i)},
        {"quantity": "newton_updates", "value": int(upd_i)},
        {"quantity": "h_min", "value": float(h_i.min())}, {"quantity": "h_max", "value": float(h_i.max())},
        {"quantity": "h_median", "value": float(np.median(h_i))},
        {"quantity": "final_y1", "value": float(Y_i[-1, 0])}, {"quantity": "final_y2", "value": float(Y_i[-1, 1])},
        {"quantity": "final_y3", "value": float(Y_i[-1, 2])},
        {"quantity": "final_error_L2", "value": final_error(t_i, Y_i, tight)},
        {"quantity": "E_inf", "value": e_inf(t_i, Y_i, tight)},
        {"quantity": "conservation_defect", "value": conservation_error(Y_i)},
        {"quantity": "min_component", "value": min_component(Y_i)},
        {"quantity": "local_err_over_tol_p05", "value": float(np.quantile(ratio_i, 0.05))},
        {"quantity": "local_err_over_tol_median", "value": float(np.median(ratio_i))},
        {"quantity": "local_err_over_tol_p95", "value": float(np.quantile(ratio_i, 0.95))},
        {"quantity": "local_err_over_tol_max", "value": float(ratio_i.max())},
        {"quantity": "h_over_EE_limit_p05_t_gt_0.1", "value": float(np.quantile(r_i, 0.05))},
        {"quantity": "h_over_EE_limit_median_t_gt_0.1", "value": float(np.median(r_i))},
        {"quantity": "h_over_EE_limit_p95_t_gt_0.1", "value": float(np.quantile(r_i, 0.95))},
    ])

    # ------------------------------------------------------------------ Jacobian / stiffness / non-normality (Section 4)
    print("[8/8] Reduced-Jacobian eigenvalues, stiffness ratio, kappa(V), h*lambda exceedance (Section 4)...")
    k2 = 3.0e7
    output_grid = common_output_grid()[1:]                 # first POSITIVE output time is 1e-8 (t=0 is degenerate)
    fine_grid = np.geomspace(1.0e-8, TF, 1000)
    def scan(times):
        Ys = tight.sol(times).T
        return np.array([eig_diagnostics(y) for y in Ys])   # columns: slow, fast, kappa, zero, mismatch
    D_fine, D_out = scan(fine_grid), scan(output_grid)
    S_fine = np.abs(D_fine[:, 1]) / np.abs(D_fine[:, 0])
    save_rows(data / "jacobian_stiffness.csv",
              ["t", "lambda_slow", "lambda_fast", "abs_slow", "abs_fast", "stiffness_ratio", "kappa_V"],
              [{"t": float(t), "lambda_slow": d[0], "lambda_fast": d[1], "abs_slow": abs(d[0]), "abs_fast": abs(d[1]),
                "stiffness_ratio": abs(d[1]) / abs(d[0]), "kappa_V": d[2]} for t, d in zip(fine_grid, D_fine)])
    save_rows(data / "kappa_output_grid.csv", ["t", "kappa_V"],
              [{"t": float(t), "kappa_V": d[2]} for t, d in zip(output_grid, D_out)])

    pack = {1.0e-4: 3.0e3, 1.0e-2: 5.4e3, 40.0: 1.58e5}     # Problem-Pack orientation values (check values only)
    o_rows = []
    for t0, s_pack in pack.items():
        d = scan(np.array([t0]))[0]
        s_ours = abs(d[1]) / abs(d[0])
        o_rows.append({"t": t0, "S_ours": s_ours, "S_problem_pack": s_pack, "rel_diff": abs(s_ours - s_pack) / s_pack,
                       "lambda_slow": d[0], "lambda_fast": d[1], "kappa_V": d[2]})
    save_rows(data / "stiffness_orientation_check.csv",
              ["t", "S_ours", "S_problem_pack", "rel_diff", "lambda_slow", "lambda_fast", "kappa_V"], o_rows)

    kap = D_out[:, 2]
    Jr40 = reduced_jacobian(tight.sol(TF))
    save_rows(data / "stiffness_metadata.csv", ["quantity", "value"], [
        {"quantity": "first_positive_output_time", "value": float(output_grid[0])},
        {"quantity": "crossing_time_analytic_1_over_2k2", "value": 1.0 / (2.0 * k2)},
        {"quantity": "crossing_time_numeric_argmin_S_fine_grid", "value": float(fine_grid[np.argmin(S_fine)])},
        {"quantity": "S_at_first_output_time", "value": float(abs(D_out[0, 1]) / abs(D_out[0, 0]))},
        {"quantity": "kappa_peak_output_grid", "value": float(kap.max())},
        {"quantity": "kappa_peak_time_output_grid", "value": float(output_grid[np.argmax(kap)])},
        {"quantity": "kappa_peak_fine_grid", "value": float(D_fine[:, 2].max())},
        {"quantity": "kappa_median_output_grid", "value": float(np.median(kap))},
        {"quantity": "kappa_fraction_above_10_output_grid", "value": float(np.mean(kap > 10.0))},
        {"quantity": "kappa_at_T", "value": float(kap[-1])},
        {"quantity": "lambda_slow_at_T", "value": float(D_out[-1, 0])},
        {"quantity": "min_abs_slow_eigenvalue_output_grid", "value": float(np.min(np.abs(D_out[:, 0])))},
        {"quantity": "structural_zero_max_abs_output_grid", "value": float(D_out[:, 3].max())},
        {"quantity": "full_vs_reduced_max_rel_mismatch", "value": float(max(D_out[:, 4].max(), D_fine[:, 4].max()))},
        {"quantity": "Jr_T_11", "value": float(Jr40[0, 0])}, {"quantity": "Jr_T_12", "value": float(Jr40[0, 1])},
        {"quantity": "Jr_T_21", "value": float(Jr40[1, 0])}, {"quantity": "Jr_T_22", "value": float(Jr40[1, 1])},
    ])

    # local hlambda位 diagnostic versus the observed Explicit-Euler outcome of the step-size sweep
    t_div = {r["h"]: r["t_diverge"] for r in b_rows}
    status_of = {r["h"]: r["status"] for r in b_rows}
    hl_rows = []
    for h in boundary_h:
        m_ee = ts_fine[h * lam_fine > 2.0]
        m_rk = ts_fine[h * lam_fine > RK4_REAL_AXIS_LIMIT]
        hl_rows.append({"h": h, "t_exceed_EE": float(m_ee[0]) if len(m_ee) else float("nan"),
                        "t_exceed_RK4": float(m_rk[0]) if len(m_rk) else float("nan"),
                        "observed_status_EE": status_of[h], "t_diverge_EE": t_div[h]})
    save_rows(data / "hlambda_exceedance.csv",
              ["h", "t_exceed_EE", "t_exceed_RK4", "observed_status_EE", "t_diverge_EE"], hl_rows)

    print("Numerical experiments complete.")
    print(f"  data directory: {data}")
    print(f"  Radau 100x refinement discrepancy = {d_ref:.3e}  (dense grid: {d_ref_dense:.3e})")
    print(f"  Radau/BDF discrepancy              = {d_bdf:.3e}  (dense grid: {d_bdf_dense:.3e})")
    print(f"  h_crit = {hcrit:.4e}, empirical h_emp = {h_emp:.4e}")
    print(f"  NOTE: wall-clock times are machine-dependent (median of {TIMING_REPEATS} warmed runs).")
    return tight


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    main(ROOT)
