#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

POS_TOL = 1.0e-12
STALE = ["table_physical_validity.tex", "table_newton_initial_guess.tex", "table_adaptive_edge_case.tex"]


# ---------------------------------------------------------------- number formatting
def _isnan(x):
    try:
        return x is None or (isinstance(x, str) and x.strip() == "") or not np.isfinite(float(x))
    except (TypeError, ValueError):
        return True


def sci_plain(x, sig=3):
    """d.dd\\times10^{e}; rounding carries (9.9996e-5 -> 1.00e-4) are handled by the formatter."""
    if _isnan(x):
        return "--"
    x = float(x)
    if x == 0.0:
        return "0"
    mant, exp = f"{x:.{sig - 1}e}".split("e")
    return f"{mant}\\times10^{{{int(exp)}}}"


def sci_tex(x, sig=3):
    s = sci_plain(x, sig)
    return s if s == "--" else f"${s}$"


def macro(name, body):
    return f"\\newcommand{{\\{name}}}{{\\ensuremath{{{body}}}}}\n"


def yn(flag):
    return "yes" if bool(flag) else "no"


def write(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------- adaptive implicit Euler (Sections 3.5, 5.5.2, 5.6)
def adaptive_implicit_block(data: Path, wp, ad, ee_wp):
    """Return (macro lines, LaTeX table, efficiency-table rows) for the adaptive implicit Euler study.

    Reads data/adaptive_implicit_*.csv (written by experiments.py step [7b/8]). Returns empty pieces if they do not exist,
    so the older pipeline keeps working."""
    f_tol = data / "adaptive_implicit_tolerance_refinement.csv"
    f_sum = data / "adaptive_implicit_demo_summary.csv"
    if not (f_tol.exists() and f_sum.exists()):
        return [], "", []
    ai = pd.read_csv(f_tol)
    ai = ai[ai.status == "completed"]
    ext = ai[ai.variant == "extrapolated"].sort_values("tol", ascending=False)
    fo = ai[ai.variant == "first-order"].sort_values("tol", ascending=False)
    ds = dict(zip(*pd.read_csv(f_sum).T.values))
    ee_ok = ad[ad.status == "completed"]
    ee_demo = ee_ok[np.isclose(ee_ok.tol, ds["tol"], rtol=1e-9, atol=0.0)].iloc[0]

    def ms(x):
        return f"{1e3 * float(x):.1f}"

    # ---- macros
    L = [
        macro("AITol", sci_plain(ds["tol"], 1)), macro("AINewtonTol", sci_plain(ds["newton_tolerance"], 1)),
        macro("AIAcc", f"{int(ds['accepted_steps'])}"), macro("AIRej", f"{int(ds['rejected_steps'])}"),
        macro("AIUpd", f"{int(ds['newton_updates']):,}".replace(",", "\\,")),
        macro("AIHmin", sci_plain(ds["h_min"], 2)), macro("AIHmax", sci_plain(ds["h_max"], 2)),
        macro("AIHmed", sci_plain(ds["h_median"], 2)),
        macro("AIEinf", sci_plain(ds["E_inf"], 3)), macro("AIFinalErr", sci_plain(ds["final_error_L2"], 3)),
        macro("AIEOverTol", f"{ds['E_inf'] / ds['tol']:.2f}"),
        macro("AIDefect", sci_plain(ds["conservation_defect"], 2)),
        macro("AIEtolLo", f"{ds['local_err_over_tol_p05']:.2f}"), macro("AIEtolMed", f"{ds['local_err_over_tol_median']:.2f}"),
        macro("AIEtolHi", f"{ds['local_err_over_tol_p95']:.2f}"), macro("AIEtolMax", f"{ds['local_err_over_tol_max']:.2f}"),
        macro("AIRatioLo", f"{ds['h_over_EE_limit_p05_t_gt_0.1']:.0f}"),
        macro("AIRatioMed", f"{ds['h_over_EE_limit_median_t_gt_0.1']:.0f}"),
        macro("AIRatioHi", f"{ds['h_over_EE_limit_p95_t_gt_0.1']:.0f}"),
        macro("AIStepsFactor", f"{ee_demo.accepted_steps / ds['accepted_steps']:.0f}"),
        macro("AIErrFactor", f"{float(ee_demo.max_L2_error) / ds['E_inf']:.0f}"),
        macro("AIEOverTolMin", f"{ext.E_over_tol.min():.2f}"), macro("AIEOverTolMax", f"{ext.E_over_tol.max():.2f}"),
        macro("AIFOOverTolMin", f"{fo.E_over_tol.min():.0f}"), macro("AIFOOverTolMax", f"{fo.E_over_tol.max():.0f}"),
        macro("AIExtMinComp", sci_plain(ext.min_component.min(), 1) if ext.min_component.min() != 0 else "0"),
        macro("AIExtDefectMax", sci_plain(ext.conservation_error.max(), 2)),
        macro("AIEEsatLo", sci_plain(ee_ok.max_L2_error.astype(float).min(), 2)),
        macro("AIEEsatHi", sci_plain(ee_ok[ee_ok.tol <= 1e-6].max_L2_error.astype(float).max(), 2)),
        macro("AIExtSmallestE", sci_plain(ext.E_inf.min(), 2)),
        macro("AIExtSmallestTol", sci_plain(ext.tol.min(), 1)),
    ]

    # ---- side-by-side table
    body = []
    for tol in sorted(set(ad.tol) | set(ext.tol), reverse=True):
        re_all = ad[np.isclose(ad.tol, tol, rtol=1e-9, atol=0.0)]
        rf = fo[np.isclose(fo.tol, tol, rtol=1e-9, atol=0.0)]
        rx = ext[np.isclose(ext.tol, tol, rtol=1e-9, atol=0.0)]
        if len(re_all) == 0:
            ee_a, ee_e = "--", "not run"
        else:
            re = re_all.iloc[0]
            ee_a = f"{int(re.accepted_steps)}" if re.status == "completed" else "--"
            ee_e = sci_tex(re.max_L2_error, 2) if re.status == "completed" else "diverged"
        c_f = (f"{int(rf.iloc[0].accepted_steps)} & {sci_tex(rf.iloc[0].E_inf, 2)}" if len(rf) else "-- & --")
        c_x = (f"{int(rx.iloc[0].accepted_steps)} & {sci_tex(rx.iloc[0].E_inf, 2)} & ${rx.iloc[0].E_over_tol:.2f}$"
               if len(rx) else "-- & -- & --")
        body.append(f"{sci_tex(tol, 1)} & {ee_a} & {ee_e} & {c_f} & {c_x}\\\\")
    table = (
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Adaptive step-doubling controllers on the same tolerances: Explicit Euler (Table~\\ref{tab:adaptive}), "
        "implicit Euler keeping the two-half-step value (first order) and implicit Euler with local extrapolation "
        "$2y_{\\rm two}-y_{\\rm full}$ (second order). ``Acc.'' counts accepted steps; all runs use $h_0=10^{-4}$ and the same "
        "absolute max-norm test $\\|y_{\\rm two}-y_{\\rm full}\\|_\\infty\\le\\mathrm{tol}$; the implicit runs use "
        "$h_{\\max}=5$ and a Newton tolerance of $10^{-13}$.}\n"
        "\\label{tab:adaptive-implicit}\n\\small\n"
        "\\resizebox{\\linewidth}{!}{\\begin{tabular}{r rr rr rrr}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Explicit Euler} & \\multicolumn{2}{c}{Implicit Euler, 1st order} & "
        "\\multicolumn{3}{c}{Implicit Euler, extrapolated}\\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-8}\n"
        "Tol. & Acc. & $E_\\infty$ & Acc. & $E_\\infty$ & Acc. & $E_\\infty$ & $E_\\infty/\\mathrm{tol}$\\\\\n\\midrule\n"
        + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")

    # ---- efficiency-table rows: fastest run meeting the target, and the run closest to the matched accuracy of Explicit Euler
    def row(r, label):
        minc = "$\\ge0$" if r.min_component >= -POS_TOL else sci_tex(r.min_component, 2)
        return (f"{label} & -- & {sci_tex(r.E_inf, 3)} & {minc} & ${ms(r.wall_clock_time)}$ & "
                f"{int(r.accepted_steps)} steps ({int(r.rejected_steps)} rejected); {int(r.newton_updates)} Newton updates\\\\")
    rows = ["\\midrule",
            "\\multicolumn{6}{@{}l}{\\emph{(c) Self-implemented adaptive Euler methods}}\\\\"]
    tgt = {}
    for var in ["extrapolated", "first-order"]:
        d = ai[(ai.variant == var) & (ai.E_inf <= 1e-4) & ai.nonnegative.astype(bool)]
        if len(d):
            r = d.loc[d.newton_updates.idxmin()]
            tgt[var] = r
            rows.append(row(r, f"Adaptive IE, {var}, tol$={sci_plain(r.tol, 1)}$"))
    E_match = float(ee_wp.E_inf)
    d = ai[(ai.variant == "extrapolated")].copy()
    d["gap"] = np.abs(np.log(d.E_inf / E_match))
    r = d.loc[d.gap.idxmin()]
    match_ok = bool(r.gap < np.log(2.0))
    if match_ok:
        rows.append(row(r, f"Adaptive IE, extrapolated, tol$={sci_plain(r.tol, 1)}$ (near-matched error)"))
    if "extrapolated" in tgt:
        r0 = tgt["extrapolated"]
        L += [macro("AITargetTol", sci_plain(r0.tol, 1)), macro("AITargetE", sci_plain(r0.E_inf, 2)),
              macro("AITargetSteps", f"{int(r0.accepted_steps)}"), macro("AITargetUpd", f"{int(r0.newton_updates)}"),
              macro("AITargetMs", ms(r0.wall_clock_time))]
        fixed_ie = wp[(wp.method == "Implicit Euler") & (wp.E_inf <= 1e-4) & wp.nonnegative.astype(bool)]
        if len(fixed_ie):
            best_ie = fixed_ie.loc[fixed_ie.newton_updates.idxmin()]
            L.append(macro("AITargetUpdVsIE", f"{float(best_ie.newton_updates) / float(r0.newton_updates):.1f}"))
            L.append(macro("AITargetTimeVsIE", f"{float(best_ie.wall_clock_time) / float(r0.wall_clock_time):.1f}"))
    if match_ok:
        L += [macro("AIMatchTol", sci_plain(r.tol, 1)), macro("AIMatchE", sci_plain(r.E_inf, 2)),
              macro("AIMatchSteps", f"{int(r.accepted_steps)}"), macro("AIMatchUpd", f"{int(r.newton_updates)}"),
              macro("AIMatchMs", ms(r.wall_clock_time)),
              macro("AIMatchStepsVsEE", f"{float(ee_wp.steps) / float(r.accepted_steps):.0f}")]
    ae = pd.read_csv(data / "adaptive_explicit_work_precision.csv")
    ae = ae[(ae.E_inf <= 1e-4) & ae.nonnegative.astype(bool)]
    if len(ae):
        r = ae.loc[ae.rhs_evaluations.idxmin()]
        rows.append(f"Adaptive EE, tol$={sci_plain(r.tol, 1)}$ & -- & {sci_tex(r.E_inf)} & $\\ge0$ & "
                    f"${ms(r.wall_clock_time)}$ & {int(r.accepted_steps)} steps ({int(r.rejected_steps)} rejected); "
                    f"{int(r.rhs_evaluations)} RHS evaluations\\\\")
    L += newton_rule_check(ext)
    return L, table, rows


# ---------------------------------------------------------------- Newton-tolerance rule check for adaptive IE (Section 5.5)
def newton_rule_check(ext: "pd.DataFrame", newton_tol_fixed: float = 1e-13):
    """Macro lines reporting, for the two finest extrapolated-IE tolerances, how far the
    fixed Newton tolerance of this study (tau_N) departs from the rule tau_N <~ E_target/N_steps
    derived in Section 5.5.1 (N taken as the accepted-step count, as there)."""
    L = []
    for tol_target, tag in [(1e-9, "9"), (1e-10, "10")]:
        d = ext[np.isclose(ext.tol, tol_target, rtol=1e-6, atol=0.0)]
        if not len(d):
            continue
        r = d.iloc[0]
        N = float(r.accepted_steps)
        ratio = (N * newton_tol_fixed) / float(r.E_inf)
        L.append(macro(f"NewtonRuleRatio{tag}", f"{ratio:.1f}"))
    return L


# ---------------------------------------------------------------- RK4/CN vs. extrapolated adaptive IE (Section 5.6)
def rk_cn_vs_extrapolated_ie(wp: "pd.DataFrame", ai: "pd.DataFrame"):
    """Compare RK4 and CN fixed-step runs with the extrapolated adaptive-IE run
    that would reach the SAME achieved error, found by log-log interpolation of
    (E_inf, wall_clock_time) over the extrapolated tolerance sweep.

    Returns (macro lines, LaTeX table). Both are empty if the extrapolated-IE
    sweep is not available, so the older pipeline keeps working."""
    ext = ai[ai.variant == "extrapolated"].sort_values("E_inf")
    if len(ext) < 2:
        return [], ""
    logE = np.log(ext.E_inf.values)
    logT = np.log(ext.wall_clock_time.values)
    order = np.argsort(logE)
    logE, logT = logE[order], logT[order]
    e_lo, e_hi = ext.E_inf.min(), ext.E_inf.max()

    targets = [("RK4", 8e-4), ("RK4", 4e-4), ("RK4", 2e-4), ("RK4", 1e-4),
               ("Trapezoidal / CN", 1.25e-4), ("Trapezoidal / CN", 1e-3)]
    rows, ratios = [], []
    for method, h in targets:
        d = wp[(wp.method == method) & np.isclose(wp.h, h)]
        if not len(d):
            continue
        r = d.iloc[0]
        E, t_ms = float(r.E_inf), float(r.wall_clock_time) * 1e3
        in_range = e_lo <= E <= e_hi
        if in_range:
            ie_ms = float(np.exp(np.interp(np.log(E), logE, logT))) * 1e3
            ratio = t_ms / ie_ms
            ratios.append(ratio)
        else:
            ie_ms, ratio = None, None
        rows.append((method, h, E, t_ms, ie_ms, ratio, in_range))

    lines = [
        "\\begin{table}[H]\n\\centering\n",
        "\\caption{Fixed-step RK4 and Trapezoidal/CN against the extrapolated adaptive-IE run "
        "reaching the same achieved error $E_\\infty$, found by log-log interpolation of the "
        "extrapolated-IE tolerance sweep (Table~\\ref{tab:adaptive-implicit}). ``IE time (interp.)'' "
        "is the interpolated, not measured, wall-clock time; ``ratio'' is fixed-step time divided by it.}\n",
        "\\label{tab:rk-cn-vs-ie}\n\\small\n",
        "\\begin{tabular}{l r r r r r}\n\\toprule\n",
        "Method & $h$ & $E_\\infty$ & time (ms) & IE time, interp.\\ (ms) & ratio \\\\\n\\midrule\n",
    ]
    for method, h, E, t_ms, ie_ms, ratio, in_range in rows:
        if in_range:
            lines.append(f"{method} & {sci_plain(h, 1)} & {sci_tex(E, 2)} & {t_ms:.1f} & {ie_ms:.2f} & {ratio:.1f}$\\times$\\\\\n")
        else:
            lines.append(f"{method} & {sci_plain(h, 1)} & {sci_tex(E, 2)} & {t_ms:.1f} & -- & outside sweep range\\\\\n")
    lines += ["\\bottomrule\n\\end{tabular}\n\\end{table}\n"]
    table = "".join(lines)

    macros = []
    if ratios:
        macros.append(macro("RKvsIEMinRatio", f"{min(ratios):.1f}"))
        macros.append(macro("RKvsIEMaxRatio", f"{max(ratios):.1f}"))
    return macros, table



def adaptive_ie_stability_macros(sr):
    """Macros for the stability functions of the two accepted states of the step-doubling IE controller.
    Reads the extra keys written by stability_region.py; returns [] for older data files."""
    if "ext_R_zero_crossing" not in sr:
        return []
    f = lambda k: float(sr[k])
    return [
        macro("ExtMaxLHP", f"{f('ext_max_abs_R_left_halfplane'):.6f}"),
        macro("IETwoMaxLHP", f"{f('ie2_max_abs_R_left_halfplane'):.6f}"),
        macro("ExtCFDev", sci_plain(f("ext_closed_form_max_deviation"), 1)),
        macro("ExtIdErr", sci_plain(f("ext_imag_axis_identity_max_error"), 1)),
        macro("ExtZero", f"{abs(f('ext_R_zero_crossing')):.3f}"),
        macro("ExtMinR", f"{abs(f('ext_R_negative_minimum')):.4f}"),
        macro("ExtMinAt", f"{abs(f('ext_R_negative_minimum_at')):.2f}"),
        macro("ExtRhpEnd", f"{f('ext_rhp_real_axis_end'):.3f}"),
        macro("ExtMinus", sci_plain(f("ext_R_at_minus_1e8"), 2)),
        macro("ExtErrConst", f"{f('ext_error_constant_over_z3'):.3f}"),
        macro("IETwoErrConst", f"{f('ie2_error_constant_over_z2'):.3f}"),
        macro("EstAmpMax", f"{f('est_ie_max_amplification'):.4f}"),
        macro("EstAmpAt", f"{abs(f('est_ie_max_amplification_at')):.3f}"),
    ]
 
 
def fast_mode_table(df):
    """Growth factors and estimator amplification at z = h*lambda_fast."""
    rows = []
    for _, r in df.iterrows():
        lab = (f"fixed $h={sci_plain(r.h, 1)}$" if r.kind == "fixed"
               else f"adaptive IE, median $h={r.h:.3f}$ ($t>0.1$)")
        rows.append(f"{lab} & ${r.z:.1f}$ & ${r.R_CN:+.3f}$ & ${r.R_IE:+.3f}$ & ${r.R_EXT:+.4f}$ & "
                    f"${r.est_IE:.3f}$ & {sci_tex(r.est_EE, 2)}\\\\")
    return (
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Growth factors and step-doubling-estimator amplification at $z=h\\lambda_{\\rm fast}$ with "
        "$\\lambda_{\\rm fast}=-2.19\\times10^{3}$. $R_{\\rm CN}$, $R_{\\rm IE}$ and $R_{\\rm ext}$ are the growth factors of "
        "Trapezoidal/CN, Implicit Euler and the locally extrapolated state $2y_{\\rm two}-y_{\\rm full}$; "
        "``est.\\ IE'' and ``est.\\ EE'' are $|R_{\\rm two}-R_{\\rm full}|$ for Implicit and Explicit Euler, i.e.\\ how much of "
        "a fast-mode amplitude of $1$ enters the error estimate $e_n$. The last row uses the median accepted macro-step of the "
        "demonstration run of Section~\\ref{sec:adaptive-method}.}\n"
        "\\label{tab:R-fast-mode}\n\\small\n"
        "\\resizebox{\\linewidth}{!}{\\begin{tabular}{l r r r r r r}\n\\toprule\n"
        "Step & $z$ & $R_{\\rm CN}$ & $R_{\\rm IE}$ & $R_{\\rm ext}$ & est.\\ IE & est.\\ EE\\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")



def main(root: Path):
    data = root / "data"
    gen = root / "generated"
    gen.mkdir(exist_ok=True)
    for name in STALE:
        (gen / name).unlink(missing_ok=True)

    rv = pd.read_csv(data / "reference_validation.csv")
    v = dict(zip(rv.check, rv.value))
    meta = pd.read_csv(data / "euler_stability_metadata.csv")
    m = dict(zip(meta.quantity, meta.value))
    fits = pd.read_csv(data / "order_fits.csv").set_index("method")
    conv = pd.read_csv(data / "four_method_results.csv")
    wp = pd.read_csv(data / "work_precision.csv")
    lib = pd.read_csv(data / "library_baseline.csv").iloc[0]
    stab = pd.read_csv(data / "euler_stability_boundary.csv")
    nt = pd.read_csv(data / "newton_tolerance.csv")
    ng = pd.read_csv(data / "newton_initial_guess_study.csv").set_index("initial_guess")
    ad = pd.read_csv(data / "adaptive_tolerance_refinement.csv")
    # Section 4 inputs
    sr = dict(zip(*pd.read_csv(data / "stability_region_checks.csv").T.values))
    sm = dict(zip(*pd.read_csv(data / "stiffness_metadata.csv").T.values))
    orient = pd.read_csv(data / "stiffness_orientation_check.csv")
    hl = pd.read_csv(data / "hlambda_exceedance.csv")
    ds = dict(zip(*pd.read_csv(data / "adaptive_demo_summary.csv").T.values))     # Section 3.5 demonstration run
    qs = dict(zip(*pd.read_csv(data / "qssa_summary.csv").T.values))               # Section 6.1 (QSSA)
    qt = pd.read_csv(data / "qssa_table.csv")

    # ------------------------------------------------------------ macros
    hcrit, hemp = m["theoretical_h_critical"], m["empirical_h_boundary_bisection"]
    pert = ng.loc["Perturbed predictor"]
    ee_wp = wp[(wp.method == "Explicit Euler") & (np.isclose(wp.h, 5e-4))].iloc[0]
    ie_match = wp[(wp.method == "Implicit Euler") & (np.isclose(wp.h, 6.5e-4))].iloc[0]

    
    rk_wp   = wp[(wp.method == "RK4") & (np.isclose(wp.h, 8e-4))].iloc[0]
    ee_fine = wp[(wp.method == "Explicit Euler") & (np.isclose(wp.h, 1.25e-4))].iloc[0]

    ad_ok = ad[ad.status == "completed"]
    ad_fail = ad[ad.status != "completed"]
    lines = ["% Auto-generated by code/make_report_tables.py. Do not hand-edit.\n"]
    lines += [
        macro("RefineDisc", sci_plain(v["Radau_100x_refinement_max_L2"])),
        macro("BDFDisc", sci_plain(v["Radau_vs_BDF_max_L2"])),
        macro("RefineDiscDense", sci_plain(v["Radau_100x_refinement_dense_max_L2"])),
        macro("BDFDiscDense", sci_plain(v["Radau_vs_BDF_dense_max_L2"])),
        macro("RefMass", sci_plain(v["reference_max_mass_defect"])),
        # 9 decimals: the 10th decimal of y3 lies inside the Radau--BDF spread
        macro("RefYOne", f"{float(v['reference_y1_T40']):.9f}"),
        macro("RefYTwo", sci_plain(v["reference_y2_T40"], 5)),
        macro("RefYThree", f"{float(v['reference_y3_T40']):.9f}"),
        macro("Hcrit", sci_plain(hcrit, 4)),
        macro("LambdaFast", sci_plain(m["max_fast_negative_real_eigenvalue"], 4)),
        macro("Hemp", sci_plain(hemp, 4)),
        macro("HempPercent", f"{100.0 * (hemp / hcrit - 1.0):.1f}"),
        macro("TExceed", f"{m['time_local_bound_exceeded_at_h_emp']:.1f}"),
        macro("HcritRK", sci_plain(m["rk4_h_critical_real_axis"], 3)),
        macro("RKPercent", f"{100.0 * 8e-4 / m['rk4_h_critical_real_axis']:.0f}"),
        macro("FitEE", f"{fits.loc['Explicit Euler', 'fit_three_finest']:.2f}"),
        macro("FitRK", f"{fits.loc['RK4', 'fit_three_finest']:.2f}"),
        macro("FitIE", f"{fits.loc['Implicit Euler', 'fit_three_finest']:.3f}"),
        macro("FitCN", f"{fits.loc['Trapezoidal / CN', 'fit_three_finest']:.3f}"),
        macro("GuessOkE", sci_plain(ng.loc["Euler predictor", "E_inf"])),
        macro("GuessPertE", f"{float(pert['E_inf']):.2f}"),
        macro("GuessPertNeg", f"{int(pert['negative_count'])}"),
        macro("EEOverTarget", f"{1e-4 / float(ee_wp['E_inf']):.0f}"),

        macro("EEBestE",      sci_plain(ee_wp["E_inf"], 2)),
        macro("EEWork",       f"{int(ee_wp['work_count'])}"),
        macro("RKBestE",      sci_plain(rk_wp["E_inf"], 2)),
        macro("RKWork",       f"{int(rk_wp['work_count'])}"),
        macro("RKErrGain",    f"{float(ee_wp['E_inf']) / float(rk_wp['E_inf']):.0f}"),
        macro("RKOverTarget", f"{1e-4 / float(rk_wp['E_inf']):.0f}"),
        macro("RKStepOverEE", f"{8e-4 / hemp:.2f}"),
        macro("EEFineWork",   f"{int(ee_fine['work_count'])}"),
        macro("RKSpeedup",    f"{float(ee_fine['wall_clock_time']) / float(rk_wp['wall_clock_time']):.1f}"),


        macro("MatchRatio", f"{float(ie_match['wall_clock_time']) / float(ee_wp['wall_clock_time']):.1f}"),
        macro("MatchStepsIE", f"{int(ie_match['steps'])}"),
        macro("LibE", sci_plain(lib["E_inf"], 2)),
        macro("LibSteps", f"{int(lib['steps'])}"),
        macro("LibNfev", f"{int(lib['nfev'])}"),
        macro("AdaptRejPercent", f"{100.0 * ad_ok[np.isclose(ad_ok.tol, 1e-6)].rejection_fraction.iloc[0]:.0f}"),
        macro("AdaptRatioLoose", f"{float(ad_ok[np.isclose(ad_ok.tol, 1e-6)].E_over_tol.iloc[0]):.0f}"),
        macro("AdaptRatioTight", f"{float(ad_ok[np.isclose(ad_ok.tol, 1e-8)].E_over_tol.iloc[0]):.0f}"),
        # ---- Section 4
        macro("RKLimit", f"{float(sr['rk4_real_axis_limit']):.4f}"),
        macro("RKMinR", f"{float(sr['rk4_real_axis_minimum_R']):.3f}"),
        macro("RKMinAt", f"{abs(float(sr['rk4_real_axis_argmin'])):.2f}"),
        macro("IEMinus", sci_plain(sr["ie_R_at_minus_1e8"], 2)),
        macro("CNMinus", f"{float(sr['cn_R_at_minus_1e8']):.4f}"),
        macro("CNDev", sci_plain(sr["cn_max_abs_deviation_from_1_on_imaginary_axis"], 1)),
        macro("TCross", sci_plain(sm["crossing_time_analytic_1_over_2k2"], 3)),
        macro("SFirst", f"{float(sm['S_at_first_output_time']):.2f}"),
        macro("KapPeak", f"{float(sm['kappa_peak_output_grid']):.1f}"),
        macro("KapPeakT", sci_plain(sm["kappa_peak_time_output_grid"], 3)),
        macro("KapPeakFine", f"{float(sm['kappa_peak_fine_grid']):.0f}"),
        macro("KapMedian", f"{float(sm['kappa_median_output_grid']):.3f}"),
        macro("KapFrac", f"{100.0 * float(sm['kappa_fraction_above_10_output_grid']):.0f}"),
        macro("KapT", f"{float(sm['kappa_at_T']):.2f}"),
        macro("LamSlowT", f"{abs(float(sm['lambda_slow_at_T'])):.4f}"),
        macro("ZeroMax", sci_plain(sm["structural_zero_max_abs_output_grid"], 1)),
        macro("MismatchMax", sci_plain(sm["full_vs_reduced_max_rel_mismatch"], 1)),
        macro("JrOneOne", f"{float(sm['Jr_T_11']):.4f}"), macro("JrOneTwo", f"{float(sm['Jr_T_12']):.1f}"),
        macro("JrTwoOne", f"{float(sm['Jr_T_21']):.4f}"), macro("JrTwoTwo", f"{float(sm['Jr_T_22']):.1f}"),
        macro("TExceedSix", f"{float(hl.loc[np.isclose(hl.h, 6e-4), 't_exceed_EE'].iloc[0]):.1f}"),
        macro("TExceedSeven", f"{float(hl.loc[np.isclose(hl.h, 7e-4), 't_exceed_EE'].iloc[0]):.1f}"),
        macro("TExceedEight", f"{float(hl.loc[np.isclose(hl.h, 8e-4), 't_exceed_EE'].iloc[0]):.1f}"),
        macro("DelaySeven", f"{float(hl.loc[np.isclose(hl.h, 7e-4), 't_diverge_EE'].iloc[0] - hl.loc[np.isclose(hl.h, 7e-4), 't_exceed_EE'].iloc[0]):.1f}"),
        macro("DelayEight", f"{float(hl.loc[np.isclose(hl.h, 8e-4), 't_diverge_EE'].iloc[0] - hl.loc[np.isclose(hl.h, 8e-4), 't_exceed_EE'].iloc[0]):.1f}"),
        # ---- Section 3.5 (adaptive Euler demonstration, tol = 1e-6)
        macro("AdAcc", f"{int(ds['accepted_steps']):,}".replace(",", "\\,")),
        macro("AdRej", f"{int(ds['rejected_steps']):,}".replace(",", "\\,")),
        macro("AdRejPct", f"{100.0 * ds['rejected_steps'] / (ds['accepted_steps'] + ds['rejected_steps']):.0f}"),
        macro("AdHmin", sci_plain(ds["h_min"], 2)), macro("AdHmax", sci_plain(ds["h_max"], 2)),
        macro("AdMean", sci_plain(ds["h_mean"], 3)), macro("AdMedian", sci_plain(ds["h_median"], 3)),
        macro("AdMedMid", sci_plain(ds["h_median_1_to_10"], 2)), macro("AdMedLate", sci_plain(ds["h_median_after_10"], 2)),
        macro("AdHlate", sci_plain(ds["h_min_after_10"], 1)),
        macro("AdStepsRise", f"{int(ds['steps_before_h_exceeds_3e-3'])}"), macro("AdTimeRise", sci_plain(ds["time_h_exceeds_3e-3"], 1)),
        macro("AdFinalOne", f"{ds['final_y1']:.9f}"), macro("AdFinalTwo", sci_plain(ds["final_y2"], 6)),
        macro("AdFinalThree", f"{ds['final_y3']:.9f}"),
        macro("AdFinalErr", sci_plain(ds["final_error_L2"], 3)), macro("AdEinf", sci_plain(ds["E_inf"], 3)),
        macro("AdErrOverTol", f"{ds['final_error_L2'] / ds['tol']:.0f}"),
        macro("AdDefect", sci_plain(ds["conservation_defect"], 3)),
        macro("AdEtolLo", f"{ds['local_err_over_tol_p05']:.2f}"), macro("AdEtolMed", f"{ds['local_err_over_tol_median']:.2f}"),
        macro("AdEtolHi", f"{ds['local_err_over_tol_p95']:.2f}"), macro("AdEtolMax", f"{ds['local_err_over_tol_max']:.2f}"),
        macro("AdSumLocal", sci_plain(ds["sum_local_error_estimates"], 2)), macro("AdNTol", sci_plain(ds["accepted_steps_times_tol"], 2)),
        macro("AdRatioLo", f"{ds['h_over_EE_limit_p05_t_gt_0.1']:.1f}"), macro("AdRatioHi", f"{ds['h_over_EE_limit_p95_t_gt_0.1']:.1f}"),
        # finest-run errors of the convergence study (cross-order consistency, Section 5.2)
        macro("EFineEE", sci_plain(conv[conv.method == "Explicit Euler"].sort_values("h").E_inf.iloc[0], 2)),
        macro("EFineRK", sci_plain(conv[conv.method == "RK4"].sort_values("h").E_inf.iloc[0], 2)),
        macro("EFineIE", sci_plain(conv[conv.method == "Implicit Euler"].sort_values("h").E_inf.iloc[0], 2)),
        macro("EFineCN", sci_plain(conv[conv.method == "Trapezoidal / CN"].sort_values("h").E_inf.iloc[0], 2)),
        # ---- Section 6.1 (QSSA / singular perturbation)
        macro("DeltaSmall", sci_plain(qs["delta"], 3)), macro("KappaThree", f"{qs['kappa3']:.2f}"),
        macro("YStar", sci_plain(qs["y2_star_early_y1_1"], 3)), macro("TFast", sci_plain(qs["t_fast_layer"], 3)),
        macro("TNinety", sci_plain(qs["t99_theory"], 3)), macro("TNinetyRef", sci_plain(qs["t99_reference"], 3)),
        macro("YMaxRef", sci_plain(qs["y2_max_reference"], 5)), macro("TYMax", sci_plain(qs["t_of_y2_max_reference"], 3)),
        macro("YMaxGap", f"{100.0 * qs['y2_max_rel_gap_to_early_balance']:.2f}"),
        macro("InnerErr", sci_plain(qs["inner_solution_max_rel_err_t_le_1e-3"], 2)),
        macro("CompErr", sci_plain(qs["composite_max_rel_err_all_t"], 2)),
        macro("QuadErrLate", sci_plain(qs["quad_qssa_max_rel_err_t_ge_5e-3"], 2)),
        macro("QuadErrT", sci_plain(qs["quad_qssa_rel_err_at_T"], 2)),
        macro("QuadErrLayer", f"{100.0 * qs['quad_qssa_rel_err_at_2e-3']:.1f}"),
        macro("EarlyErrMid", f"{100.0 * qs['early_balance_max_rel_err_5e-3_to_1e-2']:.2f}"),
        macro("EarlyErrOne", f"{100.0 * qs['early_balance_rel_err_at_1']:.0f}"),
        macro("EarlyRatioT", f"{qs['early_balance_ratio_at_T']:.1f}"),
        macro("LateErrT", f"{100.0 * qs['late_balance_rel_err_at_T']:.1f}"),
        macro("LamPlateau", sci_plain(qs["lambda_plateau_prediction_2sqrt_k1k2"], 4)),
        macro("LamPlateauNum", sci_plain(qs["lambda_numeric_at_1e-2"], 4)),
        macro("LamFormT", sci_plain(qs["lambda_formula_at_T"], 5)), macro("LamNumT", sci_plain(qs["lambda_numeric_at_T"], 5)),
        macro("HcritForm", sci_plain(qs["hcrit_from_formula"], 4)),
        macro("RedDYOne", sci_plain(qs["reduced_model_max_abs_dy1_t_ge_1e-2"], 2)),
        macro("RedDYThree", sci_plain(qs["reduced_model_max_abs_dy3_t_ge_1e-2"], 2)),
        macro("KTf", sci_plain(qs["k1_times_t_fast"], 3)),
        macro("AdaptFailTime", f"{float(ad_fail.t_reached.iloc[0]):.2f}" if len(ad_fail) else "\\mathrm{n/a}"),
    ]
    aie_lines, aie_table, aie_eff_rows = adaptive_implicit_block(data, wp, ad, ee_wp)
    lines += aie_lines

    lines += adaptive_ie_stability_macros(sr)

    f_ai = data / "adaptive_implicit_tolerance_refinement.csv"
    if f_ai.exists():
        ai_ok = pd.read_csv(f_ai)
        ai_ok = ai_ok[ai_ok.status == "completed"]
        rc_macros, rc_table = rk_cn_vs_extrapolated_ie(wp, ai_ok)
        lines += rc_macros
        if rc_table:
            write(gen / "table_rk_cn_vs_ie.tex", rc_table)
    write(gen / "macros.tex", "".join(lines))
    if aie_table:
        write(gen / "table_adaptive_implicit.tex", aie_table)

    f_fm = data / "stability_R_fast_mode.csv"
    if f_fm.exists():
        write(gen / "table_R_fast_mode.tex", fast_mode_table(pd.read_csv(f_fm)))       

    # ------------------------------------------------------------ Table: reference
    n_pts = int(v["dense_check_grid_points"])
    ref_table = rf"""\begin{{table}}[H]
\centering
\caption{{Validation checks for the reference solution (maximum Euclidean discrepancy).}}
\label{{tab:reference}}
\small
\begin{{tabular}}{{@{{}}llr@{{}}}}
\toprule
Check & Compared at & Discrepancy\\
\midrule
Radau, 100 * looser & 201 output times & {sci_tex(v['Radau_100x_refinement_max_L2'])}\\
Radau vs.\ BDF (same tolerances) & 201 output times & {sci_tex(v['Radau_vs_BDF_max_L2'])}\\
Radau, 100 * looser & ${n_pts // 1000}{{,}}{n_pts % 1000:03d}$ fine-grid points (dense output) & {sci_tex(v['Radau_100x_refinement_dense_max_L2'])}\\
Radau vs.\ BDF & ${n_pts // 1000}{{,}}{n_pts % 1000:03d}$ fine-grid points (dense output) & {sci_tex(v['Radau_vs_BDF_dense_max_L2'])}\\
Invariant defect $\max|y_1+y_2+y_3-1|$ & 201 output times & {sci_tex(v['reference_max_mass_defect'])}\\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    write(gen / "table_reference.tex", ref_table)

    # ------------------------------------------------------------ Table: convergence
    order = ["Explicit Euler", "RK4", "Implicit Euler", "Trapezoidal / CN"]
    rows = []
    for method in order:
        d = conv[conv.method == method].sort_values("h", ascending=False).reset_index(drop=True)
        for j, r in d.iterrows():
            obs = "--" if _isnan(r.observed_order) else f"${float(r.observed_order):.3f}$"
            fit = f"${fits.loc[method, 'fit_three_finest']:.3f}$" if j == len(d) - 1 else ""
            rows.append(f"{method if j == 0 else ''} & {sci_tex(r.h)} & {sci_tex(r.E_inf, 4)} & {obs} & "
                        f"{yn(r.nonnegative)} & {fit}\\\\")
        if method != order[-1]:
            rows.append("\\addlinespace[0.25em]")
    all_reached = bool(conv.reached_T.all() and conv.finite.all())
    note = ("All 16 runs reached $T=40$ with finite values (stable in this sense); "
            if all_reached else "Runs that did not reach $T=40$ are flagged in the data files; ")
    conv_table = (
        "\\begin{table}[H]\n\\centering\n"
        f"\\caption{{Four-method convergence study. {note}``Non-neg.'' states whether $\\min_{{n,j}}y_{{j,n}}\\ge-10^{{-12}}$; "
        "$p_{\\mathrm{fit}}$ is the least-squares slope over the three finest step sizes.}\n"
        "\\label{tab:convergence}\n\\small\n\\begin{tabular}{lrrrcr}\n\\toprule\n"
        "Method & $h$ & $E_\\infty$ & Pairwise order & Non-neg. & $p_{\\mathrm{fit}}$\\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(gen / "table_convergence.tex", conv_table)

    # ------------------------------------------------------------ Table: Explicit-Euler boundary
    body = []
    for _, r in stab.iterrows():
        if r.status == "diverged":
            status = f"diverged ($t\\approx{float(r.t_diverge):.1f}$)"
            minc, ferr = "--", "--"
        else:
            status = "admissible" if r.status == "admissible" else "negative values"
            minc, ferr = sci_tex(r.min_component), sci_tex(pd.to_numeric(r.final_error, errors="coerce"))
        body.append(f"{sci_tex(r.h)} & {status} & {minc} & {sci_tex(r.conservation_error)} & {ferr}\\\\")
    boundary_table = (
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Explicit Euler near the predicted limit $h_{\\mathrm{crit}}$. "
        "Admissible: reaches $T$ with $\\min y_j\\ge-10^{-12}$; negative: reaches $T$ but violates non-negativity; "
        "diverged: blow-up before $T$ (time at which $\\max_j|y_j|$ first exceeds $2$). "
        "The invariant defect is evaluated only over steps with $\\max_j|y_j|\\le2$, "
        "because beyond that it measures round-off of huge numbers.}\n"
        "\\label{tab:boundary}\n\\small\n\\begin{tabular}{rlrrr}\n\\toprule\n"
        "$h$ & Status & Minimum $y_j$ & Invariant defect & Final-time error\\\\\n\\midrule\n"
        + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(gen / "table_boundary.tex", boundary_table)

    # ------------------------------------------------------------ Table: Newton tolerance
    body = []
    for _, r in nt.iterrows():
        if r.status == "failed":
            status = f"failed ($t={float(r.fail_time):.2f}$)"
        elif r.status == "completed_nonphysical":
            status = "non-physical"
        else:
            status = "completed"
        body.append(f"{sci_tex(r.newton_tolerance, 1)} & {status} & {sci_tex(r.final_y2, 4)} & {sci_tex(r.abs_y2_error, 3)} & "
                    f"{sci_tex(r.final_reference_error, 3)} & "
                    f"{'--' if _isnan(r.total_newton_updates) else int(r.total_newton_updates)} & "
                    f"{sci_tex(r.max_newton_residual, 2)} & {sci_tex(r.conservation_error, 2)}\\\\")
    newton_table = (
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Newton-tolerance sensitivity of Implicit Euler at fixed $h=10^{-2}$ (4000 steps). "
        "The stopping test is $\\|F\\|_\\infty\\le\\tau_N$ on the absolute residual; ``Newton updates'' counts linear solves "
        "summed over all steps.}\n"
        "\\label{tab:newton}\n\\scriptsize\n\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{r l r r r r r r}\n\\toprule\n"
        "$\\tau_N$ & Status & $y_2(40)$ & $|\\Delta y_2|$ & Final $L^2$ error & Newton updates & Max residual & Invariant defect\\\\\n\\midrule\n"
        + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    write(gen / "table_newton.tex", newton_table)

    # ------------------------------------------------------------ Table: adaptive
    body = []
    for _, r in ad.iterrows():
        if r.status == "completed":
            body.append(f"{sci_tex(r.tol, 1)} & completed & {int(r.accepted_steps)} & {int(r.rejected_steps)} & "
                        f"{sci_tex(r.max_L2_error, 3)} & ${float(r.E_over_tol):.0f}$ & {sci_tex(r.mean_accepted_h, 3)}\\\\")
        else:
            body.append(f"{sci_tex(r.tol, 1)} & diverged ($t\\approx{float(r.t_reached):.2f}$) & {int(r.accepted_steps)} & "
                        f"{int(r.rejected_steps)} & -- & -- & --\\\\")
    adaptive_table = (
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Adaptive Explicit Euler (step doubling, $h_0=10^{-4}$, $h_{\\max}=10^{-2}$). "
        "The tolerance bounds the \\emph{local} error estimate per step; $E_\\infty$ is the accumulated \\emph{global} error.}\n"
        "\\label{tab:adaptive}\n\\small\n\\begin{tabular}{rlrrrrr}\n\\toprule\n"
        "Tol. & Status & Accepted & Rejected & $E_\\infty$ & $E_\\infty/\\mathrm{tol}$ & Mean accepted $h$\\\\\n\\midrule\n"
        + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(gen / "table_adaptive.tex", adaptive_table)

    # ------------------------------------------------------------ Table: efficiency
    def pick(method, cond):
        d = wp[(wp.method == method) & cond(wp)]
        return d.loc[d.wall_clock_time.idxmin()] if len(d) else None

    ok = lambda d: (d.E_inf <= 1e-4) & d.nonnegative.astype(bool)
    anyneg = lambda d: (d.E_inf <= 1e-4)

    def work_info(r):
        if r.work_type == "RHS evaluations":
            return f"{int(r.steps)} steps; {int(r.work_count)} RHS evaluations"
        return f"{int(r.steps)} steps; {int(r.newton_updates)} Newton updates"

    def line(r, label=None, flag=""):
        minc = "$\\ge0$" if r.min_component >= -POS_TOL else sci_tex(r.min_component, 2)
        return (f"{label or r.method}{flag} & {sci_tex(r.h, 3)} & {sci_tex(r.E_inf, 3)} & {minc} & "
                f"${1e3 * r.wall_clock_time:.1f}$ & {work_info(r)}\\\\")

    grpA = [pick(mm, ok) for mm in ["Explicit Euler", "Implicit Euler", "RK4", "Trapezoidal / CN"]]
    cn_fast = pick("Trapezoidal / CN", anyneg)
    ee_row = wp[(wp.method == "Explicit Euler") & np.isclose(wp.h, 5e-4)].iloc[0]
    ie_row = wp[(wp.method == "Implicit Euler") & np.isclose(wp.h, 6.5e-4)].iloc[0]
    rowsE = ["\\multicolumn{6}{@{}l}{\\emph{(a) Target $E_\\infty\\le10^{-4}$: fastest tested run per method}}\\\\"]
    rowsE += [line(r) for r in grpA if r is not None]
    if cn_fast is not None and cn_fast.h != grpA[3].h:
        rowsE.append(line(cn_fast, "Trapezoidal / CN", " (non-physical)"))
    rowsE += ["\\midrule",
              "\\multicolumn{6}{@{}l}{\\emph{(b) Matched accuracy $E_\\infty\\approx7\\times10^{-6}$}}\\\\",
              line(ee_row), line(ie_row)]
    rowsE += aie_eff_rows
    rowsE += ["\\midrule",
              "\\multicolumn{6}{@{}l}{\\emph{(%s) Library reference: adaptive implicit solver}}\\\\" % ("d" if aie_eff_rows else "c"),
              (f"SciPy Radau, rtol$=10^{{-6}}$ & -- & {sci_tex(lib.E_inf, 3)} & $\\ge0$ & ${1e3 * lib.wall_clock_time:.1f}$ & "
               f"{int(lib.steps)} steps; {int(lib.nfev)} RHS evaluations; {int(lib.njev)} Jacobians\\\\")]
    eff_table = (
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Cost at comparable accuracy (median of warmed runs, this machine). Times are implementation- and "
        "machine-dependent; the SciPy row is not JIT-compiled and its time is not like-for-like with the compiled kernels, "
        "so compare its work counts.}\n"
        "\\label{tab:efficiency}\n\\small\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}p{0.245\\linewidth} r r c r X@{}}\n\\toprule\n"
        "Method & $h$ & $E_\\infty$ & Min.\\ $y_j$ & Time (ms) & Work\\\\\n\\midrule\n"
        + "\n".join(rowsE) + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}\n")
    write(gen / "table_efficiency.tex", eff_table)

    # ------------------------------------------------------------ Table: stiffness orientation check (Section 4.2)
    body = []
    for _, r in orient.iterrows():
        tt = "$40$" if float(r.t) == 40.0 else sci_tex(r.t, 1)
        sp = sci_tex(r.S_problem_pack, 3 if float(r.S_problem_pack) > 1e5 else 2)
        body.append(f"{tt} & {sci_tex(r.S_ours, 3)} & {sp} & ${100.0 * float(r.rel_diff):.2f}\\%$\\\\")
    stiff_table = (
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Stiffness ratio $S(t)$ from the reduced Jacobian along our own reference, evaluated at the exact orientation "
        "times, compared with the Problem-Pack check values (which are quoted to two or three digits and used only as a "
        "self-consistency target).}\n"
        "\\label{tab:orientation}\n\\small\n\\begin{tabular}{rrrr}\n\\toprule\n"
        "$t$ & our $S(t)$ & Problem-Pack value & relative difference\\\\\n\\midrule\n"
        + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(gen / "table_stiffness_check.tex", stiff_table)

    # ------------------------------------------------------------ Table: local h*lambda exceedance vs observed outcome
    body = []
    for _, r in hl.iterrows():
        texc = "never" if _isnan(r.t_exceed_EE) else f"${float(r.t_exceed_EE):.1f}$"
        outcome = {"admissible": "admissible", "negative": "reaches $T$, negative values", "diverged": "diverged"}[r.observed_status_EE]
        tdiv = "--" if _isnan(r.t_diverge_EE) else f"${float(r.t_diverge_EE):.1f}$"
        body.append(f"{sci_tex(r.h)} & {texc} & {outcome} & {tdiv}\\\\")
    hl_table = (
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Explicit Euler: time at which the local limit $h|\\lambda_{\\rm fast}(t)|>2$ is first exceeded along the "
        "reference trajectory, compared with the outcome of the step-size sweep of Section~\\ref{sec:step-experiment}.}\n"
        "\\label{tab:hlambda}\n\\small\n\\begin{tabular}{rlll}\n\\toprule\n"
        "$h$ & local limit exceeded from $t=$ & observed outcome & divergence time\\\\\n\\midrule\n"
        + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(gen / "table_hlambda.tex", hl_table)

    # ------------------------------------------------------------ Table: QSSA predictions at three times (Section 6.1)
    body = []
    for _, r in qt.iterrows():
        tt = "$40$" if float(r.t) == 40.0 else sci_tex(r.t, 1) if float(r.t) != 1.0 else "$1$"
        rat = lambda x: "$" + (f"{float(x):.0f}" if float(x) >= 100 else f"{float(x):.4f}" if abs(float(x) - 1) < 0.01 else f"{float(x):.2f}") + "$"
        body.append(f"{tt} & {sci_tex(r.y2_ref, 4)} & {rat(r.y2_early / r.y2_ref)} & {rat(r.y2_late / r.y2_ref)} & "
                    f"${float(r.y2_quad / r.y2_ref):.5f}$\\\\")
    qssa_table = (
        "\\begin{table}[H]\n\\centering\n"
        "\\caption{Quasi-steady predictions of $y_2$ divided by the reference value $y_2(t)$, evaluated with the reference $y_1$, $y_3$. "
        "Early balance: $\\sqrt{k_1y_1/k_2}$; late balance: $k_1y_1/(k_3y_3)$; quadratic QSSA: positive root of $k_2y_2^2+k_3y_3y_2=k_1y_1$.}\n"
        "\\label{tab:qssa}\n\\small\n\\begin{tabular}{rrrrr}\n\\toprule\n"
        "$t$ & reference $y_2$ & early balance & late balance & quadratic QSSA\\\\\n\\midrule\n"
        + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write(gen / "table_qssa.tex", qssa_table)

    print(f"Generated LaTeX tables/macros in {gen}")


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])