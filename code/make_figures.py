#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MARK = {"Explicit Euler": "o", "RK4": "s", "Implicit Euler": "^", "Trapezoidal / CN": "D"}
COLOR = {"Explicit Euler": "C0", "RK4": "C1", "Implicit Euler": "C2", "Trapezoidal / CN": "C3"}


def meta_value(meta, name):
    return float(meta.loc[meta.quantity == name, "value"].iloc[0])


def main(root: Path):
    data = root / "data"
    figdir = root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(data / "euler_stability_metadata.csv")
    hcrit = meta_value(meta, "theoretical_h_critical")
    hemp = meta_value(meta, "empirical_h_boundary_bisection")

    # ---------------- Figure 1: reference trajectory (values below atol=1e-14 are meaningless -> not drawn)
    ref = pd.read_csv(data / "reference_trajectory.csv")
    mask = ref["t"] > 0
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for col, lab in [("y1", r"$y_1$"), ("y2", r"$y_2$"), ("y3", r"$y_3$")]:
        m = mask & (ref[col] > 1e-14)
        ax.loglog(ref.loc[m, "t"], ref.loc[m, col], lw=2, label=lab)
    ax.set_xlabel("Time"); ax.set_ylabel("Concentration")
    ax.set_title("Tight Radau reference trajectory")
    ax.grid(True, which="both", alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(figdir / "01_reference_trajectory.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # ---------------- Figure 2: convergence, guides anchored on the finest data point of a representative method
    conv = pd.read_csv(data / "four_method_results.csv")
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for method, marker in MARK.items():
        d = conv[conv.method == method].sort_values("h")
        ax.loglog(d.h, d.E_inf, marker=marker, color=COLOR[method], lw=2, label=method)
    def guide(method, p, ls, label):
        d = conv[conv.method == method].sort_values("h")
        h0, e0 = d.h.iloc[0], d.E_inf.iloc[0]
        x = np.array([h0, 8 * h0])
        ax.loglog(x, e0 * (x / h0) ** p * 0.6, ls, color="k", lw=1.1, alpha=.7, label=label)
    guide("Explicit Euler", 1, "--", r"$O(h)$")
    guide("Trapezoidal / CN", 2, "-.", r"$O(h^2)$")
    guide("RK4", 4, ":", r"$O(h^4)$")
    ax.set_xlabel("Step size $h$"); ax.set_ylabel(r"Maximum trajectory error $E_\infty$")
    ax.set_title("Convergence order verification"); ax.grid(True, which="both", alpha=.3)
    ax.legend(fontsize=8.5, ncol=2); fig.tight_layout()
    fig.savefig(figdir / "02_four_method_convergence.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # ---------------- Figure 3: Explicit-Euler boundary, two panels (x-axis: h / h_crit)
    stab = pd.read_csv(data / "euler_stability_boundary.csv")
    reached = stab[stab.reached_T.astype(bool)].copy()
    reached["final_error"] = pd.to_numeric(reached.final_error, errors="coerce")
    xr = reached.h / hcrit
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.1))
    for ax in (a1, a2):
        ax.axvline(1.0, ls="--", lw=1.3, color="k", label=rf"$h_{{crit}}=2/|\lambda|_{{max}}$ ({hcrit:.3e})")
        ax.axvline(hemp / hcrit, ls=":", lw=1.6, color="C3", label=rf"$h_{{emp}}$ by bisection ({hemp:.3e})")
        ax.axvspan(1.12, 1.4, color="0.9", zorder=0)
        ax.set_xlim(0.65, 1.4); ax.set_xlabel(r"Explicit-Euler step size $h/h_{crit}$")
        ax.grid(True, which="both", alpha=.3)
    a1.semilogy(xr, reached.final_error, marker="o", lw=2)
    a1.set_ylim(1e-6, 1e-1)
    a1.text(1.26, 1.5e-5, "diverged\nbefore $T$\n(table)", ha="center", fontsize=9)
    a1.set_ylabel(r"Final-time error $\|\mathbf{y}(40)-\mathbf{y}_{ref}(40)\|_2$")
    a1.set_title("(a) Accuracy"); a1.legend(fontsize=8, loc="upper left")
    a2.plot(xr, reached.min_component, marker="o", lw=2, color="C1")
    a2.axhline(0.0, lw=1.0, color="k")
    a2.set_yscale("symlog", linthresh=1e-7); a2.set_ylim(-3e-5, 3e-6)
    a2.text(1.26, -2e-7, "diverged\nbefore $T$\n(table)", ha="center", fontsize=9)
    a2.set_ylabel(r"Minimum concentration $\min_{t,j}y_j$"); a2.set_title("(b) Physical admissibility")
    a2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(figdir / "03_euler_stability_boundary.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # ---------------- Figure S3: adaptive-Euler demonstration of Section 3.5 (tol = 1e-6), three panels
    hist = pd.read_csv(data / "adaptive_demo_history.csv")
    tol_demo = float(pd.read_csv(data / "adaptive_demo_summary.csv").set_index("quantity").loc["tol", "value"])
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(7.4, 9.0), sharex=True,
                                     gridspec_kw={"height_ratios": [1.0, 1.25, 1.0]})
    a1.loglog(hist.t, hist.y2_ref, "k-", lw=1.4, label="reference (Radau)")
    a1.loglog(hist.t, hist.y2, ".", ms=2.5, color="C0", label="adaptive Euler, accepted steps")
    a1.set_ylabel(r"$y_2$"); a1.set_title(r"(a) Intermediate species $y_2$"); a1.legend(fontsize=8.5, loc="lower right")
    a1.grid(True, which="both", alpha=.3)
    a2.loglog(hist.t, hist.h, ".", ms=2.0, alpha=.6, color="C0", label=r"accepted step $h$")
    a2.loglog(hist.t, 2.0 / hist.lam_max, "k--", lw=1.3, label=r"$2/|\lambda_{\max}(t)|$ (Explicit-Euler limit)")
    a2.loglog(hist.t, 4.0 / hist.lam_max, "k-.", lw=1.3, label=r"$4/|\lambda_{\max}(t)|$ (limit for two half-steps)")
    a2.set_ylabel("step size"); a2.set_title("(b) Accepted step sizes against local stability bounds")
    a2.legend(fontsize=8.2, loc="upper right"); a2.grid(True, which="both", alpha=.3)
    a3.loglog(hist.t, hist.local_error_estimate / tol_demo, ".", ms=2.0, alpha=.6, color="C2")
    a3.axhline(1.0, ls="--", color="k", lw=1.2, label=r"tolerance ($e_n/\mathrm{tol}=1$)")
    a3.set_ylim(1e-3, 3.0)
    a3.set_xlabel("Time"); a3.set_ylabel(r"$e_n/\mathrm{tol}$")
    a3.set_title("(c) Local error estimate of every accepted step"); a3.legend(fontsize=8.5, loc="lower right")
    a3.grid(True, which="both", alpha=.3)
    fig.tight_layout()
    fig.savefig(figdir / "S3_adaptive_demo.png", dpi=250, bbox_inches="tight"); plt.close(fig)

    # ---------------- Figure 9: work-precision (non-physical runs hollow)
    wp = pd.read_csv(data / "work_precision.csv")
    lib = pd.read_csv(data / "library_baseline.csv")
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for method, marker in MARK.items():
        d = wp[wp.method == method].sort_values("wall_clock_time")
        ax.loglog(d.wall_clock_time * 1e3, d.E_inf, marker=marker, color=COLOR[method], lw=1.8, label=method)
        bad = d[~d.nonnegative.astype(bool)]
        if len(bad):
            ax.loglog(bad.wall_clock_time * 1e3, bad.E_inf, marker=marker, ls="none", ms=9,
                      mfc="white", mec=COLOR[method], mew=1.8)
    ax.loglog([lib.wall_clock_time.iloc[0] * 1e3], [lib.E_inf.iloc[0]], marker="*", ms=13, ls="none",
              color="k", label="SciPy Radau (adaptive, library)")
    ax.plot([], [], marker="o", ls="none", mfc="white", mec="k", label="hollow: min $y<0$")
    ax.axhline(1e-4, ls="--", lw=1.2, color="0.4", label=r"target $E_\infty=10^{-4}$")
    ax.set_xlabel("Wall-clock time (ms; median of warmed runs)")
    ax.set_ylabel(r"Maximum trajectory error $E_\infty$")
    ax.set_title("Work--precision comparison"); ax.grid(True, which="both", alpha=.3)
    ax.legend(fontsize=8.5); fig.tight_layout()
    fig.savefig(figdir / "09_work_precision.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    # ---------------- Section 4: eigenvalues + stiffness ratio (reduced Jacobian), kappa(V), h*lambda overlay
    js = pd.read_csv(data / "jacobian_stiffness.csv")
    orient = pd.read_csv(data / "stiffness_orientation_check.csv")
    sm = pd.read_csv(data / "stiffness_metadata.csv")
    smv = dict(zip(sm.quantity, sm.value))
    t_c = smv["crossing_time_analytic_1_over_2k2"]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 7.0), sharex=True)
    a1.loglog(js.t, js.abs_slow, lw=2, label=r"$|\lambda_{\rm slow}(t)|$")
    a1.loglog(js.t, js.abs_fast, lw=2, label=r"$|\lambda_{\rm fast}(t)|$")
    a1.axvline(t_c, ls=":", color="0.4", lw=1.2)
    a1.text(t_c * 1.15, 3e-1, r"$t_c=1/(2k_2)$", fontsize=8.5, color="0.3")
    a1.set_ylabel(r"$|\lambda|$"); a1.grid(True, which="both", alpha=.3); a1.legend(loc="upper left")
    a1.set_title(r"Nonzero eigenvalues of the reduced Jacobian $J_r(t)$")
    a2.loglog(js.t, js.stiffness_ratio, lw=2, color="C3")
    a2.loglog(orient.t, orient.S_ours, "ko", ms=6)
    for _, r in orient.iterrows():
        a2.annotate("${:.2f}\\times10^{{{}}}$".format(r.S_ours / 10 ** int(np.floor(np.log10(r.S_ours))), int(np.floor(np.log10(r.S_ours)))), (r.t, r.S_ours), textcoords="offset points", xytext=(-6, 8), fontsize=8.5, ha="right")
    a2.axvline(t_c, ls=":", color="0.4", lw=1.2)
    a2.set_xlabel("Time (log scale)"); a2.set_ylabel(r"Stiffness ratio $S(t)$")
    a2.grid(True, which="both", alpha=.3); a2.set_title(r"$S(t)=\max|\lambda_j|/\min|\lambda_j|$ (markers: orientation times)")
    fig.tight_layout()
    fig.savefig(figdir / "S4_eigenvalues_stiffness.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    kout = pd.read_csv(data / "kappa_output_grid.csv")
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.loglog(js.t, js.kappa_V, lw=1.6, color="C1", label=r"$\kappa(V)$, 1000-point grid")
    ax.loglog(kout.t, kout.kappa_V, ".", ms=4, color="k", label=r"200 output times")
    ax.axhline(1.0, ls="--", color="0.4", lw=1.0, label=r"$\kappa(V)=1$ (normal matrix)")
    ax.axvline(t_c, ls=":", color="0.4", lw=1.2)
    ax.set_xlabel("Time (log scale)"); ax.set_ylabel(r"$\kappa(V(t))=\|V\|_2\|V^{-1}\|_2$")
    ax.set_title("Condition number of the eigenvector matrix of $J_r(t)$")
    ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=8.5, loc="upper right"); fig.tight_layout()
    fig.savefig(figdir / "S4_condition_number.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    lim_rk = float(pd.read_csv(data / "stability_region_checks.csv").set_index("quantity").loc["rk4_real_axis_limit", "value"])
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for h, c in [(5.0e-4, "C0"), (6.0e-4, "C1"), (7.0e-4, "C2"), (8.0e-4, "C3")]:
        ax.semilogx(js.t, h * js.abs_fast, lw=1.8, color=c, label=rf"$h={h * 1e4:.1f}\times10^{{-4}}$")
    ax.axhline(2.0, ls="--", color="k", lw=1.3, label=r"Explicit Euler limit $|z|=2$")
    ax.axhline(lim_rk, ls="-.", color="k", lw=1.3, label=rf"RK4 limit $|z|={lim_rk:.3f}$")
    ax.set_xlim(1e-3, 45); ax.set_ylim(0, 3.2)
    ax.set_xlabel("Time (log scale)"); ax.set_ylabel(r"$h\,|\lambda_{\rm fast}(t)|$")
    ax.set_title(r"Local $h\lambda$ diagnostic on the real axis (frozen Jacobian along the reference)")
    ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=8, loc="lower left", ncol=2); fig.tight_layout()
    fig.savefig(figdir / "S4_hlambda_overlay.png", dpi=300, bbox_inches="tight"); plt.close(fig)

    print(f"Figures written to {figdir}")


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])