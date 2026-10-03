#!/usr/bin/env python3
"""Absolute-stability regions of the integrators (Section 4.1).

Writes
  figures/S4_stability_regions.png        2x3 panels, |R(z)| <= 1 shaded, domain Re z in [-6,6], Im z in [-4,4]
                                          (EE, RK4, IE, CN + the two accepted states of the adaptive IE controller)
  figures/S4_adaptive_ie_real_axis.png    R(z) and the step-doubling estimator amplification on the negative real axis
  data/stability_region_checks.csv        every number quoted in the text of Section 4.1
  data/stability_R_fast_mode.csv          growth factors at z = h*lambda_fast for the step sizes used in Section 6.2
The growth factors R(z) are those of the scalar test equation y' = lambda*y, z = h*lambda.
For the step-doubling IE controller one macro-step of size h gives
  y_full = R_IE(z) y,   y_two = R_IE(z/2)^2 y,   extrapolated state 2*y_two - y_full  (Section 3.5).
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar

R_EE = lambda z: 1.0 + z
R_RK4 = lambda z: 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0
R_IE = lambda z: 1.0 / (1.0 - z)
R_CN = lambda z: (1.0 + z / 2.0) / (1.0 - z / 2.0)
# --- accepted states of the step-doubling Implicit-Euler controller (Section 3.5) ---------------------------------------
R_IE2 = lambda z: 1.0 / (1.0 - z / 2.0) ** 2                  # y_two: two half steps of IE (first-order adaptive IE)
R_EXT = lambda z: 2.0 * R_IE2(z) - R_IE(z)                    # 2*y_two - y_full (locally extrapolated adaptive IE)
R_EXT_CF = lambda z: (1.0 - z - z**2 / 4.0) / ((1.0 - z / 2.0) ** 2 * (1.0 - z))   # closed form of R_EXT
EST_EE = lambda z: z**2 / 4.0                                  # (1+z/2)^2 - (1+z): step-doubling estimator of Explicit Euler
EST_IE = lambda z: R_IE2(z) - R_IE(z)                          # step-doubling estimator of Implicit Euler = -(z^2/4)/((1-z/2)^2 (1-z))
METHODS = [("Explicit Euler", R_EE), ("RK4", R_RK4), ("Implicit Euler", R_IE), ("Trapezoidal / CN", R_CN),
           ("IE, two half-steps\n(adaptive IE, first order)", R_IE2),
           ("IE, extrapolated $2y_{two}-y_{full}$\n(adaptive IE, second order)", R_EXT)]
LAMBDA_FAST = 2.19e3        # plateau value of |lambda_fast| quoted in Sections 4.2 and 6.2


def numerical_checks():
    np.seterr(divide="ignore", invalid="ignore")
    rk4_limit = brentq(lambda x: R_RK4(x) - 1.0, -3.0, -2.5)
    x_min = brentq(lambda x: 1.0 + x + x**2 / 2.0 + x**3 / 6.0, -2.0, -1.0)   # R_RK4'(x) = 0
    y = np.logspace(-3, 3, 20001)
    cn_dev = float(np.max(np.abs(np.abs(R_CN(1j * y)) - 1.0)))
    # A-stability checked on a grid of the closed left half-plane
    xr = np.linspace(-50.0, 0.0, 801)
    yi = np.linspace(-50.0, 50.0, 1601)
    Z = xr[None, :] + 1j * yi[:, None]
    ie_max = float(np.max(np.abs(R_IE(Z))))
    cn_max = float(np.max(np.abs(R_CN(Z))))
    # exact characterisation of the Implicit-Euler region: |R|<=1  <=>  |z-1| >= 1
    xg = np.linspace(-6.0, 3.0, 901)
    yg = np.linspace(-4.0, 4.0, 801)
    Zg = xg[None, :] + 1j * yg[:, None]
    mismatch = int(np.sum((np.abs(R_IE(Zg)) <= 1.0 + 1e-14) != (np.abs(Zg - 1.0) >= 1.0 - 1e-14)))

    # ---- adaptive Implicit Euler: two half steps and local extrapolation (Section 4.1, Proposition on R_EXT)
    ie2_max = float(np.max(np.abs(R_IE2(Z))))
    ext_max = float(np.max(np.abs(R_EXT(Z))))
    cf_dev = float(np.max(np.abs(R_EXT(Z) - R_EXT_CF(Z))))                         # R_EXT = closed form
    yy = np.logspace(-3, 3, 20001)
    id_rhs = (yy**4 / 2.0 + yy**6 / 16.0) / ((1.0 + yy**2 / 4.0) ** 2 * (1.0 + yy**2))
    ext_id_err = float(np.max(np.abs(1.0 - np.abs(R_EXT(1j * yy)) ** 2 - id_rhs)))  # 1-|R(iy)|^2 identity
    ext_im_max = float(np.max(np.abs(R_EXT(1j * yy))))
    z0 = brentq(lambda x: 1.0 - x - x**2 / 4.0, -6.0, -4.0)                         # numerator zero = -2-2*sqrt(2)
    res = minimize_scalar(R_EXT, bounds=(-60.0, z0), method="bounded", options={"xatol": 1e-10})
    zz = -np.logspace(-3, 6, 400001)
    amp = np.abs(EST_IE(zz))
    k = int(np.argmax(amp))
    zs = 1.0e-3
    rhp_end = brentq(lambda x: abs(R_EXT(x)) - 1.0, 5.0, 6.0)                       # right end of the unstable real interval
    return [
        ("ee_real_axis_limit", 2.0),
        ("rk4_real_axis_limit", abs(rk4_limit)),
        ("rk4_real_axis_minimum_R", float(R_RK4(x_min))),
        ("rk4_real_axis_argmin", float(x_min)),
        ("ie_R_at_minus_1e8", float(R_IE(-1.0e8))),
        ("cn_R_at_minus_1e8", float(R_CN(-1.0e8))),
        ("cn_max_abs_deviation_from_1_on_imaginary_axis", cn_dev),
        ("ie_max_abs_R_left_halfplane", ie_max),
        ("cn_max_abs_R_left_halfplane", cn_max),
        ("ie_region_indicator_mismatches", mismatch),
        # adaptive IE (two half steps / extrapolated)
        ("ie2_max_abs_R_left_halfplane", ie2_max),
        ("ext_max_abs_R_left_halfplane", ext_max),
        ("ext_closed_form_max_deviation", cf_dev),
        ("ext_imag_axis_identity_max_error", ext_id_err),
        ("ext_max_abs_R_imaginary_axis", ext_im_max),
        ("ext_R_zero_crossing", float(z0)),
        ("ext_R_negative_minimum", float(res.fun)),
        ("ext_R_negative_minimum_at", float(res.x)),
        ("ext_R_times_z_at_minus_1e8", float(R_EXT(-1.0e8) * (-1.0e8))),
        ("ie2_R_times_z2_at_minus_1e8", float(R_IE2(-1.0e8) * 1.0e16)),
        ("ext_R_at_minus_1e8", float(R_EXT(-1.0e8))),
        ("ext_error_constant_over_z3", float((R_EXT(zs) - np.exp(zs)) / zs**3)),
        ("ie2_error_constant_over_z2", float((R_IE2(zs) - np.exp(zs)) / zs**2)),
        ("est_ie_max_amplification", float(amp[k])),
        ("est_ie_max_amplification_at", float(zz[k])),
        ("ext_rhp_real_axis_end", float(rhp_end)),
    ]


def real_axis_figure(figdir: Path, data: Path):
    """Negative real axis: (a) R(z), (b) |R(z)| for very stiff modes, (c) amplification of the step-doubling estimator."""
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14.0, 4.3))
    z = -np.linspace(1e-6, 30.0, 6001)
    for name, R, col in [("Implicit Euler", R_IE, "C2"), ("Trapezoidal / CN", R_CN, "C3"),
                         ("IE, two half-steps", R_IE2, "C4"), ("IE, extrapolated", R_EXT, "C5")]:
        a1.plot(z, R(z), color=col, lw=1.8, label=name)
    z0 = -2.0 - 2.0 * np.sqrt(2.0)
    a1.axhline(0, color="0.4", lw=0.8)
    a1.axvline(z0, color="0.5", ls=":", lw=1.1)
    a1.annotate(r"$R_{\rm ext}=0$ at $z=-2-2\sqrt{2}$", xy=(z0, 0.0), xytext=(-29.0, 0.35), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="0.4"))
    a1.set_xlim(-30, 0); a1.set_ylim(-1.1, 1.1)
    a1.set_xlabel(r"$z=h\lambda$ (real, negative)"); a1.set_ylabel(r"$R(z)$")
    a1.set_title("(a) Growth factor on the negative real axis", fontsize=10); a1.legend(fontsize=8, loc="lower right")
    a1.grid(alpha=.3)

    zl = -np.logspace(-2, 6, 4000)
    for name, R, col in [("Implicit Euler", R_IE, "C2"), ("Trapezoidal / CN", R_CN, "C3"),
                         ("IE, two half-steps", R_IE2, "C4"), ("IE, extrapolated", R_EXT, "C5")]:
        a2.loglog(-zl, np.abs(R(zl)), color=col, lw=1.8, label=name)
    a2.loglog(-zl, 1.0 / (-zl), "k:", lw=1.0, label=r"$1/|z|$")
    for lab, h in [("fixed $h=10^{-3}$", 1e-3), ("fixed $h=10^{-2}$", 1e-2)]:
        a2.axvline(LAMBDA_FAST * h, color="0.6", ls="--", lw=0.9)
        a2.text(LAMBDA_FAST * h * 1.1, 2e-7, lab, rotation=90, fontsize=7.5, color="0.35", va="bottom")
    f = data / "adaptive_implicit_demo_history.csv"
    if f.exists():
        import pandas as pd
        d = pd.read_csv(f)
        d = d[d.t > 0.1]
        zm = float(np.median(d.h * d.lam_max))
        a2.axvline(zm, color="C5", ls="--", lw=0.9)
        a2.text(zm * 1.1, 2e-7, "adaptive IE, median $h$", rotation=90, fontsize=7.5, color="C5", va="bottom")
    a2.set_ylim(1e-7, 2.0)
    a2.set_xlabel(r"$|z|=h|\lambda|$"); a2.set_ylabel(r"$|R(z)|$")
    a2.set_title("(b) Very stiff modes: CN does not damp", fontsize=10)
    a2.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 0.90), framealpha=0.92)
    a2.grid(True, which="both", alpha=.3)

    zl = -np.logspace(-2, 6, 4000)
    a3.loglog(-zl, np.abs(EST_EE(zl)), color="C0", lw=1.8, label=r"Explicit Euler: $z^2/4$")
    a3.loglog(-zl, np.abs(EST_IE(zl)), color="C2", lw=1.8, label=r"Implicit Euler: $|R_{\rm two}-R_{\rm full}|$")
    a3.axhline(1.0, color="0.5", ls=":", lw=1.0)
    a3.set_ylim(1e-7, 1e8)
    a3.set_xlabel(r"$|z|=h|\lambda|$"); a3.set_ylabel("contribution of a mode of amplitude 1 to $e_n$")
    a3.set_title("(c) What the step-doubling estimator sees", fontsize=10); a3.legend(fontsize=8, loc="upper left")
    a3.grid(True, which="both", alpha=.3)
    fig.tight_layout()
    fig.savefig(figdir / "S4_adaptive_ie_real_axis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_fast_mode_table(data: Path):
    """Growth factors at z = h*lambda_fast (lambda_fast = -2.19e3) for the step sizes of Section 6.2,
    plus the median macro-step of the adaptive-IE demonstration run (if its history exists)."""
    hs = [("fixed", 1.0e-3, -LAMBDA_FAST * 1.0e-3), ("fixed", 2.0e-3, -LAMBDA_FAST * 2.0e-3),
          ("fixed", 5.0e-3, -LAMBDA_FAST * 5.0e-3), ("fixed", 1.0e-2, -LAMBDA_FAST * 1.0e-2)]
    f = data / "adaptive_implicit_demo_history.csv"
    if f.exists():
        import pandas as pd
        d = pd.read_csv(f)
        d = d[d.t > 0.1]
        hs.append(("adaptive IE, median (t>0.1)", float(np.median(d.h)), -float(np.median(d.h * d.lam_max))))
    with (data / "stability_R_fast_mode.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "h", "z", "R_IE", "R_CN", "R_IE2", "R_EXT", "est_IE", "est_EE"])
        for kind, h, z in hs:
            w.writerow([kind, h, z, R_IE(z), R_CN(z), R_IE2(z), R_EXT(z), abs(EST_IE(z)), EST_EE(z)])


def main(root: Path):
    np.seterr(divide="ignore", invalid="ignore")
    figdir = root / "figures"
    data = root / "data"
    figdir.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-6.0, 6.0, 1201)
    y = np.linspace(-4.0, 4.0, 801)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.4), sharex=True, sharey=True)
    for ax, (name, R) in zip(axes.ravel(), METHODS):
        A = np.abs(R(Z))
        ax.contourf(X, Y, A, levels=[0.0, 1.0], colors=["#a8cff0"])
        ax.contour(X, Y, A, levels=[1.0], colors=["#08306b"], linewidths=1.3)
        ax.axhline(0, color="0.4", lw=0.8); ax.axvline(0, color="0.4", lw=0.8)
        ax.set_title(name, fontsize=9.5); ax.set_aspect("equal"); ax.grid(alpha=.25)
    for ax in axes[1]:
        ax.set_xlabel("Re$(z)$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Im$(z)$")
    fig.tight_layout()
    fig.savefig(figdir / "S4_stability_regions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    real_axis_figure(figdir, data)

    rows = numerical_checks()
    with (data / "stability_region_checks.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["quantity", "value"]); w.writerows(rows)
    write_fast_mode_table(data)
    print(f"Stability-region figures and checks written ({figdir}, {data})")


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])