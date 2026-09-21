#!/usr/bin/env python3
"""Absolute-stability regions of the four integrators (Section 4.1).

Writes
  figures/S4_stability_regions.png     2x2 panels, |R(z)| <= 1 shaded, domain Re z in [-6,3], Im z in [-4,4]
  data/stability_region_checks.csv     every number quoted in the text of Section 4.1
The growth factors R(z) are those of the scalar test equation y' = lambda*y, z = h*lambda.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

R_EE = lambda z: 1.0 + z
R_RK4 = lambda z: 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0
R_IE = lambda z: 1.0 / (1.0 - z)
R_CN = lambda z: (1.0 + z / 2.0) / (1.0 - z / 2.0)
METHODS = [("Explicit Euler", R_EE), ("RK4", R_RK4), ("Implicit Euler", R_IE), ("Trapezoidal / CN", R_CN)]


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
    ]


def main(root: Path):
    np.seterr(divide="ignore", invalid="ignore")
    figdir = root / "figures"
    data = root / "data"
    figdir.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-6.0, 3.0, 901)
    y = np.linspace(-4.0, 4.0, 801)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.2), sharex=True, sharey=True)
    for ax, (name, R) in zip(axes.ravel(), METHODS):
        A = np.abs(R(Z))
        ax.contourf(X, Y, A, levels=[0.0, 1.0], colors=["#a8cff0"])
        ax.contour(X, Y, A, levels=[1.0], colors=["#08306b"], linewidths=1.3)
        ax.axhline(0, color="0.4", lw=0.8); ax.axvline(0, color="0.4", lw=0.8)
        ax.set_title(name, fontsize=10); ax.set_aspect("equal"); ax.grid(alpha=.25)
    for ax in axes[1]:
        ax.set_xlabel("Re$(z)$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Im$(z)$")
    fig.tight_layout()
    fig.savefig(figdir / "S4_stability_regions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    rows = numerical_checks()
    with (data / "stability_region_checks.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["quantity", "value"]); w.writerows(rows)
    print(f"Stability-region figure and checks written ({figdir}, {data})")


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])