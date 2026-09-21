#!/usr/bin/env python3
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from models import TF, jacobian

K1, K2, K3 = 0.04, 3.0e7, 1.0e4


def y2_early(y1):
    return np.sqrt(K1 * y1 / K2)


def y2_late(y1, y3):
    return K1 * y1 / (K3 * y3)


def y2_quad(y1, y3):
    return (-K3 * y3 + np.sqrt((K3 * y3) ** 2 + 4.0 * K2 * K1 * y1)) / (2.0 * K2)


def lam_fast_formula(y1, y3):
    return np.sqrt((K3 * y3) ** 2 + 4.0 * K1 * K2 * y1)


def lam_fast_numeric(y):
    J = jacobian(0.0, np.asarray(y, dtype=float))
    w = np.linalg.eigvals(J[:2, :2] - J[:2, 2:3])     # reduced Jacobian (Section 4.2)
    return float(np.max(np.abs(w.real)))


def reduced_rhs(t, z):
    """Slow flow on the quadratic QSSA manifold: y2 eliminated algebraically."""
    y1, y3 = z
    p = y2_quad(y1, max(y3, 0.0))
    return [-K1 * y1 + K3 * p * y3, K2 * p * p]


def write_rows(path: Path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)


def main(root: Path):
    from experiments import reference       # same reference as the rest of the report
    data = root / "data"; figdir = root / "figures"
    data.mkdir(parents=True, exist_ok=True); figdir.mkdir(parents=True, exist_ok=True)
    tight = reference("Radau", 1e-12, 1e-14)

    delta = 1.0 / np.sqrt(K2)
    kappa3 = K3 * delta
    ys0 = np.sqrt(K1 / K2)                   # sqrt(k1/k2) with y1 = 1
    t_f = 1.0 / np.sqrt(K1 * K2)             # initial-layer time scale
    inner = lambda t: ys0 * np.tanh(t / t_f)

    tg = np.geomspace(1e-8, TF, 4000)
    y1, y2, y3 = tight.sol(tg)
    q = y2_quad(y1, y3)
    comp = q - ys0 * (1.0 - np.tanh(tg / t_f))               # quadratic QSSA + initial-layer correction
    rel_q, rel_c = np.abs(q - y2) / y2, np.abs(comp - y2) / y2

    tz = np.linspace(0.0, 8e-3, 8001)
    y2z = tight.sol(tz)[1]
    i_max = int(np.argmax(y2))
    t99_ref = float(tz[np.argmax(y2z >= 0.99 * ys0)])
    m1e3 = tg <= 1e-3

    # eigenvalue: prediction versus computation
    j2 = int(np.argmin(np.abs(tg - 1e-2)))
    y_T = tight.sol(TF)
    lam_T_num, lam_T_form = lam_fast_numeric(y_T), float(lam_fast_formula(y_T[0], y_T[2]))
    lam_2_num = lam_fast_numeric(tight.sol(1e-2))

    # reduced (slow) model: y2 eliminated; initial values (1, 0); differs from the full model by an initial layer
    red = solve_ivp(reduced_rhs, (0.0, TF), [1.0, 0.0], method="Radau", rtol=1e-12, atol=1e-14, dense_output=True)
    late = tg >= 1e-2
    r = red.sol(tg[late])
    d_y1 = float(np.max(np.abs(r[0] - y1[late]))); d_y3 = float(np.max(np.abs(r[1] - y3[late])))

    e_at = lambda arr, t0: float(np.interp(t0, tg, arr))
    summary = [
        ("k1", K1), ("k2", K2), ("k3", K3), ("delta", delta), ("kappa3", kappa3),
        ("y2_star_early_y1_1", ys0), ("t_fast_layer", t_f), ("t99_theory", t_f * np.arctanh(0.99)), ("t99_reference", t99_ref),
        ("y2_max_reference", float(y2[i_max])), ("t_of_y2_max_reference", float(tg[i_max])),
        ("y2_max_rel_gap_to_early_balance", float((ys0 - y2[i_max]) / y2[i_max])),
        ("inner_solution_max_rel_err_t_le_1e-3", float((np.abs(inner(tg) - y2) / y2)[m1e3].max())),
        ("composite_max_rel_err_all_t", float(rel_c.max())),
        ("quad_qssa_max_rel_err_t_ge_5e-3", float(rel_q[tg >= 5e-3].max())),
        ("quad_qssa_rel_err_at_T", float(rel_q[-1])),
        ("quad_qssa_rel_err_at_2e-3", e_at(rel_q, 2e-3)),
        ("early_balance_max_rel_err_5e-3_to_1e-2", float((np.abs(y2_early(y1) - y2) / y2)[(tg > 5e-3) & (tg < 1e-2)].max())),
        ("early_balance_rel_err_at_1", e_at(np.abs(y2_early(y1) - y2) / y2, 1.0)),
        ("early_balance_ratio_at_T", float(y2_early(y1[-1]) / y2[-1])),
        ("late_balance_rel_err_at_T", float(abs(y2_late(y1[-1], y3[-1]) - y2[-1]) / y2[-1])),
        ("lambda_plateau_prediction_2sqrt_k1k2", 2.0 * np.sqrt(K1 * K2)), ("lambda_numeric_at_1e-2", lam_2_num),
        ("lambda_formula_at_1e-2", float(lam_fast_formula(y1[j2], y3[j2]))),
        ("lambda_numeric_at_T", lam_T_num), ("lambda_formula_at_T", lam_T_form),
        ("hcrit_from_formula", 2.0 / lam_T_form),
        ("reduced_model_max_abs_dy1_t_ge_1e-2", d_y1), ("reduced_model_max_abs_dy3_t_ge_1e-2", d_y3),
        ("k1_times_t_fast", K1 * t_f),
    ]
    write_rows(data / "qssa_summary.csv", ["quantity", "value"], summary)

    rows = []
    for t0 in (1e-2, 1.0, 40.0):
        a, b, c = tight.sol(t0)
        p = {"ref": b, "early": float(y2_early(a)), "late": float(y2_late(a, c)), "quad": float(y2_quad(a, c))}
        rows.append([t0, p["ref"], p["early"], p["late"], p["quad"],
                     (p["early"] - b) / b, (p["late"] - b) / b, (p["quad"] - b) / b])
    write_rows(data / "qssa_table.csv",
               ["t", "y2_ref", "y2_early", "y2_late", "y2_quad", "rel_early", "rel_late", "rel_quad"], rows)

    # ------------------------------------------------------------------ figure
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(7.4, 10.2), gridspec_kw={"height_ratios": [1.0, 1.0, 0.9]})
    s = 1e5
    a1.plot(tz * 1e3, y2z * s, "k-", lw=1.8, label="reference (Radau)")
    a1.plot(tz * 1e3, inner(tz) * s, "C3--", lw=1.6, label=r"initial layer $\sqrt{k_1/k_2}\,\tanh(\sqrt{k_1k_2}\,t)$")
    a1.axhline(ys0 * s, color="0.4", ls=":", lw=1.2, label=r"early balance $y_2^\star=\sqrt{k_1/k_2}$")
    a1.axvline(t_f * 1e3, color="0.6", ls="-.", lw=1.0)
    a1.text(t_f * 1e3 * 1.04, 0.35, r"$t_f=1/\sqrt{k_1k_2}$", fontsize=8.5, color="0.35")
    a1.plot([tg[i_max] * 1e3], [y2[i_max] * s], "ko", ms=5)
    a1.annotate("maximum", (tg[i_max] * 1e3, y2[i_max] * s), textcoords="offset points", xytext=(8, -16), fontsize=8.5)
    a1.set_xlabel(r"Time $t$ ($10^{-3}$)"); a1.set_ylabel(r"$y_2\ (10^{-5})$"); a1.set_xlim(0, 8); a1.set_ylim(0, 4.2)
    a1.set_title(r"(a) Initial layer: $y_2$ on $0\leq t\leq8\times10^{-3}$"); a1.legend(fontsize=8.2, loc="lower right"); a1.grid(alpha=.3)

    a2.loglog(tg, y2, "k-", lw=1.8, label="reference (Radau)")
    a2.loglog(tg, q, "C0--", lw=1.6, label=r"quadratic QSSA: $k_2y_2^2+k_3y_3y_2=k_1y_1$")
    a2.loglog(tg, y2_early(y1), "C1:", lw=1.6, label=r"early balance $\sqrt{k_1y_1/k_2}$")
    a2.loglog(tg, np.where(y3 > 1e-7, y2_late(y1, np.maximum(y3, 1e-7)), np.nan), "C2-.", lw=1.4, label=r"late balance $k_1y_1/(k_3y_3)$")
    a2.set_ylim(3e-6, 1e-4); a2.set_xlim(1e-4, 45)
    a2.set_xlabel("Time"); a2.set_ylabel(r"$y_2$"); a2.set_title("(b) Quasi-steady values along the reference trajectory")
    a2.legend(fontsize=8.2, loc="lower left"); a2.grid(True, which="both", alpha=.3)

    a3.loglog(tg, np.maximum(rel_q, 1e-9), "C0-", lw=1.6, label="quadratic QSSA")
    a3.loglog(tg, np.maximum(rel_c, 1e-9), "C3-", lw=1.6, label="QSSA + initial-layer correction")
    a3.axvline(t_f, color="0.6", ls="-.", lw=1.0)
    a3.set_ylim(1e-7, 1e6); a3.set_xlabel("Time"); a3.set_ylabel(r"$|y_2^{\rm pred}-y_2|/y_2$")
    a3.set_title("(c) Relative error of the predictions"); a3.legend(fontsize=8.5, loc="upper right"); a3.grid(True, which="both", alpha=.3)
    fig.tight_layout()
    fig.savefig(figdir / "S6_qssa.png", dpi=250, bbox_inches="tight"); plt.close(fig)
    print(f"QSSA analysis written ({data}, {figdir})")


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])