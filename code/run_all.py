#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-only", action="store_true", help="reuse existing CSV data; do not rerun solvers")
    args = parser.parse_args()

    t0 = time.perf_counter()

    if not args.figures_only:
        from experiments import main as run_experiments
        run_experiments(ROOT)
    else:
        print("Reusing existing CSV files (--figures-only).")

    from stability_region import main as make_stability
    from qssa_analysis import main as make_qssa
    from make_figures import main as make_figures
    from make_report_tables import main as make_tables

    make_stability(ROOT)      # stability regions + numeric checks of Section 4.1
    make_qssa(ROOT)           # QSSA / singular-perturbation analysis of Section 6.1
    make_figures(ROOT)
    make_tables(ROOT)

    dt = time.perf_counter() - t0
    print(f"\nDONE in {dt:.1f} s")
    print(f"Data:    {ROOT/'data'}")
    print(f"Figures: {ROOT/'figures'}")


if __name__ == "__main__":
    main()
