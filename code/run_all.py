#!/usr/bin/env python3
"""One-command reproducibility driver for Sections 3.5, 4, 5 and 6.1.

Default behaviour:
  1. recompute every numerical experiment from the Robertson ODE;
  2. regenerate all Section-4 and Section-5 figures (incl. the stability regions);
  3. regenerate LaTeX tables/macros from those exact CSV files;
  4. optionally compile main.tex when pdflatex is available.

Use --figures-only only when you intentionally want to reuse existing CSV data.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))


def run_cmd(cmd, cwd=ROOT):
    print("$", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-only", action="store_true", help="reuse existing CSV data; do not rerun solvers")
    parser.add_argument("--no-pdf", action="store_true", help="skip local pdflatex compilation")
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
    make_stability(ROOT)      # stability regions + numeric checks of Section 4.1 (needed by make_figures)
    make_qssa(ROOT)           # QSSA / singular-perturbation analysis of Section 6.1 (needed by make_report_tables)
    make_figures(ROOT)
    make_tables(ROOT)

    if not args.no_pdf:
        pdflatex = shutil.which("pdflatex")
        if pdflatex:
            print("Compiling main.tex...")
            for _ in range(2):
                run_cmd([pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"])
        else:
            print("pdflatex not found; skipping local PDF compilation. Upload the folder to Overleaf instead.")

    dt = time.perf_counter() - t0
    print(f"\nDONE in {dt:.1f} s")
    print(f"Data:    {ROOT/'data'}")
    print(f"Figures: {ROOT/'figures'}")
    print(f"Report:  {ROOT/'main.tex'}")
    if (ROOT/'main.pdf').exists():
        print(f"PDF:     {ROOT/'main.pdf'}")

if __name__ == "__main__":
    main()