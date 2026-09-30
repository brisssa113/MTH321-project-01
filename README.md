# Robertson stiff ODE experiments

From this directory, install dependencies and run:

```sh
python -m pip install -r requirements.txt
python code/run_all.py
python code/smoke_test.py
python code/test_adaptive_implicit.py
```

Python 3.12 is recommended. `python code/run_all.py --figures-only` rebuilds
figures and LaTeX tables from saved CSV files. Results are in `data/`, `figures/`
and `generated/`. Tables are LaTeX fragments using booktabs, float and tabularx;
include them in your report with `\input{generated/table_adaptive_implicit}`.

## Added: Adaptive Implicit Euler

The shared backward-Euler Newton routine uses the original analytic Jacobian.
Fixed Implicit Euler retains its original predictor; the adaptive method uses
the previous state to initialize each nonlinear solve at potentially large steps.
One full step and two half steps estimate the local error as
`max(abs(y_two_half_steps - y_full_step)) / (2**1 - 1)`.
The two-half-step state is accepted when this is at most the absolute `tol`.
No Richardson extrapolation is applied. The method remains first order.

The controller uses `0.9 * sqrt(tol / error)`, with accepted-step factors in
[0.5, 2] and rejected-step factors in [0.1, 0.5]. Newton failure rejects the trial
and halves the step. There is no forced acceptance at the minimum step.
Nonlinear residual tolerance is `min(1e-13, 0.001*tol)`. Default bounds are
`h0=1e-4`, `h_min=1e-10`, `h_max=1`; the explicit method retains its original
`h_max=1e-2`. Local tolerance does not guarantee a global error bound.

New outputs:
- `data/adaptive_implicit_tolerance_refinement.csv`: eight tolerances, errors,
  step counts, conservation, residuals and Newton work including rejected trials.
- `data/adaptive_implicit_demo_history.csv`: full accepted history at tol=1e-6.
- `data/adaptive_work_precision.csv`: both adaptive methods, warmed timings.
- `figures/S3_adaptive_implicit_demo.png` and `10_adaptive_comparison.png`.
- `figures/09_work_precision.png`: all six self-implemented methods plus Radau.
- `generated/table_adaptive_implicit.tex` and `table_adaptive_comparison.tex`.

Existing fixed-step and adaptive-explicit studies and filenames are retained.
The original archive contained source and caches, but no saved experimental
outputs; the delivered data are regenerated. Runtime values vary by machine.
The existing deliberate unstable/nonphysical experiments retain their original
statuses and are not project execution failures.
