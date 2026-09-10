# MTH321-project-01

Numerical Analysis of ODEs and PDEs — Project 1
Topic: Chemical Kinetics — The Robertson Problem (stiff ODE benchmark)

## Team
- 队员1 (GitHub: xxx) — 求解器核心库 / Newton迭代
- 队员2 (GitHub: xxx) — 稳定性 & 刚性分析
- 队员3 (GitHub: xxx) — 验证 & 参考解
- 队员4 (GitHub: xxx) — 自适应步长 & 可视化
- 队员5 (GitHub: xxx) — 报告整合 & 幻灯片

## How to run

1. Clone the repo and enter it:
   git clone https://github.com/brisssa113/MTH321-project-01.git
   cd MTH321-project-01

2. Activate the virtual environment (create one if you don't have it):
   source env/bin/activate      # Windows: env\Scripts\activate

3. Install dependencies:
   pip install numpy scipy matplotlib

4. Reproduce all figures:
   python code/run_all.py

Figures will be generated in figures/.

## Project structure
- code/    — all source code (integrators.py, robertson.py, run_all.py, ...)
- figures/ — generated figures (only final report figures are committed)
- report/  — LaTeX report and slides
- notes/   — meeting notes, task tracker

## Status
Week 1: repo set up, topic chosen (Robertson stiff ODE problem)