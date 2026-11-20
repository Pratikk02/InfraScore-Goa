"""Compare regenerated outputs with the headline numbers reported in the 23B-V2 paper.

Run after executing InfraScore_Goa_Final.ipynb. Exits with status 1 if any value
differs from the reported figure by more than its tolerance.
"""
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent / "outputs"
r = pd.read_csv(OUT / "infrascore_rankings.csv").set_index("town")
gm = pd.read_csv(OUT / "gap_model_comparison.csv").set_index("model")
gap = pd.read_csv(OUT / "gap_analysis.csv").set_index("town")
cov = pd.read_csv(OUT / "interval_coverage_test.csv")
k10 = pd.read_csv(OUT / "allocation_k10_spillover.csv")
bud = pd.read_csv(OUT / "allocation_realistic_budget.csv")
mc = pd.read_csv(OUT / "realistic_cost_scenarios.csv")
now = pd.read_csv(OUT / "contemporary_access.csv")
ua = pd.read_csv(OUT / "uncertainty_analysis.csv")
val = pd.read_csv(OUT / "validation_literacy.csv").set_index("measure")


def cv(scheme, rule, col="empirical"):
    return cov[(cov.scheme == scheme) & (cov.rule == rule) & (cov.nominal == 0.9)][col].iloc[0]


def k(spill, strat):
    return k10[(k10.spillover == spill) & (k10.strategy == strat)].pop_weighted_gain.iloc[0]


def b(strat, budget=250):
    return bud[(bud.budget_cr == budget) & (bud.strategy.str.startswith(strat))].pop_weighted_gain.iloc[0]


CHECKS = [  # (label, regenerated value, value in paper, tolerance)
    ("InfraScore Panaji", r.loc["Panaji", "infrascore"], 74.2, 0.06),
    ("InfraScore mean", r.infrascore.mean(), 40.4, 0.06),
    ("Uncertainty median rho", ua.median_rho.median(), 0.87, 0.006),
    ("Lasso MAE", gm.loc["Lasso", "MAE"], 6.80, 0.006),
    ("Baseline MAE", gm.loc["Mean baseline", "MAE"], 8.53, 0.006),
    ("Lasso R2", gm.loc["Lasso", "R2"], 0.24, 0.006),
    ("Gap Cuncolim", gap.loc["Cuncolim", "gap"], -19.3, 0.06),
    ("Coverage first-review, random", cv("random 5-fold x10", "plain (V1)"), 0.896, 0.0006),
    ("Coverage first-review, statutory", cv("random 5-fold x10", "plain (V1)", "cov_statutory"), 0.771, 0.0006),
    ("Coverage conformal, random", cv("random 5-fold x10", "conformal"), 0.941, 0.0006),
    ("Coverage Mondrian, statutory", cv("random 5-fold x10", "Mondrian", "cov_statutory"), 0.914, 0.0006),
    ("Experiment A greedy", k(False, "Greedy optimiser"), 0.76, 0.006),
    ("Experiment A random", k(False, "Random (mean of 200)"), 0.19, 0.006),
    ("Experiment A greedy, spillover", k(True, "Greedy optimiser"), 1.10, 0.006),
    ("Experiment B cost-aware", b("Cost-aware"), 0.88, 0.006),
    ("Experiment B cost-blind", b("Greedy (ignores"), 0.80, 0.006),
    ("MC win vs cost-blind", (mc["Cost-aware greedy"] > mc["Greedy (ignores cost)"]).mean(), 0.97, 0.006),
    ("Median km to public inpatient care", now.inpatient_now.median(), 4.8, 0.06),
    ("Partial-update Spearman", pd.Series(now.is_2011).corr(pd.Series(now.is_partial_update), method="spearman"), 0.987, 0.0006),
    ("Literacy partial rho", val.loc["infrascore", "partial_rho_ctrl_size"], 0.31, 0.006),
]

failed = 0
for label, got, want, tol in CHECKS:
    ok = abs(got - want) <= tol
    failed += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {label:<38} regenerated {got:8.3f}   paper {want:8.3f}")
print(f"\n{len(CHECKS) - failed} of {len(CHECKS)} headline numbers reproduced")
sys.exit(1 if failed else 0)
