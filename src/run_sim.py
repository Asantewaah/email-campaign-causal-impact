"""Repeat the targeting simulation many times to check each estimator's bias and coverage.

Usage: python run_sim.py [n_reps]
"""
import sys
from pathlib import Path

import pandas as pd

from causal import estimate_all, load_hillstrom, naive, simulate_targeting

df = load_hillstrom("../data/hillstrom.csv")
truth = naive(df["treated"], df["visit"])["estimate"]
start, stop = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (0, int(sys.argv[1]) if len(sys.argv) > 1 else 100)
out_path = Path("../results/simulation_results.csv")
if start == 0 and out_path.exists():
    out_path.unlink()  # fresh run
for seed in range(start, stop):
    res = estimate_all(simulate_targeting(df, seed=seed), seed=seed).rename_axis("estimator").reset_index()
    res["rep"], res["truth"] = seed, truth
    res.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
