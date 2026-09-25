"""House style and figure functions for the campaign impact project.

Every chart follows the same rules: a bold headline that states the finding,
a grey subtitle that says what is plotted, direct labels instead of legends
where possible, and a small source note.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, GOLD, BLUE, GREY, LIGHT = "#1D1B4C", "#F2A900", "#2F45C9", "#9A99B8", "#ECEBF5"
SOURCE = "Source: Hillstrom MineThatData e-mail experiment (64,000 customers)."


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.25,
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": GREY, "axes.labelcolor": INK, "axes.labelsize": 10,
        "axes.grid": True, "axes.grid.axis": "x", "grid.color": LIGHT, "grid.linewidth": 1,
        "axes.axisbelow": True, "xtick.color": INK, "ytick.color": INK,
        "ytick.major.size": 0, "xtick.major.size": 0, "legend.frameon": False,
    })


def _frame(fig, ax, title: str, subtitle: str, source: str = SOURCE) -> None:
    """Headline, subtitle and source note, aligned to the left edge of the figure."""
    fig.text(0.0, 1.0, title, fontsize=14, fontweight="bold", color=INK, ha="left", va="bottom",
             transform=fig.transFigure)
    fig.text(0.0, 0.965, subtitle, fontsize=10.5, color="#5B5A7E", ha="left", va="top",
             transform=fig.transFigure)
    fig.text(0.0, -0.02, source, fontsize=8, color=GREY, ha="left", va="top", transform=fig.transFigure)


def balance_plot(balance: pd.DataFrame, path: str | None = None):
    """Dumbbell chart of standardised differences: experiment vs targeted campaign."""
    b = balance.reindex(balance["Targeted campaign"].abs().sort_values().index)
    y = np.arange(len(b))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    fig.subplots_adjust(top=0.84)
    ax.axvspan(-0.1, 0.1, color=LIGHT, lw=0, zorder=0)
    ax.hlines(y, b["Randomised experiment"], b["Targeted campaign"], color=GREY, lw=1.5, zorder=1)
    ax.scatter(b["Randomised experiment"], y, s=45, color=BLUE, zorder=3)
    ax.scatter(b["Targeted campaign"], y, s=70, color=GOLD, edgecolor=INK, lw=0.6, zorder=3)
    ax.set_yticks(y, b.index)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Standardised difference in means, emailed minus not emailed")
    top = b.index[-1]
    ax.annotate("Targeted campaign", (b.loc[top, "Targeted campaign"], len(b) - 1), xytext=(-8, 10),
                textcoords="offset points", ha="right", color="#9A7400", fontsize=9, fontweight="bold")
    ax.annotate("Experiment", (b.loc[top, "Randomised experiment"], len(b) - 1), xytext=(8, 10),
                textcoords="offset points", ha="left", color=BLUE, fontsize=9, fontweight="bold")
    ax.set_ylim(-0.6, len(b) + 0.4)
    _frame(fig, ax, "Targeting makes emailed customers look very different",
           "Shaded band = well balanced (|difference| < 0.1). In the targeted campaign, emailed\n"
           "customers spent more last year and bought more recently.")
    if path:
        fig.savefig(path)
    return fig


def forest_plot(res: pd.DataFrame, truth: float, order: list[str], path: str | None = None):
    """Point estimates and 95% intervals from one dataset, with values labelled."""
    fig, ax = plt.subplots(figsize=(8, 3.9))
    fig.subplots_adjust(top=0.8)
    ax.axvline(truth * 100, color=GOLD, lw=3, zorder=1)
    for i, name in enumerate(order[::-1]):
        r = res.loc[name]
        col = GREY if name == "Naive comparison" else INK
        if not np.isnan(r.se):
            ax.hlines(i, r.lower * 100, r.upper * 100, color=col, lw=2.5)
        ax.scatter(r.estimate * 100, i, color=col, s=60, zorder=3)
        ax.text(r.estimate * 100, i + 0.22, f"{r.estimate * 100:.1f}", ha="center", va="bottom", fontsize=9, color=col)
    ax.set_yticks(range(len(order)), order[::-1])
    ax.text(truth * 100, len(order) - 0.45, f"  truth {truth * 100:.1f}", color="#9A7400", fontsize=9,
            fontweight="bold", va="bottom")
    ax.set_ylim(-0.6, len(order))
    ax.set_xlabel("Estimated lift in visit rate (percentage points), with 95% confidence interval")
    _frame(fig, ax, "Adjusting for how customers were chosen recovers the truth",
           "Estimates from one simulated targeted campaign. Outcome regression has no simple interval.")
    if path:
        fig.savefig(path)
    return fig


def simulation_plot(sim: pd.DataFrame, truth: float, order: list[str], coverage: pd.Series,
                    path: str | None = None, seed: int = 0):
    """Every estimate from every simulated campaign, with median and CI coverage per method."""
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    fig.subplots_adjust(top=0.82, right=0.83)
    ax.axvline(truth * 100, color=GOLD, lw=3, zorder=1)
    for i, name in enumerate(order[::-1]):
        e = sim.loc[sim.estimator == name, "estimate"].to_numpy() * 100
        col = GREY if name == "Naive comparison" else INK
        ax.scatter(e, i + rng.uniform(-0.22, 0.22, len(e)), s=11, color=col, alpha=0.35, lw=0, zorder=2)
        ax.plot([np.median(e)] * 2, [i - 0.3, i + 0.3], color=col, lw=3, zorder=3)
        c = coverage.get(name, np.nan)
        ax.text(1.02, i, "n/a" if pd.isna(c) else f"{c:.0%}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=10, fontweight="bold",
                color=INK if pd.isna(c) or abs(c - 0.95) <= 0.03 else "#C0392B")
    ax.text(1.02, len(order) - 0.35, "95% CI\ncoverage", transform=ax.get_yaxis_transform(),
            fontsize=8, color=GREY, va="bottom")
    ax.text(truth * 100, len(order) - 0.45, "  truth", color="#9A7400", fontsize=9, fontweight="bold", va="bottom")
    ax.set_yticks(range(len(order)), order[::-1])
    ax.set_ylim(-0.6, len(order))
    ax.set_xlabel("Estimated lift in visit rate (percentage points)")
    _frame(fig, ax, "Doubly robust methods land on the truth, the naive comparison never does",
           f"Each dot is one of {sim.rep.nunique()} simulated targeted campaigns; bars mark the median.")
    if path:
        fig.savefig(path)
    return fig


def heterogeneity_plot(hte: pd.DataFrame, groups: list[str], path: str | None = None):
    """Lift from each email within each shopper group, as dots with 95% intervals."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.subplots_adjust(top=0.8)
    styles = [("Men's email", INK, 0.13), ("Women's email", GOLD, -0.13)]
    for email, col, dy in styles:
        t = hte[hte.email == email].set_index("shopper").loc[groups[::-1]]
        y = np.arange(len(groups)) + dy
        ax.hlines(y, t.lower * 100, t.upper * 100, color=col, lw=2.5)
        ax.scatter(t.estimate * 100, y, s=70, color=col, edgecolor=INK, lw=0.6, zorder=3)
        for yy, v in zip(y, t.estimate * 100):
            ax.text(v, yy + 0.08 * np.sign(dy), f"{v:.1f}", ha="center", va="bottom" if dy > 0 else "top",
                    fontsize=8.5, color=INK)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(groups)), groups[::-1])
    ax.set_xlabel("Lift in visit rate versus no email (percentage points), with 95% confidence interval")
    ax.set_ylim(-0.6, len(groups) - 0.3)
    fig.text(0.0, 0.905, "●", color=INK, fontsize=12, transform=fig.transFigure, va="center")
    fig.text(0.025, 0.905, "Men's email", color=INK, fontsize=9.5, fontweight="bold", transform=fig.transFigure, va="center")
    fig.text(0.16, 0.905, "●", color=GOLD, fontsize=12, transform=fig.transFigure, va="center")
    fig.text(0.185, 0.905, "Women's email", color="#9A7400", fontsize=9.5, fontweight="bold", transform=fig.transFigure, va="center")
    _frame(fig, ax, "The men's email works for everyone; the women's email misses men's shoppers",
           "")
    if path:
        fig.savefig(path)
    return fig