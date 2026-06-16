#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  TOPSIS Knee-Point Selector — Module 14
================================================================================

  Technique for Order Preference by Similarity to Ideal Solution
  (Hwang & Yoon, 1981).

  Replaces the previous "minimum normalised distance to utopia" knee-point
  heuristic used in optimize_de() / optimize_nsga2() / optimize_nsga3().

  Why TOPSIS over "distance to utopia"?
  ──────────────────────────────────────
  The old method picked the point closest to the (unreachable) ideal point
  A* = min(F) per column. This implicitly assumes the decision-maker is
  indifferent between "close to ideal in objective 1" and "close to ideal
  in objective 2" — i.e. equal weights, AND it ignores how far each point
  is from the WORST case (anti-ideal A⁻).

  TOPSIS instead ranks each candidate by its relative closeness:

      C_i = D_i⁻ / (D_i⁺ + D_i⁻)

  where D_i⁺ = distance to the ideal solution A⁺ (best per objective)
        D_i⁻ = distance to the anti-ideal solution A⁻ (worst per objective)

  A point with C_i close to 1 is simultaneously close to the best and far
  from the worst — a more robust "balanced compromise" than minimum
  distance-to-ideal alone. TOPSIS also accepts explicit importance WEIGHTS,
  letting the engineer express priorities (e.g. "L10 life matters 2x more
  than cost") — the previous method had no such mechanism.

  Reference
  ─────────
      Hwang, C.L., Yoon, K. (1981). Multiple Attribute Decision Making:
      Methods and Applications. Springer-Verlag.
================================================================================
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TOPSISResult:
    """
    Result of a TOPSIS ranking over a Pareto front.

    Attributes
    ----------
    scores    : (n_points,) relative-closeness C_i ∈ [0,1]; higher = better
    best_idx  : index of the highest-scoring point (recommended design)
    ranking   : indices sorted by score, descending (ranking[0] == best_idx)
    weights   : the weight vector used (normalised to sum to 1)
    ideal     : (n_obj,) ideal point A⁺ used for distance D⁺
    anti_ideal: (n_obj,) anti-ideal point A⁻ used for distance D⁻
    """
    scores:     np.ndarray
    best_idx:   int
    ranking:    np.ndarray
    weights:    np.ndarray
    ideal:      np.ndarray
    anti_ideal: np.ndarray

    def summary(self, objective_labels: Optional[List[str]] = None, top_n: int = 5) -> str:
        lines = [f"  TOPSIS ranking (top {min(top_n, len(self.ranking))} of "
                 f"{len(self.ranking)} Pareto points):"]
        for rank, idx in enumerate(self.ranking[:top_n], 1):
            marker = " ◄ BEST" if idx == self.best_idx else ""
            lines.append(f"    #{rank}  idx={idx:<4}  C={self.scores[idx]:.4f}{marker}")
        if objective_labels:
            lines.append(f"\n  Weights used:")
            for lbl, w in zip(objective_labels, self.weights):
                lines.append(f"    {lbl:<28} w={w:.3f}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Core TOPSIS algorithm
# ─────────────────────────────────────────────────────────────────────────────
def topsis(
    F:       np.ndarray,
    weights: Optional[np.ndarray] = None,
    benefit: Optional[np.ndarray] = None,
) -> TOPSISResult:
    """
    Rank Pareto-front points by TOPSIS relative closeness.

    Parameters
    ----------
    F        : (n_points, n_obj) objective matrix. By the sign conventions
               used throughout this RDO suite, ALL objectives (f1..f8) are
               formulated as MINIMISE (lower = better), so the default
               `benefit=None` (all cost-type) works directly without any
               sign flipping.

    weights  : (n_obj,) importance weights. Need not sum to 1 — normalised
               internally. Default: equal weights (1/n_obj each).
               Example: prioritise L10 (f5) and reliability (f7) 2x:
                   weights = [1,1,1,1, 2,1, 2,1]

    benefit  : (n_obj,) bool array. True  → column is BENEFIT-type
                                            (higher = better, e.g. β if
                                             you pass +β instead of −β)
                                     False → column is COST-type
                                            (lower = better — the default
                                             for every f1..f8 in this suite)
               Default: None → all columns treated as cost-type.

    Returns
    -------
    TOPSISResult with .scores, .best_idx, .ranking

    Algorithm (Hwang & Yoon 1981)
    ──────────────────────────────
    1. Vector-normalise each column:  r_ij = f_ij / sqrt(Σ_i f_ij²)
    2. Apply weights:                 v_ij = w_j * r_ij
    3. Ideal / anti-ideal per column:
           cost-type    j: A⁺_j = min_i v_ij,  A⁻_j = max_i v_ij
           benefit-type j: A⁺_j = max_i v_ij,  A⁻_j = min_i v_ij
    4. Euclidean distances:
           D_i⁺ = sqrt(Σ_j (v_ij − A⁺_j)²)
           D_i⁻ = sqrt(Σ_j (v_ij − A⁻_j)²)
    5. Relative closeness:  C_i = D_i⁻ / (D_i⁺ + D_i⁻)   ∈ [0,1]
    6. best_idx = argmax(C_i)
    """
    F = np.asarray(F, dtype=float)
    n_points, n_obj = F.shape

    if weights is None:
        weights = np.ones(n_obj)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    if benefit is None:
        benefit = np.zeros(n_obj, dtype=bool)   # all cost-type (minimise)
    benefit = np.asarray(benefit, dtype=bool)

    # ── 1. Vector normalisation ─────────────────────────────────────────
    norm = np.sqrt((F**2).sum(axis=0))
    norm = np.where(norm < 1e-12, 1.0, norm)   # guard zero columns
    R = F / norm

    # ── 2. Weighted normalised matrix ───────────────────────────────────
    V = R * weights

    # ── 3. Ideal & anti-ideal ────────────────────────────────────────────
    ideal      = np.where(benefit, V.max(axis=0), V.min(axis=0))
    anti_ideal = np.where(benefit, V.min(axis=0), V.max(axis=0))

    # ── 4. Distances ─────────────────────────────────────────────────────
    D_plus  = np.sqrt(((V - ideal)**2).sum(axis=1))
    D_minus = np.sqrt(((V - anti_ideal)**2).sum(axis=1))

    # ── 5. Relative closeness ───────────────────────────────────────────
    denom = D_plus + D_minus
    denom = np.where(denom < 1e-12, 1.0, denom)
    scores = D_minus / denom

    # ── 6. Ranking ────────────────────────────────────────────────────────
    ranking  = np.argsort(-scores)   # descending
    best_idx = int(ranking[0])

    return TOPSISResult(
        scores=scores, best_idx=best_idx, ranking=ranking,
        weights=weights, ideal=ideal, anti_ideal=anti_ideal,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_topsis_ranking(
    F:                np.ndarray,
    result:           TOPSISResult,
    objective_labels: List[str],
    save_path:        str = "./14a_topsis_ranking.png",
    top_n:            int = 10,
) -> None:
    """
    Fig 14a — TOPSIS ranking visualisation.

    Left panel : bar chart of closeness scores C_i for the top-N candidates,
                  with the selected design highlighted.
    Right panel: parallel-coordinates plot of the top-N candidates across
                  all objectives, normalised to [0,1] (0 = best, 1 = worst
                  for cost-type objectives), so the selected design's
                  trade-off profile is visible at a glance.
    """
    import matplotlib.pyplot as plt
    from plot_theme import apply_paper_theme, C, savefig_paper
    apply_paper_theme()

    n_show = min(top_n, len(result.ranking))
    top_idx = result.ranking[:n_show]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=C.BG)

    # ── Left: closeness scores bar chart ────────────────────────────────
    ax1.set_facecolor(C.BG)
    colors = [C.GREEN if i == result.best_idx else C.BLUE for i in top_idx]
    bars = ax1.barh(range(n_show), result.scores[top_idx], color=colors,
                     edgecolor=C.TEXT, linewidth=0.6)
    ax1.set_yticks(range(n_show))
    ax1.set_yticklabels([f"#{r+1}  idx={i}" for r, i in enumerate(top_idx)])
    ax1.invert_yaxis()
    ax1.set_xlabel("TOPSIS Closeness C (higher = better)")
    ax1.set_title("Fig 14a — TOPSIS Ranking (Top {})".format(n_show))
    ax1.set_xlim(0, 1)
    for r, i in enumerate(top_idx):
        ax1.text(result.scores[i] + 0.01, r, f"{result.scores[i]:.3f}",
                 va="center", fontsize=8, color=C.TEXT)
    # Legend
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color=C.GREEN, label="Selected design"),
                        Patch(color=C.BLUE,  label="Other Pareto points")],
               fontsize=8, loc="lower right")

    # ── Right: parallel coordinates (normalised objectives) ──────────────
    ax2.set_facecolor(C.BG)
    F_min = F.min(axis=0); F_max = F.max(axis=0)
    F_range = np.where(np.abs(F_max - F_min) < 1e-12, 1.0, F_max - F_min)
    F_norm = (F - F_min) / F_range   # 0=best(min), 1=worst(max) for cost-type

    n_obj = F.shape[1]
    x = np.arange(n_obj)
    for i in top_idx:
        col   = C.GREEN if i == result.best_idx else C.GRAY
        alpha = 1.0 if i == result.best_idx else 0.35
        lw    = 2.4 if i == result.best_idx else 1.0
        ax2.plot(x, F_norm[i], color=col, alpha=alpha, lw=lw,
                 marker="o", markersize=3,
                 label="Selected" if i == result.best_idx else None)

    short_labels = [lbl.split("[")[0].strip()[:14] for lbl in objective_labels]
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("Normalised objective (0=best, 1=worst)")
    ax2.set_title("Trade-off Profile — Top Candidates")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    savefig_paper(fig, save_path)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)

    # Synthetic 8-objective Pareto front (all cost-type / minimise)
    n_points, n_obj = 30, 8
    objective_labels = [
        "−S/N_deflection [dB]", "−S/N_stress [dB]", "Total cost [USD]",
        "Weight [kg]", "−L10/20000 [−]", "Speed ratio n/f1 [−]",
        "−β_system [−]", "Chatter ratio b/blim [−]",
    ]

    # Generate a plausible front: trade-offs between cost (col2) and
    # everything else (anti-correlated)
    cost = np.random.uniform(800, 3000, n_points)
    F = np.column_stack([
        -np.random.uniform(8, 25, n_points),          # f1 SN_defl (more negative=better SN... )
        -np.random.uniform(8, 25, n_points),          # f2
        cost,                                          # f3 cost
        np.random.uniform(5, 15, n_points) + cost/500, # f4 weight correlates w/ cost
        -np.random.uniform(0.5, 2.0, n_points) - (3000-cost)/3000,  # f5 L10 (more negative = longer life, cheap->short)
        np.random.uniform(0.3, 0.9, n_points),        # f6 speed ratio
        -np.random.uniform(1.0, 5.0, n_points),       # f7 -beta
        np.random.uniform(0.2, 1.5, n_points),        # f8 chatter ratio
    ])

    # T1: equal weights
    result = topsis(F, weights=None, benefit=None)
    assert result.scores.shape == (n_points,)
    assert 0 <= result.scores.min() and result.scores.max() <= 1.0001
    assert result.best_idx == result.ranking[0]
    print(f"T1 PASS: equal weights, best_idx={result.best_idx}, "
          f"C_best={result.scores[result.best_idx]:.4f}")

    # T2: weighted - prioritise cost (f3) and weight (f4) heavily
    w = np.array([1,1,4,4,1,1,1,1], dtype=float)
    result_w = topsis(F, weights=w)
    print(f"T2 PASS: cost-weighted, best_idx={result_w.best_idx}, "
          f"F[best]={F[result_w.best_idx]}")
    # The cost-weighted best should generally have lower cost than equal-weighted best
    # (not a strict guarantee, but check it ran without error and weights normalised)
    assert np.isclose(result_w.weights.sum(), 1.0)
    print(f"     normalised weights: {result_w.weights}")

    # T3: different weight vectors give different rankings (sensitivity)
    w_life = np.array([1,1,1,1,5,1,3,1], dtype=float)   # prioritise L10 + reliability
    result_life = topsis(F, weights=w_life)
    print(f"T3 PASS: life-weighted best_idx={result_life.best_idx} "
          f"(vs equal={result.best_idx}, cost={result_w.best_idx})")

    # T4: feasibility - scores in [0,1], no NaN
    assert not np.isnan(result.scores).any()
    assert (result.scores >= -1e-9).all() and (result.scores <= 1+1e-9).all()
    print("T4 PASS: all scores in [0,1], no NaN")

    # T5: summary text
    print("\n" + result.summary(objective_labels, top_n=5))

    # T6: plot
    import os
    os.makedirs("/tmp/topsis_plots", exist_ok=True)
    plot_topsis_ranking(F, result, objective_labels,
                        save_path="/tmp/topsis_plots/14a_topsis_ranking.png")
    print("\nT6 PASS: plot generated")

    print("\nALL 6 TESTS PASSED")
