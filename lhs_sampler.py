#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  LHS Sampler v2 — Space-Filling Design of Experiments
================================================================================

  Bug Fixes Applied (vs. v1):
      BUG-4 [CRITICAL]: pyDOE2 criterion strings ("maximin", "center", …) were
                         forwarded directly to scipy.stats.qmc.LatinHypercube
                         (optimization= parameter), which only accepts
                         "random-cd", "lloyd", or None → ValueError at runtime.
                         Fix: explicit criterion_map lookup before calling scipy.
      BUG-5 [MEDIUM]:   generate_hybrid() return type annotation used lowercase
                         tuple[…] (Python 3.9+ only).
                         Fix: uses Tuple from typing for 3.8 compatibility.
================================================================================
"""

from __future__ import annotations
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple   # BUG-5 FIX

from scipy.stats.qmc import LatinHypercube, Sobol

from design_variables import DesignSpace
from plot_theme import apply_paper_theme, C, savefig_paper

log = logging.getLogger("LHSSampler")
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")


# BUG-4 FIX — mapping from user-facing names to scipy optimisation strings
_CRITERION_MAP: Dict[str, Optional[str]] = {
    "maximin":       "random-cd",   # centred-discrepancy ≈ maximin in practice
    "center":        None,          # plain LHS, no extra optimisation
    "centermaximin": "lloyd",       # Lloyd's algorithm → centroidal Voronoi
    "correlation":   "random-cd",   # CD also reduces correlation
    # scipy native names pass through unchanged
    "random-cd":     "random-cd",
    "lloyd":         "lloyd",
}


class LHSSampler:
    """
    Space-filling sampler: Latin Hypercube, Sobol, or Monte Carlo.

    All methods return an (n_samples, n_vars) float64 array scaled to the
    actual variable bounds defined in *design_space*.
    """

    def __init__(self, design_space: DesignSpace):
        self.design_space = design_space
        self.bounds       = design_space.get_bounds()          # (n_vars, 2)
        self.n_vars       = len(self.bounds)
        self.names        = design_space.get_variable_names()

    # ─────────────────────────────────────────────────────────────────────────
    # LHS (primary method)
    # ─────────────────────────────────────────────────────────────────────────
    def generate_lhs(
        self,
        n_samples: int,
        criterion: Literal[
            "maximin", "center", "centermaximin",
            "correlation", "random-cd", "lloyd"
        ] = "maximin",
        iterations: int = 100,     # kept for API compatibility; scipy ignores it
        seed: Optional[int] = 42,
    ) -> np.ndarray:
        """
        Generate Latin Hypercube samples in the design space.

        Args:
            n_samples: Number of points.
            criterion: Space-filling criterion.
                "maximin"       → scipy "random-cd"   (best general choice)
                "centermaximin" → scipy "lloyd"
                "center"        → plain LHS (no optimisation)
                "correlation"   → scipy "random-cd"
            seed:      Reproducibility seed.

        Returns:
            X: (n_samples, n_vars) in actual units.
        """
        # BUG-4 FIX
        if criterion not in _CRITERION_MAP:
            raise ValueError(
                f"Unknown criterion '{criterion}'. "
                f"Valid: {list(_CRITERION_MAP)}"
            )
        scipy_opt = _CRITERION_MAP[criterion]

        # "lloyd" (Voronoi CVT) requires n_samples > n_vars; fall back
        # gracefully rather than crashing with a QhullError.
        if scipy_opt == "lloyd" and n_samples <= self.n_vars:
            log.warning(
                f"criterion='centermaximin' (lloyd) needs n_samples > n_vars "
                f"({n_samples} ≤ {self.n_vars}). Falling back to 'random-cd'."
            )
            scipy_opt = "random-cd"

        sampler = LatinHypercube(d=self.n_vars, seed=seed, optimization=scipy_opt)
        X_norm  = sampler.random(n=n_samples)
        return self._scale(X_norm)

    # ─────────────────────────────────────────────────────────────────────────
    # Sobol quasi-random sequence
    # ─────────────────────────────────────────────────────────────────────────
    def generate_sobol(
        self,
        n_samples: int,
        scramble: bool = True,
        seed: Optional[int] = 42,
    ) -> np.ndarray:
        """
        Quasi-random Sobol sequence (low discrepancy, excellent for >10 dims).

        Best when n_samples is a power of two (64, 128, 256, …).
        """
        sampler = Sobol(d=self.n_vars, scramble=scramble, seed=seed)
        X_norm  = sampler.random(n_samples)
        return self._scale(X_norm)

    # ─────────────────────────────────────────────────────────────────────────
    # Monte Carlo (uniform or normal)
    # ─────────────────────────────────────────────────────────────────────────
    def generate_montecarlo(
        self,
        n_samples: int,
        distribution: Literal["uniform", "normal"] = "uniform",
        seed: Optional[int] = 42,
    ) -> np.ndarray:
        """
        Pure random Monte Carlo samples.

        "uniform" — exploration (DoE phase).
        "normal"  — centered at nominal with σ = tolerance/3 (manufacturing
                    variation; use for robustness verification).
        """
        rng = np.random.default_rng(seed)

        if distribution == "uniform":
            X_norm = rng.random((n_samples, self.n_vars))
            return self._scale(X_norm)

        if distribution == "normal":
            nominal    = self.design_space.get_nominal()
            tolerances = self.design_space.get_tolerances()
            sigma      = tolerances / 3.0

            X = rng.normal(loc=nominal, scale=sigma, size=(n_samples, self.n_vars))
            # Hard-clip to physical bounds
            lo, hi = self.bounds[:, 0], self.bounds[:, 1]
            return np.clip(X, lo, hi)

        raise ValueError(f"Unknown distribution '{distribution}'")

    # ─────────────────────────────────────────────────────────────────────────
    # Hybrid (LHS nominal + MC perturbations for robustness)
    # ─────────────────────────────────────────────────────────────────────────
    def generate_hybrid(
        self,
        n_lhs: int,
        n_mc_per_lhs: int = 5,
        seed: Optional[int] = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:      # BUG-5 FIX: Tuple from typing
        """
        Hybrid sampling: LHS nominal designs + MC tolerance perturbations.

        Returns:
            X_nominal:   (n_lhs, n_vars)               space-filling designs
            X_perturbed: (n_lhs * n_mc_per_lhs, n_vars) manufacturing variation
        """
        X_nominal  = self.generate_lhs(n_lhs, seed=seed)
        tolerances = self.design_space.get_tolerances()
        sigma      = tolerances / 3.0
        lo, hi     = self.bounds[:, 0], self.bounds[:, 1]

        rng        = np.random.default_rng(seed + 1 if seed is not None else None)
        X_pert_lst: List[np.ndarray] = []

        for x_nom in X_nominal:
            noise  = rng.normal(0.0, sigma, size=(n_mc_per_lhs, self.n_vars))
            x_pert = np.clip(x_nom + noise, lo, hi)
            X_pert_lst.append(x_pert)

        return X_nominal, np.vstack(X_pert_lst)

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────
    def _scale(self, X_norm: np.ndarray) -> np.ndarray:
        """Map [0,1]^n_vars → actual variable bounds."""
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + X_norm * (hi - lo)

    def save_samples(
        self,
        X: np.ndarray,
        filepath: str | Path,
        fmt: Literal["csv", "json"] = "csv",
    ) -> None:
        """Write sample matrix to CSV or JSON with column headers."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "csv":
            import csv
            with open(filepath, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(self.names)
                writer.writerows(X)

        elif fmt == "json":
            records = [
                {n: float(v) for n, v in zip(self.names, row)}
                for row in X
            ]
            with open(filepath, "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2)

        else:
            raise ValueError(f"Unknown format '{fmt}'")

        print(f"✅ Saved {len(X)} samples → {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────


def plot_lhs_samples(X, ds, save_dir="."):
    """Fig 02a/b/c — LHS coverage, 2D scatter pairs, empirical CDF."""
    import matplotlib.pyplot as plt, os
    from matplotlib.patches import Patch

    NAVY=C.NAVY; TEAL=C.TEAL; CORAL=C.RED; GOLD=C.ORANGE
    MINT=C.GREEN; GRAY=C.GRAY; PURPLE=C.PURPLE
    os.makedirs(save_dir, exist_ok=True)
    apply_paper_theme()

    names  = ds.get_variable_names()
    bounds = ds.get_bounds()
    X_norm = (X - bounds[:,0]) / (bounds[:,1] - bounds[:,0])

    # 02a: coverage strip chart
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=C.BG)
    ax.set_facecolor(C.BG)
    for i, name in enumerate(names):
        cv = np.std(X_norm[:,i]) / max(np.mean(X_norm[:,i]), 1e-12)
        col = TEAL if cv < 0.6 else CORAL
        ax.barh(i, 1.0, color=col, alpha=0.12, height=0.8)
        ax.scatter(X_norm[:,i], [i]*len(X), s=10, color=col, alpha=0.55)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Normalised [0,1]")
    ax.set_title(f"Fig 02a — LHS Coverage per Variable  (n={X.shape[0]})", pad=6)
    ax.set_xlim(0,1)
    ax.legend(handles=[Patch(color=TEAL,label="uniform"),Patch(color=CORAL,label="clustered")], fontsize=7)
    plt.tight_layout()
    p = os.path.join(save_dir,"02a_lhs_coverage.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=C.BG); plt.close(fig)
    print(f"  Saved → {p}")

    # 02b: 2D scatter pairs (geometric vars)
    geom = [n for n in names if n in ["L1","L2","R1","R2"]]
    gidx = [names.index(n) for n in geom]
    pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    fig, axes = plt.subplots(2, 3, figsize=(12,7), facecolor=C.BG)
    fig.suptitle("Fig 02b — LHS 2D Projections", color=C.TEXT, y=1.01)
    for ax,(i,j) in zip(axes.flat, pairs):
        ax.set_facecolor(C.BG)
        ax.scatter(X[:,gidx[i]], X[:,gidx[j]], s=14, c=TEAL, alpha=0.6, edgecolors="none")
        ax.set_xlabel(geom[i], fontsize=8); ax.set_ylabel(geom[j], fontsize=8)
        ax.tick_params(labelsize=7)
    plt.tight_layout()
    p = os.path.join(save_dir,"02b_lhs_scatter.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=C.BG); plt.close(fig)
    print(f"  Saved → {p}")

    # 02c: empirical CDF vs ideal
    fig, axes = plt.subplots(3, 7, figsize=(14,6), facecolor=C.BG)
    fig.suptitle("Fig 02c — Empirical CDF vs Ideal Uniform", color=C.TEXT, y=1.01)
    ideal = np.linspace(0,1,200)
    for ax, name, i in zip(axes.flat, names, range(len(names))):
        ax.set_facecolor(C.BG)
        ax.step(np.sort(X_norm[:,i]), np.arange(1,len(X)+1)/len(X), color=TEAL, lw=1.1)
        ax.plot(ideal, ideal, color=GOLD, lw=0.8, linestyle="--")
        ax.set_title(name, fontsize=6.5, color=C.TEXT); ax.tick_params(labelsize=5)
    plt.tight_layout()
    p = os.path.join(save_dir,"02c_lhs_cdf.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=C.BG); plt.close(fig)
    print(f"  Saved → {p}")


if __name__ == "__main__":
    import os
    from design_variables import DesignSpace

    ds      = DesignSpace()
    sampler = LHSSampler(ds)

    print(f"\nDesign space: {sampler.n_vars} variables\n")

    X_lhs   = sampler.generate_lhs(50, criterion="maximin")
    X_sobol = sampler.generate_sobol(64, scramble=True)
    X_mc    = sampler.generate_montecarlo(100, distribution="normal")
    X_n, X_p = sampler.generate_hybrid(n_lhs=20, n_mc_per_lhs=5)

    print(f"LHS    shape: {X_lhs.shape}")
    print(f"Sobol  shape: {X_sobol.shape}")
    print(f"MC     shape: {X_mc.shape}")
    print(f"Hybrid nominal={X_n.shape}  perturbed={X_p.shape}")

    # All criteria should work now (BUG-4 fix)
    for crit in ["maximin", "center", "centermaximin", "correlation"]:
        X = sampler.generate_lhs(10, criterion=crit, seed=0)
        print(f"  criterion='{crit}' → shape {X.shape}  ✓")

    sampler.save_samples(X_lhs, "/tmp/lhs_test.csv", fmt="csv")

    os.makedirs("/tmp/spindle_plots", exist_ok=True)
    print("\nGenerating LHS plots...")
    plot_lhs_samples(X_lhs, ds, save_dir="/tmp/spindle_plots")
    print("\n✅ LHS sampler v2 OK")
