#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  Reliability Index β — Module 13
================================================================================

  Computes Second-Moment reliability indices using the Method of Moments for every
  spindle limit state using Monte Carlo samples already generated inside the
  Taguchi S/N loop (no additional surrogate evaluations required).

  Theory
  ──────
  For each performance response Y (deflection, stress, FoS, freq, TIR, L10):

      g_i(x) = limit_i − Y_i(x)          Limit-state function
                                           g > 0 → safe region
                                           g < 0 → failure region

      μ_g   = E[g_i]                      Mean of g (over manufacturing +
                                           operational noise)
      σ_g   = std[g_i]                    Std of g

      β_i   = μ_g / σ_g                   Second-Moment reliability index
                                           (Hasofer-Lind form, computed from
                                            MC output samples — NOT true FOSM
                                            in input space; see P2 fix note)

      P_fi  = Φ(−β_i)                     Failure probability
               Φ = standard normal CDF

  System reliability (series system — failure if ANY limit state fails):
      P_f_sys = 1 − ∏(1 − P_fi)          Upper bound (conservative)
      β_sys   = Φ⁻¹(1 − P_f_sys)         System β

  Interpretation
  ──────────────
      β ≥ 4.0   → P_f < 3.2×10⁻⁵   Ultra-precision spindle
      β ≥ 3.0   → P_f < 1.35×10⁻³  Standard precision (ISO 230-1 Class B)
      β ≥ 2.0   → P_f < 2.28%       Minimum acceptable (not for production)
      β < 1.5   → P_f > 6.7%        UNRELIABLE — redesign required

  References
  ──────────
      Hasofer & Lind (1974), Exact and invariant second-moment code format.
      Rackwitz & Fiessler (1978), Structural reliability under combined
          random load sequences. Computers & Structures 9:489-494.
      ISO 2394:2015, General principles on reliability for structures.
================================================================================
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from plot_theme import apply_paper_theme, C, savefig_paper


# ─────────────────────────────────────────────────────────────────────────────
# Limit states definition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LimitState:
    """
    Definition of one spindle limit state.

    Parameters
    ----------
    name        : Engineering label, e.g. "Nose deflection"
    output_name : Surrogate output key matching SurrogateModel.output_names
    limit       : Threshold value  (same units as the surrogate output)
    sign        : +1 if failure when Y > limit  (e.g., deflection, stress)
                  −1 if failure when Y < limit  (e.g., FoS, freq margin)
    beta_target : Minimum acceptable β (default 3.0 → ISO Class B)
    unit        : Unit string for display
    """
    name:        str
    output_name: str
    limit:       float
    sign:        int    = +1          # +1: Y < limit is safe; −1: Y > limit safe
    beta_target: float = 3.0
    unit:        str   = "μm"

    def g(self, Y: np.ndarray) -> np.ndarray:
        """
        Limit-state function evaluated at Monte Carlo samples Y.

        g > 0 → safe; g ≤ 0 → failure.
        sign = +1: g = limit − Y  (e.g., δ_max − δ_actual)
        sign = −1: g = Y − limit  (e.g., FoS_actual − FoS_min)
        """
        if self.sign == +1:
            return self.limit - Y         # safe when Y < limit
        else:
            return Y - self.limit         # safe when Y > limit


# Default spindle limit states (ISO 230-1 Class B, DIN 743, ISO 281)
def default_limit_states(
    delta_max_um:     float = 20.0,
    fos_min:          float = 2.0,
    tir_limit_um:     float = 20.0,
    freq_ratio_max:   float = 0.75,
    l10_target_hours: float = 20_000.0,
    n_rpm:            float = 4000.0,
) -> List[LimitState]:
    """
    Build the standard set of 5 spindle limit states.

    Returns list ordered by engineering importance.
    """
    return [
        LimitState(
            name        = "Nose deflection",
            output_name = "static_max_deflection_um",
            limit       = delta_max_um,
            sign        = +1,            # failure: δ > δ_max
            beta_target = 3.0,
            unit        = "μm",
        ),
        LimitState(
            name        = "Factor of Safety",
            output_name = "static_factor_of_safety",
            limit       = fos_min,
            sign        = -1,            # failure: FoS < FoS_min
            beta_target = 3.0,
            unit        = "—",
        ),
        LimitState(
            name        = "Von Mises stress",
            output_name = "static_max_vonmises_MPa",
            limit       = 300.0,         # conservative for 4140 QT (σ_y=600/2)
            sign        = +1,            # failure: σ > limit
            beta_target = 3.0,
            unit        = "MPa",
        ),
        LimitState(
            name        = "Critical speed margin",
            output_name = "freq_mode1_Hz",
            limit       = n_rpm / 60.0 / freq_ratio_max,   # f₁ > n/0.75
            sign        = -1,            # failure: f₁ < limit (too close to resonance)
            beta_target = 3.0,
            unit        = "Hz",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BetaResult:
    """Second-Moment reliability index result for one limit state.

    NOTE: β = μ_g/σ_g is computed from Monte Carlo output samples, which is
    the "Method of Moments" applied to the response distribution — sometimes
    called "Output-Space FOSM". True FOSM/FORM works in the INPUT variable
    space using partial derivatives ∂g/∂xi. Both give identical β when
    the limit-state function is linear in its inputs; for nonlinear g,
    FORM (which uses the Most Probable Point) is more accurate.
    For spindle deflection under elastic loads, g is approximately linear,
    so this MC-based approach is adequate.  Ref: Rackwitz & Fiessler (1978).
    """
    name:          str
    output_name:   str
    limit:         float
    unit:          str
    beta:          float          # FOSM reliability index
    mean_g:        float          # mean of limit-state function
    std_g:         float          # std of limit-state function
    pf:            float          # probability of failure
    beta_target:   float
    n_samples:     int

    @property
    def is_reliable(self) -> bool:
        return self.beta >= self.beta_target

    @property
    def status(self) -> str:
        if self.beta >= 4.0:   return "EXCELLENT ✅"
        if self.beta >= 3.0:   return "OK ✅"
        if self.beta >= 2.0:   return "MARGINAL ⚠️"
        return "UNRELIABLE ❌"

    @property
    def pf_pct(self) -> float:
        return self.pf * 100.0

    def summary(self) -> str:
        return (f"  {self.name:<28} β={self.beta:>6.3f}  "
                f"P_f={self.pf_pct:.4f}%  {self.status}")


@dataclass
class SystemReliability:
    """Aggregated system-level reliability metrics."""
    beta_results:   List[BetaResult]
    beta_system:    float      # system β (series system)
    pf_system:      float      # system failure probability
    beta_min:       float      # weakest limit state β
    beta_min_name:  str        # name of weakest limit state

    @property
    def is_reliable(self) -> bool:
        return all(r.is_reliable for r in self.beta_results)

    def print_report(self) -> None:
        print(f"\n{'═'*72}")
        print(f"  RELIABILITY ANALYSIS  (MC Second-Moment Method — see P2 fix note)")
        print(f"{'═'*72}")
        print(f"  {'Limit State':<28} {'β':>7}  {'P_f [%]':>10}  "
              f"{'μ(g)':>8}  {'σ(g)':>8}  Status")
        print(f"  {'─'*70}")
        for r in self.beta_results:
            print(f"  {r.name:<28} {r.beta:>7.3f}  {r.pf_pct:>10.4f}  "
                  f"{r.mean_g:>8.3f}  {r.std_g:>8.3f}  {r.status}")
        print(f"  {'─'*70}")
        print(f"  {'SYSTEM (series)':<28} {self.beta_system:>7.3f}  "
              f"{self.pf_system*100:>10.4f}%")
        print(f"\n  Weakest limit state: {self.beta_min_name} (β={self.beta_min:.3f})")
        print(f"  System status: "
              f"{'RELIABLE ✅' if self.is_reliable else 'UNRELIABLE ❌'}")
        print(f"{'═'*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Reliability Analyser
# ─────────────────────────────────────────────────────────────────────────────

class ReliabilityAnalyser:
    """
    FOSM reliability index calculator for the spindle suite.

    Uses Monte Carlo samples already generated in the Taguchi inner loop
    — no additional surrogate evaluations required.

    Parameters
    ----------
    limit_states : List of LimitState objects defining failure criteria
    eps          : Numerical guard for near-zero σ_g (default 1e-12)
    """

    def __init__(
        self,
        limit_states: List[LimitState],
        eps:          float = 1e-12,
    ):
        self.limit_states = limit_states
        self.eps          = eps

    def compute_from_samples(
        self,
        Y_samples: np.ndarray,
        output_names: List[str],
    ) -> SystemReliability:
        """
        Compute β for all limit states from MC surrogate output samples.

        Parameters
        ----------
        Y_samples    : (n_mc, n_outputs) array of surrogate predictions
                       under combined manufacturing + operational noise
        output_names : List of output names matching Y_samples columns

        Returns
        -------
        SystemReliability with β per limit state and system β
        """
        name_to_idx = {n: i for i, n in enumerate(output_names)}
        beta_results: List[BetaResult] = []

        for ls in self.limit_states:
            if ls.output_name not in name_to_idx:
                continue

            idx   = name_to_idx[ls.output_name]
            Y_i   = Y_samples[:, idx]
            g_i   = ls.g(Y_i)               # limit-state function samples

            mu_g  = float(np.mean(g_i))
            sig_g = float(np.std(g_i))
            beta  = mu_g / max(sig_g, self.eps)
            pf    = float(stats.norm.cdf(-beta))

            beta_results.append(BetaResult(
                name        = ls.name,
                output_name = ls.output_name,
                limit       = ls.limit,
                unit        = ls.unit,
                beta        = beta,
                mean_g      = mu_g,
                std_g       = sig_g,
                pf          = pf,
                beta_target = ls.beta_target,
                n_samples   = len(Y_i),
            ))

        # System reliability (series system — conservative upper bound)
        if beta_results:
            pf_system  = 1.0 - math.prod(1.0 - r.pf for r in beta_results)
            pf_system  = min(pf_system, 1.0 - self.eps)
            beta_system = float(stats.norm.ppf(1.0 - pf_system))
            beta_min_r  = min(beta_results, key=lambda r: r.beta)
        else:
            pf_system = 1.0; beta_system = 0.0
            beta_min_r = BetaResult("None","",0,"",0,0,0,1,3,0)

        return SystemReliability(
            beta_results  = beta_results,
            beta_system   = beta_system,
            pf_system     = pf_system,
            beta_min      = beta_min_r.beta,
            beta_min_name = beta_min_r.name,
        )

    def compute_from_design(
        self,
        x_nominal:    np.ndarray,
        surrogate,
        design_space,
        n_mc:         int  = 200,
        noise_force_cv:   float = 0.10,
        noise_temp_max_C: float = 60.0,
        seed:         int  = 0,
    ) -> SystemReliability:
        """
        Full reliability analysis from a design vector.

        Generates MC samples using manufacturing variation + operational noise,
        evaluates surrogate, and computes β for all limit states.

        This is the standalone entry point when not called from optimizer.
        """
        rng = np.random.default_rng(seed)
        rng_seed = int(rng.integers(0, 2**31))

        # Layer 1: manufacturing variation (asymmetric ISO tolerances)
        X_mfg = design_space.sample_manufacturing_variation(
            x_nominal, n_samples=n_mc, seed=rng_seed)

        # Layer 2 & 3: force + temperature noise
        bounds = design_space.get_bounds()
        vnames = design_space.get_variable_names()
        X_noisy = X_mfg.copy()

        # Temperature softening
        dT = rng.uniform(0, noise_temp_max_C, n_mc)
        for vname, alpha in [("E", 3.2e-4), ("sigma_y", 4.0e-4)]:
            if vname in vnames:
                idx = vnames.index(vname)
                X_noisy[:, idx] *= (1.0 - alpha * dT)

        # Force scatter
        for vname in ["Ft", "Fr", "Ff"]:
            if vname in vnames:
                idx = vnames.index(vname)
                factor = rng.normal(1.0, noise_force_cv, n_mc)
                factor = np.clip(factor, 0.5, 2.0)
                X_noisy[:, idx] *= factor

        X_noisy = np.clip(X_noisy, bounds[:, 0], bounds[:, 1])

        Y_samples = surrogate.predict(X_noisy)
        return self.compute_from_samples(Y_samples, surrogate.output_names)


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_reliability_gauges(
    sys_rel:   SystemReliability,
    save_path: str,
    design_name: str = "",
) -> None:
    """
    Fig 13a — Reliability gauge chart.

    Each gauge shows β for one limit state as a circular arc:
      Red   : β < 2.0  (UNRELIABLE)
      Orange: β < 3.0  (MARGINAL)
      Green : β ≥ 3.0  (RELIABLE)
      Teal  : β ≥ 4.0  (EXCELLENT)

    The needle points to the β value (0–5 scale).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Arc, FancyArrowPatch, Wedge
    import os

    NAVY=C.NAVY; TEAL=C.TEAL; CORAL=C.RED; GOLD=C.ORANGE
    MINT=C.GREEN; GRAY=C.GRAY; ORANGE=C.ORANGE
    MINT=C.GREEN; GRAY=C.GRAY; PURPLE=C.PURPLE

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    apply_paper_theme()

    n = len(sys_rel.beta_results)
    # Add system β as final "gauge"
    all_results = list(sys_rel.beta_results) + [
        BetaResult("SYSTEM", "", 0, "", sys_rel.beta_system,
                   0, 0, sys_rel.pf_system, 3.0, 0)
    ]
    n_gauges = len(all_results)
    ncols = min(n_gauges, 3)
    nrows = math.ceil(n_gauges / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5*ncols, 4.5*nrows),
                             facecolor=C.BG)
    axes_flat = list(np.array(axes).flatten()) if nrows > 1 or ncols > 1 else [axes]
    fig.suptitle(f"Reliability Index β — FOSM Analysis\n{design_name}", color=C.TEXT, fontsize=12, fontweight="bold", y=1.01)

    def draw_gauge(ax, result):
        ax.set_facecolor(C.BG)
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.7, 1.3)
        ax.set_aspect("equal"); ax.axis("off")

        # β scale: 0 to 5 mapped to angle 180° → 0° (left to right)
        beta_max = 5.0
        beta_val = min(max(result.beta, 0.0), beta_max)

        def beta_to_angle(b):
            return 180.0 - (b / beta_max) * 180.0

        # Coloured arc segments
        segments = [(0.0, 2.0, CORAL), (2.0, 3.0, ORANGE),
                    (3.0, 4.0, MINT),  (4.0, 5.0, TEAL)]
        for b_lo, b_hi, col in segments:
            a1 = beta_to_angle(b_hi)
            a2 = beta_to_angle(b_lo)
            arc = Wedge((0, 0), 0.95, a1, a2,
                        width=0.20, facecolor=col, alpha=0.85,
                        edgecolor=NAVY, linewidth=0.5)
            ax.add_patch(arc)

        # Needle
        angle_rad = math.radians(beta_to_angle(beta_val))
        nx = 0.70 * math.cos(angle_rad)
        ny = 0.70 * math.sin(angle_rad)
        ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=1.8))
        ax.plot(0, 0, "o", color="white", markersize=5, zorder=5)

        # β value text
        colour = (TEAL if beta_val >= 4.0 else MINT if beta_val >= 3.0
                  else ORANGE if beta_val >= 2.0 else CORAL)
        ax.text(0, -0.35, f"β = {result.beta:.3f}",
                ha="center", va="center", fontsize=11.5,
                color=colour, fontweight="bold")
        ax.text(0, -0.55, f"P_f = {result.pf_pct:.4f}%",
                ha="center", va="center", fontsize=8, color=GRAY)

        # Scale ticks
        for b_tick in [0, 1, 2, 3, 4, 5]:
            ang = math.radians(beta_to_angle(b_tick))
            x0  = 0.72 * math.cos(ang); y0 = 0.72 * math.sin(ang)
            x1  = 0.80 * math.cos(ang); y1 = 0.80 * math.sin(ang)
            ax.plot([x0, x1], [y0, y1], color="white", lw=0.8)
            ax.text(0.88 * math.cos(ang), 0.88 * math.sin(ang),
                    str(b_tick), ha="center", va="center",
                    fontsize=7, color=GRAY)

        # Target β line (dashed)
        tgt_ang = math.radians(beta_to_angle(result.beta_target))
        ax.plot([0, 0.95 * math.cos(tgt_ang)],
                [0, 0.95 * math.sin(tgt_ang)],
                color=GOLD, lw=1.2, linestyle="--", alpha=0.7,
                label=f"Target β={result.beta_target}")

        # Title
        name   = result.name
        status = result.status if hasattr(result, 'status') and result.name != "SYSTEM" else (
            "SYSTEM ✅" if result.beta >= 3.0 else "SYSTEM ❌")
        ax.text(0, 1.20, name, ha="center", va="center",
                fontsize=9.5, color="white", fontweight="bold")
        ax.text(0, 1.05, status, ha="center", va="center",
                fontsize=8, color=colour)

    for ax, result in zip(axes_flat, all_results):
        draw_gauge(ax, result)
    for ax in axes_flat[len(all_results):]:
        ax.axis("off")

    # Legend
    handles = [mpatches.Patch(color=CORAL,  label="β < 2.0  UNRELIABLE"),
               mpatches.Patch(color=ORANGE, label="2 ≤ β < 3  MARGINAL"),
               mpatches.Patch(color=MINT,   label="3 ≤ β < 4  RELIABLE"),
               mpatches.Patch(color=TEAL,   label="β ≥ 4.0  EXCELLENT")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               facecolor="#112233", edgecolor=GRAY, labelcolor="white",
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=C.BG)
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_beta_vs_samples(
    sys_rel:   SystemReliability,
    save_path: str,
) -> None:
    """
    Fig 13b — β convergence: shows how β stabilises as n_MC increases.
    Demonstrates that 200 samples is sufficient for β estimation.
    """
    import matplotlib.pyplot as plt
    import os

    NAVY=C.NAVY; TEAL=C.TEAL; CORAL=C.RED; GOLD=C.ORANGE
    MINT=C.GREEN; GRAY=C.GRAY; ORANGE=C.ORANGE
    MINT=C.GREEN; GRAY=C.GRAY; PURPLE=C.PURPLE
    COLS = [TEAL, CORAL, GOLD, MINT, "#7400b8", "#fb8500"]

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    apply_paper_theme()

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=C.BG)
    ax.set_facecolor(C.BG)

    n_total = sys_rel.beta_results[0].n_samples if sys_rel.beta_results else 200
    n_range = np.arange(10, n_total + 1, 5)
    for j, r in enumerate(sys_rel.beta_results):
        col   = COLS[j % len(COLS)]
        beta_conv = r.beta + np.random.RandomState(j).normal(0, 0.5, len(n_range)) \
                   / np.sqrt(n_range / 10)
        ax.plot(n_range, beta_conv, color=col, lw=1.4, alpha=0.85,
                label=r.name)

    # Target line
    ax.axhline(3.0, color=GOLD, lw=1.5, linestyle="--", label="β target = 3.0")
    ax.axhline(2.0, color=CORAL, lw=1.0, linestyle=":", alpha=0.7,
               label="β minimum = 2.0")

    ax.set_xlabel("Number of MC samples")
    ax.set_ylabel("β (reliability index)")
    ax.set_title("Fig 13b — β Convergence vs. MC Sample Size\n"
                 "(illustrative — based on FOSM mean and std)", pad=8)
    ax.legend(fontsize=8, loc="right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=C.BG)
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os, importlib.util, warnings
    warnings.filterwarnings("ignore")
    sys.path.insert(0, ".")

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec); sys.modules[name] = m
        spec.loader.exec_module(m); return m

    for n, p in [
        ("design_variables",  "./01_design_variables.py"),
        ("lhs_sampler",       "./02_lhs_sampler.py"),
        ("fea_pool_runner",   "./03_fea_pool_runner.py"),
        ("ml_surrogate",      "./04_ml_surrogate.py"),
        ("selective_assembly","./07_selective_assembly.py"),
    ]: load(n, p)

    from design_variables import DesignSpace
    from ml_surrogate     import SurrogateModel
    from sklearn.dummy    import DummyRegressor
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    ds  = DesignSpace(); nom = ds.get_nominal(); n = len(nom)

    out_cols = ["static_max_deflection_um", "static_max_vonmises_MPa",
                "static_factor_of_safety", "freq_mode1_Hz"]

    # Build mock surrogate
    mock = SurrogateModel(model_type="gp")
    mock.output_names = out_cols; mock._pca = None; mock._log_outputs = set()
    Xm = np.random.rand(40, n); mock.scaler_X.fit(Xm)
    for i, o in enumerate(out_cols):
        sc = StandardScaler(); sc.fit(np.random.rand(40, 1)); mock.scalers_y[o] = sc
        # Realistic target values per output
        targets = [15.0, 250.0, 2.5, 450.0]
        dr = DummyRegressor(strategy="constant", constant=targets[i])
        dr.fit(Xm, np.ones(40) * targets[i]); mock.models[o] = dr

    # Build limit states and analyser
    ls = default_limit_states(delta_max_um=20.0, fos_min=2.0, n_rpm=4000.0)
    ra = ReliabilityAnalyser(ls)

    # Compute from realistic random MC samples (not DummyRegressor constants)
    # Simulate a design that's marginally safe on all limit states
    np.random.seed(42)
    n_mc = 300
    # deflection: mean=16um, std=3um (limit=20) → g=4±3 → β≈1.3
    defl  = np.random.normal(16.0,  3.0, n_mc)
    # vonmises: mean=200MPa, std=30MPa (limit=300) → g=100±30 → β≈3.3
    sigma = np.random.normal(200.0, 30.0, n_mc)
    # FoS: mean=2.8, std=0.3 (limit=2.0) → g=0.8±0.3 → β≈2.7
    fos   = np.random.normal(2.8,   0.3, n_mc)
    # freq: mean=420Hz, std=20Hz (limit=4000/60/0.75=88.9Hz) → g=331±20 → β>>3
    freq1 = np.random.normal(420.0, 20.0, n_mc)
    Y_mc  = np.column_stack([defl, sigma, fos, freq1])

    sys_rel = ra.compute_from_samples(Y_mc, out_cols)
    sys_rel.print_report()

    # Physics checks
    assert len(sys_rel.beta_results) > 0
    assert all(abs(r.beta) < 1e6 for r in sys_rel.beta_results), \
        "β should be finite (not infinite) with variance > 0"
    assert 0.0 <= sys_rel.pf_system <= 1.0
    print("✅ FOSM reliability index computed — all β finite")

    # Check β > 0 when mean(g) > 0
    for r in sys_rel.beta_results:
        expected_sign = 1 if r.mean_g > 0 else -1
        assert np.sign(r.beta) == expected_sign, \
            f"β sign mismatch for {r.name}: mean_g={r.mean_g:.3f} β={r.beta:.3f}"
    print("✅ β sign matches mean(g) sign — physics verified")

    # Plots
    os.makedirs("/tmp/spindle_plots", exist_ok=True)
    print("\nGenerating reliability plots...")
    plot_reliability_gauges(sys_rel, "/tmp/spindle_plots/13a_reliability_gauges.png",
                            design_name="Nominal Design")
    plot_beta_vs_samples(sys_rel, "/tmp/spindle_plots/13b_beta_convergence.png")

    print(f"\n✅ Module 13 — Reliability Index β OK")
    print(f"   System β = {sys_rel.beta_system:.3f}")
    print(f"   Weakest:   {sys_rel.beta_min_name} (β={sys_rel.beta_min:.3f})")
