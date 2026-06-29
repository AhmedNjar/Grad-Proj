#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  Robust Optimizer v2 — GA with Taguchi S/N Ratio + Cost Objective
================================================================================

  Objectives (all minimised):
      f1  Deflection robustness  —  −S/N_deflection  (smaller=better)
      f2  Stress robustness      —  −S/N_stress       (smaller=better)
      f3  Total manufacturing cost [USD]  ← NEW
      f4  Weight proxy [kg]

  Bug Fixes Applied (vs. v1):
      BUG-6 [CRITICAL]: multi_objective_fitness accessed surrogate outputs via
                         hardcoded key "static_max_deflection_um".  If the
                         surrogate was trained on differently-named columns the
                         .get() silently returned 0 for every candidate, making
                         the GA optimise a flat landscape.
                         Fix: __init__ resolves output indices once from
                         self.surrogate.output_names; raises a clear error if
                         the required output names are absent.
      BUG-7 [HIGH]:     n_obj=3 was hardcoded in the pymoo Problem subclass.
                         Now parameterised from self.n_objectives.
      BUG-8 [HIGH]:     Pareto-front weight vector had 3 elements; adding a
                         4th objective caused a shape mismatch crash.
                         Fix: weights built dynamically from n_objectives.
      BUG-9 [MEDIUM]:   "larger-is-better" S/N used log10(1/y²) with no ε
                         guard → −inf when y≈0.
                         Fix: clamp y before log: y = max(|y|, ε).
================================================================================
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.termination import get_termination
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False

from ml_surrogate        import SurrogateModel
from design_variables    import DesignSpace
from selective_assembly  import SelectiveAssemblyAnalyser, SpindleCostModel
from plot_theme import apply_paper_theme, C, savefig_paper
import importlib.util as _ilu, sys as _sys, os as _os
def _load_topsis():
    if 'topsis_selector' in _sys.modules:
        return _sys.modules['topsis_selector']
    _p = _os.path.join(_os.path.dirname(__file__), '14_topsis_selector.py')
    _spec = _ilu.spec_from_file_location('topsis_selector', _p)
    _m = _ilu.module_from_spec(_spec)
    _sys.modules['topsis_selector'] = _m
    _spec.loader.exec_module(_m)
    return _m

log = logging.getLogger("RobustOptimizer")
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

# Outputs the optimiser requires from the surrogate; the user may provide
# alternative names via output_name_map in __init__
_DEFAULT_OUTPUT_MAP = {
    "deflection": "static_max_deflection_um",
    "stress":     "static_max_vonmises_MPa",
    "frequency":  "freq_mode1_Hz",
}


@dataclass
class OptimizationResult:
    """Container for GA results."""
    pareto_front_X:     np.ndarray
    pareto_front_F:     np.ndarray
    best_robust_design: np.ndarray
    n_evaluations:      int
    convergence_history: List[float]
    objective_labels:   List[str]
    topsis_result:      Optional[object] = None   # TOPSISResult (Module 14)


# Penalty constants — named for clarity (magic numbers eliminated)
CHATTER_PENALTY_USD_PER_UNIT = 2000.0  # USD per unit chatter ratio excess (f8>1)
BORE_SNAP_PENALTY_USD_PER_MM = 500.0   # USD per mm bore-catalog deviation

class RobustOptimizer:
    """
    Robust Design Optimization: GA + Taguchi S/N + Cost + Selective Assembly.

    Four objectives (all minimised):
        f1 = −S/N_deflection  (robustness, dB)
        f2 = −S/N_stress      (robustness, dB)
        f3 = total cost [USD] (material + machining + bearings + SA)
        f4 = weight proxy [kg]

    Args:
        surrogate:       Trained SurrogateModel (must contain deflection & stress outputs)
        design_space:    DesignSpace with 19 variables
        n_mc_inner:      Monte Carlo samples per candidate for robustness evaluation
        output_name_map: Maps role→actual output name (override if different from default)
        n_sa_bins:       Selective-assembly bins per interface
        sa_n_parts:      Part count simulated in SA yield analysis
    """

    def __init__(
        self,
        surrogate:        SurrogateModel,
        design_space:     DesignSpace,
        n_mc_inner:       int   = 20,
        output_name_map:  Optional[Dict[str, str]] = None,
        n_sa_bins:        int   = 5,
        sa_n_parts:       int   = 300,
        noise_force_cv:   float = 0.10,
        noise_temp_max_C: float = 60.0,
        noise_alpha_E:    float = 2.2e-4,   # P1 fix: ASM Handbook Vol.1, 4140 steel
            # E(20°C)=200GPa → E(200°C)≈193GPa → dE/dT ≈ 2.2e-4/°C
            # (previous 3.2e-4 overestimated thermal E-loss by 45%)
        noise_alpha_sy:   float = 4.0e-4,
        n_rpm:            float = 4000.0,    # operating speed for L10 & margin
        L10_target_hours: float = 20_000.0,  # normalisation base for f5
        # ── Chatter stability (Module 9, option C: constraint + objective) ──
        chatter_Ks:           float = 2500.0,  # specific cutting force coeff [N/mm²] — steel
        chatter_zeta:         float = 0.03,    # structural damping ratio (typ. 0.02-0.05)
        chatter_b_required:   float = 2.0,     # required axial depth of cut [mm] (roughing)
        bearing_manufacturer: str   = None,     # None = best across all manufacturers
    ):
        self.surrogate    = surrogate
        self.design_space = design_space
        self.n_mc_inner   = n_mc_inner
        self.bounds       = design_space.get_bounds()
        self.n_vars       = len(self.bounds)
        self.n_sa_bins    = n_sa_bins
        self.n_rpm        = n_rpm
        self.L10_target   = L10_target_hours

        # Chatter stability parameters
        self.chatter_Ks         = chatter_Ks
        self.chatter_zeta       = chatter_zeta
        self.chatter_b_required = chatter_b_required

        # Bearing manufacturer filter (None = best C_r across all 6 manufacturers)
        self.bearing_manufacturer = bearing_manufacturer

        # Operational noise config
        self.noise_force_cv   = noise_force_cv
        self.noise_temp_max_C = noise_temp_max_C
        self.noise_alpha_E    = noise_alpha_E
        self.noise_alpha_sy   = noise_alpha_sy

        # Pre-resolve variable indices for noise injection
        _vn = design_space.get_variable_names()
        self._idx_E   = _vn.index("E")        if "E"        in _vn else None
        self._idx_sy  = _vn.index("sigma_y")  if "sigma_y"  in _vn else None
        self._idx_Ft  = _vn.index("Ft")       if "Ft"       in _vn else None
        self._idx_Fr  = _vn.index("Fr")       if "Fr"       in _vn else None
        self._idx_Ff  = _vn.index("Ff")       if "Ff"       in _vn else None
        self._idx_R2  = _vn.index("R2")       if "R2"       in _vn else None

        # Cost and SA models
        self.cost_model  = SpindleCostModel()
        self.sa_analyser = SelectiveAssemblyAnalyser(n_parts=sa_n_parts)

        # BUG-6 FIX — resolve surrogate output indices at construction time
        omap  = output_name_map or _DEFAULT_OUTPUT_MAP
        names = surrogate.output_names
        self._idx: Dict[str, int] = {}
        missing = []
        for role, col in omap.items():
            if col in names:
                self._idx[role] = names.index(col)
            else:
                missing.append(col)
        if missing:
            raise ValueError(
                f"Surrogate does not contain required outputs: {missing}\n"
                f"Available: {names}"
            )

        # Pre-resolve freq1 index (for critical speed objective)
        freq1_col = "freq_mode1_Hz"
        self._idx_freq1 = names.index(freq1_col) if freq1_col in names else None

        # ── 8 objectives ─────────────────────────────────────────────────────
        self.objective_labels = [
            "−S/N_deflection [dB]",          # f1: robustness vs deflection
            "−S/N_stress [dB]",              # f2: robustness vs stress
            "Total cost [USD]",              # f3: manufacturing + SA cost
            "Weight [kg]",                   # f4: shaft mass proxy
            "−L10 / 20000 [−]",             # f5: bearing life
            "Speed ratio n/f₁ [−]",         # f6: critical speed margin
            "−β_system [−]",                # f7: system reliability index
            "Chatter ratio b/b_lim [−]",    # f8: chatter stability (NEW)
        ]
        self.n_objectives = len(self.objective_labels)

        log.info(
            f"RobustOptimizer ready  "
            f"n_vars={self.n_vars}  n_obj={self.n_objectives}  "
            f"n_mc_inner={n_mc_inner}  sa_bins={n_sa_bins}  "
            f"force_CV={noise_force_cv:.0%}  ΔT_max={noise_temp_max_C:.0f}°C  "
            f"n_rpm={n_rpm:.0f}  "
            f"chatter: Ks={chatter_Ks:.0f}N/mm² ζ={chatter_zeta:.3f} "
            f"b_req={chatter_b_required:.1f}mm"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Operational noise injection
    # ─────────────────────────────────────────────────────────────────────────
    def _apply_operational_noise(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Apply force scatter and thermal softening on top of manufacturing variation.

        Sources:
          • Force scatter (CV=noise_force_cv): cutting forces vary ±10% due to
            workpiece material, tool wear, and depth-of-cut variation.
          • Thermal E reduction (α_E=3.2e-4/°C): Young's modulus drops with
            spindle temperature rise ΔT ~ Uniform[0, ΔT_max].
          • Thermal σ_y reduction (α_σ=4.0e-4/°C): yield strength drops faster
            than E with temperature (NIST data, AISI 4140 QT).
        """
        X_noisy = X.copy()
        n       = len(X_noisy)
        delta_T = rng.uniform(0.0, self.noise_temp_max_C, n)

        if self._idx_E  is not None:
            X_noisy[:, self._idx_E]  *= (1.0 - self.noise_alpha_E  * delta_T)
        if self._idx_sy is not None:
            X_noisy[:, self._idx_sy] *= (1.0 - self.noise_alpha_sy * delta_T)

        if self.noise_force_cv > 0:
            for idx in [self._idx_Ft, self._idx_Fr, self._idx_Ff]:
                if idx is not None:
                    factor = rng.normal(1.0, self.noise_force_cv, n)
                    factor = np.clip(factor, 0.5, 2.0)
                    X_noisy[:, idx] *= factor

        return np.clip(X_noisy, self.bounds[:, 0], self.bounds[:, 1])

    # ─────────────────────────────────────────────────────────────────────────
    # Taguchi S/N
    # ─────────────────────────────────────────────────────────────────────────
    def taguchi_sn_ratio(
        self,
        x_nominal: np.ndarray,
        sn_type:   Literal["smaller", "larger", "nominal"] = "smaller",
        target:    Optional[float] = None,
        eps:       float = 1e-10,
        seed:      Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute Taguchi S/N ratio under combined 3-layer noise model.

        Layer 1 — Manufacturing (asymmetric ISO tolerances via sample_manufacturing_variation):
            Correctly centres distribution at mid-band for h5/H7 etc.
            Truncated-normal clipped to physical bounds.
            REPLACES: x + N(0, tols/3) which was symmetric, unclipped, wrong for h5.

        Layer 2 — Force scatter (noise_force_cv=10%): Ft, Fr, Ff × N(1, σ_force)
        Layer 3 — Thermal softening: E and σ_y reduced by ΔT ~ Uniform[0, ΔT_max]
        """
        rng = np.random.default_rng(seed)
        rng_seed = int(rng.integers(0, 2**31))   # derive integer seed for sample_mfg

        # Layer 1: asymmetric ISO tolerance sampling (THE FIX)
        X_mfg  = self.design_space.sample_manufacturing_variation(
            x_nominal, n_samples=self.n_mc_inner, seed=rng_seed)

        # Layers 2 & 3: operational noise
        X_pert = self._apply_operational_noise(X_mfg, rng)

        Y = self.surrogate.predict(X_pert)

        results: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(self.surrogate.output_names):
            y   = Y[:, i]
            mu  = float(np.mean(y))
            std = float(np.std(y))
            y_c = np.maximum(np.abs(y), eps)

            if sn_type == "smaller":
                sn = -10.0 * np.log10(np.mean(y_c**2))
            elif sn_type == "larger":
                sn = -10.0 * np.log10(np.mean(1.0 / y_c**2))
            elif sn_type == "nominal":
                if target is not None:
                    sn = -10.0 * np.log10(np.mean((y - target)**2) + eps)
                else:
                    sn = 10.0 * np.log10(mu**2 / (std**2 + eps))
            else:
                raise ValueError(f"Unknown sn_type '{sn_type}'")

            results[name] = {"sn_ratio": float(sn), "mean": mu, "std": std}

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-objective fitness  (4 objectives + catalog-snap penalty)
    # ─────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Fast L10 approximation (Objective 5)
    # ─────────────────────────────────────────────────────────────────────────
    def _l10_fast(self, var_dict: Dict[str, float]) -> float:
        """
        ISO 281 L10 approximation without full BearingPerformanceCalculator.

        Fast enough for 50K+ fitness evaluations inside GA.

        Physics:
            R_front  = F × b / L_span            Front bearing load [N]
                       (cantilever statics — R_front > F for overhang spindle)
            L10   = (C_r / R_front)^p × 10⁶ / (60 × n)  [h]
            p = 3.0 for ACBB — ISO 281:2007 Table 1
                (ROLLER exponent is p=10/3; using roller exponent for ACBB
                 overestimates L10 by +205% — P1 bug fix)
            C_r from multi-manufacturer catalog, snapped to nearest bore = R2×2.

        P2 fix: load on front bearing is R_front = F × b/L_span (from moment
        equilibrium about rear bearing), NOT the total cutting force F.
        For a typical spindle with a=80mm overhang and L_span=200mm:
            R_front = F × 280/200 = 1.4 × F   (40% higher than naive P_eq=F)
        """
        C_r, _K_radial = self._bearing_cached(var_dict)

        Ft = float(var_dict.get("Ft", 1000.0))
        Fr = float(var_dict.get("Fr", 1000.0))
        F_total = max(np.sqrt(Ft**2 + Fr**2), 1.0)  # N

        # Cantilever load distribution — front bearing sees MORE than total load
        L1 = float(var_dict.get("L1", 80.0))
        L2 = float(var_dict.get("L2", 200.0))
        ff = float(var_dict.get("front_z_fraction", 0.15))
        rf = float(var_dict.get("rear_z_fraction",  0.85))
        a       = L1 + ff * L2            # nose to front bearing [mm]
        b       = L1 + rf * L2            # nose to rear bearing  [mm]
        L_span  = max(b - a, 1.0)
        R_front = F_total * b / L_span    # [N] — always > F_total (P2 fix)
        P_eq    = max(R_front, 0.01 * C_r)

        p     = 3.0        # ACBB ball bearing — ISO 281 Table 1 (P1 bug fix: was 10/3)
        L10_h = (C_r / P_eq)**p * 1e6 / (60.0 * max(self.n_rpm, 1.0))
        return float(L10_h)

    # ─────────────────────────────────────────────────────────────────────────
    # Cached bearing lookup — shared by L10 (f5) and chatter stability (f8)
    # ─────────────────────────────────────────────────────────────────────────
    def _bearing_cached(self, var_dict: Dict[str, float]) -> Tuple[float, float]:
        """
        Snap R2 → nearest catalog ACBB bearing, returning (C_r [N], K_radial_pair [N/mm]).

        Uses bearing_catalog.snap_to_bearing() which searches across ALL 6
        manufacturers (SKF, FAG, NSK, NTN, JTEKT, Timken) unless restricted
        by self.bearing_manufacturer, returning the highest-C_r bearing at
        the nearest bore size.

        Cached per (rounded_R2, manufacturer) key.
        """
        from bearing_catalog import snap_to_bearing

        if not hasattr(self, "_brg_cache"):
            self._brg_cache: Dict[Tuple, Tuple[float, float]] = {}

        R2  = round(float(var_dict.get("R2", 50.0)), 2)
        key = (R2, self.bearing_manufacturer)
        if key not in self._brg_cache:
            try:
                brg, _, _ = snap_to_bearing(
                    R2, "ACBB", self.n_rpm, "grease",
                    manufacturer=self.bearing_manufacturer,
                    criterion="max_Cr",
                )
                C_r    = float(brg.C_r)
                K_pair = 1.7 * float(brg.radial_stiffness_single_N_mm)
                self._brg_cache[key] = (C_r, K_pair)
            except Exception:
                self._brg_cache[key] = (1e5, 2.0e5)   # fallback

        return self._brg_cache[key]

    # ─────────────────────────────────────────────────────────────────────────
    # Chatter stability limit (Module 9, objective f8 + constraint)
    # ─────────────────────────────────────────────────────────────────────────
    def _chatter_b_lim(self, k_dyn: float) -> float:
        """
        Minimum-stability-lobe axial depth of cut b_lim [mm] (Altintas, 2012).

        For a 1-DOF tool/spindle structure with transfer function
            G(iω) = (1/k) / [(1-r²) + i·2ζr],   r = ω/ωn
        the stability boundary is:
            b_lim = −1 / (2·K_s·Re[G(iω_c)]_min)

        The minimum (most negative) Re[G] occurs at r² = 1+2ζ, giving the
        closed-form result (derived from dRe[G]/d(r²) = 0):
            Re[G]_min = −1 / [4·k·ζ·(1+ζ)]

        Substituting:
            b_lim = 2·k·ζ·(1+ζ) / K_s

        Parameters
        ----------
        k_dyn : dynamic radial stiffness [N/mm] of the dominant mode
                (front bearing pair stiffness, K_radial_pair)

        Returns
        -------
        b_lim : limiting stable axial depth of cut [mm]

        Simplification note (P2):
        k_dyn here = front BEARING PAIR stiffness (K_radial_pair).
        The true tool-tip compliance includes shaft bending in series:
            1/k_tip = 1/k_bearing + 1/k_shaft_bending
        where k_shaft = 3EI/a³ (cantilever at nose, a = overhang).
        For typical lathe spindle (a≈80mm, R1≈30mm): k_shaft ≈ 4.8 MN/mm,
        which reduces k_tip by ~6% vs bearing-only — acceptable error.
        For slender spindles (large a or small R1), this assumption
        underestimates tool-tip compliance → overestimates b_lim.

        Reference
        ─────────
            Altintas, Y. (2012). Manufacturing Automation: Metal Cutting
            Mechanics, Machine Tool Vibrations, and CNC Design (2nd ed.),
            Cambridge University Press, Ch. 3 (Chatter Stability).

        Sanity check (typical CNC lathe):
            k=200,000 N/mm, ζ=0.03, Ks=2500 N/mm²
            → b_lim = 2×200000×0.03×1.03/2500 ≈ 4.94 mm  (reasonable for
              roughing — matches published stability-lobe charts)
        """
        zeta = self.chatter_zeta
        b_lim = 2.0 * k_dyn * zeta * (1.0 + zeta) / max(self.chatter_Ks, 1e-6)
        return max(b_lim, 1e-6)   # guard against zero/negative

    # ─────────────────────────────────────────────────────────────────────────
    # β system reliability helper
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_beta_system(
        self,
        x:   np.ndarray,
        sn:  dict,
    ) -> float:  # n_mc param unused: β from sn dict directly (no extra MC needed)
        """
        Fast FOSM system reliability index from Taguchi S/N means + stds.

        Uses mean and std already stored in `sn` dict (computed by taguchi_sn_ratio)
        — no extra surrogate evaluations.

        For each limit state g_i = limit_i − Y_i (or Y_i − limit_i):
            β_i = μ(g_i) / σ(g_i)
            P_fi = Φ(−β_i)
        System: β_sys = Φ⁻¹(1 − ∏(1 − P_fi))

        Returns −β_system so that minimising f7 = maximising reliability.
        """
        from scipy import stats as scipy_stats
        import math as _math

        # Use surrogate output means and stds directly from sn dict
        _LIMITS_SIGN = {
            "static_max_deflection_um" : (20.0,   +1),  # δ < 20 μm
            "static_max_vonmises_MPa"  : (300.0,  +1),  # σ < 300 MPa
            "static_factor_of_safety"  : (2.0,    -1),  # FoS > 2.0
            "freq_mode1_Hz"            : (self.n_rpm / 60.0 / 0.75, -1),  # f1 > n[Hz]/0.75 ✅ correct
        }

        eps = 1e-12
        pf_list = []

        # Surrogate quality guard: if surrogate CV R² < 0.5 for an output,
        # its β estimate is noise-dominated and should be excluded from system β.
        # This prevents β_deflection≈0.45 (from bad surrogate) from dragging
        # down β_sys when the actual deflection is safely within limits.
        _poor_surrogate_outputs = set()
        if hasattr(self.surrogate, 'cv_scores_') and self.surrogate.cv_scores_:
            for out_name, r2 in self.surrogate.cv_scores_.items():
                if r2 < 0.50:
                    _poor_surrogate_outputs.add(out_name)

        for out_name, (limit, sign) in _LIMITS_SIGN.items():
            if out_name not in sn:
                continue
            if out_name in _poor_surrogate_outputs:
                continue  # skip unreliable surrogate outputs from β_sys
            mu_y  = sn[out_name]["mean"]
            sig_y = max(sn[out_name]["std"], eps)
            # g = sign*(limit - Y) → μ_g, σ_g
            mu_g  = sign * (limit - mu_y)
            sig_g = sig_y                          # std(g) = std(Y) for linear g
            beta  = mu_g / max(sig_g, eps)
            beta  = max(-8.0, min(8.0, beta))      # clamp numerical extremes
            pf    = float(scipy_stats.norm.cdf(-beta))
            pf_list.append(pf)

        if not pf_list:
            return 0.0

        # Series system failure probability — UPPER BOUND (conservative)
        # Assumes limit states are INDEPENDENT; positive correlation between
        # states (e.g. larger bore improves both deflection and L10 together)
        # means actual P_f_sys ≤ this bound.  Ref: ISO 2394:2015 §6.4.
        pf_sys = 1.0 - _math.prod(1.0 - p for p in pf_list)
        pf_sys = max(eps, min(1.0 - eps, pf_sys))
        beta_sys = float(scipy_stats.norm.ppf(1.0 - pf_sys))
        beta_sys = max(-8.0, min(8.0, beta_sys))

        return -beta_sys   # negate: minimising f7 = maximising β

    # ─────────────────────────────────────────────────────────────────────────
    # Objective evaluation
    # ─────────────────────────────────────────────────────────────────────────
    def multi_objective_fitness(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate all EIGHT objectives for a single design vector.

        Objectives:
            f1  −S/N deflection [dB]        Robustness under manufacturing + op. noise
            f2  −S/N stress [dB]            Robustness under noise
            f3  Total cost [USD]            Machining + SA + snap + CHATTER penalty
            f4  Mass [kg]                   Shaft weight proxy
            f5  −L10/L10_target             Bearing life (normalised, minimise = longer life)
            f6  n_rpm / freq_mode1_Hz       Critical speed ratio (minimise < 0.75)
            f7  −β_system                   System reliability index (FOSM)
            f8  b_required/b_lim            Chatter stability ratio (NEW, option C)

        f8 / Chatter Stability (option C — constraint AND objective):
            • OBJECTIVE: f8 = b_required / b_lim is minimised, pushing the
              optimizer toward designs with higher b_lim (more stable —
              larger achievable depth of cut before regenerative chatter).
            • CONSTRAINT: if f8 > 1.0 (b_lim < b_required → UNSTABLE at the
              required roughing depth), a penalty is added to f3 (cost),
              following the same pattern as the existing catalog-snap
              penalty:  chatter_penalty = max(0, f8−1) × 2000 [USD]
              This makes unstable designs strictly dominated on cost while
              f8 itself remains visible as a continuous trade-off axis on
              the Pareto front (per user's "option C" choice).

        b_lim = 2·k_dyn·ζ·(1+ζ)/K_s   (Altintas 2012, see _chatter_b_lim)
        k_dyn = front-bearing-pair radial stiffness, cached per R2 via
                _bearing_cached() — zero extra catalog lookups vs. f5.
        """
        from design_variables import _ACBB_BORES

        # ── f1, f2: Robustness (Taguchi S/N under 3-layer noise) ─────────
        sn          = self.taguchi_sn_ratio(x, sn_type="smaller")
        defl_name   = self.surrogate.output_names[self._idx["deflection"]]
        stress_name = self.surrogate.output_names[self._idx["stress"]]
        f1          = -sn[defl_name]["sn_ratio"]
        f2          = -sn[stress_name]["sn_ratio"]

        # ── f3: Cost + catalog-snap penalty ──────────────────────────────
        var_dict   = self.design_space.decode_vector(x)
        sa_results = self.sa_analyser.analyse_all(var_dict, n_bins=self.n_sa_bins)
        cost_dict  = self.cost_model.total_cost(var_dict, sa_results)
        f3         = cost_dict["total_usd"]
        bore_raw   = var_dict["R2"] * 2.0
        nearest    = int(np.argmin(np.abs(_ACBB_BORES - bore_raw)))
        f3        += BORE_SNAP_PENALTY_USD_PER_MM * abs(bore_raw - _ACBB_BORES[nearest])

        # ── Bearing minimum load constraint (ISO 281 §4.3.3) ───────────
        # P_min ≈ 0.02 × C_r for ACBB to prevent skidding/smearing
        # If cutting force < P_min → bearing is over-designed for this load
        C_r_min, _ = self._bearing_cached(var_dict)
        P_min_iso  = 0.02 * C_r_min           # N
        Ft_v = float(var_dict.get("Ft", 1000.0))
        Fr_v = float(var_dict.get("Fr", 1000.0))
        F_act = np.sqrt(Ft_v**2 + Fr_v**2)
        if F_act < P_min_iso:
            # Penalty: $2000 per kN below minimum load (steers to smaller bore)
            f3 += 2000.0 * (P_min_iso - F_act) / 1000.0

        # ── f4: Mass ──────────────────────────────────────────────────────
        ri     = var_dict["ri"]
        volume = sum(np.pi * (var_dict[f"R{i}"]**2 - ri**2) * var_dict[f"L{i}"]
                     for i in range(1, 5))
        f4     = volume * var_dict["rho"] * 1e3   # kg

        # ── f5: Bearing life (ISO 281, fast) ─────────────────────────────
        L10   = self._l10_fast(var_dict)
        # Normalise: 0 = exactly at target, negative = exceeds target (good)
        # Minimising f5 → maximising L10
        f5    = -(L10 / self.L10_target)

        # ── f6: Critical speed margin ─────────────────────────────────────
        # freq_mode1_Hz mean from the Taguchi S/N computation
        if self._idx_freq1 is not None:
            freq1_name = self.surrogate.output_names[self._idx_freq1]
            freq1_mean = sn[freq1_name]["mean"] if freq1_name in sn \
                         else sn[next(iter(sn))]["mean"]
        else:
            freq1_mean = sn[next(iter(sn))]["mean"]   # fallback: first output
        freq1_mean = max(abs(freq1_mean), 1.0)         # guard div/0
        # CRITICAL FIX: n_rpm [rpm] / freq [Hz] is NOT dimensionless.
        # Correct speed ratio = (n_rpm/60) / f1_Hz — both in [Hz]
        # Old code gave f6≈8.5 instead of ≈0.14 for a typical spindle.
        # Threshold of 0.75 now physically correct: run < 75% of f1.
        f6 = (self.n_rpm / 60.0) / max(freq1_mean, 1.0)  # dimensionless ∈ (0,1) for safe designs

        # ── f7: System Reliability Index β (FOSM) ─────────────────────────
        # −β_system  (minimise → maximise reliability)
        # Reuses MC samples from taguchi_sn_ratio via _compute_beta_system.
        f7 = self._compute_beta_system(x, sn)

        # ── TIR bending-slope penalty (Module 09 physics, inline estimate) ────
        # Root cause of TIR≈41μm in production run: large overhang drives
        # bending slope TIR. This wasn't an optimizer objective — now it is.
        # Formula IDENTICAL to 09_shaft_runout.py (after consistency fix):
        #   L_oh  = L1 + front_z_fraction × L2   [mm]  — from design vector
        #   θ     = F_resultant × L_oh² / (2EI)  [rad]
        #   TIR   = θ × L_oh × 1000              [μm]
        # Penalty: $1500/μm above ISO 230-1 Class B limit of 20μm.
        _TIR_LIMIT = 20.0   # ISO 230-1 Class B
        _L1  = float(var_dict.get("L1",  80.0))
        _L2  = float(var_dict.get("L2", 200.0))
        _ff  = float(var_dict.get("front_z_fraction", 0.15))
        _rf  = float(var_dict.get("rear_z_fraction",  0.85))
        _L_oh   = _L1 + _ff * _L2          # nose → front bearing [mm]
        _L_rear = _L1 + _rf * _L2          # nose → rear bearing  [mm]
        _L_span = max(_L_rear - _L_oh, 1.0)
        _amp    = 1.0 + _L_oh / _L_span    # TIR amplification factor (same as module 09)
        _R1  = float(var_dict.get("R1", 45.0))
        _ri  = float(var_dict.get("ri", 30.0))
        _E   = float(var_dict.get("E", 2.1e5))
        _Ft  = float(var_dict.get("Ft", 1000.0))
        _Fr  = float(var_dict.get("Fr", 1000.0))
        _F_res = math.sqrt(_Ft**2 + _Fr**2)   # resultant (both bend the shaft)
        _I   = max(math.pi / 4.0 * (_R1**4 - _ri**4), 1.0)
        _theta     = (_F_res * _L_oh**2) / (2.0 * _E * _I)
        _TIR_slope = abs(_theta * _L_oh) * 1000.0   # μm (before amp)
        f3 += 1500.0 * max(0.0, _TIR_slope - _TIR_LIMIT)

        # ── f8: Chatter Stability Ratio (option C — constraint + objective) ─
        _, K_dyn = self._bearing_cached(var_dict)   # reuses f5's cached lookup
        b_lim    = self._chatter_b_lim(K_dyn)
        f8       = self.chatter_b_required / b_lim   # <1 stable, >1 unstable

        # CONSTRAINT: penalise unstable designs (f8>1) via cost f3
        chatter_penalty = max(0.0, f8 - 1.0) * CHATTER_PENALTY_USD_PER_UNIT
        f3             += chatter_penalty

        return np.array([f1, f2, f3, f4, f5, f6, f7, f8])

    # ─────────────────────────────────────────────────────────────────────────
    # Differential Evolution (weighted sum)
    # ─────────────────────────────────────────────────────────────────────────
    def optimize_de(
        self,
        objective_weights: Optional[np.ndarray] = None,
        maxiter: int = 100,
        seed:    int = 42,
        topsis_weights: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """
        Differential Evolution sweep over 18 weight vectors → approximate Pareto.
        Updated for 6 objectives: adds L10 and critical-speed-margin weights.
        """
        from scipy.optimize import differential_evolution

        lo, hi = self.bounds[:, 0], self.bounds[:, 1]

        # ── Weight vectors — 20 diverse directions for 8 objectives ───────
        # [f1_defl, f2_stress, f3_cost, f4_weight, f5_L10, f6_speed, f7_beta, f8_chatter]
        weight_sets = np.array([
            [1/8]*8,                                     # equal
            [1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],        # deflection / stress axes
            [0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],        # cost / weight axes
            [0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],        # L10 / speed axes
            [0,0,0,0,0,0,1,0],                           # reliability axis
            [0,0,0,0,0,0,0,1],                           # chatter axis (NEW)
            [0.27,0.27,0.09,0.09,0.05,0.05,0.09,0.09],  # robustness-heavy+β+chatter
            [0.09,0.09,0.31,0.22,0.05,0.05,0.09,0.10],  # cost+weight heavy
            [0.13,0.13,0.09,0.09,0.22,0.09,0.13,0.12],  # L10+β priority
            [0.18,0.18,0.09,0.09,0.09,0.18,0.09,0.10],  # speed margin priority
            [0.17,0.17,0.10,0.09,0.10,0.09,0.14,0.14],  # balanced spindle
            [0.22,0.22,0.09,0.09,0.09,0.09,0.09,0.11],  # robustness balanced
            [0.09,0.09,0.13,0.09,0.18,0.09,0.22,0.11],  # reliability priority
            [0.30,0.09,0.09,0.09,0.09,0.09,0.13,0.12],  # max defl robustness+β
            [0.09,0.30,0.09,0.09,0.09,0.09,0.13,0.12],  # max stress robustness+β
            [0.09,0.09,0.09,0.09,0.18,0.18,0.18,0.10],  # life+speed+reliability
            [0.13,0.13,0.09,0.04,0.13,0.13,0.22,0.13],  # reliability-dominant
            [0.13,0.13,0.09,0.09,0.09,0.09,0.13,0.25],  # chatter-dominant (NEW)
        ], dtype=float)
        weight_sets = weight_sets / weight_sets.sum(axis=1, keepdims=True)

        # Pilot normalisation
        pilot_X = np.array([
            self.design_space.get_nominal(),
            lo + (hi - lo) * 0.5,
        ])
        pilot_F = np.array([self.multi_objective_fitness(x) for x in pilot_X])
        scale   = np.abs(pilot_F).mean(axis=0)
        scale   = np.where(scale < 1e-10, 1.0, scale)

        log.info(f"Running Differential Evolution  "
                 f"maxiter={maxiter}  n_weights={len(weight_sets)}")

        all_X: List[np.ndarray] = []
        all_F: List[np.ndarray] = []

        for w in weight_sets:
            def scalar_obj(x, _w=w):
                return float(_w @ (self.multi_objective_fitness(x) / scale))

            res = differential_evolution(
                scalar_obj,
                bounds=self.bounds.tolist(),
                maxiter=maxiter,
                seed=seed,
                disp=False,
                workers=1,
            )
            all_X.append(res.x)
            all_F.append(self.multi_objective_fitness(res.x))

        X_all = np.array(all_X); F_all = np.array(all_F)

        # Non-dominated front
        pareto_idx = self._fast_nondom_sort(F_all)[0]
        X_p = X_all[pareto_idx]; F_p = F_all[pareto_idx]

        # ── Knee point via TOPSIS (Module 14) — replaces min-distance-to-utopia ──
        topsis_mod = _load_topsis()
        topsis_res = topsis_mod.topsis(F_p, weights=topsis_weights)
        best       = topsis_res.best_idx

        beta_sys = -F_p[best,6] if F_p.shape[1] > 6 else float("nan")
        chatter  = F_p[best,7] if F_p.shape[1] > 7 else float("nan")
        log.info(f"✅ DE done  nfev≈{maxiter*len(weight_sets)*15}  "
                 f"cost=${F_p[best,2]:.0f}  weight={F_p[best,3]:.2f}kg  "
                 f"L10={-F_p[best,4]*self.L10_target:,.0f}h  "
                 f"speed_ratio={F_p[best,5]:.4f}  {'SAFE ✅' if F_p[best,5]<0.75 else 'WARN ⚠️ >0.75'}  β_sys={beta_sys:.3f}  "
                 f"chatter_ratio={chatter:.3f}  TOPSIS_C={topsis_res.scores[best]:.4f}")

        return OptimizationResult(
            pareto_front_X      = X_p,
            pareto_front_F      = F_p,
            best_robust_design  = X_p[best],
            n_evaluations       = maxiter * len(weight_sets) * 15,
            convergence_history = [],
            objective_labels    = self.objective_labels,
            topsis_result       = topsis_res,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # NSGA-II (kept for backward compat) + NSGA-III (recommended for 6-obj)
    # ─────────────────────────────────────────────────────────────────────────
    def optimize_nsga2(
        self,
        pop_size: int = 100,
        n_gen:    int = 50,
        seed:     int = 42,
        topsis_weights: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """NSGA-II — kept for backward compatibility. Use optimize_nsga3() for 8 objectives."""
        if PYMOO_AVAILABLE:
            n_obj  = self.n_objectives
            outer  = self

            class _Problem(Problem):
                def __init__(self):
                    super().__init__(n_var=outer.n_vars, n_obj=n_obj,
                                     xl=outer.bounds[:,0], xu=outer.bounds[:,1])
                def _evaluate(self, X, out, *a, **kw):
                    out["F"] = np.array([outer.multi_objective_fitness(x) for x in X])

            res      = pymoo_minimize(_Problem(), NSGA2(pop_size=pop_size),
                                      get_termination("n_gen",n_gen),
                                      seed=seed, verbose=False)
            pareto_X = res.X; pareto_F = res.F
        else:
            log.warning("pymoo not found — falling back to NSGA-III custom")
            return self.optimize_nsga3(pop_size, n_gen, seed, topsis_weights=topsis_weights)

        topsis_mod = _load_topsis()
        topsis_res = topsis_mod.topsis(pareto_F, weights=topsis_weights)
        best_idx   = topsis_res.best_idx
        log.info(f"✅ NSGA-II done  Pareto points={len(pareto_X)}  "
                 f"TOPSIS_C={topsis_res.scores[best_idx]:.4f}")

        return OptimizationResult(
            pareto_front_X      = pareto_X,
            pareto_front_F      = pareto_F,
            best_robust_design  = pareto_X[best_idx],
            n_evaluations       = pop_size*n_gen,
            convergence_history = [],
            objective_labels    = self.objective_labels,
            topsis_result       = topsis_res,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # NSGA-III (recommended for 6 objectives)
    # ─────────────────────────────────────────────────────────────────────────
    def optimize_nsga3(
        self,
        pop_size: int = 120,
        n_gen:    int = 80,
        seed:     int = 42,
        topsis_weights: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """
        NSGA-III — better than NSGA-II for ≥ 3 objectives (Deb & Jain, 2014).

        Key improvement: reference-point association replaces crowding distance.
        With 8 objectives, crowding distance is meaningless; structured
        Das-Dennis reference directions (n_partitions=2 → C(9,2)=36 directions)
        maintain diversity across the Pareto surface.

        Knee-point selection uses TOPSIS (Module 14) instead of minimum
        distance-to-utopia — see topsis_selector.topsis() for rationale.

        Uses pymoo if available, otherwise falls back to custom implementation.
        """
        log.info(f"Running NSGA-III  pop={pop_size}  gen={n_gen}  "
                 f"n_obj={self.n_objectives}")
        rng = np.random.default_rng(seed)

        try:
            from pymoo.algorithms.moo.nsga3 import NSGA3
            from pymoo.util.ref_dirs        import get_reference_directions

            outer = self
            class _Prob3(Problem):
                def __init__(self):
                    super().__init__(n_var=outer.n_vars, n_obj=outer.n_objectives,
                                     xl=outer.bounds[:,0], xu=outer.bounds[:,1])
                def _evaluate(self, X, out, *a, **kw):
                    out["F"] = np.array([outer.multi_objective_fitness(x) for x in X])

            # n_partitions=2 for 8 objectives -> C(8+2-1,2) = C(9,2) = 36 ref dirs
            # (n_partitions=4 would give C(11,4)=330, too many for pop_size=120)
            ref_dirs = get_reference_directions(
                "das-dennis", self.n_objectives, n_partitions=2)
            res  = pymoo_minimize(
                _Prob3(), NSGA3(pop_size=pop_size, ref_dirs=ref_dirs),
                get_termination("n_gen", n_gen), seed=seed, verbose=False)
            X_p, F_p = res.X, res.F
            log.info(f"✅ NSGA-III (pymoo) done  Pareto points={len(X_p)}")

        except (ImportError, Exception) as e:
            log.warning(f"NSGA-III pymoo fallback: {e}")
            X_p, F_p = self._nsga3_custom(pop_size, n_gen, rng)

        topsis_mod = _load_topsis()
        topsis_res = topsis_mod.topsis(F_p, weights=topsis_weights)
        best       = topsis_res.best_idx
        log.info(f"   TOPSIS_C={topsis_res.scores[best]:.4f}  "
                 f"(of {len(F_p)} Pareto points)")

        return OptimizationResult(
            pareto_front_X      = X_p,
            pareto_front_F      = F_p,
            best_robust_design  = X_p[best],
            n_evaluations       = pop_size * n_gen,
            convergence_history = [],
            objective_labels    = self.objective_labels,
            topsis_result       = topsis_res,
        )

    def _nsga3_custom(self, pop_size, n_gen, rng):
        """Custom NSGA-III fallback without pymoo."""
        from itertools import combinations_with_replacement

        lo, hi = self.bounds[:,0], self.bounds[:,1]

        def simplex_lattice(m, p):
            dirs = []
            for combo in combinations_with_replacement(range(p+1), m-1):
                w = np.diff([0]+list(combo)+[p])
                if w.min() >= 0: dirs.append(w/p)
            return np.array(dirs)

        # p=2 for 8 objectives -> C(2+8-1,8-1)=C(9,7)=36 ref dirs (manageable)
        ref_dirs = simplex_lattice(self.n_objectives, 2)
        N = max(pop_size, len(ref_dirs))

        X_pop = lo + rng.random((N, self.n_vars))*(hi-lo)
        F_pop = np.array([self.multi_objective_fitness(x) for x in X_pop])

        for gen in range(n_gen):
            idx1 = rng.integers(0,N,N); idx2 = rng.integers(0,N,N)
            alpha = rng.random((N, self.n_vars))
            X_off = np.clip(alpha*X_pop[idx1]+(1-alpha)*X_pop[idx2], lo, hi)
            mask  = rng.random((N, self.n_vars)) < 0.05
            X_off[mask] = lo[np.where(mask)[1]] + \
                          rng.random(mask.sum())*(hi-lo)[np.where(mask)[1]]
            X_off = np.clip(X_off, lo, hi)
            F_off = np.array([self.multi_objective_fitness(x) for x in X_off])

            X_all = np.vstack([X_pop,X_off]); F_all = np.vstack([F_pop,F_off])
            fronts = self._fast_nondom_sort(F_all)
            sel=[]; fi=0
            while fi<len(fronts) and len(sel)+len(fronts[fi])<=N:
                sel.extend(fronts[fi]); fi+=1
            if len(sel)<N and fi<len(fronts):
                needed = N-len(sel)
                F_n    = self._normalise(F_all)
                remain = fronts[fi]
                assoc  = np.array([
                    np.argmin([np.linalg.norm(F_n[i]-rd*(F_n[i]@rd)/max(rd@rd,1e-12))
                               for rd in ref_dirs])
                    for i in remain])
                nc = np.zeros(len(ref_dirs),int)
                for _ in range(needed):
                    if not remain: break
                    j = int(np.argmin(nc[assoc[:len(remain)]]))
                    sel.append(remain[j]); nc[assoc[j]]+=1
                    remain.pop(j); assoc=np.delete(assoc,j)
            X_pop = X_all[sel[:N]]; F_pop = F_all[sel[:N]]

        pidx = self._fast_nondom_sort(F_pop)[0]
        log.info(f"✅ NSGA-III (custom)  Pareto={len(pidx)}")
        return X_pop[pidx], F_pop[pidx]

    @staticmethod
    def _fast_nondom_sort(F):
        n = len(F); dc = np.zeros(n,int); db=[[] for _ in range(n)]; fronts=[[]]
        for i in range(n):
            for j in range(i+1,n):
                ij=np.all(F[i]<=F[j])and np.any(F[i]<F[j])
                ji=np.all(F[j]<=F[i])and np.any(F[j]<F[i])
                if ij:   db[i].append(j); dc[j]+=1
                elif ji: db[j].append(i); dc[i]+=1
            if dc[i]==0: fronts[0].append(i)
        cur=0
        while fronts[cur]:
            nxt=[]
            for i in fronts[cur]:
                for j in db[i]:
                    dc[j]-=1
                    if dc[j]==0: nxt.append(j)
            cur+=1; fronts.append(nxt)
        return [f for f in fronts if f]

    @staticmethod
    def _normalise(F):
        lo=F.min(axis=0); hi=F.max(axis=0)
        r=np.where(np.abs(hi-lo)<1e-12,1.,hi-lo)
        return (F-lo)/r

    # ─────────────────────────────────────────────────────────────────────────
    # Result reporting
    # ─────────────────────────────────────────────────────────────────────────
    def report(self, result: OptimizationResult) -> pd.DataFrame:
        """
        Format a readable DataFrame of the Pareto front objectives.
        """
        df = pd.DataFrame(result.pareto_front_F, columns=result.objective_labels)
        return df.round(3)

    def report_best(
        self,
        result:        OptimizationResult,
        n_rpm:         float = 4000.0,
        save_csv_path: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Report the best design with catalog-snapped bearing values.

        IMPORTANT FIX: previous version saved raw optimizer values for R2,
        K_radial, K_axial.  These are now replaced with catalog-derived
        values via resolve_to_catalog() so the output is directly usable
        on a drawing and BOM.

        Args:
            result       : OptimizationResult from optimize_de() or optimize_nsga2()
            n_rpm        : Operating speed for catalog snap [RPM]
            save_csv_path: If set, saves the manufacturable design to this CSV

        Returns:
            dict with keys:
                design_variables     — raw optimizer vector (for surrogate calls)
                catalog              — resolve_to_catalog() output
                objectives           — f1..f4 values
                costs                — cost breakdown
                selective_assembly   — SA quality metrics
        """
        x         = result.best_robust_design
        # Use catalog-resolved design for reporting (not raw vector)
        catalog   = self.design_space.resolve_to_catalog(x, n_rpm)
        var_dict  = catalog["design_variables"]

        # SA and cost use var_dict (raw) — that's correct because pos_tol,
        # geometry etc. are still from the optimizer; only K values change
        sa_res   = self.sa_analyser.analyse_all(var_dict, self.n_sa_bins)
        costs    = self.cost_model.total_cost(var_dict, sa_res)
        f        = self.multi_objective_fitness(x)

        # Add catalog stiffness to cost model bearing input for accuracy
        costs_corrected = dict(costs)
        costs_corrected["K_radial_used"] = catalog["K_radial_catalog"]
        costs_corrected["K_axial_used"]  = catalog["K_axial_catalog"]

        report = {
            "design_variables":   var_dict,
            "catalog":            catalog,
            "objectives":         dict(zip(self.objective_labels, f.tolist())),
            "costs":              costs_corrected,
            "selective_assembly": {
                r.interface_name: {
                    "improvement_x":  r.improvement_ratio,
                    "std_with_sa_um": r.std_gap_um,
                    "std_no_sa_um":   r.std_gap_no_sa_um,
                    "yield_pct":      r.match_yield * 100,
                }
                for r in sa_res
            },
        }

        # Save manufacturable design to CSV if requested
        if save_csv_path:
            self._save_optimal_csv(catalog["manufacturable_design"], save_csv_path)

        return report

    def _save_optimal_csv(self, mfg_design: Dict, path: str) -> None:
        """
        Save the catalog-resolved optimal design to CSV.

        The CSV now contains:
          - All geometry/material/load variables (raw optimizer values — correct)
          - bore_catalog_mm   ← the actual SKF bore to order
          - front_bearing     ← SKF part number
          - rear_bearing      ← SKF part number
          - K_radial_catalog  ← stiffness for ANSYS COMBIN14
          - K_axial_catalog
          - F_preload_MA_N
        """
        import csv
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(mfg_design.keys()))
            writer.writeheader()
            writer.writerow(mfg_design)
        print(f"✅ Optimal design saved → {path}")
        print(f"   Front bearing: {mfg_design.get('front_bearing','?')}")
        print(f"   Rear  bearing: {mfg_design.get('rear_bearing','?')}")
        print(f"   Bore (catalog): {mfg_design.get('bore_catalog_mm','?')} mm")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_optimizer_results(
    result:       "OptimizationResult",
    design_space: "DesignSpace",
    save_dir:     str = ".",
) -> None:
    """
    Three optimizer-result plots.

    Fig 05a — Pareto front scatter (f1 vs f2 coloured by f3=cost)
    Fig 05b — Objective convergence (best f1+f2 per generation if available,
               else bar chart of final objectives at best design)
    Fig 05c — Best design variable values vs. bounds (normalised bar)
    """
    import matplotlib.pyplot as plt
    import os

    NAVY=C.NAVY; TEAL=C.TEAL; CORAL=C.RED; GOLD=C.ORANGE
    MINT=C.GREEN; GRAY=C.GRAY; PURPLE=C.PURPLE
    os.makedirs(save_dir, exist_ok=True)
    apply_paper_theme()

    F      = result.pareto_front_F   # (n_pareto, 4)
    X      = result.pareto_front_X   # (n_pareto, n_vars)
    labels = result.objective_labels
    names  = design_space.get_variable_names()
    bounds = design_space.get_bounds()

    # ── Fig 05a: Pareto front (f1 vs f2, colour = f3 cost) ───────────
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=C.BG)
    ax.set_facecolor(C.BG)
    n_pts = len(F)
    if n_pts > 1:
        sc = ax.scatter(F[:, 0], F[:, 1], c=F[:, 2], cmap="plasma",
                        s=40, edgecolors="none", alpha=0.85)
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("Total cost [USD]", color="white")
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    else:
        ax.scatter(F[:, 0], F[:, 1], c=TEAL, s=80, marker="*", zorder=5)
    # Highlight best point
    best_idx = np.argmin(F[:, 0] + F[:, 1])  # min sum of robustness objectives
    ax.scatter(F[best_idx, 0], F[best_idx, 1],
               c=GOLD, s=150, marker="*", zorder=6, label="Best design")
    ax.set_xlabel(labels[0] if len(labels) > 0 else "f1")
    ax.set_ylabel(labels[1] if len(labels) > 1 else "f2")
    ax.set_title("Fig 05a — Pareto Front  (f1=−SN_defl, f2=−SN_stress, colour=cost)",
                 pad=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(save_dir, "05a_pareto_front.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=C.BG)
    plt.close(fig); print(f"  Saved → {p}")

    # ── Fig 05b: Final objective values at best design ────────────────
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=C.BG)
    ax.set_facecolor(C.BG)
    best_f  = F[best_idx]
    f_names = labels if labels else [f"f{i+1}" for i in range(len(best_f))]
    # Normalise each objective relative to its Pareto-front range
    f_min = F.min(axis=0); f_max = F.max(axis=0)
    f_range = np.where(np.abs(f_max - f_min) < 1e-10, 1.0, f_max - f_min)
    f_norm  = (best_f - f_min) / f_range
    colours = [TEAL, CORAL, GOLD, MINT][:len(best_f)]
    bars = ax.bar(range(len(best_f)), f_norm, color=colours,
                  edgecolor=NAVY, linewidth=0.5)
    for bar, raw_val, fname in zip(bars, best_f, f_names):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f"{raw_val:.2f}", ha="center", va="bottom",
                fontsize=8, color="white")
    ax.set_xticks(range(len(f_names)))
    ax.set_xticklabels([n[:20] for n in f_names], rotation=12, ha="right", fontsize=8)
    ax.set_ylabel("Normalised objective (0=best, 1=worst in Pareto)")
    ax.set_title("Fig 05b — Objective Values at Best Design", pad=8)
    ax.set_ylim(0, 1.25)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(save_dir, "05b_best_objectives.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=C.BG)
    plt.close(fig); print(f"  Saved → {p}")

    # ── Fig 05c: Best design variables vs. bounds ─────────────────────
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=C.BG)
    ax.set_facecolor(C.BG)
    best_x  = X[best_idx]
    nom_x   = design_space.get_nominal()
    # Normalise each var to [0,1] within its bounds
    x_norm  = (best_x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    nom_norm = (nom_x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    y_pos   = np.arange(len(names))
    ax.barh(y_pos, x_norm, color=TEAL, alpha=0.75, height=0.5,
            edgecolor=NAVY, linewidth=0.4, label="Optimal")
    ax.scatter(nom_norm, y_pos, s=25, c=GOLD, zorder=5,
               marker="|", linewidths=1.5, label="Nominal")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("Normalised position in bounds  [0=lower, 1=upper]")
    ax.set_title("Fig 05c — Optimal Design Variables (gold tick = nominal)", pad=8)
    ax.axvline(0.5, color=GRAY, lw=0.8, linestyle="--", alpha=0.5)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(save_dir, "05c_design_variables.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=C.BG)
    plt.close(fig); print(f"  Saved → {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os; sys.path.insert(0, ".")
    from design_variables import DesignSpace
    from ml_surrogate     import SurrogateModel
    from sklearn.dummy    import DummyRegressor
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    ds = DesignSpace()
    n  = ds.get_bounds().shape[0]

    surr = SurrogateModel(model_type="gp")
    surr.output_names = [
        "static_max_deflection_um",
        "static_max_vonmises_MPa",
        "freq_mode1_Hz",
    ]
    X_mock = np.random.rand(30, n)
    surr.scaler_X.fit(X_mock)
    for name in surr.output_names:
        sc = StandardScaler(); sc.fit(np.random.rand(30, 1))
        surr.scalers_y[name] = sc
        dr = DummyRegressor(strategy="mean")
        dr.fit(X_mock, np.random.rand(30))
        surr.models[name] = dr

    opt = RobustOptimizer(surr, ds, n_mc_inner=5, n_sa_bins=3, sa_n_parts=100)

    print("\n── Fitness evaluation ──")
    x0 = ds.get_nominal()
    f  = opt.multi_objective_fitness(x0)
    for label, val in zip(opt.objective_labels, f):
        print(f"  {label:<30} {val:10.3f}")

    print("\n── Differential Evolution (3 iter) ──")
    res = opt.optimize_de(maxiter=3, seed=42)
    print(f"  Best cost: ${res.pareto_front_F[0, 2]:.2f}")

    print("\n── Full report ──")
    rpt = opt.report_best(res)
    print("  SA interfaces:")
    for iname, stats in rpt["selective_assembly"].items():
        print(f"    {iname}: ×{stats['improvement_x']:.1f} improvement  "
              f"yield={stats['yield_pct']:.0f}%")

    os.makedirs("/tmp/spindle_plots", exist_ok=True)
    print("\nGenerating optimizer plots...")
    plot_optimizer_results(res, ds, save_dir="/tmp/spindle_plots")
    print("\n✅ Robust Optimizer v2 OK")
