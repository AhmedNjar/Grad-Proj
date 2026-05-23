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
        noise_alpha_E:    float = 3.2e-4,
        noise_alpha_sy:   float = 4.0e-4,
    ):
        self.surrogate    = surrogate
        self.design_space = design_space
        self.n_mc_inner   = n_mc_inner
        self.bounds       = design_space.get_bounds()
        self.n_vars       = len(self.bounds)
        self.n_sa_bins    = n_sa_bins

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

        # Cost and SA models
        self.cost_model  = SpindleCostModel()
        self.sa_analyser = SelectiveAssemblyAnalyser(n_parts=sa_n_parts)

        # BUG-6 FIX — resolve surrogate output indices at construction time
        omap = output_name_map or _DEFAULT_OUTPUT_MAP
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

        # Objective labels (BUG-7/8 FIX — dynamic)
        self.objective_labels = [
            "−S/N_deflection (dB)",
            "−S/N_stress (dB)",
            "Total cost (USD)",
            "Weight proxy (kg)",
        ]
        self.n_objectives = len(self.objective_labels)

        log.info(
            f"RobustOptimizer ready  "
            f"n_vars={self.n_vars}  n_obj={self.n_objectives}  "
            f"n_mc_inner={n_mc_inner}  sa_bins={n_sa_bins}  "
            f"force_CV={noise_force_cv:.0%}  ΔT_max={noise_temp_max_C:.0f}°C"
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
    def multi_objective_fitness(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate all four objectives for a single design vector.

        Catalog-snap penalty:
            The optimizer works in continuous R2 space [35, 60] mm.
            At each evaluation, the actual bore is snapped to the nearest
            catalog value.  The snap residual:
                δ_bore = |R2_optimizer × 2 − bore_catalog| [mm]
            is penalised by adding to f3 (cost):
                penalty = 500 × δ_bore   [USD/mm]
            This steers the optimizer toward catalog-exact bore values
            (where δ_bore = 0) making the raw and snapped designs identical.

        Returns:
            f: (4,) array [−SN_defl, −SN_stress, total_cost_USD, weight_kg]
        """
        from design_variables import _ACBB_BORES

        # ── Robustness objectives (f1, f2) ────────────────────────────────
        sn = self.taguchi_sn_ratio(x, sn_type="smaller")

        # BUG-6 FIX — use pre-resolved indices, not hardcoded key lookup
        defl_name   = self.surrogate.output_names[self._idx["deflection"]]
        stress_name = self.surrogate.output_names[self._idx["stress"]]
        f1 = -sn[defl_name]["sn_ratio"]
        f2 = -sn[stress_name]["sn_ratio"]

        # ── Cost objective (f3) with catalog-snap penalty ─────────────────
        var_dict   = self.design_space.decode_vector(x)
        sa_results = self.sa_analyser.analyse_all(var_dict, n_bins=self.n_sa_bins)
        cost_dict  = self.cost_model.total_cost(var_dict, sa_results)
        f3         = cost_dict["total_usd"]

        # Catalog-snap penalty: steers optimizer to exact catalog bores
        bore_raw    = var_dict["R2"] * 2.0
        nearest_idx = int(np.argmin(np.abs(_ACBB_BORES - bore_raw)))
        snap_bore   = _ACBB_BORES[nearest_idx]
        delta_bore  = abs(bore_raw - snap_bore)          # mm gap
        snap_penalty = 500.0 * delta_bore                # USD/mm — makes
        f3 += snap_penalty                               # exact bores cheaper

        # ── Weight proxy (f4) ──────────────────────────────────────────────
        R_vals  = [var_dict["R1"], var_dict["R2"], var_dict["R3"], var_dict["R4"]]
        L_vals  = [var_dict["L1"], var_dict["L2"], var_dict["L3"], var_dict["L4"]]
        ri      = var_dict["ri"]
        volume  = sum(np.pi * (R**2 - ri**2) * L for R, L in zip(R_vals, L_vals))
        mass_kg = volume * var_dict["rho"] * 1e3   # ton→kg
        f4      = mass_kg

        return np.array([f1, f2, f3, f4])

    # ─────────────────────────────────────────────────────────────────────────
    # Differential Evolution (weighted sum)
    # ─────────────────────────────────────────────────────────────────────────
    def optimize_de(
        self,
        objective_weights: Optional[np.ndarray] = None,
        maxiter: int = 100,
        seed:    int = 42,
    ) -> OptimizationResult:
        """
        Weighted-sum single-objective GA via scipy Differential Evolution.

        Default weights: [0.30, 0.30, 0.25, 0.15]
        (robustness prioritised; cost and weight secondary).
        """
        n_obj = self.n_objectives
        if objective_weights is None:
            # BUG-8 FIX — weights built dynamically from n_objectives
            objective_weights = np.array([0.30, 0.30, 0.25, 0.15])
            if len(objective_weights) != n_obj:
                objective_weights = np.ones(n_obj) / n_obj

        # Normalise weights
        w = objective_weights / objective_weights.sum()

        # Cache normalisation factors from a small pilot evaluation
        pilot_X = np.array([
            self.design_space.get_nominal(),
            self.bounds[:, 0] + (self.bounds[:, 1] - self.bounds[:, 0]) * 0.5,
        ])
        pilot_F = np.array([self.multi_objective_fitness(x) for x in pilot_X])
        scale   = np.abs(pilot_F).mean(axis=0)
        scale   = np.where(scale < 1e-10, 1.0, scale)

        def scalar_obj(x):
            f = self.multi_objective_fitness(x)
            return float(w @ (f / scale))

        log.info(f"Running Differential Evolution  maxiter={maxiter}")
        res = differential_evolution(
            scalar_obj,
            bounds=self.bounds.tolist(),
            maxiter=maxiter,
            seed=seed,
            disp=False,
            workers=1,
        )

        best = res.x
        best_F = self.multi_objective_fitness(best)
        log.info(
            f"✅ DE done  nfev={res.nfev}  "
            f"cost=${best_F[2]:.0f}  weight={best_F[3]:.2f}kg"
        )

        return OptimizationResult(
            pareto_front_X      = best.reshape(1, -1),
            pareto_front_F      = best_F.reshape(1, -1),
            best_robust_design  = best,
            n_evaluations       = res.nfev,
            convergence_history = [],
            objective_labels    = self.objective_labels,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # NSGA-II (true multi-objective Pareto front)
    # ─────────────────────────────────────────────────────────────────────────
    def optimize_nsga2(
        self,
        pop_size: int = 100,
        n_gen:    int = 50,
        seed:     int = 42,
    ) -> OptimizationResult:
        """
        NSGA-II via pymoo.  Returns full Pareto front.
        """
        if not PYMOO_AVAILABLE:
            raise ImportError("Install pymoo: pip install pymoo")

        n_obj = self.n_objectives   # BUG-7 FIX — dynamic

        outer = self   # captured in inner class

        class _Problem(Problem):
            def __init__(self):
                super().__init__(
                    n_var=outer.n_vars,
                    n_obj=n_obj,              # BUG-7 FIX
                    xl=outer.bounds[:, 0],
                    xu=outer.bounds[:, 1],
                )

            def _evaluate(self, X, out, *args, **kwargs):
                out["F"] = np.array([
                    outer.multi_objective_fitness(x) for x in X
                ])

        res = pymoo_minimize(
            _Problem(),
            NSGA2(pop_size=pop_size),
            get_termination("n_gen", n_gen),
            seed=seed,
            verbose=False,
        )

        pareto_X = res.X
        pareto_F = res.F

        # BUG-8 FIX — weights built from n_objectives, not hardcoded length 3
        w = np.array([0.30, 0.30, 0.25, 0.15][:n_obj])
        w = w / w.sum()
        # Normalise Pareto front columns before weighting
        scale = np.abs(pareto_F).mean(axis=0)
        scale = np.where(scale < 1e-10, 1.0, scale)
        best_idx = np.argmin((pareto_F / scale) @ w)

        log.info(f"✅ NSGA-II done  Pareto points={len(pareto_X)}")

        return OptimizationResult(
            pareto_front_X      = pareto_X,
            pareto_front_F      = pareto_F,
            best_robust_design  = pareto_X[best_idx],
            n_evaluations       = pop_size * n_gen,
            convergence_history = [],
            objective_labels    = self.objective_labels,
        )

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
        with open(path, "w", newline="") as fh:
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

    NAVY="#0d1b2a"; TEAL="#00b4d8"; CORAL="#e63946"; GOLD="#ffd166"
    MINT="#06d6a0"; GRAY="#8d99ae"
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": NAVY, "axes.facecolor": "#112233",
        "axes.edgecolor": GRAY, "axes.labelcolor": "white",
        "xtick.color": GRAY, "ytick.color": GRAY,
        "text.color": "white", "grid.color": "#2d4060",
        "grid.alpha": 0.4, "font.size": 9,
    })

    F      = result.pareto_front_F   # (n_pareto, 4)
    X      = result.pareto_front_X   # (n_pareto, n_vars)
    labels = result.objective_labels
    names  = design_space.get_variable_names()
    bounds = design_space.get_bounds()

    # ── Fig 05a: Pareto front (f1 vs f2, colour = f3 cost) ───────────
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=NAVY)
    ax.set_facecolor("#112233")
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
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig); print(f"  Saved → {p}")

    # ── Fig 05b: Final objective values at best design ────────────────
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=NAVY)
    ax.set_facecolor("#112233")
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
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig); print(f"  Saved → {p}")

    # ── Fig 05c: Best design variables vs. bounds ─────────────────────
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=NAVY)
    ax.set_facecolor("#112233")
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
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=NAVY)
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
