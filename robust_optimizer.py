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
        n_mc_inner:       int = 20,
        output_name_map:  Optional[Dict[str, str]] = None,
        n_sa_bins:        int = 5,
        sa_n_parts:       int = 300,
    ):
        self.surrogate    = surrogate
        self.design_space = design_space
        self.n_mc_inner   = n_mc_inner
        self.bounds       = design_space.get_bounds()
        self.n_vars       = len(self.bounds)
        self.n_sa_bins    = n_sa_bins

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
            f"n_mc_inner={n_mc_inner}  sa_bins={n_sa_bins}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Taguchi S/N
    # ─────────────────────────────────────────────────────────────────────────
    def taguchi_sn_ratio(
        self,
        x_nominal: np.ndarray,
        sn_type:   Literal["smaller", "larger", "nominal"] = "smaller",
        target:    Optional[float] = None,
        eps:       float = 1e-10,     # BUG-9 FIX guard
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute S/N ratio for every surrogate output.

        Returns:
            Dict[output_name → {"sn_ratio": float, "mean": float, "std": float}]
        """
        tols  = self.design_space.get_tolerances()
        sigma = tols / 3.0
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]

        noise = np.random.normal(0.0, sigma,
                                 size=(self.n_mc_inner, self.n_vars))
        X_pert = np.clip(x_nominal + noise, lo, hi)

        Y = self.surrogate.predict(X_pert)   # (n_mc_inner, n_outputs)

        results: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(self.surrogate.output_names):
            y    = Y[:, i]
            mu   = float(np.mean(y))
            std  = float(np.std(y))
            y_c  = np.maximum(np.abs(y), eps)    # BUG-9 FIX

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
    # Multi-objective fitness  (4 objectives)
    # ─────────────────────────────────────────────────────────────────────────
    def multi_objective_fitness(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate all four objectives for a single design vector.

        Returns:
            f: (4,) array [−SN_defl, −SN_stress, total_cost_USD, weight_kg]
        """
        # ── Robustness objectives (f1, f2) ────────────────────────────────
        sn = self.taguchi_sn_ratio(x, sn_type="smaller")

        # BUG-6 FIX — use pre-resolved indices, not hardcoded key lookup
        defl_name  = self.surrogate.output_names[self._idx["deflection"]]
        stress_name = self.surrogate.output_names[self._idx["stress"]]
        f1 = -sn[defl_name]["sn_ratio"]
        f2 = -sn[stress_name]["sn_ratio"]

        # ── Cost objective (f3) — NEW ──────────────────────────────────────
        var_dict   = self.design_space.decode_vector(x)
        sa_results = self.sa_analyser.analyse_all(var_dict, n_bins=self.n_sa_bins)
        cost_dict  = self.cost_model.total_cost(var_dict, sa_results)
        f3         = cost_dict["total_usd"]

        # ── Weight proxy (f4) ──────────────────────────────────────────────
        L_total  = var_dict["L1"] + var_dict["L2"] + var_dict["L3"] + var_dict["L4"]
        R_vals   = [var_dict["R1"], var_dict["R2"], var_dict["R3"], var_dict["R4"]]
        L_vals   = [var_dict["L1"], var_dict["L2"], var_dict["L3"], var_dict["L4"]]
        ri       = var_dict["ri"]
        volume   = sum(
            np.pi * (R**2 - ri**2) * L
            for R, L in zip(R_vals, L_vals)
        )
        mass_kg  = volume * var_dict["rho"] * 1e3   # ton→kg
        f4       = mass_kg

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
        result: OptimizationResult,
    ) -> Dict[str, object]:
        """
        Decode and report the best design with all cost and SA details.
        """
        x        = result.best_robust_design
        var_dict = self.design_space.decode_vector(x)
        sa_res   = self.sa_analyser.analyse_all(var_dict, self.n_sa_bins)
        costs    = self.cost_model.total_cost(var_dict, sa_res)
        f        = self.multi_objective_fitness(x)

        report = {
            "design_variables": var_dict,
            "objectives":       dict(zip(self.objective_labels, f.tolist())),
            "costs":            costs,
            "selective_assembly": {
                r.interface_name: {
                    "improvement_x":    r.improvement_ratio,
                    "std_with_sa_um":   r.std_gap_um,
                    "std_no_sa_um":     r.std_gap_no_sa_um,
                    "yield_pct":        r.match_yield * 100,
                }
                for r in sa_res
            },
        }
        return report


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from design_variables import DesignSpace
    from ml_surrogate     import SurrogateModel
    from sklearn.dummy    import DummyRegressor
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    ds = DesignSpace()
    n  = ds.get_bounds().shape[0]

    # Build a mock surrogate with correct output names
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
    print("\n✅ Robust Optimizer v2 OK")
