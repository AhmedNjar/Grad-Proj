#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  Master Workflow — TechPulse Spindle RDO  (v2 — fully integrated)
================================================================================

  Usage:
      # Quick dry-run (no ANSYS, ~60 seconds)
      python master_workflow.py --n_samples 50 --dry_run

      # Full production run (requires ANSYS)
      python master_workflow.py --n_samples 500 --surrogate gp

      # From an existing FEA CSV
      python master_workflow.py --mode ml_only --fea_csv rdo_results/fea_results.csv

      # Only re-run optimization with a saved surrogate
      python master_workflow.py --mode opt_only --surrogate_pkl rdo_results/surrogate.pkl

  Flags:
      --n_samples     Number of LHS DoE points           [default 50]
      --dry_run       Use analytical beam model (no ANSYS) [flag]
      --surrogate     gp | xgb | mlp                     [default gp]
      --opt_method    de | nsga2                         [default de]
      --output_dir    Output directory                   [default rdo_results]
      --no_plots      Skip matplotlib plot generation    [flag]
      --n_rpm         Operating speed for analysis       [default 4000]
      --mode          full | ml_only | opt_only          [default full]
      --fea_csv       Path to existing FEA results CSV
      --surrogate_pkl Path to pre-trained surrogate .pkl

  Output files in <output_dir>/:
      lhs_samples.csv
      fea_results.csv
      surrogate_<type>.pkl
      optimal_design.csv          ← catalog-resolved, ready for drawing
      optimal_design_report.txt   ← full engineering report
      plots/01a_*.png … 11d_*.png ← 34 diagnostic plots
================================================================================
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("RDO_Master")


# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports — all modules loaded here so errors show clearly
# ─────────────────────────────────────────────────────────────────────────────

def _import_all():
    """Import all framework modules and return them as a namespace dict."""
    import importlib

    mods = {}
    mod_paths = {
        "design_variables":    "01_design_variables",
        "lhs_sampler":         "02_lhs_sampler",
        "fea_pool_runner":     "03_fea_pool_runner",
        "ml_surrogate":        "04_ml_surrogate",
        "selective_assembly":  "07_selective_assembly",   # must be before robust_optimizer
        "robust_optimizer":    "05_robust_optimizer",
        "inverse_design":      "06_inverse_design",
        "bearing_performance": "08_bearing_performance",
        "shaft_runout":        "09_shaft_runout",
        "rotor_eccentricity":  "10_rotor_eccentricity",
        "final_report":        "11_final_report",
        "tolerance_optimizer": "12_tolerance_optimizer",
    }

    # Add script directory to path so bare imports work
    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    for alias, mod_name in mod_paths.items():
        # Try both the numbered name and the plain name
        for try_name in [mod_name, alias]:
            try:
                spec = importlib.util.spec_from_file_location(
                    alias,
                    script_dir / f"{mod_name}.py",
                )
                m = importlib.util.module_from_spec(spec)
                sys.modules[alias] = m
                spec.loader.exec_module(m)
                mods[alias] = m
                break
            except Exception:
                pass
        if alias not in mods:
            log.warning(f"Could not import {mod_name} — skipping")

    return mods


# ─────────────────────────────────────────────────────────────────────────────
# Master orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class RDOMasterOrchestrator:

    def __init__(self, config: dict):
        self.config     = config
        self.out_dir    = Path(config["output_dir"])
        self.plot_dir   = self.out_dir / "plots"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        log.info("=" * 72)
        log.info("  TechPulse Spindle RDO — Master Workflow v2")
        log.info("=" * 72)
        for k, v in config.items():
            log.info(f"  {k:<20} = {v}")
        log.info("=" * 72)

        # Import all modules
        self.m = _import_all()

        # Core objects
        M = self.m
        self.ds  = M["design_variables"].DesignSpace()
        self.arr = M["design_variables"].SpindleBearingArrangement.default_lathe()

        log.info(f"Design space: {len(self.ds.get_variable_names())} variables")
        log.info(f"Bearing arrangement:\n{self.arr.description()}")

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1 — LHS Sampling
    # ─────────────────────────────────────────────────────────────────────────
    def stage_lhs(self) -> np.ndarray:
        log.info("\n── STAGE 1: LHS SAMPLING ──────────────────────────────────────")
        M       = self.m
        sampler = M["lhs_sampler"].LHSSampler(self.ds)
        n       = self.config["n_samples"]
        X       = sampler.generate_lhs(n, criterion="maximin", seed=42)
        path    = self.out_dir / "lhs_samples.csv"
        sampler.save_samples(X, str(path), fmt="csv")
        log.info(f"✅ {n} LHS samples saved → {path}")

        if not self.config["no_plots"]:
            log.info("   Generating LHS plots...")
            M["lhs_sampler"].plot_lhs_samples(X, self.ds, save_dir=str(self.plot_dir))

        return X

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2 — FEA Pool
    # ─────────────────────────────────────────────────────────────────────────
    def stage_fea(self, X: np.ndarray) -> pd.DataFrame:
        log.info("\n── STAGE 2: FEA POOL ──────────────────────────────────────────")
        M      = self.m
        dry    = self.config.get("dry_run", True)
        runner = M["fea_pool_runner"].FEAPoolRunner(
            X, self.ds, dry_run=dry,
            output_dir=str(self.out_dir / "fea_cache"),
        )
        t0 = time.time()
        df = runner.execute_batch(max_failures=10, save_interval=20)
        elapsed = time.time() - t0

        path = self.out_dir / "fea_results.csv"
        df.to_csv(path, index=False)
        log.info(f"✅ {len(df)} FEA cases in {elapsed:.1f}s → {path}")
        log.info(f"   δ: {df['static_max_deflection_um'].mean():.1f} μm  "
                 f"σ: {df['static_max_vonmises_MPa'].mean():.1f} MPa  "
                 f"f1: {df['freq_mode1_Hz'].mean():.0f} Hz")

        if not self.config["no_plots"]:
            log.info("   Generating FEA plots...")
            M["fea_pool_runner"].plot_fea_results(df, self.ds, save_dir=str(self.plot_dir))

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3 — ML Surrogate
    # ─────────────────────────────────────────────────────────────────────────
    def stage_surrogate(self, df: pd.DataFrame) -> object:
        log.info("\n── STAGE 3: ML SURROGATE ──────────────────────────────────────")
        M        = self.m
        out_cols = [c for c in [
            "static_max_deflection_um", "static_max_vonmises_MPa",
            "static_factor_of_safety",  "freq_mode1_Hz",
            "freq_mode2_Hz",            "freq_mode3_Hz",
        ] if c in df.columns]
        var_cols = [c for c in df.columns if c.startswith("var_")]

        X_tr = df[var_cols].values
        y_tr = df[out_cols].values

        surr = M["ml_surrogate"].SurrogateModel(
            model_type=self.config.get("surrogate", "gp")
        )
        surr.train(X_tr, y_tr, output_names=out_cols, verbose=True)

        pkl_path = self.out_dir / f"surrogate_{surr.model_type}.pkl"
        surr.save(str(pkl_path))
        log.info(f"✅ Surrogate trained → {pkl_path}")

        # Validation
        metrics = surr.evaluate(X_tr[:10], y_tr[:10])
        log.info(f"   CV R² per output:")
        for _, row in metrics.iterrows():
            log.info(f"     {row['output']:<35} R²={row['R2']:.4f}")

        if not self.config["no_plots"]:
            log.info("   Generating surrogate plots...")
            n_te = min(20, len(X_tr) // 5)
            M["ml_surrogate"].plot_surrogate_performance(
                surr, X_tr, y_tr, X_tr[:n_te], y_tr[:n_te],
                save_dir=str(self.plot_dir),
            )

        return surr

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 4 — Robust Optimization
    # ─────────────────────────────────────────────────────────────────────────
    def stage_optimize(self, surr: object) -> object:
        log.info("\n── STAGE 4: ROBUST OPTIMIZATION ───────────────────────────────")
        M   = self.m
        opt = M["robust_optimizer"].RobustOptimizer(
            surr, self.ds, n_mc_inner=20, n_sa_bins=5, sa_n_parts=500,
        )

        method = self.config.get("opt_method", "de")
        if method == "nsga2":
            log.info("Running NSGA-II multi-objective (100 pop × 50 gen)...")
            result = opt.optimize_nsga2(pop_size=100, n_gen=50, seed=42)
        else:
            log.info("Running Differential Evolution (200 iter)...")
            result = opt.optimize_de(maxiter=200, seed=42)

        n_rpm = float(self.config.get("n_rpm", 4000))

        # ── Catalog-resolved report ───────────────────────────────────────
        rpt = opt.report_best(result, n_rpm=n_rpm)
        cat = rpt["catalog"]
        mfg = cat["manufacturable_design"]

        # Save correct CSV (catalog values, not raw optimizer)
        csv_path = self.out_dir / "optimal_design.csv"
        import csv
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(mfg.keys()))
            writer.writeheader(); writer.writerow(mfg)
        log.info(f"✅ Optimal design saved → {csv_path}")
        log.info(f"   Front bearing : {mfg.get('front_bearing','?')}")
        log.info(f"   Rear  bearing : {mfg.get('rear_bearing','?')}")
        log.info(f"   Bore (catalog): {mfg.get('bore_catalog_mm','?')} mm")
        log.info(f"   K_radial      : {mfg.get('K_radial_catalog',0):,.0f} N/mm")

        if not self.config["no_plots"]:
            log.info("   Generating optimizer plots...")
            M["robust_optimizer"].plot_optimizer_results(
                result, self.ds, save_dir=str(self.plot_dir),
            )

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 5 — Selective Assembly
    # ─────────────────────────────────────────────────────────────────────────
    def stage_selective_assembly(self, var_dict: dict) -> tuple:
        log.info("\n── STAGE 5: SELECTIVE ASSEMBLY ─────────────────────────────────")
        M = self.m
        sa_analyser = M["selective_assembly"].SelectiveAssemblyAnalyser(n_parts=1000)
        cost_model  = M["selective_assembly"].SpindleCostModel()

        sa_results = sa_analyser.analyse_all(var_dict, n_bins=5)
        costs      = cost_model.total_cost(var_dict, sa_results)

        log.info("   SA Results (5 bins):")
        for r in sa_results:
            log.info(f"     {r.interface_name:<30} ×{r.improvement_ratio:.1f}  "
                     f"yield={r.match_yield*100:.0f}%")
        log.info(f"   Total cost: ${costs.get('total_usd',0):.2f}")

        if not self.config["no_plots"]:
            M["selective_assembly"].plot_selective_assembly(
                sa_results, costs, var_dict,
                n_bins_list=[3,5,7,10], analyser=sa_analyser,
                save_dir=str(self.plot_dir),
            )

        return sa_results, costs

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 6 — Bearing Performance
    # ─────────────────────────────────────────────────────────────────────────
    def stage_bearing_performance(self, x_opt: np.ndarray, n_rpm: float) -> object:
        log.info("\n── STAGE 6: BEARING PERFORMANCE ────────────────────────────────")
        M    = self.m
        calc = M["bearing_performance"].BearingPerformanceCalculator(
            self.ds, self.arr,
            n_nom_rpm=n_rpm, n_max_rpm=6000,
            l10_target_hours=20000,
        )
        state = calc.evaluate(x_opt, n_rpm=n_rpm)
        con   = calc.check_constraints(state)

        log.info(f"   System L10 : {state.L10_system_hours:,.0f} h")
        log.info(f"   Constraints: {'ALL OK' if con.all_satisfied else str(con.violated_names())}")

        # ANSYS spring table
        springs = calc.stiffness_for_ansys(state)
        log.info("   ANSYS COMBIN14 spring table:")
        for r in springs:
            log.info(f"     z={r['z_mm']:>7.1f}mm  K_r={r['K_radial']:>8.0f}  "
                     f"K_a={r['K_axial']:>8.0f}  {r['role']}")

        if not self.config["no_plots"]:
            M["bearing_performance"].plot_bearing_performance(
                calc, self.ds, x_opt,
                speeds=[1000,2000,3000,4000,5000,6000],
                save_dir=str(self.plot_dir),
            )

        return state

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 7 — Runout
    # ─────────────────────────────────────────────────────────────────────────
    def stage_runout(self, var_dict: dict, delta_nose_um: float, n_rpm: float,
                     z_f: float = None, z_r: float = None) -> object:
        log.info("\n── STAGE 7: SHAFT RUNOUT ───────────────────────────────────────")
        M = self.m
        if z_f is None or z_r is None:
            z_f, z_r = M["shaft_runout"].get_bearing_positions_from_design(var_dict, self.arr)
        Fr_N = float(np.sqrt(var_dict["Ft"]**2 + var_dict["Fr"]**2))

        analyser = M["shaft_runout"].ShaftRunoutAnalyser(
            precision_class="P5", straightness_grade="precision",
            tir_geometric_limit=10.0, tir_loaded_limit=20.0,  # Option C: Class B limit
            preload_class="MA", lubrication="grease",
        )
        bd  = analyser.analyse(var_dict, z_f, z_r,
                               delta_nose_ansys_um=delta_nose_um,
                               Fr_N=Fr_N, n_rpm=n_rpm)
        con = analyser.check_constraints(bd)

        log.info(f"   TIR (RSS)  : {bd.TIR_rss_um:.2f} μm")
        log.info(f"   TIR (linear): {bd.TIR_linear_um:.2f} μm")
        log.info(f"   Dominant source: "
                 f"{max(bd.sources_dict, key=bd.sources_dict.get)}")
        log.info(f"   Constraints: {'OK' if con.all_satisfied else 'VIOLATED'}")

        if not self.config["no_plots"]:
            M["shaft_runout"].plot_runout_breakdown(
                bd, save_path=str(self.plot_dir / "09a_runout_breakdown.png"))
            M["shaft_runout"].plot_runout_vs_speed(
                var_dict, z_f, z_r, analyser,
                delta_nose_um=delta_nose_um, Fr_N=Fr_N,
                save_path=str(self.plot_dir / "09b_runout_vs_speed.png"))
            M["shaft_runout"].plot_tir_sensitivity(
                var_dict, z_f, z_r, analyser,
                n_rpm=n_rpm, delta_nose=delta_nose_um, Fr_N=Fr_N,
                save_path=str(self.plot_dir / "09c_tir_sensitivity.png"))

        return bd

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 8 — Eccentricity
    # ─────────────────────────────────────────────────────────────────────────
    def stage_eccentricity(self, var_dict: dict, n_rpm: float,
                           bearing_span_mm: float) -> object:
        log.info("\n── STAGE 8: ROTOR ECCENTRICITY ─────────────────────────────────")
        M   = self.m
        ecc = M["rotor_eccentricity"].RotorEccentricityAnalyser(
            balance_grade="G2.5", bore_offset_mm=0.005,
            F_imbal_limit_N=50.0, bearing_span_mm=bearing_span_mm,
        )
        result = ecc.analyse(var_dict, n_rpm=n_rpm,
                              bearing_span_mm=bearing_span_mm)
        con    = ecc.check_constraints(result)

        log.info(f"   e_static : {result.e_static_um:.2f} μm")
        log.info(f"   U_static : {result.U_static_gmm:.1f} g·mm  "
                 f"(allow {result.U_allow_gmm:.1f})")
        log.info(f"   F_imbal  : {result.F_imbalance_N:.3f} N")
        log.info(f"   Couple C : {result.couple_gmm2:.1f} g·mm²")
        log.info(f"   Constraints: {'OK' if con.all_satisfied else 'VIOLATED'}")

        if not self.config["no_plots"]:
            M["rotor_eccentricity"].plot_eccentricity(
                ecc, var_dict,
                speeds=list(range(500, 6500, 500)),
                save_dir=str(self.plot_dir),
            )

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 9 — Inverse Design
    # ─────────────────────────────────────────────────────────────────────────
    def stage_inverse_design(self, df: pd.DataFrame) -> object:
        log.info("\n── STAGE 9: INVERSE DESIGN ─────────────────────────────────────")
        M = self.m
        out_cols = [c for c in [
            "static_max_deflection_um", "static_max_vonmises_MPa", "freq_mode1_Hz"
        ] if c in df.columns]
        var_cols = [c for c in df.columns if c.startswith("var_")]

        eng = M["inverse_design"].InverseDesignEngine(self.ds, use_tensorflow=False)
        metrics = eng.train(
            df[var_cols].values, df[out_cols].values,
            performance_names=out_cols, verbose=False,
        )
        pkl_path = self.out_dir / "inverse_engine.pkl"
        eng.save(str(pkl_path))
        log.info(f"   val R² = {metrics.get('val_r2',0):.4f}  → {pkl_path}")

        # Test with a target
        target = {
            "static_max_deflection_um": 10.0,
            "static_max_vonmises_MPa": 300.0,
            "freq_mode1_Hz": 600.0,
        }
        target_filtered = {k: v for k, v in target.items() if k in out_cols}
        if target_filtered:
            x_pred = eng.predict_design(target_filtered)
            v_pred = self.ds.decode_vector(x_pred)
            log.info(f"   Inverse target: {target_filtered}")
            log.info(f"   Predicted R2={v_pred.get('R2',0):.2f}mm  L1={v_pred.get('L1',0):.1f}mm")

            if not self.config["no_plots"]:
                M["inverse_design"].plot_inverse_design(
                    eng, target_filtered, save_dir=str(self.plot_dir))

        return eng

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 11 — Tolerance Optimization  (Module 12)
    # ─────────────────────────────────────────────────────────────────────────
    def stage_tolerance_optimization(
        self,
        x_opt:        np.ndarray,
        delta_nose_um: float,
        L10_base:     float,
        z_f:          float,
        z_r:          float,
    ) -> object:
        log.info("\n── STAGE 11: TOLERANCE OPTIMISATION (Module 12) ────────────────")
        M    = self.m
        if "tolerance_optimizer" not in M:
            log.warning("   tolerance_optimizer not loaded — skipping")
            return None

        v    = self.ds.decode_vector(x_opt)
        span = max(z_r - z_f, 1.0)

        ev = M["tolerance_optimizer"].ToleranceEvaluator(
            d_journal_mm  = v["R2"] * 2,
            d_bore_mm     = v["ri"] * 2,
            R_outer_mm    = v["R2"],
            L_overhang_mm = z_f,
            L_span_mm     = span,
            delta_nose_um = delta_nose_um,
            L10_base_hours= L10_base,
        )

        opt    = M["tolerance_optimizer"].ToleranceOptimizer(
            ev, tir_limit_um=12.0, l10_loss_max_pct=20.0, n_weights=15,
        )
        pareto = opt.run(verbose=True)
        best   = opt.best_by_priority(pareto, priority="cost")

        M["tolerance_optimizer"].print_tolerance_report(pareto, best, v)

        # ── Deviation optimizer (within-grade band positioning) ────────────
        amp     = 1.0 + z_f / max(z_r - z_f, 1.0)
        dev_opt = M["tolerance_optimizer"].DeviationOptimizer(
            amp_factor=amp, L10_base_hours=L10_base)
        dev_result = dev_opt.optimise_all(best, v, verbose=True)
        M["tolerance_optimizer"].print_full_tolerance_spec(best, dev_result, v)

        if not self.config["no_plots"]:
            M["tolerance_optimizer"].plot_tolerance_pareto(
                pareto, best, save_dir=str(self.plot_dir),
            )

        # Save tolerance recommendation to CSV
        import csv
        tol_path = self.out_dir / "optimal_tolerances.csv"
        sp_bal   = dev_result.get("journal_knee")
        sp_bore  = dev_result.get("inner_bore")
        sp_pf    = dev_result.get("pos_tol_front")
        sp_pr    = dev_result.get("pos_tol_rear")
        with open(tol_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["feature","it_grade","it_value_um","upper_dev_um",
                        "lower_dev_um","iso_fit","clearance_um","tir_fit_um",
                        "assembly","standard"])
            if sp_bal:
                w.writerow(["journals_R1R2R3R4", sp_bal.it_grade,
                            round(sp_bal.it_value_um,1),
                            round(sp_bal.upper_dev_um,1), round(sp_bal.lower_dev_um,1),
                            sp_bal.iso_fit_equiv, round(sp_bal.clearance_um,1),
                            round(sp_bal.tir_from_fit_um,3),
                            sp_bal.assembly_note, "ISO 286-1"])
            if sp_bore:
                w.writerow(["inner_bore_ri", sp_bore.it_grade,
                            round(sp_bore.it_value_um,1),
                            round(sp_bore.upper_dev_um,1), 0,
                            sp_bore.iso_fit_equiv, round(sp_bore.clearance_um,1),
                            0, sp_bore.assembly_note, "ISO 286-1"])
            if sp_pf:
                w.writerow(["pos_tol_front", "ISO 1101",
                            round(sp_pf.it_value_um,1), "", "",
                            sp_pf.iso_fit_equiv, "", "", "circular zone", "ISO 1101:2017"])
            if sp_pr:
                w.writerow(["pos_tol_rear", "ISO 1101",
                            round(sp_pr.it_value_um,1), "", "",
                            sp_pr.iso_fit_equiv, "", "", "circular zone", "ISO 1101:2017"])
        log.info(f"   Tolerance spec saved → {tol_path}")

        return pareto, best
    def stage_final_report(self, x_opt: np.ndarray, bearing_state: object,
                           runout_bd: object, ecc_result: object,
                           fea_row, n_rpm: float) -> None:
        log.info("\n── STAGE 10: FINAL ENGINEERING REPORT ──────────────────────────")
        M       = self.m
        builder = M["final_report"].FinalReportBuilder(
            FoS_min=2.0, delta_max_um=20.0, tir_limit_um=20.0, L10_target_hours=20000,  # Option C
        )

        # Print to console
        builder.print_report(x_opt, self.ds, bearing_state, runout_bd,
                              ecc_result, fea_row, n_rpm=n_rpm,
                              design_name="Optimised Spindle Design")

        # Save report text
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            builder.print_report(x_opt, self.ds, bearing_state, runout_bd,
                                  ecc_result, fea_row, n_rpm=n_rpm,
                                  design_name="Optimised Spindle Design")
        rpt_path = self.out_dir / "optimal_design_report.txt"
        rpt_path.write_text(buf.getvalue())
        log.info(f"   Report saved → {rpt_path}")

        if not self.config["no_plots"]:
            builder.generate_plots(x_opt, self.ds, bearing_state, runout_bd,
                                    ecc_result, fea_row, n_rpm=n_rpm,
                                    save_dir=str(self.plot_dir))

    # ─────────────────────────────────────────────────────────────────────────
    # Full workflow
    # ─────────────────────────────────────────────────────────────────────────
    def run_full(self) -> None:
        n_rpm = float(self.config.get("n_rpm", 4000))

        # 1. Design space plot
        if not self.config["no_plots"]:
            log.info("   Generating design-space plots...")
            self.m["design_variables"].plot_design_space(
                self.ds, save_dir=str(self.plot_dir))

        # 2–4. DoE → FEA → Surrogate → Optimization
        X   = self.stage_lhs()
        df  = self.stage_fea(X)
        surr = self.stage_surrogate(df)
        opt_result = self.stage_optimize(surr)

        # Resolve best design
        x_opt = opt_result.best_robust_design
        var_d = self.ds.decode_vector(x_opt)
        cat   = self.ds.resolve_to_catalog(x_opt, n_rpm)

        # Compute bearing positions once — used by stages 7, 8, 9, 11
        z_f, z_r = self.m["shaft_runout"].get_bearing_positions_from_design(
            var_d, self.arr)
        span = max(z_r - z_f, 1.0)

        # 5. Selective Assembly
        self.stage_selective_assembly(var_d)

        # 6. Bearing Performance
        bearing_state = self.stage_bearing_performance(x_opt, n_rpm)

        # 7. FEA single-point for reporting
        M   = self.m
        fea_df = M["fea_pool_runner"].FEAPoolRunner(
            np.array([x_opt]), self.ds, dry_run=True).execute_batch()
        fea_row = fea_df.iloc[0]
        delta_nose_um = float(fea_row["static_max_deflection_um"])

        # 8. Runout (uses precomputed z_f, z_r)
        runout_bd = self.stage_runout(var_d, delta_nose_um, n_rpm, z_f, z_r)

        # 9. Eccentricity (uses precomputed span)
        ecc_result = self.stage_eccentricity(var_d, n_rpm, span)

        # 10. Inverse design
        self.stage_inverse_design(df)

        # 11. Tolerance Optimization (Option C — Module 12)
        self.stage_tolerance_optimization(
            x_opt, delta_nose_um, bearing_state.L10_system_hours, z_f, z_r,
        )

        # 12. Final report
        self.stage_final_report(x_opt, bearing_state, runout_bd,
                                 ecc_result, fea_row, n_rpm)

        # Summary
        plots = sorted([f for f in os.listdir(self.plot_dir) if f.endswith(".png")])
        log.info("\n" + "="*72)
        log.info(f"  ✅ WORKFLOW COMPLETE")
        log.info(f"  Output dir  : {self.out_dir}")
        log.info(f"  Plots dir   : {self.plot_dir}")
        log.info(f"  Plots generated: {len(plots)}")
        for p in plots:
            log.info(f"    {p}")
        log.info("="*72)

    # ─────────────────────────────────────────────────────────────────────────
    # ml_only — load existing FEA CSV
    # ─────────────────────────────────────────────────────────────────────────
    def run_ml_only(self) -> None:
        fea_csv = self.config.get("fea_csv")
        if not fea_csv or not Path(fea_csv).exists():
            log.error(f"FEA CSV not found: {fea_csv}")
            return
        df   = pd.read_csv(fea_csv)
        surr = self.stage_surrogate(df)
        self.stage_inverse_design(df)
        self.stage_optimize(surr)

    # ─────────────────────────────────────────────────────────────────────────
    # opt_only — load pre-trained surrogate
    # ─────────────────────────────────────────────────────────────────────────
    def run_opt_only(self) -> None:
        pkl = self.config.get("surrogate_pkl")
        if not pkl or not Path(pkl).exists():
            log.error(f"Surrogate pkl not found: {pkl}")
            return
        surr = self.m["ml_surrogate"].SurrogateModel.load(pkl)
        self.stage_optimize(surr)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="TechPulse Spindle RDO — Master Workflow v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1].split("Output files")[0] if "Usage:" in __doc__ else "",
    )
    p.add_argument("--mode",          choices=["full","ml_only","opt_only"], default="full")
    p.add_argument("--n_samples",     type=int,   default=50)
    p.add_argument("--dry_run",       action="store_true")
    p.add_argument("--surrogate",     choices=["gp","xgb","mlp"], default="gp")
    p.add_argument("--opt_method",    choices=["de","nsga2"],     default="de")
    p.add_argument("--output_dir",    default="rdo_results")
    p.add_argument("--no_plots",      action="store_true")
    p.add_argument("--n_rpm",         type=float, default=4000.0)
    p.add_argument("--fea_csv",       default=None)
    p.add_argument("--surrogate_pkl", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    config = vars(args)
    config["dry_run"] = args.dry_run   # ensure bool

    orch = RDOMasterOrchestrator(config)

    if   args.mode == "full":     orch.run_full()
    elif args.mode == "ml_only":  orch.run_ml_only()
    elif args.mode == "opt_only": orch.run_opt_only()


if __name__ == "__main__":
    main()
