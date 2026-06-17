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
        "design_variables":    "design_variables",
        "lhs_sampler":         "lhs_sampler",
        "fea_pool_runner":     "fea_pool_runner",
        "ml_surrogate":        "ml_surrogate",
        "selective_assembly":  "selective_assembly",   # must be before robust_optimizer
        "robust_optimizer":    "robust_optimizer",
        "inverse_design":      "inverse_design",
        "bearing_performance": "bearing_performance",
        "shaft_runout":        "shaft_runout",
        "rotor_eccentricity":  "rotor_eccentricity",
        "final_report":        "final_report",
        "tolerance_optimizer": "tolerance_optimizer",
        "reliability_index":   "reliability_index",
        "topsis_selector":     "topsis_selector",
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

        X_all = df[var_cols].values
        y_all = df[out_cols].values

        # ── FIX: proper train/test split — NEVER evaluate on training data ──
        # R²=1.0 on training data is meaningless (GP interpolates exactly).
        # Always use a held-out test set (20%) for honest evaluation.
        from sklearn.model_selection import train_test_split
        n_test = max(int(len(X_all) * 0.20), 3)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all, y_all, test_size=n_test,
            random_state=42, shuffle=True,
        )
        log.info(f"   Train: {len(X_tr)} samples  |  Hold-out test: {len(X_te)} samples")

        surr = M["ml_surrogate"].SurrogateModel(
            model_type=self.config.get("surrogate", "gp")
        )
        surr.train(X_tr, y_tr, output_names=out_cols, verbose=True)

        pkl_path = self.out_dir / f"surrogate_{surr.model_type}.pkl"
        surr.save(str(pkl_path))
        log.info(f"✅ Surrogate trained → {pkl_path}")

        # ── Honest evaluation on HELD-OUT test set ───────────────────────
        metrics = surr.evaluate(X_te, y_te)
        log.info(f"   Hold-out test R² per output (n={len(X_te)}):")
        low_r2_outputs = []
        for _, row in metrics.iterrows():
            r2    = float(row["R2"])
            flag  = "⚠️  LOW" if r2 < 0.70 else "✅"
            log.info(f"     {row['output']:<35} R²={r2:.4f}  {flag}")
            if r2 < 0.70:
                low_r2_outputs.append(row["output"])

        if low_r2_outputs:
            log.warning(
                f"   ⚠️  {len(low_r2_outputs)} output(s) have R² < 0.70 on held-out test: "
                f"{low_r2_outputs}\n"
                f"   → Consider: more samples (--n_samples 200+), "
                f"model_type=xgb, or feature engineering."
            )

        if not self.config["no_plots"]:
            log.info("   Generating surrogate plots...")
            M["ml_surrogate"].plot_surrogate_performance(
                surr, X_tr, y_tr, X_te, y_te,
                save_dir=str(self.plot_dir),
            )

        # ── Active Learning / Adaptive DoE (Module 5) ─────────────────────
        if self.config.get("active_learning", False):
            log.info("\n── ACTIVE LEARNING (Adaptive DoE) ──────────────────────────")
            if surr.model_type not in ("gp", "ensemble"):
                log.warning(f"   model_type='{surr.model_type}' has no uncertainty "
                            f"estimate — Active Learning needs 'gp' or 'ensemble'. "
                            f"Skipping.")
            else:
                al_rounds   = int(self.config.get("al_rounds", 2))
                al_n_select = int(self.config.get("al_n_select", 8))
                al_pool     = int(self.config.get("al_pool_size", 300))

                al = M["ml_surrogate"].ActiveLearner(self.ds, surr)
                # Record round-0 (initial) R² for the convergence plot
                m0 = surr.evaluate(X_te, y_te)
                al.history.append({
                    "n_train": len(X_tr),
                    **{f"R2_{row['output']}": row["R2"] for _, row in m0.iterrows()},
                })

                X_cur, y_cur = X_tr.copy(), y_tr.copy()
                for r in range(al_rounds):
                    log.info(f"   Round {r+1}/{al_rounds}: selecting "
                             f"{al_n_select} candidates from pool of {al_pool}...")
                    X_cur, y_cur, surr, r2 = al.run_round(
                        X_cur, y_cur, out_cols, M["fea_pool_runner"].FEAPoolRunner,
                        n_pool=al_pool, n_select=al_n_select, seed=100+r,
                        X_test=X_te, y_test=y_te,
                    )
                    r2_str = "  ".join(f"{k}={v:.3f}" for k,v in r2.items())
                    log.info(f"     n_train={len(X_cur)}  R²: {r2_str}")

                # Re-save the improved surrogate (overwrites initial pkl)
                pkl_path = self.out_dir / f"surrogate_{surr.model_type}.pkl"
                surr.save(str(pkl_path))
                log.info(f"✅ Active Learning complete → {pkl_path} "
                         f"({len(X_cur)} total samples, "
                         f"+{len(X_cur)-len(X_tr)} from AL)")

                if not self.config["no_plots"]:
                    M["ml_surrogate"].plot_active_learning_convergence(
                        al.history,
                        save_path=str(self.plot_dir / "04c_active_learning_convergence.png"),
                    )

        return surr

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 4 — Robust Optimization
    # ─────────────────────────────────────────────────────────────────────────
    def stage_optimize(self, surr: object) -> object:
        log.info("\n── STAGE 4: ROBUST OPTIMIZATION ───────────────────────────────")
        M      = self.m
        n_rpm  = float(self.config.get("n_rpm", 4000))
        method = self.config.get("opt_method", "de")

        # TOPSIS weights (Module 14) — parse "--topsis_weights" CSV string
        topsis_weights = None
        tw_str = self.config.get("topsis_weights")
        if tw_str:
            try:
                topsis_weights = np.array([float(v) for v in tw_str.split(",")])
                if len(topsis_weights) != 8:
                    log.warning(f"   --topsis_weights expected 8 values, "
                                f"got {len(topsis_weights)} — using equal weights")
                    topsis_weights = None
                else:
                    log.info(f"   TOPSIS weights: {topsis_weights}")
            except ValueError:
                log.warning(f"   Could not parse --topsis_weights='{tw_str}' "
                            f"— using equal weights")

        mfr = self.config.get("manufacturer", None)
        if mfr:
            log.info(f"   Bearing manufacturer locked to: {mfr}")

        opt = M["robust_optimizer"].RobustOptimizer(
            surr, self.ds,
            n_mc_inner       = 20,
            n_sa_bins        = 5,
            sa_n_parts       = 500,
            noise_force_cv   = self.config.get("noise_force_cv",   0.10),
            noise_temp_max_C = self.config.get("noise_temp_max_c", 60.0),
            n_rpm            = n_rpm,
            chatter_Ks           = self.config.get("chatter_Ks",         2500.0),
            chatter_zeta         = self.config.get("chatter_zeta",       0.03),
            chatter_b_required   = self.config.get("chatter_b_required", 2.0),
            bearing_manufacturer = mfr,
        )

        if method == "nsga3":
            log.info("Running NSGA-III (8-obj, pop=120 × 80 gen — recommended)...")
            result = opt.optimize_nsga3(pop_size=120, n_gen=80, seed=42,
                                         topsis_weights=topsis_weights)
        elif method == "nsga2":
            log.info("Running NSGA-II (legacy, pop=100 × 50 gen)...")
            result = opt.optimize_nsga2(pop_size=100, n_gen=50, seed=42,
                                         topsis_weights=topsis_weights)
        else:
            log.info("Running DE sweep (21 weight vectors × 200 iter)...")
            result = opt.optimize_de(maxiter=200, seed=42,
                                      topsis_weights=topsis_weights)

        # ── Catalog-resolved report ───────────────────────────────────────
        rpt = opt.report_best(result, n_rpm=n_rpm)
        cat = rpt["catalog"]
        mfg = cat["manufacturable_design"]

        # Save correct CSV (catalog values, not raw optimizer)
        csv_path = self.out_dir / "optimal_design.csv"
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(mfg.keys()))
            writer.writeheader(); writer.writerow(mfg)
        log.info(f"✅ Optimal design saved → {csv_path}")
        log.info(f"   Front bearing : {mfg.get('front_bearing','?')}")
        log.info(f"   Rear  bearing : {mfg.get('rear_bearing','?')}")
        log.info(f"   Bore (catalog): {mfg.get('bore_catalog_mm','?')} mm")
        log.info(f"   K_radial      : {mfg.get('K_radial_catalog',0):,.0f} N/mm")

        # ── Selected design metrics (TOPSIS-chosen point) ──────────────────
        if result.topsis_result is not None:
            best_idx = result.topsis_result.best_idx
            topsis_c = result.topsis_result.scores[best_idx]
        else:
            best_idx = 0
            topsis_c = float("nan")
        best_F = result.pareto_front_F[best_idx]

        if len(best_F) >= 6:
            L10_h = -best_F[4] * opt.L10_target
            log.info(f"   L10 (est.)    : {L10_h:,.0f} h")
            log.info(f"   Speed ratio   : {best_F[5]:.3f}  "
                     f"({'SAFE ✅' if best_F[5]<0.75 else 'WARN ⚠️ >0.75'})")
        if len(best_F) >= 7:
            beta_sys = -best_F[6]
            log.info(f"   β_system      : {beta_sys:.3f}  "
                     f"({'RELIABLE ✅' if beta_sys>=3.0 else 'CHECK ⚠️'})")
        if len(best_F) >= 8:
            chatter_ratio = best_F[7]
            b_lim = opt.chatter_b_required / max(chatter_ratio, 1e-9)
            log.info(f"   Chatter ratio : {chatter_ratio:.3f}  "
                     f"(b_lim={b_lim:.2f}mm, b_req={opt.chatter_b_required:.1f}mm)  "
                     f"{'STABLE ✅' if chatter_ratio<=1.0 else 'UNSTABLE ❌'}")
        log.info(f"   TOPSIS score  : C={topsis_c:.4f}  "
                 f"(Module 14, knee-point of {len(result.pareto_front_F)} Pareto points)")

        if not self.config["no_plots"]:
            log.info("   Generating optimizer plots...")
            M["robust_optimizer"].plot_optimizer_results(
                result, self.ds, save_dir=str(self.plot_dir),
            )
            # TOPSIS ranking plot (Module 14)
            if result.topsis_result is not None:
                try:
                    M["topsis_selector"].plot_topsis_ranking(
                        result.pareto_front_F, result.topsis_result,
                        opt.objective_labels,
                        save_path=str(self.plot_dir / "14a_topsis_ranking.png"),
                    )
                except Exception as e:
                    log.warning(f"   TOPSIS plot skipped: {e}")

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
            # ── SA comparison table (σ and spread before/after) ───────────
            log.info("   Generating SA comparison table...")
            self._plot_sa_comparison_table(sa_results, var_dict)

        return sa_results, costs

    def _plot_sa_comparison_table(self, sa_results, var_dict: dict) -> None:
        """Generate the SA before/after comparison table plot."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
        import os

        NAVY="#0d1b2a"; TEAL="#00b4d8"; CORAL="#e63946"
        GOLD="#ffd166"; MINT="#06d6a0"; GRAY="#8d99ae"
        plt.rcParams.update({
            "figure.facecolor": NAVY, "axes.facecolor": "#112233",
            "axes.edgecolor": GRAY, "axes.labelcolor": "white",
            "xtick.color": GRAY, "ytick.color": GRAY,
            "text.color": "white", "grid.color": "#2d4060",
            "grid.alpha": 0.4, "font.size": 9,
        })

        from matplotlib.gridspec import GridSpec

        en_names   = [r.interface_name.replace("_", " / ") for r in sa_results]
        std_before = np.array([r.std_gap_no_sa_um for r in sa_results])
        std_after  = np.array([r.std_gap_um       for r in sa_results])
        spread_b   = std_before * 6
        spread_a   = std_after  * 6
        improve    = np.array([r.improvement_ratio for r in sa_results])
        yields     = np.array([r.match_yield * 100 for r in sa_results])
        means      = np.array([r.mean_gap_um       for r in sa_results])
        reduc_pct  = (1 - std_after / std_before) * 100

        fig = plt.figure(figsize=(16, 14), facecolor=NAVY)
        fig.suptitle(
            "جدول مقارنة: الانحراف المعياري وتشتت التجاوزات عند واجهات المغزل الحرجة\n"
            "قبل وبعد تطبيق استراتيجية التجميع الانتقائي — 5 فئات",
            color="white", fontsize=13, fontweight="bold", y=0.98,
        )
        gs = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.38,
                      top=0.91, bottom=0.06, left=0.07, right=0.97)
        n  = len(sa_results)
        x  = np.arange(n); w = 0.32

        # ── σ bar chart ───────────────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, 0]); ax0.set_facecolor("#112233")
        b1 = ax0.bar(x - w/2, std_before, w, color=CORAL, edgecolor=NAVY, linewidth=0.5,
                     label="قبل SA (Before)")
        b2 = ax0.bar(x + w/2, std_after,  w, color=TEAL,  edgecolor=NAVY, linewidth=0.5,
                     label="بعد SA (After 5 bins)")
        for bar, val in zip(list(b1)+list(b2), list(std_before)+list(std_after)):
            ax0.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.08,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=8, color="white")
        ax0.set_xticks(x); ax0.set_xticklabels([f"I{i+1}" for i in range(n)])
        ax0.set_ylabel("σ [μm]")
        ax0.set_title("الانحراف المعياري σ — قبل / بعد", fontsize=10, pad=6)
        ax0.legend(fontsize=8); ax0.grid(axis="y", alpha=0.35)

        # ── 6σ spread bar chart ───────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 1]); ax1.set_facecolor("#112233")
        b3 = ax1.bar(x - w/2, spread_b, w, color=CORAL, edgecolor=NAVY, linewidth=0.5)
        b4 = ax1.bar(x + w/2, spread_a, w, color=TEAL,  edgecolor=NAVY, linewidth=0.5)
        for bar, val in zip(list(b3)+list(b4), list(spread_b)+list(spread_a)):
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=8, color="white")
        ax1.set_xticks(x); ax1.set_xticklabels([f"I{i+1}" for i in range(n)])
        ax1.set_ylabel("6σ Spread [μm]")
        ax1.set_title("تشتت التجاوزات (6σ) — قبل / بعد", fontsize=10, pad=6)
        ax1.legend(handles=[mpatches.Patch(color=CORAL, label="قبل SA"),
                             mpatches.Patch(color=TEAL,  label="بعد SA (5 bins)")],
                   fontsize=8)
        ax1.grid(axis="y", alpha=0.35)

        # ── improvement horizontal bar ────────────────────────────────────
        ax2 = fig.add_subplot(gs[1, 0]); ax2.set_facecolor("#112233")
        colours_imp = [MINT if v >= 3 else GOLD for v in improve]
        bars = ax2.barh(np.arange(n), improve, color=colours_imp,
                        edgecolor=NAVY, linewidth=0.5, height=0.45)
        for bar, val in zip(bars, improve):
            ax2.text(bar.get_width()+0.05, bar.get_y()+bar.get_height()/2,
                     f"×{val:.2f}", va="center", fontsize=9, color="white",
                     fontweight="bold")
        ax2.axvline(1.0, color=GRAY, lw=0.9, linestyle="--", alpha=0.6)
        ax2.set_yticks(np.arange(n))
        ax2.set_yticklabels([f"I{i+1}" for i in range(n)])
        ax2.set_xlabel("نسبة التحسين ×")
        ax2.set_title("نسبة تحسين التشتت (σ_before / σ_after)", fontsize=10, pad=6)
        ax2.grid(axis="x", alpha=0.35)

        # ── yield + reduction dual axis ───────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 1]); ax3.set_facecolor("#112233")
        ax3b = ax3.twinx(); ax3b.set_facecolor("#112233")
        b5 = ax3.bar(x - w/2,  yields,    w, color=GOLD, edgecolor=NAVY, linewidth=0.5,
                     label="Yield %")
        b6 = ax3b.bar(x + w/2, reduc_pct, w, color=MINT, edgecolor=NAVY, linewidth=0.5,
                      label="تخفيض σ %")
        for bar, val in zip(b5, yields):
            ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=8, color="white")
        for bar, val in zip(b6, reduc_pct):
            ax3b.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                      f"{val:.0f}%", ha="center", va="bottom", fontsize=8, color=MINT)
        ax3.set_xticks(x); ax3.set_xticklabels([f"I{i+1}" for i in range(n)])
        ax3.set_ylabel("Yield [%]", color=GOLD); ax3b.set_ylabel("تخفيض σ [%]", color=MINT)
        ax3.tick_params(axis="y", colors=GOLD); ax3b.tick_params(axis="y", colors=MINT)
        ax3.set_ylim(85, 102); ax3b.set_ylim(0, 100)
        ax3.set_title("نسبة التطابق + نسبة تخفيض σ", fontsize=10, pad=6)
        ax3.legend(handles=[mpatches.Patch(color=GOLD, label="Yield %"),
                             mpatches.Patch(color=MINT, label="σ reduction %")],
                   fontsize=8)

        # ── summary table ─────────────────────────────────────────────────
        ax4 = fig.add_subplot(gs[2, :]); ax4.axis("off")
        cols = ["الواجهة (Interface)", "μ_gap [μm]",
                "σ قبل SA [μm]", "σ بعد SA [μm]",
                "6σ قبل [μm]", "6σ بعد [μm]",
                "تخفيض σ [%]", "تحسين ×", "تطابق %", "تصنيف"]
        rows = []
        for i, r in enumerate(sa_results):
            grade = ("ممتاز ✅" if improve[i] >= 4 else
                     "جيد جداً ✅" if improve[i] >= 2.5 else "مقبول ⚠️")
            rows.append([en_names[i], f"{means[i]:.2f}",
                          f"{std_before[i]:.3f}", f"{std_after[i]:.3f}",
                          f"{spread_b[i]:.2f}", f"{spread_a[i]:.2f}",
                          f"{reduc_pct[i]:.1f}%", f"×{improve[i]:.2f}",
                          f"{yields[i]:.1f}%", grade])
        tbl = ax4.table(cellText=rows, colLabels=cols,
                        cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
        row_colours = ["#1a2c3d", "#162436"]
        for j in range(len(cols)):
            cell = tbl[0, j]
            cell.set_facecolor(TEAL)
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_edgecolor(NAVY)
        for i, row in enumerate(rows):
            for j in range(len(cols)):
                cell = tbl[i+1, j]
                cell.set_facecolor(row_colours[i % 2])
                cell.set_edgecolor(NAVY)
                cell.set_text_props(color="white")
                if j == 7:   # improvement column
                    cell.set_facecolor(MINT if improve[i] >= 4 else
                                       GOLD if improve[i] >= 2.5 else CORAL)
                    cell.set_text_props(color=NAVY, fontweight="bold")
                if j == 6:
                    cell.set_text_props(color=MINT, fontweight="bold")
        ax4.set_title("جدول ملخص المقارنة الكاملة — الواجهات الحرجة",
                      color="white", fontsize=10, pad=8)
        imap = " | ".join([f"I{i+1}={nm}" for i, nm in enumerate(en_names)])
        fig.text(0.5, 0.025, imap, ha="center", color=GRAY, fontsize=8)

        p = str(self.plot_dir / "07_sa_comparison_table.png")
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=NAVY)
        plt.close(fig)
        log.info(f"   Saved → {p}")

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 6 — Bearing Performance
    # ─────────────────────────────────────────────────────────────────────────
    def stage_manufacturer_comparison(
        self,
        bore_radius_mm: float,
        n_rpm:          float = 4000.0,
    ) -> None:
        """Print side-by-side comparison of all 6 manufacturers for the optimal bore."""
        try:
            from bearing_catalog import print_manufacturer_comparison
            log.info("\n── BEARING MANUFACTURER COMPARISON ──────────────────────────────")
            log.info(f"   Bore radius = {bore_radius_mm:.1f}mm → bore Ø{bore_radius_mm*2:.0f}mm")
            print_manufacturer_comparison(bore_radius_mm * 2, "ACBB", n_rpm)
            print_manufacturer_comparison(bore_radius_mm * 2, "CRB",  n_rpm)
        except ImportError as e:
            log.warning(f"bearing_catalog not found: {e}")

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
    def stage_reliability(
        self, x_opt: np.ndarray, surr: object,
        delta_nose_um: float, n_rpm: float,
    ) -> object:
        log.info("\n── RELIABILITY ANALYSIS (β — Module 13) ──────────────────────")
        M = self.m
        if "reliability_index" not in M:
            log.warning("   reliability_index not loaded — skipping"); return None
        ls = M["reliability_index"].default_limit_states(
            delta_max_um=20.0, fos_min=2.0, n_rpm=n_rpm)
        ra = M["reliability_index"].ReliabilityAnalyser(ls)
        sys_rel = ra.compute_from_design(
            x_opt, surr, self.ds, n_mc=300,
            noise_force_cv   = self.config.get("noise_force_cv",   0.10),
            noise_temp_max_C = self.config.get("noise_temp_max_c", 60.0),
            seed=42,
        )
        sys_rel.print_report()
        log.info(f"   β_sys={sys_rel.beta_system:.3f}  "
                 f"P_f={sys_rel.pf_system*100:.4f}%  "
                 f"Weakest: {sys_rel.beta_min_name} β={sys_rel.beta_min:.3f}")
        if sys_rel.beta_system >= 3.0:   log.info("   ✅ RELIABLE")
        elif sys_rel.beta_system >= 2.0: log.warning("   ⚠️  MARGINAL")
        else:                            log.warning("   ❌ UNRELIABLE")
        if not self.config["no_plots"]:
            M["reliability_index"].plot_reliability_gauges(
                sys_rel, str(self.plot_dir/"13a_reliability_gauges.png"),
                design_name=f"Optimal Design n={n_rpm:.0f}RPM")
            M["reliability_index"].plot_beta_vs_samples(
                sys_rel, str(self.plot_dir/"13b_beta_convergence.png"))
        return sys_rel

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
        # sp_bal is a DeviationParetoPoint — use best.it_journal for grade
        with open(tol_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["feature","it_grade","it_value_um","upper_dev_um",
                        "lower_dev_um","iso_fit","clearance_um","tir_fit_um",
                        "assembly","standard"])
            if sp_bal:   # DeviationParetoPoint (knee point)
                it_val = M["tolerance_optimizer"].it_value_um(
                    best.it_journal, v["R2"] * 2)
                # Housing IT grade from best TolerancePoint
                _H_IT = {"IT5":18.0,"IT6":25.0,"IT7":40.0,"IT8":63.0}
                h_grade = getattr(best, "it_housing", "IT6")
                w.writerow(["housing_bore", h_grade,
                            _H_IT.get(h_grade, 25.0), 0, 0,
                            f"H{h_grade[-1]}", 0, 0,
                            "non-rotating outer ring", "ISO 286-1+ISO 15"])
                w.writerow(["journals_R1R2R3R4", best.it_journal,
                            round(it_val, 1),
                            round(sp_bal.upper_dev_um, 1),
                            round(sp_bal.lower_dev_um, 1),
                            sp_bal.iso_fit,
                            round(sp_bal.clearance_um, 1),
                            round(sp_bal.tir_fit_um, 3),
                            sp_bal.assembly_note, "ISO 286-1"])
            if sp_bore:   # OptimalDeviationSpec
                w.writerow(["inner_bore_ri", sp_bore.it_grade,
                            round(sp_bore.it_value_um, 1),
                            round(sp_bore.upper_dev_um, 1), 0,
                            sp_bore.iso_fit_equiv,
                            round(sp_bore.clearance_um, 1),
                            0, sp_bore.assembly_note, "ISO 286-1"])
            if sp_pf:
                w.writerow(["pos_tol_front", "ISO 1101",
                            round(sp_pf.it_value_um, 1), "", "",
                            sp_pf.iso_fit_equiv, "", "", "circular zone", "ISO 1101:2017"])
            if sp_pr:
                w.writerow(["pos_tol_rear", "ISO 1101",
                            round(sp_pr.it_value_um, 1), "", "",
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
            chatter_Ks         = self.config.get("chatter_Ks",         2500.0),
            chatter_zeta       = self.config.get("chatter_zeta",       0.03),
            chatter_b_required = self.config.get("chatter_b_required", 2.0),
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

        # 10. Reliability Index β
        self.stage_reliability(x_opt, surr, delta_nose_um, n_rpm)

        # 10b. Inverse design
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
        description="TechPulse Spindle RDO — Master Workflow v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── Core ──────────────────────────────────────────────────────────────
    p.add_argument("--mode", default="full",
                   choices=["full", "ml_only", "opt_only", "from_surrogate"],
                   help="full=complete pipeline | ml_only=train surrogate only | "
                        "opt_only=optimise only | from_surrogate=skip FEA, resume "
                        "from saved surrogate pkl + fea csv")
    p.add_argument("--n_samples",     type=int,   default=200,
                   help="Number of LHS samples for FEA pool (default 200)")
    p.add_argument("--dry_run",       action="store_true",
                   help="Use fast analytical solver instead of ANSYS")
    p.add_argument("--output_dir",    default="rdo_results",
                   help="Directory for all outputs (default rdo_results)")
    p.add_argument("--no_plots",      action="store_true",
                   help="Skip all matplotlib figure generation")
    p.add_argument("--n_rpm",         type=float, default=4000.0,
                   help="Operating spindle speed [rpm] (default 4000)")

    # ── Noise / robustness ────────────────────────────────────────────────
    p.add_argument("--noise_force_cv",   type=float, default=0.10,
                   help="Coefficient of variation for cutting-force scatter "
                        "(default 0.10 = 10%%)")
    p.add_argument("--noise_temp_max_c", type=float, default=60.0,
                   help="Maximum operating temperature rise ΔT [°C] "
                        "(default 60.0)")

    # ── Surrogate ─────────────────────────────────────────────────────────
    p.add_argument("--surrogate", default="ensemble",
                   choices=["gp", "xgb", "mlp", "ensemble"],
                   help="Surrogate model type (default gp; "
                        "ensemble=GP+GBR recommended with --active_learning)")
    p.add_argument("--fea_csv",       default=None,
                   help="Path to saved FEA results CSV "
                        "(required for --mode from_surrogate)")
    p.add_argument("--surrogate_pkl", default=None,
                   help="Path to saved surrogate .pkl "
                        "(required for --mode from_surrogate)")

    # ── Active Learning ───────────────────────────────────────────────────
    p.add_argument("--active_learning", action="store_true",
                   help="Enable uncertainty-driven Adaptive DoE after initial training")
    p.add_argument("--al_rounds",     type=int, default=2,
                   help="Number of Active Learning rounds (default 2)")
    p.add_argument("--al_n_select",   type=int, default=8,
                   help="New FEA samples added per AL round (default 8)")
    p.add_argument("--al_pool_size",  type=int, default=300,
                   help="LHS candidate pool size per AL round (default 300)")

    # ── Optimisation ──────────────────────────────────────────────────────
    p.add_argument("--opt_method", default="nsga3",
                   choices=["de", "nsga2", "nsga3"],
                   help="Optimisation algorithm "
                        "(nsga3=recommended for 8 objectives | "
                        "de=fast weighted-sum sweep | nsga2=legacy)")
    p.add_argument("--topsis_weights", type=str, default=None,
                   help="Comma-separated 8 floats for TOPSIS weights "
                        "(order: defl,stress,cost,weight,L10,speed,beta,chatter). "
                        "Default: equal weights. Example: '1,1,1,1,2,1,2,1'")

    # ── Chatter stability ─────────────────────────────────────────────────
    p.add_argument("--chatter_Ks",          type=float, default=2500.0,
                   help="Specific cutting force [N/mm²] (default 2500 = steel)")
    p.add_argument("--chatter_zeta",        type=float, default=0.03,
                   help="Structural damping ratio (default 0.03)")
    p.add_argument("--chatter_b_required",  type=float, default=2.0,
                   help="Required stable axial depth of cut [mm] (default 2.0)")

    # ── Bearing manufacturer ──────────────────────────────────────────────
    p.add_argument("--manufacturer", type=str, default=None,
                   choices=["SKF", "FAG", "NSK", "NTN", "JTEKT", "Timken", None],
                   help="Lock bearing selection to one manufacturer. "
                        "Default None = best C_r across all 6 manufacturers.")

    return p.parse_args()


def main():
    args   = parse_args()
    config = vars(args)   # all CLI args already in config via vars()

    orch = RDOMasterOrchestrator(config)

    if   args.mode == "full":            orch.run_full()
    elif args.mode == "ml_only":         orch.run_ml_only()
    elif args.mode == "from_surrogate":  orch.run_from_surrogate()
    elif args.mode == "opt_only":        orch.run_opt_only()
    else:
        print(f"Unknown mode: {args.mode}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()