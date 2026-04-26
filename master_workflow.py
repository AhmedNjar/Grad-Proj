#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  Master Orchestration Script — Complete RDO Workflow
================================================================================

  Purpose:
      End-to-end demonstration of the Robust Design Optimization framework
      for lathe spindle analysis. This script orchestrates all modules:
      
      1. Design Variable Extraction
      2. LHS Sampling (Space-Filling DoE)
      3. FEA Pool Execution (ANSYS or Analytical)
      4. ML Surrogate Training (Gaussian Process)
      5. Robust Optimization (GA + Taguchi)
      6. Inverse Design (DNN)

  Workflow:
      
      ┌────────────────────────────────────────────────────────────┐
      │ 1. Define Design Space & Extract Variables                 │
      └────────────────────────┬───────────────────────────────────┘
                               │
      ┌────────────────────────▼───────────────────────────────────┐
      │ 2. Generate LHS Samples (n=100-500)                        │
      └────────────────────────┬───────────────────────────────────┘
                               │
      ┌────────────────────────▼───────────────────────────────────┐
      │ 3. Run FEA Pool (ANSYS or Analytical Beam Theory)          │
      │    • Extract: Deflection, Stress, Frequencies              │
      └────────────────────────┬───────────────────────────────────┘
                               │
      ┌────────────────────────▼───────────────────────────────────┐
      │ 4. Train ML Surrogate (GP/XGB/MLP)                         │
      │    • Input: Design variables                               │
      │    • Output: FEA results                                   │
      │    • Validation: R² > 0.95 target                          │
      └────────────────────────┬───────────────────────────────────┘
                               │
                       ┌───────┴──────┐
                       │              │
      ┌────────────────▼───┐   ┌──────▼────────────────────────────┐
      │ 5a. Robust         │   │ 5b. Inverse Design                │
      │     Optimization   │   │     • Input: Target Performance   │
      │     (GA + Taguchi) │   │     • Output: Required Dimensions │
      └────────────────────┘   └───────────────────────────────────┘

  Usage:
      python master_workflow.py --mode full --n_samples 200 --dry_run
      
      Modes:
        • "full": Complete workflow (DoE → FEA → ML → Optimization)
        • "ml_only": Train ML from existing FEA results
        • "opt_only": Run optimization with pre-trained surrogate

  Author: Manus AI
  Date: April 2026
================================================================================
"""

from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Import RDO modules
from design_variables import DesignSpace
from lhs_sampler import LHSSampler
from fea_pool_runner import FEAPoolRunner
from ml_surrogate import SurrogateModel
from robust_optimizer import RobustOptimizer
from inverse_design import InverseDesignEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("RDO_Master")


class RDOMasterOrchestrator:
    """
    Master orchestrator for the complete RDO workflow.
    
    Attributes:
        config: Configuration dictionary
        design_space: DesignSpace instance
        output_dir: Path to output directory
        surrogate: Trained ML surrogate model
        results: Dictionary storing all results
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.design_space = DesignSpace()
        self.surrogate = None
        self.inverse_engine = None
        self.results = {}
        
        log.info("=" * 80)
        log.info("  TechPulse Spindle RDO Framework — Master Orchestrator")
        log.info("=" * 80)
        log.info(f"Configuration:")
        for k, v in config.items():
            log.info(f"  • {k}: {v}")
        log.info("=" * 80)
    
    # -------------------------------------------------------------------------
    # STAGE 1: LHS Sampling
    # -------------------------------------------------------------------------
    def run_lhs_sampling(self) -> np.ndarray:
        """
        Generate space-filling LHS samples.
        
        Returns:
            X_samples: (n_samples, n_vars) design matrix
        """
        log.info("\n" + "=" * 80)
        log.info("STAGE 1: LHS SAMPLING")
        log.info("=" * 80)
        
        sampler = LHSSampler(self.design_space)
        
        n_samples = self.config["n_samples"]
        criterion = self.config.get("lhs_criterion", "maximin")
        
        log.info(f"Generating {n_samples} LHS samples (criterion: {criterion})...")
        
        X_samples = sampler.generate_lhs(
            n_samples=n_samples,
            criterion=criterion,
            iterations=100,
            seed=42,
        )
        
        # Save samples
        samples_path = self.output_dir / "lhs_samples.csv"
        sampler.save_samples(X_samples, samples_path, fmt="csv")
        
        self.results["X_samples"] = X_samples
        
        log.info(f"✅ LHS sampling complete: {X_samples.shape}")
        return X_samples
    
    # -------------------------------------------------------------------------
    # STAGE 2: FEA Pool Execution
    # -------------------------------------------------------------------------
    def run_fea_pool(self, X_samples: np.ndarray) -> pd.DataFrame:
        """
        Execute FEA simulations in batch.
        
        Args:
            X_samples: Design samples from LHS
        
        Returns:
            results_df: DataFrame with FEA results
        """
        log.info("\n" + "=" * 80)
        log.info("STAGE 2: FEA POOL EXECUTION")
        log.info("=" * 80)
        
        dry_run = self.config.get("dry_run", False)
        
        runner = FEAPoolRunner(
            X_samples,
            design_space=self.design_space,
            dry_run=dry_run,
            output_dir=self.output_dir / "fea_results",
        )
        
        log.info(f"Executing {len(X_samples)} FEA cases...")
        log.info(f"Mode: {'DRY RUN (Analytical)' if dry_run else 'FULL ANSYS'}")
        
        start_time = time.time()
        results_df = runner.execute_batch(max_failures=10, save_interval=20)
        elapsed = time.time() - start_time
        
        self.results["fea_results"] = results_df
        
        log.info(f"✅ FEA pool complete: {len(results_df)} successful cases")
        log.info(f"   Elapsed time: {elapsed:.1f} seconds ({elapsed/len(results_df):.2f} s/case)")
        
        return results_df
    
    # -------------------------------------------------------------------------
    # STAGE 3: ML Surrogate Training
    # -------------------------------------------------------------------------
    def train_surrogate(self, results_df: pd.DataFrame) -> SurrogateModel:
        """
        Train ML surrogate model.
        
        Args:
            results_df: FEA results from run_fea_pool()
        
        Returns:
            Trained SurrogateModel
        """
        log.info("\n" + "=" * 80)
        log.info("STAGE 3: ML SURROGATE TRAINING")
        log.info("=" * 80)
        
        # Extract design variables (columns starting with "var_")
        var_cols = [c for c in results_df.columns if c.startswith("var_")]
        X_train = results_df[var_cols].values
        
        # Extract FEA outputs
        output_cols = [
            "static_max_deflection_um",
            "static_max_vonmises_MPa",
            "static_factor_of_safety",
            "freq_mode1_Hz",
            "freq_mode2_Hz",
            "freq_mode3_Hz",
        ]
        
        # Filter to only existing columns
        output_cols = [c for c in output_cols if c in results_df.columns]
        y_train = results_df[output_cols]
        
        log.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        log.info(f"Outputs: {output_cols}")
        
        # Initialize and train
        model_type = self.config.get("surrogate_type", "gp")
        surrogate = SurrogateModel(model_type=model_type)
        
        cv_scores = surrogate.train(X_train, y_train, verbose=True)
        
        # Save model
        model_path = self.output_dir / f"surrogate_{model_type}.pkl"
        surrogate.save(model_path)
        
        self.surrogate = surrogate
        self.results["surrogate"] = surrogate
        self.results["cv_scores"] = cv_scores
        
        log.info(f"✅ Surrogate training complete")
        log.info(f"   Cross-validation R² scores:")
        for name, score in cv_scores.items():
            log.info(f"      • {name}: {score:.4f}")
        
        return surrogate
    
    # -------------------------------------------------------------------------
    # STAGE 4: Robust Optimization
    # -------------------------------------------------------------------------
    def run_robust_optimization(self, surrogate: SurrogateModel):
        """
        Run robust optimization with GA + Taguchi.
        
        Args:
            surrogate: Trained surrogate model
        """
        log.info("\n" + "=" * 80)
        log.info("STAGE 4: ROBUST OPTIMIZATION")
        log.info("=" * 80)
        
        optimizer = RobustOptimizer(
            surrogate,
            self.design_space,
            n_mc_inner=self.config.get("n_mc_robustness", 20),
        )
        
        # Run optimization
        opt_method = self.config.get("opt_method", "de")
        
        if opt_method == "nsga2":
            log.info("Running NSGA-II multi-objective optimization...")
            result = optimizer.optimize_nsga2(
                pop_size=100,
                n_gen=50,
                seed=42,
            )
        else:
            log.info("Running Differential Evolution (single-objective)...")
            result = optimizer.optimize_de(
                maxiter=100,
                seed=42,
            )
        
        # Decode best design
        best_design = result.best_robust_design
        best_design_dict = self.design_space.decode_vector(best_design)
        
        log.info(f"✅ Optimization complete!")
        log.info(f"   Number of evaluations: {result.n_evaluations}")
        log.info(f"   Best robust design:")
        for name, val in list(best_design_dict.items())[:8]:
            log.info(f"      • {name}: {val:.2f}")
        
        # Save results
        result_df = pd.DataFrame([best_design_dict])
        result_df.to_csv(self.output_dir / "optimal_design.csv", index=False)
        
        self.results["optimization"] = result
        
        return result
    
    # -------------------------------------------------------------------------
    # STAGE 5: Inverse Design
    # -------------------------------------------------------------------------
    def train_inverse_design(self, results_df: pd.DataFrame):
        """
        Train inverse design engine.
        
        Args:
            results_df: FEA results with design variables
        """
        log.info("\n" + "=" * 80)
        log.info("STAGE 5: INVERSE DESIGN ENGINE")
        log.info("=" * 80)
        
        # Extract design variables
        var_cols = [c for c in results_df.columns if c.startswith("var_")]
        X_design = results_df[var_cols].values
        
        # Extract performance
        output_cols = [
            "static_max_deflection_um",
            "static_max_vonmises_MPa",
            "freq_mode1_Hz",
        ]
        output_cols = [c for c in output_cols if c in results_df.columns]
        y_performance = results_df[output_cols]
        
        # Train inverse model
        engine = InverseDesignEngine(self.design_space, use_tensorflow=True)
        
        metrics = engine.train(
            X_design,
            y_performance,
            epochs=100,
            verbose=True,
        )
        
        # Save model
        engine.save(self.output_dir / "inverse_design_model.pkl")
        
        self.inverse_engine = engine
        self.results["inverse_engine"] = engine
        
        log.info(f"✅ Inverse design training complete")
        log.info(f"   Validation R²: {metrics.get('val_r2', 0):.4f}")
        
        # Test inverse design
        log.info("\n📋 Testing Inverse Design:")
        target = {
            "static_max_deflection_um": 8.0,
            "static_max_vonmises_MPa": 350.0,
            "freq_mode1_Hz": 800.0,
        }
        log.info(f"   Target: {target}")
        
        x_pred = engine.predict_design(target)
        x_pred_dict = self.design_space.decode_vector(x_pred)
        
        log.info(f"   Predicted design:")
        for name, val in list(x_pred_dict.items())[:6]:
            log.info(f"      • {name}: {val:.2f}")
        
        return engine
    
    # -------------------------------------------------------------------------
    # Execute Full Workflow
    # -------------------------------------------------------------------------
    def execute_full_workflow(self):
        """Run all stages in sequence."""
        
        # Stage 1: LHS Sampling
        X_samples = self.run_lhs_sampling()
        
        # Stage 2: FEA Pool
        results_df = self.run_fea_pool(X_samples)
        
        # Stage 3: Surrogate Training
        surrogate = self.train_surrogate(results_df)
        
        # Stage 4: Robust Optimization
        opt_result = self.run_robust_optimization(surrogate)
        
        # Stage 5: Inverse Design
        inverse_engine = self.train_inverse_design(results_df)
        
        # Final Summary
        log.info("\n" + "=" * 80)
        log.info("🎉 COMPLETE RDO WORKFLOW FINISHED!")
        log.info("=" * 80)
        log.info(f"All results saved to: {self.output_dir}")
        log.info("")
        log.info("Generated files:")
        log.info(f"  • lhs_samples.csv")
        log.info(f"  • fea_results/fea_batch_results.csv")
        log.info(f"  • surrogate_{self.config['surrogate_type']}.pkl")
        log.info(f"  • optimal_design.csv")
        log.info(f"  • inverse_design_model.pkl")
        log.info("=" * 80)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  COMMAND-LINE INTERFACE                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def parse_args():
    parser = argparse.ArgumentParser(
        description="TechPulse Spindle RDO Master Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--mode",
        choices=["full", "ml_only", "opt_only"],
        default="full",
        help="Workflow mode: full (all stages), ml_only (train ML from existing FEA), opt_only (optimize with pre-trained model)"
    )
    
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of LHS samples for DoE (default: 100)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Use analytical beam theory instead of ANSYS (fast prototyping)"
    )
    
    parser.add_argument(
        "--surrogate_type",
        choices=["gp", "xgb", "mlp"],
        default="gp",
        help="ML surrogate type: gp (Gaussian Process), xgb (XGBoost), mlp (Neural Network)"
    )
    
    parser.add_argument(
        "--opt_method",
        choices=["de", "nsga2"],
        default="nsga2",
        help="Optimization method: de (Differential Evolution), nsga2 (Multi-objective GA)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="rdo_results",
        help="Output directory for all results"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Build configuration
    config = {
        "mode": args.mode,
        "n_samples": args.n_samples,
        "dry_run": args.dry_run,
        "surrogate_type": args.surrogate_type,
        "opt_method": args.opt_method,
        "output_dir": args.output_dir,
        "lhs_criterion": "maximin",
        "n_mc_robustness": 20,
    }
    
    # Initialize orchestrator
    orchestrator = RDOMasterOrchestrator(config)
    
    # Execute workflow
    if config["mode"] == "full":
        orchestrator.execute_full_workflow()
    
    elif config["mode"] == "ml_only":
        # Load existing FEA results
        fea_path = Path(config["output_dir"]) / "fea_results" / "fea_batch_results.csv"
        if not fea_path.exists():
            log.error(f"FEA results not found at {fea_path}")
            return
        
        results_df = pd.read_csv(fea_path)
        orchestrator.train_surrogate(results_df)
        orchestrator.train_inverse_design(results_df)
    
    elif config["mode"] == "opt_only":
        # Load pre-trained surrogate
        model_path = Path(config["output_dir"]) / f"surrogate_{config['surrogate_type']}.pkl"
        if not model_path.exists():
            log.error(f"Surrogate model not found at {model_path}")
            return
        
        surrogate = SurrogateModel.load(model_path)
        orchestrator.surrogate = surrogate
        orchestrator.run_robust_optimization(surrogate)


if __name__ == "__main__":
    main()
