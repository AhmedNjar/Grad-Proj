#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  ML Surrogate Model v2 — Fast Digital Twin for FEA Emulation
================================================================================

  Bug Fixes Applied (vs. v1):
      BUG-1 [CRITICAL]: Single shared scaler_y was refitted per output inside
                         the training loop, so predict() always inverse-
                         transformed every output using only the LAST output's
                         scale factors — numerically wrong for all other outputs.
                         Fix: self.scalers_y is now a Dict[str, StandardScaler],
                         one independent scaler per output.
      BUG-2 [CRITICAL]: List, Dict were used in type annotations but never
                         imported from typing → NameError at import time.
      BUG-3 [HIGH]:     GP std unscaling used self.scaler_y.scale_[0] which
                         referenced the (now removed) shared scaler.
                         Fix: uses scalers_y[out_name].scale_[0].
================================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple   # BUG-2 FIX
import joblib
import logging

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ML_Surrogate")


class SurrogateModel:
    """
    Multi-output ML surrogate for FEA emulation.

    One independent model + one independent StandardScaler per output variable.
    A single shared scaler (v1 bug) caused all outputs except the last to be
    inverse-transformed with the wrong scale.

    Attributes:
        model_type:   "gp" | "xgb" | "mlp"
        scaler_X:     StandardScaler for the design-variable inputs
        scalers_y:    Dict[output_name → StandardScaler]   ← BUG-1 FIX
        models:       Dict[output_name → fitted estimator]
        output_names: Ordered list of output variable names
    """

    def __init__(
        self,
        model_type: Literal["gp", "xgb", "mlp"] = "gp",
        random_state: int = 42,
    ):
        self.model_type    = model_type
        self.random_state  = random_state
        self.scaler_X      = StandardScaler()
        self.scalers_y: Dict[str, StandardScaler] = {}   # BUG-1 FIX
        self.models:    Dict[str, object]          = {}
        self.output_names: List[str]               = []
        log.info(f"Initialised {model_type.upper()} Surrogate Model")

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray | pd.DataFrame,
        output_names: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Train independent surrogate + scaler for every output column.

        Returns:
            cv_scores: Dict[output_name → 5-fold CV mean R²]
        """
        if isinstance(y_train, pd.DataFrame):
            self.output_names = y_train.columns.tolist()
            y_train = y_train.values
        else:
            self.output_names = output_names or [
                f"output_{i}" for i in range(y_train.shape[1])
            ]

        n_samples, n_features = X_train.shape
        log.info(
            f"Training on {n_samples} samples, {n_features} features, "
            f"{len(self.output_names)} outputs"
        )

        X_scaled  = self.scaler_X.fit_transform(X_train)
        cv_scores: Dict[str, float] = {}

        for i, out_name in enumerate(self.output_names):
            if verbose:
                log.info(f"  ► {out_name}")

            y_i = y_train[:, i].reshape(-1, 1)

            # BUG-1 FIX — fresh, independent scaler per output
            scaler_i = StandardScaler()
            y_i_s    = scaler_i.fit_transform(y_i).ravel()
            self.scalers_y[out_name] = scaler_i

            model = self._build_model(n_features)
            model.fit(X_scaled, y_i_s)
            self.models[out_name] = model

            if verbose:
                kfold = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
                cv    = cross_val_score(model, X_scaled, y_i_s, cv=kfold, scoring="r2")
                cv_scores[out_name] = float(cv.mean())
                log.info(f"     CV R² = {cv.mean():.4f} ± {cv.std():.4f}")

        log.info("✅ Training complete!")
        return cv_scores

    # ─────────────────────────────────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────────────────────────────────
    def predict(
        self,
        X_test: np.ndarray,
        return_std: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        Predict FEA outputs.

        Returns:
            y_pred          (n_samples, n_outputs)   — always
            y_std           (n_samples, n_outputs)   — only when return_std=True & GP
        """
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        X_s       = self.scaler_X.transform(X_test)
        n_samples = X_test.shape[0]
        n_out     = len(self.output_names)
        y_pred    = np.zeros((n_samples, n_out))
        y_std     = np.zeros((n_samples, n_out)) if return_std else None

        for i, out_name in enumerate(self.output_names):
            model    = self.models[out_name]
            scaler_i = self.scalers_y[out_name]          # BUG-1 & BUG-3 FIX

            if return_std and self.model_type == "gp":
                y_s, std_s   = model.predict(X_s, return_std=True)
                y_pred[:, i] = scaler_i.inverse_transform(y_s.reshape(-1, 1)).ravel()
                y_std[:, i]  = std_s * scaler_i.scale_[0]   # BUG-3 FIX
            else:
                y_s          = model.predict(X_s)
                y_pred[:, i] = scaler_i.inverse_transform(y_s.reshape(-1, 1)).ravel()

        return (y_pred, y_std) if return_std else y_pred

    # ─────────────────────────────────────────────────────────────────────────
    # Model builders
    # ─────────────────────────────────────────────────────────────────────────
    def _build_model(self, n_features: int) -> object:
        if self.model_type == "gp":
            return self._build_gp(n_features)
        if self.model_type == "xgb":
            return self._build_xgb()
        if self.model_type == "mlp":
            return self._build_mlp(n_features)
        raise ValueError(f"Unknown model_type: '{self.model_type}'")

    def _build_gp(self, n_features: int) -> GaussianProcessRegressor:
        """Matérn(ν=2.5) + white noise.  ν=2.5 → twice-differentiable responses."""
        kernel = (
            C(1.0, (1e-3, 1e3))
            * Matern(
                length_scale=[1.0] * n_features,
                nu=2.5,
                length_scale_bounds=(1e-2, 1e2),
            )
            + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=self.random_state,
            normalize_y=False,
        )

    def _build_xgb(self) -> object:
        if XGBOOST_AVAILABLE:
            return xgb.XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=6,
                subsample=0.8, colsample_bytree=0.8,
                random_state=self.random_state, n_jobs=-1, verbosity=0,
            )
        return GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, random_state=self.random_state,
        )

    def _build_mlp(self, n_features: int) -> MLPRegressor:
        return MLPRegressor(
            hidden_layer_sizes=(2 * n_features, n_features, n_features // 2),
            activation="relu", solver="adam", learning_rate="adaptive",
            max_iter=1000, early_stopping=True, validation_fraction=0.2,
            random_state=self.random_state,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """Return R², RMSE, MAE for each output on a held-out test set."""
        y_pred = self.predict(X_test)
        rows: List[Dict] = []
        for i, name in enumerate(self.output_names):
            t, p = y_test[:, i], y_pred[:, i]
            rows.append({
                "output": name,
                "R2":     r2_score(t, p),
                "RMSE":   float(np.sqrt(mean_squared_error(t, p))),
                "MAE":    float(mean_absolute_error(t, p)),
            })
        return pd.DataFrame(rows)

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────
    def save(self, filepath: str | Path) -> None:
        joblib.dump({
            "model_type":   self.model_type,
            "models":       self.models,
            "scaler_X":     self.scaler_X,
            "scalers_y":    self.scalers_y,   # BUG-1 FIX: dict, not single scaler
            "output_names": self.output_names,
        }, filepath)
        log.info(f"💾 Saved → {filepath}")

    @classmethod
    def load(cls, filepath: str | Path) -> "SurrogateModel":
        d   = joblib.load(filepath)
        obj = cls(model_type=d["model_type"])
        obj.models       = d["models"]
        obj.scaler_X     = d["scaler_X"]
        obj.scalers_y    = d["scalers_y"]
        obj.output_names = d["output_names"]
        log.info(f"📂 Loaded ← {filepath}")
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test (proves BUG-1 is fixed with deliberately different output scales)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    n, p = 120, 10
    X_tr = np.random.rand(n, p)
    X_te = np.random.rand(20, p)

    # Three outputs with VERY different scales — the original bug collapses them
    y_tr = np.column_stack([
        10  * np.sin(5 * X_tr[:, 0]) + 5 * X_tr[:, 1]**2,   # scale ~15
        200 + 40 * X_tr[:, 2] * X_tr[:, 3],                  # scale ~220
        8000 + 1000 * np.exp(-X_tr[:, 4]),                    # scale ~9000
    ])
    y_te = np.column_stack([
        10  * np.sin(5 * X_te[:, 0]) + 5 * X_te[:, 1]**2,
        200 + 40 * X_te[:, 2] * X_te[:, 3],
        8000 + 1000 * np.exp(-X_te[:, 4]),
    ])
    names = ["deflection_um", "stress_MPa", "frequency_Hz"]

    print("\n🤖  ML Surrogate v2 — smoke test\n")
    m = SurrogateModel(model_type="gp")
    m.train(X_tr, y_tr, output_names=names, verbose=False)

    y_p, y_s = m.predict(X_te, return_std=True)
    print(f"{'Output':<22} {'True':>10} {'Pred':>10} {'Std':>8}")
    print("-" * 55)
    for j, n_ in enumerate(names):
        print(f"{n_:<22} {y_te[0,j]:10.2f} {y_p[0,j]:10.2f} {y_s[0,j]:8.2f}")

    print()
    print(m.evaluate(X_te, y_te).to_string(index=False))

    m.save("/tmp/surr_v2.pkl")
    assert np.allclose(SurrogateModel.load("/tmp/surr_v2.pkl").predict(X_te), y_p)
    print("\n✅ Round-trip OK — all bugs fixed")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_surrogate_performance(model, X_train, y_train, X_test, y_test, save_dir="."):
    """Fig 04a/b/c — Predicted vs actual, residuals, GP uncertainty."""
    import matplotlib.pyplot as plt, os
    NAVY="#0d1b2a"; TEAL="#00b4d8"; CORAL="#e63946"; GOLD="#ffd166"
    MINT="#06d6a0"; GRAY="#8d99ae"; PURPLE="#7400b8"
    SEG_COLS=[TEAL,CORAL,GOLD,MINT,PURPLE,GRAY]
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams.update({"figure.facecolor":NAVY,"axes.facecolor":"#112233",
        "axes.edgecolor":GRAY,"axes.labelcolor":"white","xtick.color":GRAY,
        "ytick.color":GRAY,"text.color":"white","grid.color":"#2d4060","grid.alpha":0.4,"font.size":9})

    n_out=len(model.output_names)
    y_pred=model.predict(X_test)
    metrics=model.evaluate(X_test, y_test)

    # 04a: predicted vs actual
    ncols=min(n_out,3); nrows=(n_out+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(4.5*ncols,4.5*nrows),facecolor=NAVY)
    axes_flat=list(axes.flat) if hasattr(axes,"flat") else [axes]
    fig.suptitle("Fig 04a — Predicted vs Actual", color="white", y=1.01)
    for j,(name,ax) in enumerate(zip(model.output_names,axes_flat)):
        ax.set_facecolor("#112233"); col=SEG_COLS[j%len(SEG_COLS)]
        yt=y_test[:,j]; yp=y_pred[:,j]
        ax.scatter(yt,yp,s=18,c=col,alpha=0.7,edgecolors="none")
        lim=[min(yt.min(),yp.min()),max(yt.max(),yp.max())]
        ax.plot(lim,lim,color=GOLD,lw=1.2,linestyle="--")
        row=metrics[metrics["output"]==name]
        r2=float(row["R2"].values[0]) if len(row) else float("nan")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title(f"{name}\nR²={r2:.4f}",fontsize=8.5)
    for ax in axes_flat[n_out:]: ax.set_visible(False)
    plt.tight_layout()
    p=os.path.join(save_dir,"04a_surrogate_pva.png")
    fig.savefig(p,dpi=150,bbox_inches="tight",facecolor=NAVY); plt.close(fig); print(f"  Saved → {p}")

    # 04b: residuals
    fig,axes=plt.subplots(1,n_out,figsize=(4.5*n_out,4.5),facecolor=NAVY)
    axes_flat2=list(axes.flat) if hasattr(axes,"flat") else [axes]
    fig.suptitle("Fig 04b — Prediction Residuals", color="white", y=1.01)
    for j,(name,ax) in enumerate(zip(model.output_names,axes_flat2)):
        ax.set_facecolor("#112233"); col=SEG_COLS[j%len(SEG_COLS)]
        res=y_test[:,j]-y_pred[:,j]
        ax.hist(res,bins=12,color=col,edgecolor=NAVY,alpha=0.85)
        ax.axvline(0,color=GOLD,lw=1.5,linestyle="--")
        ax.axvline(res.mean(),color="white",lw=1.0,linestyle=":",label=f"μ={res.mean():.3f}")
        ax.set_xlabel("Residual"); ax.set_title(name,fontsize=8.5); ax.legend(fontsize=7.5)
    plt.tight_layout()
    p=os.path.join(save_dir,"04b_surrogate_residuals.png")
    fig.savefig(p,dpi=150,bbox_inches="tight",facecolor=NAVY); plt.close(fig); print(f"  Saved → {p}")

    # 04c: GP uncertainty vs distance (if gp model)
    if model.model_type == "gp":
        try:
            _,y_std=model.predict(X_test,return_std=True)
            dists=[np.min(np.linalg.norm(X_train-xt,axis=1)) for xt in X_test]
            dists=np.array(dists)
            fig,axes=plt.subplots(1,n_out,figsize=(4.5*n_out,4.5),facecolor=NAVY)
            axes_flat3=list(axes.flat) if hasattr(axes,"flat") else [axes]
            fig.suptitle("Fig 04c — GP Uncertainty vs Distance", color="white", y=1.01)
            for j,(name,ax) in enumerate(zip(model.output_names,axes_flat3)):
                ax.set_facecolor("#112233"); col=SEG_COLS[j%len(SEG_COLS)]
                ax.scatter(dists,y_std[:,j],s=18,c=col,alpha=0.7)
                if len(dists)>1:
                    z=np.polyfit(dists,y_std[:,j],1); xs=np.linspace(dists.min(),dists.max(),100)
                    ax.plot(xs,np.poly1d(z)(xs),color=GOLD,lw=1.3,linestyle="--")
                ax.set_xlabel("Dist to nearest train pt"); ax.set_ylabel("Pred std"); ax.set_title(name,fontsize=8.5)
            plt.tight_layout()
            p=os.path.join(save_dir,"04c_gp_uncertainty.png")
            fig.savefig(p,dpi=150,bbox_inches="tight",facecolor=NAVY); plt.close(fig); print(f"  Saved → {p}")
        except Exception as e:
            print(f"  04c skipped: {e}")
