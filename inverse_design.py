#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  Inverse Design Engine v2 — DNN for Performance → Geometry Mapping
================================================================================

  Purpose:
      Map desired performance targets back to required design dimensions.

      Forward:  Design (X)  → FEA  → Performance (Y)
      Inverse:  Target (Y*) → DNN  → Design (X*)

  Bug Fixes Applied (vs. v1):
      BUG-10 [CRITICAL]: List[str] used in train() signature and locally but
                          List was never imported from typing → NameError at
                          runtime on any Python version.
================================================================================
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional    # BUG-10 FIX: List added

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks as k_cb
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from design_variables import DesignSpace

log = logging.getLogger("InverseDesign")
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")


class InverseDesignEngine:
    """
    Deep Neural Network for inverse design.

    Maps: Performance targets Y* → Required design X*

    Training data = (Y_fea, X_fea) pairs from the FEA pool — note the swap:
    the DNN is trained with performance as input and design as output.
    After training, pass target performance in and get predicted geometry out.
    The forward surrogate is used for optional validation.

    Attributes:
        design_space:  DesignSpace for bounds clipping
        input_names:   Performance variable names (DNN inputs)
        output_names:  Design variable names (DNN outputs)
        scaler_X:      StandardScaler for DNN inputs (performance)
        scaler_y:      StandardScaler for DNN outputs (design variables)
        model:         Trained Keras or sklearn model
    """

    def __init__(
        self,
        design_space:    DesignSpace,
        use_tensorflow:  bool = True,
    ):
        self.design_space    = design_space
        self.use_tensorflow  = use_tensorflow and TF_AVAILABLE
        self.scaler_X        = StandardScaler()
        self.scaler_y        = StandardScaler()
        self.input_names: List[str]  = []     # BUG-10 FIX: annotation is valid now
        self.output_names: List[str] = design_space.get_variable_names()
        self.model           = None

        backend = "TensorFlow" if self.use_tensorflow else "sklearn MLP"
        log.info(f"InverseDesignEngine  backend={backend}")

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────
    def train(
        self,
        X_design:          np.ndarray,
        y_performance:     np.ndarray | pd.DataFrame,
        performance_names: Optional[List[str]] = None,    # BUG-10 FIX
        epochs:            int = 200,
        batch_size:        int = 32,
        validation_split:  float = 0.2,
        verbose:           bool = True,
    ) -> Dict[str, float]:
        """
        Train inverse model: Performance → Design.

        Args:
            X_design:          (n_samples, n_design_vars)  design vectors
            y_performance:     (n_samples, n_perf_vars)    FEA performance
            performance_names: Column names of y_performance
            epochs, batch_size, validation_split: TF training params
            verbose: Print training log

        Returns:
            Metrics dict: val_r2, val_mse
        """
        if isinstance(y_performance, pd.DataFrame):
            self.input_names = y_performance.columns.tolist()
            y_performance    = y_performance.values
        else:
            self.input_names = performance_names or [
                f"perf_{i}" for i in range(y_performance.shape[1])
            ]

        n_in  = y_performance.shape[1]
        n_out = X_design.shape[1]

        log.info(
            f"Training inverse model  "
            f"inputs(perf)={n_in}  outputs(design)={n_out}  "
            f"samples={len(X_design)}"
        )

        # NOTE: roles are swapped compared to forward surrogate
        # DNN input  = performance  → scaled by scaler_X
        # DNN output = design vars  → scaled by scaler_y
        X_tr, X_val, y_tr, y_val = train_test_split(
            y_performance, X_design,
            test_size=validation_split, random_state=42,
        )

        X_tr_s  = self.scaler_X.fit_transform(X_tr)
        X_val_s = self.scaler_X.transform(X_val)
        y_tr_s  = self.scaler_y.fit_transform(y_tr)
        y_val_s = self.scaler_y.transform(y_val)

        if self.use_tensorflow:
            metrics = self._train_keras(
                X_tr_s, y_tr_s, X_val_s, y_val_s, y_val,
                epochs=epochs, batch_size=batch_size, verbose=verbose,
            )
        else:
            metrics = self._train_sklearn(
                X_tr_s, y_tr_s, X_val_s, y_val,
            )

        log.info(f"✅ Inverse model trained  val_R²={metrics.get('val_r2',0):.4f}")
        return metrics

    # ─────────────────────────────────────────────────────────────────────────
    # Keras DNN
    # ─────────────────────────────────────────────────────────────────────────
    def _train_keras(
        self,
        X_tr: np.ndarray, y_tr: np.ndarray,
        X_val: np.ndarray, y_val_raw: np.ndarray, y_val_original: np.ndarray,
        epochs: int, batch_size: int, verbose: bool,
    ) -> Dict[str, float]:
        n_in, n_out = X_tr.shape[1], y_tr.shape[1]

        model = keras.Sequential([
            layers.Input(shape=(n_in,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(64,  activation="relu"),
            layers.Dense(n_out, activation="linear"),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss="mse", metrics=["mae"],
        )

        cb_list = [
            k_cb.EarlyStopping(monitor="val_loss", patience=20,
                               restore_best_weights=True),
            k_cb.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                   patience=10, min_lr=1e-6),
        ]

        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val_raw),
            epochs=epochs, batch_size=batch_size,
            callbacks=cb_list,
            verbose=1 if verbose else 0,
        )
        self.model = model

        y_pred_s = model.predict(X_val, verbose=0)
        y_pred   = self.scaler_y.inverse_transform(y_pred_s)
        r2  = r2_score(y_val_original, y_pred, multioutput="variance_weighted")
        mse = mean_squared_error(y_val_original, y_pred)
        return {"val_r2": float(r2), "val_mse": float(mse)}

    # ─────────────────────────────────────────────────────────────────────────
    # sklearn MLP fallback
    # ─────────────────────────────────────────────────────────────────────────
    def _train_sklearn(
        self,
        X_tr: np.ndarray, y_tr: np.ndarray,
        X_val: np.ndarray, y_val_original: np.ndarray,
    ) -> Dict[str, float]:
        n_in = X_tr.shape[1]
        model = MLPRegressor(
            hidden_layer_sizes=(128, 256, 128, 64),
            activation="relu", solver="adam",
            learning_rate_init=0.001,
            max_iter=500, early_stopping=True,
            validation_fraction=0.2, random_state=42,
        )
        model.fit(X_tr, y_tr)
        self.model = model

        y_pred_s = model.predict(X_val)
        y_pred   = self.scaler_y.inverse_transform(y_pred_s)
        r2  = r2_score(y_val_original, y_pred, multioutput="variance_weighted")
        mse = mean_squared_error(y_val_original, y_pred)
        return {"val_r2": float(r2), "val_mse": float(mse)}

    # ─────────────────────────────────────────────────────────────────────────
    # Predict design from target performance
    # ─────────────────────────────────────────────────────────────────────────
    def predict_design(
        self,
        target_performance: np.ndarray | Dict[str, float],
        clip_to_bounds: bool = True,
    ) -> np.ndarray:
        """
        Predict design dimensions from target performance.

        Args:
            target_performance: dict {"deflection_um": 8.0, …} or 1-D array
            clip_to_bounds:     Hard-clip predictions to design-space limits

        Returns:
            x_pred: (n_design_vars,) predicted design vector
        """
        if isinstance(target_performance, dict):
            arr = np.array([
                target_performance[n] for n in self.input_names
            ])
        else:
            arr = np.asarray(target_performance, dtype=float)

        arr_2d  = arr.reshape(1, -1)
        arr_s   = self.scaler_X.transform(arr_2d)

        if self.use_tensorflow and isinstance(self.model, keras.Sequential):
            x_s = self.model.predict(arr_s, verbose=0)
        else:
            x_s = self.model.predict(arr_s)

        x_pred = self.scaler_y.inverse_transform(x_s).ravel()

        if clip_to_bounds:
            lo, hi = self.design_space.get_bounds().T
            x_pred = np.clip(x_pred, lo, hi)

        return x_pred

    # ─────────────────────────────────────────────────────────────────────────
    # Validation via forward surrogate
    # ─────────────────────────────────────────────────────────────────────────
    def validate(
        self,
        x_pred: np.ndarray,
        forward_surrogate,
        target: np.ndarray,
    ) -> pd.DataFrame:
        """
        Check predicted design by running it through the forward surrogate.

        Returns DataFrame: target | achieved | abs_error | rel_error_%
        """
        y_achieved = forward_surrogate.predict(x_pred.reshape(1, -1)).ravel()
        rows: List[Dict] = []
        for i, name in enumerate(self.input_names):
            t, a = target[i], y_achieved[i]
            rows.append({
                "variable":      name,
                "target":        t,
                "achieved":      a,
                "abs_error":     abs(a - t),
                "rel_error_%":   abs(a - t) / (abs(t) + 1e-10) * 100,
            })
        return pd.DataFrame(rows)

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────
    def save(self, filepath: str | Path) -> None:
        filepath = Path(filepath)
        payload: Dict = {
            "scaler_X":     self.scaler_X,
            "scaler_y":     self.scaler_y,
            "input_names":  self.input_names,
            "output_names": self.output_names,
            "use_tensorflow": self.use_tensorflow,
        }
        if self.use_tensorflow:
            keras_path = filepath.with_suffix(".keras")
            self.model.save(keras_path)
            payload["keras_path"] = str(keras_path)
        else:
            payload["model"] = self.model
        joblib.dump(payload, filepath)
        log.info(f"💾 Saved → {filepath}")

    @classmethod
    def load(cls, filepath: str | Path, design_space: DesignSpace) -> "InverseDesignEngine":
        filepath = Path(filepath)
        d   = joblib.load(filepath)
        eng = cls(design_space, use_tensorflow=d["use_tensorflow"])
        eng.scaler_X     = d["scaler_X"]
        eng.scaler_y     = d["scaler_y"]
        eng.input_names  = d["input_names"]
        eng.output_names = d["output_names"]
        if d["use_tensorflow"]:
            eng.model = keras.models.load_model(d["keras_path"])
        else:
            eng.model = d["model"]
        log.info(f"📂 Loaded ← {filepath}")
        return eng


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_inverse_design(
    engine:       "InverseDesignEngine",
    target:       Dict[str, float],
    forward_surr: Optional[object] = None,
    save_dir:     str = ".",
) -> None:
    """
    Three inverse-design diagnostic plots.

    Fig 06a — Target vs. predicted performance (validation bar chart)
    Fig 06b — Predicted design vector: normalised position in bounds
    Fig 06c — Prediction error: per-output (actual − predicted) if forward surrogate available
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

    ds     = engine.design_space
    x_pred = engine.predict_design(target)
    names  = ds.get_variable_names()
    bounds = ds.get_bounds()
    tgt_names  = list(target.keys())
    tgt_values = list(target.values())

    # ── Fig 06a: Target vs forward-predicted performance ─────────────
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=NAVY)
    ax.set_facecolor("#112233")

    # If forward surrogate provided, get achieved performance
    achieved = []
    if forward_surr is not None:
        try:
            y_fwd = forward_surr.predict(x_pred.reshape(1, -1))[0]
            for tname in tgt_names:
                idx_o = forward_surr.output_names.index(tname) \
                        if tname in forward_surr.output_names else None
                achieved.append(float(y_fwd[idx_o]) if idx_o is not None else None)
        except Exception:
            achieved = [None] * len(tgt_names)
    else:
        achieved = [None] * len(tgt_names)

    x_pos  = np.arange(len(tgt_names))
    ax.bar(x_pos - 0.18, tgt_values, 0.32, color=TEAL, edgecolor=NAVY,
           linewidth=0.5, label="Target")
    if any(v is not None for v in achieved):
        achv_vals = [v if v is not None else 0 for v in achieved]
        ax.bar(x_pos + 0.18, achv_vals, 0.32, color=GOLD, edgecolor=NAVY,
               linewidth=0.5, label="Achieved (fwd surrogate)")
        for i, (t, a) in enumerate(zip(tgt_values, achv_vals)):
            err_pct = abs(a - t) / max(abs(t), 1e-12) * 100
            ax.text(i, max(t, a) + max(tgt_values) * 0.02,
                    f"{err_pct:.1f}%", ha="center", fontsize=8, color="white")
    ax.set_xticks(x_pos); ax.set_xticklabels(tgt_names, fontsize=9)
    ax.set_ylabel("Performance value"); ax.set_title("Fig 06a — Inverse Design Validation", pad=8)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(save_dir, "06a_inverse_validation.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig); print(f"  Saved → {p}")

    # ── Fig 06b: Predicted design variables (normalised in bounds) ────
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=NAVY)
    ax.set_facecolor("#112233")
    x_norm = (x_pred - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    nom_x  = ds.get_nominal()
    nom_norm = (nom_x - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    y_pos  = np.arange(len(names))
    ax.barh(y_pos, x_norm, color=CORAL, alpha=0.8, height=0.5,
            edgecolor=NAVY, linewidth=0.4, label="Inverse predicted")
    ax.scatter(nom_norm, y_pos, s=25, c=GOLD, zorder=5,
               marker="|", linewidths=1.5, label="Nominal baseline")
    ax.set_yticks(y_pos); ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("Normalised position in bounds  [0=lower, 1=upper]")
    ax.set_title("Fig 06b — Inverse Design: Predicted Variables (gold = nominal)", pad=8)
    ax.axvline(0.5, color=GRAY, lw=0.8, linestyle="--", alpha=0.5)
    ax.set_xlim(0, 1); ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(save_dir, "06b_inverse_design_vars.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig); print(f"  Saved → {p}")

    # ── Fig 06c: Error per performance target ─────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=NAVY)
    ax.set_facecolor("#112233")
    if any(v is not None for v in achieved):
        errs = [(a - t) / max(abs(t), 1e-12) * 100 if a is not None else 0
                for a, t in zip(achieved, tgt_values)]
        colours_e = [TEAL if abs(e) < 10 else CORAL for e in errs]
        ax.bar(tgt_names, errs, color=colours_e, edgecolor=NAVY, linewidth=0.5)
        ax.axhline(0,  color="white", lw=1.0)
        ax.axhline(+10, color=GOLD, lw=1.0, linestyle="--", alpha=0.7)
        ax.axhline(-10, color=GOLD, lw=1.0, linestyle="--", alpha=0.7, label="±10% limit")
        ax.set_ylabel("Relative error  [%]")
        ax.set_title("Fig 06c — Inverse Design Prediction Error per KPI", pad=8)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Forward surrogate not provided\n(pass forward_surr= for error plot)",
                ha="center", va="center", color=GRAY, fontsize=10, transform=ax.transAxes)
        ax.set_title("Fig 06c — Prediction Error (surrogate required)", pad=8)
    plt.tight_layout()
    p = os.path.join(save_dir, "06c_inverse_error.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig); print(f"  Saved → {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os; sys.path.insert(0, ".")
    from design_variables import DesignSpace

    np.random.seed(42)
    ds = DesignSpace()
    n  = len(ds.get_variable_names())

    X_design = np.random.rand(200, n) * 100
    y_perf   = np.random.rand(200, 3) * 50
    names    = ["deflection_um", "stress_MPa", "frequency_Hz"]

    print("\n🔮 Inverse Design Engine v2 — smoke test\n")
    engine = InverseDesignEngine(ds, use_tensorflow=False)
    m      = engine.train(X_design, y_perf, performance_names=names, verbose=False)
    print(f"val_R² = {m['val_r2']:.4f}")

    target = {"deflection_um": 8.0, "stress_MPa": 350.0, "frequency_Hz": 800.0}
    x_pred = engine.predict_design(target)
    print(f"Predicted L1 = {x_pred[ds.get_variable_names().index('L1')]:.2f} mm")

    engine.save("/tmp/inv_v2.pkl")
    eng2 = InverseDesignEngine.load("/tmp/inv_v2.pkl", ds)
    assert np.allclose(eng2.predict_design(target), x_pred)
    print("✅ Round-trip OK — BUG-10 fixed")

    os.makedirs("/tmp/spindle_plots", exist_ok=True)
    print("\nGenerating inverse design plots...")
    plot_inverse_design(engine, target, save_dir="/tmp/spindle_plots")
    print("✅ Inverse Design v2 OK")
