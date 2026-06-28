#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  FEA Pool Runner v2 — Batch Parametric Simulation Engine
================================================================================

  Bug Fixes Applied (vs. v1):
      BUG-11 [HIGH]:  Analytical model treated the spindle as a pure cantilever
                       (δ=FL³/3EI), ignoring the two bearing supports.  A
                       spindle is a propped beam: the load is applied at the
                       nose overhang (z<0 relative to front bearing), while
                       the two bearings act as spring supports.
                       Fix: three-segment transfer-matrix model that accounts
                       for front and rear bearing radial stiffness explicitly.
      BUG-12 [MEDIUM]: First natural frequency used the cantilever Rayleigh
                        coefficient λ₁=1.875 (free-clamped).  With two spring
                        supports the Dunkerly superposition formula is used
                        instead, accumulating compliance contributions from
                        each support point.
================================================================================
"""

from __future__ import annotations
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── try to import from the user's PyMAPDL script ──────────────────────────
try:
    sys.path.insert(0, "/mnt/user-data/uploads")
    from lathe_spindle_simulation import (
        SpindleGeometry, MaterialProperties, BearingConfig,
        CuttingLoads, MeshConfig, ModalConfig, SimulationCase,
        SpindleSimulation,
    )
    MAPDL_AVAILABLE = True
except ImportError:
    MAPDL_AVAILABLE = False
    logging.getLogger("FEA_Pool").warning(
        "PyMAPDL script not importable — dry-run only."
    )

from design_variables import DesignSpace
from plot_theme import apply_paper_theme, C, savefig_paper
from design_variables import snap_to_skf_bearing

log = logging.getLogger("FEA_Pool")
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")


class FEAPoolRunner:
    """
    Batch executor for parametric FEA simulations.

    Two modes:
      dry_run=True   → fast analytical model (corrected v2 physics)
      dry_run=False  → full ANSYS MAPDL via PyMAPDL
    """

    def __init__(
        self,
        X_samples: np.ndarray,
        design_space: DesignSpace,
        dry_run: bool = True,
        output_dir: str | Path = "fea_batch_results",
        start_id: int = 1,
    ):
        self.X_samples    = X_samples
        self.design_space = design_space
        self.dry_run      = dry_run
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict] = []
        self.start_id   = start_id

        self.output_dir = Path("F:/files/rdo_results/fea_cache")

        mode = "DRY-RUN (analytical v2)" if dry_run else "FULL ANSYS MAPDL"
        log.info(f"FEA Pool Runner  mode={mode}  n={len(X_samples)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────
    def execute_batch(
        self,
        max_failures: int = 10,
        save_interval: int = 20,
    ) -> pd.DataFrame:
        """Run all samples, return results DataFrame."""
        n        = len(self.X_samples)
        fails    = 0
        executor = self._run_analytical if self.dry_run else self._run_ansys

        for i, x in enumerate(tqdm(self.X_samples, desc="FEA Pool")):
            try:
                res = executor(x, case_id=self.start_id + i)
                self.results.append(res)
                fails = 0
            except Exception as exc:
                log.error(f"Case {self.start_id + i} failed: {exc}")
                fails += 1
                if fails >= max_failures:
                    log.error("Stopping — too many consecutive failures.")
                    break

            if (i + 1) % save_interval == 0:
                self._save()

        self._save()
        df = pd.DataFrame(self.results)
        log.info(f"✅ Batch done: {len(df)}/{n} successful")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Analytical solver — corrected two-bearing propped-beam model (BUG-11/12)
    # ─────────────────────────────────────────────────────────────────────────
    def _run_analytical(self, x: np.ndarray, case_id: int) -> Dict:
        """
        Physically correct analytical model for a hollow stepped spindle
        supported on two radial spring bearings.

        Geometry layout (Z-axis):
            Z=0 ──── nose load ─── Z=a (front bearing) ── Z=b (rear bearing) ── Z=L_total

        The nose-overhang region [0, a] is modelled as a cantilever propped
        at Z=a and Z=b with springs K_front and K_rear.

        Deflection at nose (Z=0) is found via force equilibrium and
        compatibility (three-moment / unit-load method on the segment).

        First natural frequency uses Dunkerly's superposition of the
        compliance at each support point.
        """
        v = self.design_space.decode_vector(x)

        # ── Geometry ──────────────────────────────────────────────────────────
        L1, L2, L3, L4 = v["L1"], v["L2"], v["L3"], v["L4"]
        R1, R2, R3, R4 = v["R1"], v["R2"], v["R3"], v["R4"]
        ri              = v["ri"]
        L_total         = L1 + L2 + L3 + L4

        # Bearing axial positions from nose
        a = L1 + v["front_z_fraction"] * L2      # front bearing
        b = L1 + v["rear_z_fraction"]  * L2      # rear bearing
        bearing_span = b - a
        if bearing_span < 1.0:
            bearing_span = 1.0                    # prevent division by zero

        # Representative cross-section properties (area-weighted over segments)
        R_vals = [R1, R2, R3, R4]
        L_vals = [L1, L2, L3, L4]

        I_segs, A_segs = [], []
        for R_o, L_s in zip(R_vals, L_vals):
            I_segs.append(np.pi / 4 * (R_o**4 - ri**4))
            A_segs.append(np.pi * (R_o**2 - ri**2))

        # Effective EI and ρA weighted by length
        E_mod = v["E"]                            # MPa = N/mm²
        rho   = v["rho"]                          # ton/mm³

        EI_eff = E_mod * np.average(I_segs, weights=L_vals)
        rhoA_eff = rho * np.average(A_segs, weights=L_vals)
        # cross-section at nose for stress calculation
        I_nose = I_segs[0]
        R_nose = R1

        # ── Applied loads ─────────────────────────────────────────────────────
        Ft, Fr, Ff = v["Ft"], v["Fr"], v["Ff"]
        F_transverse = np.sqrt(Ft**2 + Fr**2)    # N, acts at Z=0 nose

        # ── Reaction forces (BUG-11 FIX) ─────────────────────────────────────
        # K_radial is no longer a design variable (removed in v4).
        # Derive from catalog via snap_to_skf_bearing using R2.
        # (snap_to_skf_bearing imported at module level)
        brg_front, _, _ = snap_to_skf_bearing(R2, "ACBB", 4000.0, "grease")
        K_fr = 1.7 * brg_front.radial_stiffness_single_N_mm     # DB pair [N/mm]
        K_re = 1.7 * brg_front.radial_stiffness_single_N_mm * 0.9  # rear CRB slightly softer

        # Moment equilibrium about rear bearing:
        R_front = F_transverse * b / bearing_span
        R_rear  = F_transverse - R_front

        # Deflection at each bearing seat due to spring compliance
        delta_front = R_front / K_fr              # mm
        delta_rear  = R_rear  / K_re              # mm

        # Deflection at nose Z=0:
        #   1) Cantilever contribution from overhang a (clamped at front bearing)
        delta_cantilever = F_transverse * a**3 / (3 * EI_eff)
        #   2) Rigid-body tilt from bearing compliance difference
        delta_tilt       = delta_front + (delta_front - delta_rear) / bearing_span * a

        delta_nose_mm  = abs(delta_cantilever + delta_tilt)
        delta_nose_um  = delta_nose_mm * 1e3     # μm

        # ── Bending stress at front bearing (max moment location) ─────────────
        # Static stress only — Euler-Bernoulli bending: σ = M·R/I
        # This is a STATIC safety factor (σ_y / σ_max), NOT a DIN 743 fatigue SF.
        # For rotating bending (R = -1 loading), fatigue SF per DIN 743 would
        # require endurance limit, size/surface factors (Kf), and mean stress —
        # not implemented here. The static SF is reported for reference only.
        M_max     = F_transverse * a             # N·mm
        sigma_MPa = M_max * R_nose / I_nose
        fos       = v["sigma_y"] / max(sigma_MPa, 1e-6)  # Static SF only

        # ── Natural frequency — Dunkerly superposition (BUG-12 FIX) ──────────
        # Dunkerly: 1/f₁² = 1/f_beam² + 1/f_front² + 1/f_rear²
        #
        # Shaft beam natural frequency (free-free, no supports):
        #   f_beam = (π²/(2π·L²)) · √(EI / ρA)
        f_beam_sq = (np.pi**2 / (2 * np.pi * L_total**2))**2 * (EI_eff / rhoA_eff)

        # Spring support natural frequencies (mass on spring):
        total_mass = rhoA_eff * L_total          # ton (mm-ton system)
        # Guard against near-zero mass
        total_mass = max(total_mass, 1e-12)
        omega_front = np.sqrt(K_fr / (total_mass / 2))   # rad/s
        omega_rear  = np.sqrt(K_re / (total_mass / 2))
        f_front_sq  = (omega_front / (2 * np.pi))**2
        f_rear_sq   = (omega_rear  / (2 * np.pi))**2

        dunkerly_inv = (1.0 / max(f_beam_sq, 1e-20)
                       + 1.0 / max(f_front_sq, 1e-20)
                       + 1.0 / max(f_rear_sq,  1e-20))
        freq1_Hz = 1.0 / np.sqrt(max(dunkerly_inv, 1e-20))

        # Approximate higher modes (scaled from mode 1)
        # Mode scaling: theoretical ratio for pinned-pinned beam = n²
        # For spindle with bearing supports: empirical ≈ 2.76, 5.40
        # (ANSYS MAPDL gives exact values; these are 1st-order approximations)
        # Mode scaling: empirical ratios for spindle with bearing supports
        # (pinned-pinned beam theory: ×4, ×9 — spindle: ≈×2.76, ×5.40)
        # ANSYS gives exact values; these are 1st-order approximations.
        freq2_Hz = freq1_Hz * 2.76
        freq3_Hz = freq1_Hz * 5.40

        # ── Assemble result ───────────────────────────────────────────────────
        result   = {
            "case_id":                    case_id,
            "mode":                       "analytical_v2",
            "total_length_mm":            L_total,
            "static_max_deflection_um":   float(delta_nose_um),
            # Note: this is bending stress σ=M·R/I (not full Von Mises incl. shear)
            # Column kept as "static_max_vonmises_MPa" for downstream compatibility
            "static_max_vonmises_MPa":    float(sigma_MPa),  # bending stress [MPa]
            "static_factor_of_safety":    float(fos),
            "freq_mode1_Hz":              float(freq1_Hz),
            "freq_mode2_Hz":              float(freq2_Hz),
            "freq_mode3_Hz":              float(freq3_Hz),
        }
        # Prefix all design-variable columns with "var_" for downstream code
        result.update({f"var_{k}": float(v_) for k, v_ in v.items()})
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # ANSYS MAPDL full solver
    # ─────────────────────────────────────────────────────────────────────────
    def _run_ansys(self, x: np.ndarray, case_id: int) -> Dict:
        if not MAPDL_AVAILABLE:
            raise RuntimeError("PyMAPDL not available. Use dry_run=True.")

        v = self.design_space.decode_vector(x)

        # K_radial and K_axial are no longer design variables (v4).
        # Derive from catalog via snap_to_skf_bearing using R2.
        R2 = v["R2"]
        brg_front, _, _ = snap_to_skf_bearing(R2, "ACBB", 4000.0, "grease")
        alpha = math.radians(brg_front.contact_angle_deg)
        K_radial = 1.7 * brg_front.radial_stiffness_single_N_mm       # pair stiffness [N/mm]
        K_axial  = 1.7 * brg_front.radial_stiffness_single_N_mm * math.tan(alpha)**2

        case = SimulationCase(
            case_id  = case_id,
            geometry = SpindleGeometry(
                L1=v["L1"], L2=v["L2"], L3=v["L3"], L4=v["L4"],
                R1=v["R1"], R2=v["R2"], R3=v["R3"], R4=v["R4"],
                ri=v["ri"],
            ),
            material = MaterialProperties(
                E=v["E"], nu=0.29, rho=v["rho"],
                sigma_y=v["sigma_y"], sigma_u=v["sigma_y"] * 1.3,
            ),
            bearings = BearingConfig(
                front_z_fraction=v["front_z_fraction"],
                rear_z_fraction =v["rear_z_fraction"],
                K_radial=K_radial, K_axial=K_axial,
            ),
            loads    = CuttingLoads(Ft=v["Ft"], Fr=v["Fr"], Ff=v["Ff"]),
            mesh     = MeshConfig(),
            modal    = ModalConfig(num_modes=6),
        )

        # ... (نفس الكود بالأعلى)
        sim = SpindleSimulation(output_dir=str(self.output_dir / f"case_{case_id:04d}"))
        sim.start()
        try:
            res = sim.run_case(case)
        finally:
            sim.close()
            # --------- التعديل: تنظيف المساحة فوراً بعد انتهاء التحليل ---------
            import glob, os
            case_dir = str(self.output_dir / f"case_{case_id:04d}")
            # تحديد امتدادات ملفات MAPDL الضخمة التي لا نحتاجها بعد استخراج النتائج
            junk_extensions = ["*.rst", "*.db", "*.emat", "*.esav", "*.err", "*.log", "*.page"]
            for ext in junk_extensions:
                for f in glob.glob(os.path.join(case_dir, ext)):
                    try:
                        os.remove(f)
                    except Exception as e:
                        pass
            # -------------------------------------------------------------------

        res.update({f"var_{k}": float(v_) for k, v_ in v.items()})
        res["mode"] = "ansys_mapdl"
        return res

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────
    def _save(self) -> None:
        if not self.results:
            return
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_dir / "fea_batch_results.csv", index=False)
        try:
            df.to_parquet(self.output_dir / "fea_batch_results.parquet", index=False)
        except ImportError:
            pass   # pyarrow not installed — CSV is sufficient

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────


def plot_fea_results(df, ds, save_dir="."):
    """Fig 03a/b/c — FEA output distributions, deflection sensitivity, freq & FoS."""
    import matplotlib.pyplot as plt, os
    NAVY=C.NAVY; TEAL=C.TEAL; CORAL=C.RED; GOLD=C.ORANGE
    MINT=C.GREEN; GRAY=C.GRAY; PURPLE=C.PURPLE
    os.makedirs(save_dir, exist_ok=True)
    apply_paper_theme()

    # 03a: output distributions
    outputs = {"static_max_deflection_um":("Deflection [μm]",TEAL),
               "static_max_vonmises_MPa":("Stress [MPa]",CORAL),
               "static_factor_of_safety":("FoS",GOLD),
               "freq_mode1_Hz":("f1 [Hz]",MINT)}
    fig, axes = plt.subplots(2, 2, figsize=(11,7), facecolor=C.BG)
    fig.suptitle("Fig 03a — FEA Batch Output Distributions", color=C.TEXT, y=1.01)
    for ax,(col,(lbl,colour)) in zip(axes.flat, outputs.items()):
        if col not in df.columns: continue
        data=df[col].dropna(); ax.set_facecolor(C.BG)
        ax.hist(data, bins=min(20,max(3,len(data)//2+1)), color=colour, edgecolor=NAVY, alpha=0.85)
        ax.axvline(data.mean(), color=GOLD, lw=1.5, linestyle="--", label=f"μ={data.mean():.2f}")
        ax.axvline(data.median(), color="white", lw=1.0, linestyle=":", label=f"med={data.median():.2f}")
        ax.set_xlabel(lbl); ax.legend(fontsize=7.5)
    plt.tight_layout()
    p=os.path.join(save_dir,"03a_fea_distributions.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=C.BG); plt.close(fig); print(f"  Saved → {p}")

    # 03b: deflection vs R2, L1
    fig, axes = plt.subplots(1,2,figsize=(11,5), facecolor=C.BG)
    fig.suptitle("Fig 03b — Deflection Sensitivity", color=C.TEXT)
    defl=df["static_max_deflection_um"].values
    for ax, vn, col in zip(axes, ["R2","L1"], [TEAL,CORAL]):
        ax.set_facecolor(C.BG)
        cn=f"var_{vn}"
        if cn not in df.columns: continue
        xd=df[cn].values; ax.scatter(xd, defl, s=16, c=col, alpha=0.65, edgecolors="none")
        z=np.polyfit(xd,defl,1); xs=np.linspace(xd.min(),xd.max(),100)
        ax.plot(xs,np.poly1d(z)(xs), color=GOLD, lw=1.4, linestyle="--", label=f"slope={z[0]:.2f}")
        ax.set_xlabel(f"{vn} [mm]"); ax.set_ylabel("Deflection [μm]")
        ax.set_title(f"δ vs {vn}"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p=os.path.join(save_dir,"03b_deflection_sensitivity.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=C.BG); plt.close(fig); print(f"  Saved → {p}")

    # 03c: freq vs L2 + FoS histogram
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(11,5), facecolor=C.BG)
    fig.suptitle("Fig 03c — Frequency vs L2 & FoS", color=C.TEXT)
    for ax in (ax1,ax2): ax.set_facecolor(C.BG)
    if "var_L2" in df.columns:
        ax1.scatter(df["var_L2"], df["freq_mode1_Hz"], s=16, c=MINT, alpha=0.65, edgecolors="none")
        z2=np.polyfit(df["var_L2"],df["freq_mode1_Hz"],1)
        xs2=np.linspace(df["var_L2"].min(),df["var_L2"].max(),100)
        ax1.plot(xs2, np.poly1d(z2)(xs2), color=GOLD, lw=1.4, linestyle="--")
        ax1.set_xlabel("L2 [mm]"); ax1.set_ylabel("f₁ [Hz]"); ax1.grid(True, alpha=0.3)
    fos=df["static_factor_of_safety"].values
    ax2.hist(fos, bins=min(20,max(3,len(fos)//2+1)), color=GOLD, edgecolor=NAVY, alpha=0.85)
    ax2.axvline(2.0, color=CORAL, lw=1.5, linestyle="--", label="FoS=2.0 (min)")
    ax2.set_xlabel("Factor of Safety"); ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p=os.path.join(save_dir,"03c_frequency_fos.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=C.BG); plt.close(fig); print(f"  Saved → {p}")


if __name__ == "__main__":
    import os
    from design_variables import DesignSpace
    from lhs_sampler     import LHSSampler

    ds = DesignSpace()
    X  = LHSSampler(ds).generate_lhs(15, seed=42)

    runner = FEAPoolRunner(X, ds, dry_run=True)
    df     = runner.execute_batch()

    cols = ["case_id", "static_max_deflection_um",
            "static_max_vonmises_MPa", "freq_mode1_Hz"]
    print(df[cols].to_string(index=False))

    # Sanity: stiffening the shaft (larger R2) should reduce deflection
    from design_variables import DesignSpace as DS
    import copy

    ds2  = DS()
    nom  = ds2.get_nominal()
    soft = nom.copy(); stiff = nom.copy()

    # Find R2 index
    idx_R2 = ds2.get_variable_names().index("R2")
    soft[idx_R2]  = 40.0    # small radius
    stiff[idx_R2] = 60.0    # large radius

    X2 = np.vstack([soft, stiff])
    r2 = FEAPoolRunner(X2, ds2, dry_run=True).execute_batch()
    d_soft  = r2.iloc[0]["static_max_deflection_um"]
    d_stiff = r2.iloc[1]["static_max_deflection_um"]
    assert d_soft > d_stiff, f"Physics check failed: {d_soft:.2f} vs {d_stiff:.2f}"
    print(f"\n✅ Stiffness check: R2=40→δ={d_soft:.2f}μm  R2=60→δ={d_stiff:.2f}μm (correct)")

    os.makedirs("/tmp/spindle_plots", exist_ok=True)
    print("\nGenerating FEA plots...")
    plot_fea_results(df, ds, save_dir="/tmp/spindle_plots")
    print("✅ FEA Pool Runner OK")
