#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  Shaft Runout Analysis v3 — Module 09
================================================================================

  New in v3:
      • Source 5 — Thermal runout  (differential thermal expansion)
      • Source 6 — Positional tolerance contribution (ISO 1101 ⊕)
      • plot_runout_breakdown()  — stacked bar chart of all 6 sources
      • plot_runout_vs_speed()   — thermal runout as function of speed
      • plot_tir_sensitivity()   — tornado chart: which source drives TIR most

  Sources of Runout (complete model):
  ─────────────────────────────────────
    1. Bearing inner-ring radial runout  (ISO 492, precision class)
    2. Shaft straightness bow            (ISO 1101 form tolerance)
    3. ANSYS elastic deflection          (static under cutting forces)
    4. Bending slope amplification       (slope × overhang)
    5. Thermal differential expansion    (NEW — shaft vs housing)
    6. Positional tolerance              (NEW — ISO 1101 ⊕ bearing seat)

  Thermal Runout Model (Source 5):
  ─────────────────────────────────
    During machining the spindle shaft heats up while the cast-iron housing
    heats more slowly.  The temperature difference ΔT causes a differential
    radial expansion of the shaft:

        Δr = α_shaft × R_shaft × ΔT   [mm]

    where:
        α_shaft  = CTE of shaft material (steel ≈ 12 × 10⁻⁶ /°C)
        R_shaft  = shaft radius at bearing seat [mm]
        ΔT       = temperature rise above ambient [°C]
                   (depends on speed, lubrication, preload)

    Speed-dependent ΔT model (empirical for grease-lubricated ACBB):
        ΔT(n) ≈ k_heat × n^0.7 × μ_preload   [°C]
        k_heat    = 0.0012  (grease, MA preload)
        n         = speed [RPM]
        μ_preload = 1.0 (LA), 1.4 (MA), 2.0 (HA)

    This thermal radial growth shifts the journal centre relative to the
    housing, directly contributing to spindle nose runout via the same
    amplification factor as bearing eccentricity:

        TIR_thermal = Δr_front × (1 + L_oh / L_span)   [μm]

    Standard reference: ISO 230-3:2020 — Test code for machine tools —
    Part 3: Determination of thermal effects.

  Positional Tolerance Model (Source 6):
  ─────────────────────────────────────
    ISO 1101 positional tolerance ⊕ defines a cylindrical zone of diameter
    ϕ_pos within which the actual bearing seat axis must lie.
    Maximum radial offset of seat centreline from nominal spindle axis:
        e_pos = ϕ_pos / 2   [mm]

    This offset is amplified to the nose by the same lever formula as the
    bearing inner-ring runout (Source 1):
        TIR_pos_front = e_pos_front × (1 + L_oh / L_span)
        TIR_pos_rear  = e_pos_rear  × L_oh / L_span
            (rear seat tilt effect, smaller than front)

    Standard reference: ISO 1101:2017 §15 — Positional tolerances.
================================================================================
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

# Bearing precision class inner-ring radial runout [μm] per ISO 492:2014
BEARING_RUNOUT_UM: Dict[str, float] = {
    "P0": 15.0, "P6": 10.0, "P5": 5.0, "P4": 2.5, "P2": 1.5,
}

SHAFT_STRAIGHTNESS_MM_PER_MM: Dict[str, float] = {
    "standard":  4e-5,   # ISO 2768-1 fine
    "precision": 1e-5,   # P5 ground shaft
    "ultra":     5e-6,   # P4/P2 lapped
}

# Thermal expansion coefficients [/°C]
CTE_STEEL_PER_C  = 12.0e-6   # shaft (4140/4340 steel)
CTE_CI_PER_C     = 10.0e-6   # cast-iron housing (often neglected, adds conservatism)

# Grease-lubricated empirical heat model: ΔT ≈ k × n^0.7 × μ_preload
# k fitted to published SKF spindle thermal rise data (P5, MA, d=80–120 mm)
_HEAT_K   = 0.0012
_MU_PRELOAD = {"LA": 1.0, "MA": 1.4, "HA": 2.0}


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunoutBreakdown:
    """All 6 TIR source contributions at the spindle nose [μm]."""

    # Source 1 — Bearing inner-ring (ISO 492)
    TIR_bearing_um:      float
    # Source 2 — Shaft straightness bow (ISO 1101)
    TIR_straightness_um: float
    # Source 3 — ANSYS elastic deflection
    TIR_elastic_um:      float
    # Source 4 — Bending slope amplification
    TIR_slope_um:        float
    # Source 5 — Thermal differential expansion (NEW)
    TIR_thermal_um:      float
    # Source 6 — Positional tolerance front seat (ISO 1101 ⊕) (NEW)
    TIR_pos_front_um:    float
    # Source 6b — Positional tolerance rear seat (tilt contribution) (NEW)
    TIR_pos_rear_um:     float

    # Combinations
    TIR_linear_um:       float   # worst-case sum of all 6
    TIR_rss_um:          float   # RSS of all 6

    # Inputs for traceability
    delta_ir_um:          float
    L_overhang_mm:        float
    L_span_mm:            float
    amp_factor:           float
    precision_class:      str
    straightness_grade:   str
    ansys_deflection_um:  float
    delta_T_C:            float   # thermal rise used
    pos_tol_front_mm:     float   # ϕ_pos front [mm]
    pos_tol_rear_mm:      float   # ϕ_pos rear  [mm]

    @property
    def TIR_geometric_um(self) -> float:
        """All analytical sources except ANSYS (pre-FEA estimate)."""
        return math.sqrt(
            self.TIR_bearing_um**2 + self.TIR_straightness_um**2
            + self.TIR_thermal_um**2
            + self.TIR_pos_front_um**2 + self.TIR_pos_rear_um**2
        )

    @property
    def TIR_loaded_um(self) -> float:
        """Full loaded TIR including ANSYS deflection (RSS)."""
        return self.TIR_rss_um

    @property
    def sources_dict(self) -> Dict[str, float]:
        return {
            "Bearing (ISO 492)":  self.TIR_bearing_um,
            "Straightness":       self.TIR_straightness_um,
            "ANSYS deflection":   self.TIR_elastic_um,
            "Bending slope":      self.TIR_slope_um,
            "Thermal ΔT":         self.TIR_thermal_um,
            "Pos.tol front":      self.TIR_pos_front_um,
            "Pos.tol rear":       self.TIR_pos_rear_um,
        }


@dataclass
class RunoutConstraints:
    g_tir_geometric: float
    g_tir_loaded:    float
    limit_geometric_um: float
    limit_loaded_um:    float

    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.g_tir_geometric, self.g_tir_loaded])

    @property
    def all_satisfied(self) -> bool:
        return bool(np.all(self.as_array <= 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Thermal ΔT helper
# ─────────────────────────────────────────────────────────────────────────────

def estimate_delta_T(
    n_rpm:          float,
    preload_class:  str  = "MA",
    lubrication:    str  = "grease",
    bearing_d_mm:   float = 80.0,
) -> float:
    """
    Empirical temperature rise of spindle bearing zone above ambient [°C].

    Model:  ΔT = k × n^0.7 × μ_preload × f_lub

    Fitted to SKF published thermal data for angular contact ball bearings
    (P5, bore 60–120 mm, grease lubrication):
        k = 0.0012  for grease, bore 80 mm

    f_lub:
        grease   : 1.0
        oil-air  : 0.5  (oil-air removes ~50% of heat)

    Note: For detailed thermal analysis, use ANSYS Steady-State Thermal
    with convection BCs on the housing.  This formula is for quick estimation.

    Standard reference: ISO 230-3:2020 §5.3 — Thermal drift.
    """
    mu  = _MU_PRELOAD.get(preload_class, 1.4)
    f_l = 0.5 if lubrication == "oil" else 1.0
    # Scale k slightly with bearing bore (larger bore → more heat)
    k   = _HEAT_K * (bearing_d_mm / 80.0) ** 0.3
    return k * (n_rpm ** 0.7) * mu * f_l


# ─────────────────────────────────────────────────────────────────────────────
# Main analyser
# ─────────────────────────────────────────────────────────────────────────────

class ShaftRunoutAnalyser:
    """
    6-source shaft runout calculator including thermal and positional tolerance.

    Parameters
    ----------
    precision_class      : ISO 492 bearing precision class
    straightness_grade   : "standard" | "precision" | "ultra"
    tir_geometric_limit  : Max TIR from geometry+thermal (pre-FEA) [μm]
    tir_loaded_limit     : Max TIR under cutting load (post-FEA) [μm]
    cte_shaft            : CTE of shaft material [/°C]   default steel
    preload_class        : "LA" | "MA" | "HA"  for thermal model
    lubrication          : "grease" | "oil"
    """

    def __init__(
        self,
        precision_class:     str   = "P5",
        straightness_grade:  str   = "precision",
        tir_geometric_limit: float = 10.0,
        tir_loaded_limit:    float = 20.0,    # Class B CNC lathe (ISO 230-1)
        cte_shaft:           float = CTE_STEEL_PER_C,
        preload_class:       str   = "MA",
        lubrication:         str   = "grease",
    ):
        self.prec_class         = precision_class
        self.straightness_grade = straightness_grade
        self.tir_geom_limit     = tir_geometric_limit
        self.tir_loaded_limit   = tir_loaded_limit
        self.cte_shaft          = cte_shaft
        self.preload_class      = preload_class
        self.lubrication        = lubrication
        self.delta_ir           = BEARING_RUNOUT_UM.get(precision_class, 5.0)
        self.t_s                = SHAFT_STRAIGHTNESS_MM_PER_MM.get(straightness_grade, 1e-5)

    # ─────────────────────────────────────────────────────────────────────────
    def analyse(
        self,
        var_dict:            Dict[str, float],
        z_front_bearing_mm:  float,
        z_rear_bearing_mm:   float,
        delta_nose_ansys_um: float = 0.0,
        Fr_N:                float = 0.0,
        n_rpm:               float = 4000.0,
        delta_T_override:    Optional[float] = None,
    ) -> RunoutBreakdown:
        """
        Compute full 6-source runout breakdown.

        Parameters
        ----------
        var_dict              : Decoded design vector (must contain pos_tol_front,
                                pos_tol_rear, R2, E, R1, ri, L1-L4)
        z_front_bearing_mm    : Absolute z of front bearing from nose [mm]
        z_rear_bearing_mm     : Absolute z of rear bearing centroid [mm]
        delta_nose_ansys_um   : ANSYS elastic deflection at nose [μm]
        Fr_N                  : Radial cutting force [N]
        n_rpm                 : Operating speed [RPM]  (used for thermal model)
        delta_T_override      : If set, override thermal model [°C]
        """
        L_oh   = z_front_bearing_mm
        L_span = max(z_rear_bearing_mm - z_front_bearing_mm, 1.0)
        L_total = (var_dict["L1"] + var_dict["L2"]
                   + var_dict["L3"] + var_dict["L4"])
        amp = 1.0 + L_oh / L_span

        # ── Source 1: Bearing inner-ring runout (ISO 492) ─────────────────
        TIR_bearing = self.delta_ir * amp

        # ── Source 2: Shaft straightness bow (ISO 1101) ───────────────────
        f_bow        = self.t_s * L_total / 2.0
        TIR_straight = f_bow * (L_oh / (L_total / 2.0)) * 1000.0   # μm

        # ── Source 3: ANSYS elastic deflection ────────────────────────────
        TIR_elastic = abs(delta_nose_ansys_um)

        # ── Source 4: Bending slope amplification ─────────────────────────
        E  = var_dict.get("E", 2.1e5)
        R1 = var_dict.get("R1", 45.0)
        ri = var_dict.get("ri", 30.0)
        I  = max(math.pi / 4.0 * (R1**4 - ri**4), 1.0)
        theta_front  = (Fr_N * L_oh**2) / (2.0 * E * I)
        TIR_slope    = abs(theta_front * L_oh) * 1000.0   # μm

        # ── Source 5: Thermal differential expansion (NEW) ────────────────
        R2 = var_dict.get("R2", 50.0)   # bearing seat radius [mm]
        if delta_T_override is not None:
            dT = delta_T_override
        else:
            dT = estimate_delta_T(n_rpm, self.preload_class,
                                  self.lubrication, bearing_d_mm=R2 * 2)

        # Differential radial expansion: shaft CTE − housing CTE
        # Net Δr at bearing seat [mm]
        delta_r_mm  = (self.cte_shaft - CTE_CI_PER_C) * R2 * dT
        # Amplified to nose by same lever as bearing eccentricity
        TIR_thermal = abs(delta_r_mm) * amp * 1000.0   # μm

        # ── Source 6: Positional tolerance (ISO 1101 ⊕) ──────────────────
        # Front seat: maximum offset = ϕ_pos_front / 2
        # Amplified to nose by factor amp (same as Source 1)
        pos_front = var_dict.get("pos_tol_front", 0.010)   # mm (ϕ)
        e_pos_front = pos_front / 2.0
        TIR_pos_front = e_pos_front * amp * 1000.0   # μm

        # Rear seat: offset creates tilt about front bearing
        # Contribution at nose ≈ e_pos_rear × L_oh / L_span (tilt)
        pos_rear = var_dict.get("pos_tol_rear", 0.015)    # mm
        e_pos_rear  = pos_rear / 2.0
        TIR_pos_rear = e_pos_rear * (L_oh / L_span) * 1000.0   # μm

        # ── Combinations ──────────────────────────────────────────────────
        vals = [TIR_bearing, TIR_straight, TIR_elastic,
                TIR_slope, TIR_thermal, TIR_pos_front, TIR_pos_rear]
        TIR_linear = sum(vals)
        TIR_rss    = math.sqrt(sum(v**2 for v in vals))

        return RunoutBreakdown(
            TIR_bearing_um      = TIR_bearing,
            TIR_straightness_um = TIR_straight,
            TIR_elastic_um      = TIR_elastic,
            TIR_slope_um        = TIR_slope,
            TIR_thermal_um      = TIR_thermal,
            TIR_pos_front_um    = TIR_pos_front,
            TIR_pos_rear_um     = TIR_pos_rear,
            TIR_linear_um       = TIR_linear,
            TIR_rss_um          = TIR_rss,
            delta_ir_um         = self.delta_ir,
            L_overhang_mm       = L_oh,
            L_span_mm           = L_span,
            amp_factor          = amp,
            precision_class     = self.prec_class,
            straightness_grade  = self.straightness_grade,
            ansys_deflection_um = delta_nose_ansys_um,
            delta_T_C           = dT,
            pos_tol_front_mm    = pos_front,
            pos_tol_rear_mm     = pos_rear,
        )

    # ─────────────────────────────────────────────────────────────────────────
    def check_constraints(self, bd: RunoutBreakdown) -> RunoutConstraints:
        g_geom   = (bd.TIR_geometric_um  - self.tir_geom_limit)  / self.tir_geom_limit
        g_loaded = (bd.TIR_loaded_um     - self.tir_loaded_limit) / self.tir_loaded_limit
        return RunoutConstraints(
            g_tir_geometric    = g_geom,
            g_tir_loaded       = g_loaded,
            limit_geometric_um = self.tir_geom_limit,
            limit_loaded_um    = self.tir_loaded_limit,
        )

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def print_report(bd: RunoutBreakdown, con: RunoutConstraints) -> None:
        print(f"\n{'═'*70}")
        print(f"  Shaft Runout Analysis  (6 sources)")
        print(f"{'═'*70}")
        print(f"  Bearing class  : {bd.precision_class}  "
              f"(δ_ir = {bd.delta_ir_um:.1f} μm, ISO 492)")
        print(f"  L_overhang     : {bd.L_overhang_mm:.1f} mm")
        print(f"  L_span         : {bd.L_span_mm:.1f} mm")
        print(f"  Amp factor     : {bd.amp_factor:.3f}×")
        print(f"  Thermal ΔT     : {bd.delta_T_C:.1f} °C  (ISO 230-3 model)")
        print(f"  pos_tol_front  : ϕ{bd.pos_tol_front_mm*1000:.0f} μm  (ISO 1101 ⊕)")
        print(f"  pos_tol_rear   : ϕ{bd.pos_tol_rear_mm*1000:.0f} μm")
        print(f"  {'─'*67}")
        print(f"  {'Source':<28} {'TIR [μm]':>10}  {'% of RSS':>10}")
        rss = max(bd.TIR_rss_um, 1e-12)
        for name, val in bd.sources_dict.items():
            pct = val**2 / rss**2 * 100
            print(f"  {name:<28} {val:>10.3f}  {pct:>9.1f}%")
        print(f"  {'─'*67}")
        print(f"  {'TIR linear (worst-case)':<28} {bd.TIR_linear_um:>10.3f}")
        print(f"  {'TIR RSS (probabilistic)':<28} {bd.TIR_rss_um:>10.3f}")
        print(f"  {'TIR geometric (pre-FEA)':<28} {bd.TIR_geometric_um:>10.3f}  "
              f"(limit {con.limit_geometric_um} μm)")
        print(f"  {'TIR loaded (post-FEA)':<28} {bd.TIR_loaded_um:>10.3f}  "
              f"(limit {con.limit_loaded_um} μm)")
        print(f"\n  Constraints:")
        for nm, gi in zip(["TIR_geometric", "TIR_loaded"], con.as_array):
            print(f"    {'✅' if gi<=0 else '❌'}  {nm:<18}  g = {gi:+.4f}")
        print(f"  {'ALL OK ✅' if con.all_satisfied else 'VIOLATIONS ❌'}")
        print(f"{'═'*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def get_bearing_positions_from_design(
    var_dict: Dict[str, float],
    arrangement,
) -> Tuple[float, float]:
    L1, L2 = var_dict["L1"], var_dict["L2"]
    z_front = z_rear = None
    for st in arrangement.stations:
        z_pos = st.z_positions_mm(L1, L2)
        if st.role == "front":
            z_front = float(np.mean(z_pos))
        elif st.role == "rear":
            z_rear = float(np.mean(z_pos))
    if z_front is None or z_rear is None:
        raise ValueError("Arrangement must have front and rear stations.")
    return z_front, z_rear


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_runout_breakdown(
    bd: RunoutBreakdown,
    save_path: str = "./09a_runout_breakdown.png",
) -> None:
    """
    Fig 1 — Stacked horizontal bar: all 6 TIR source contributions.
    Shows both RSS and linear totals with limit lines.
    """
    import matplotlib.pyplot as plt

    NAVY = "#0d1b2a"; TEAL="#00b4d8"; CORAL="#e63946"; GOLD="#ffd166"
    MINT="#06d6a0"; PURPLE="#7400b8"; PINK="#f72585"; GRAY="#8d99ae"
    colours = [TEAL, MINT, GOLD, CORAL, PINK, PURPLE, GRAY]

    sources = bd.sources_dict
    names   = list(sources.keys())
    values  = list(sources.values())
    total_v = sum(values)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    facecolor=NAVY, gridspec_kw={"width_ratios":[2,1]})
    fig.suptitle("Spindle Nose TIR — 6-Source Breakdown", color="white",
                 fontsize=12, y=1.02)

    # Left: stacked bar
    left = 0.0
    for val, col, lbl in zip(values, colours, names):
        ax1.barh(["TIR Sources"], [val], left=left, color=col,
                 edgecolor=NAVY, linewidth=0.5, label=f"{lbl}: {val:.2f} μm")
        if val > 0.3:
            ax1.text(left + val/2, 0, f"{val:.2f}", ha="center", va="center",
                     fontsize=7.5, color="white", fontweight="bold")
        left += val

    ax1.axvline(bd.TIR_rss_um,    color=GOLD,  linestyle="--", lw=1.5,
                label=f"RSS  = {bd.TIR_rss_um:.2f} μm")
    ax1.axvline(bd.TIR_linear_um, color=CORAL, linestyle=":",  lw=1.5,
                label=f"Linear = {bd.TIR_linear_um:.2f} μm")
    ax1.set_facecolor("#112233"); ax1.set_xlabel("TIR contribution [μm]", color="white")
    ax1.tick_params(colors="white"); ax1.legend(fontsize=7.5, loc="lower right")

    # Right: pie of % variance contribution (RSS basis)
    pcts = [(v**2 / max(bd.TIR_rss_um**2, 1e-12))*100 for v in values]
    wedges, texts, autotexts = ax2.pie(
        pcts, labels=names, colors=colours,
        autopct=lambda p: f"{p:.0f}%" if p > 3 else "",
        textprops={"fontsize": 7, "color": "white"},
        wedgeprops={"edgecolor": NAVY, "linewidth": 0.6},
    )
    for at in autotexts:
        at.set_color("white"); at.set_fontsize(7)
    ax2.set_facecolor(NAVY)
    ax2.set_title("Variance share (RSS basis)", color="white", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_runout_vs_speed(
    var_dict:       Dict[str, float],
    z_front:        float,
    z_rear:         float,
    analyser:       "ShaftRunoutAnalyser",
    speeds:         Optional[List[float]] = None,
    delta_nose_um:  float = 0.0,
    Fr_N:           float = 0.0,
    save_path:      str   = "./09b_runout_vs_speed.png",
) -> None:
    """
    Fig 2 — TIR vs operating speed.
    Shows total TIR (RSS) and thermal component separately.
    """
    import matplotlib.pyplot as plt

    NAVY="#0d1b2a"; TEAL="#00b4d8"; CORAL="#e63946"; GOLD="#ffd166"; GRAY="#8d99ae"
    if speeds is None:
        speeds = np.linspace(500, 6000, 60).tolist()

    tir_total, tir_thermal, tir_geom = [], [], []
    for n in speeds:
        bd = analyser.analyse(var_dict, z_front, z_rear,
                              delta_nose_ansys_um=delta_nose_um, Fr_N=Fr_N, n_rpm=n)
        tir_total.append(bd.TIR_rss_um)
        tir_thermal.append(bd.TIR_thermal_um)
        tir_geom.append(bd.TIR_geometric_um)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=NAVY)
    ax.set_facecolor("#112233")
    ax.plot(speeds, tir_total,   color=TEAL,  lw=2,   label="Total TIR (RSS, with ANSYS)")
    ax.plot(speeds, tir_geom,    color=GOLD,  lw=1.5, linestyle="--", label="Geometric TIR (pre-FEA)")
    ax.fill_between(speeds, tir_thermal, 0, alpha=0.25, color=CORAL, label="Thermal contribution")
    ax.axhline(analyser.tir_loaded_limit,  color=CORAL, lw=1.2, linestyle=":", label=f"Loaded limit = {analyser.tir_loaded_limit} μm")
    ax.axhline(analyser.tir_geom_limit,    color=GOLD,  lw=1.0, linestyle=":", label=f"Geom. limit = {analyser.tir_geom_limit} μm")
    ax.axvline(4000, color=GRAY, lw=0.8, linestyle="--", alpha=0.7, label="Nominal 4,000 RPM")
    ax.axvline(6000, color=GRAY, lw=0.6, linestyle=":",  alpha=0.6, label="Max 6,000 RPM")

    ax.set_xlabel("Operating speed [RPM]", color="white")
    ax.set_ylabel("TIR at spindle nose [μm]", color="white")
    ax.set_title("Fig 2 — Spindle Nose TIR vs. Speed\n(ISO 230-3 thermal model + analytical sources)",
                 color="white")
    ax.tick_params(colors="white")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(min(speeds), max(speeds))
    ax.grid(True, alpha=0.3, color=GRAY)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig)
    print(f"  Saved → {save_path}")


def plot_tir_sensitivity(
    var_dict:    Dict[str, float],
    z_front:     float,
    z_rear:      float,
    analyser:    "ShaftRunoutAnalyser",
    n_rpm:       float = 4000.0,
    delta_nose:  float = 35.0,
    Fr_N:        float = 1581.0,
    save_path:   str   = "./09c_tir_sensitivity.png",
) -> None:
    """
    Fig 3 — Tornado sensitivity chart.
    Each source is perturbed ±20% of its nominal value.
    Shows which TIR driver has the biggest impact (ISO 230-1 §A4 method).
    """
    import matplotlib.pyplot as plt

    NAVY="#0d1b2a"; TEAL="#00b4d8"; CORAL="#e63946"; GOLD="#ffd166"; GRAY="#8d99ae"

    bd_nom = analyser.analyse(var_dict, z_front, z_rear,
                              delta_nose_ansys_um=delta_nose, Fr_N=Fr_N, n_rpm=n_rpm)
    nom_tir = bd_nom.TIR_rss_um

    # Parameters to perturb and their nominal values
    params = {
        "L_overhang (L1)":  ("L1",          +20, -20),
        "Journal R2":        ("R2",          +20, -20),
        "Bearing bore ε":    ("_bearing_ir", +20, -20),   # special: vary precision class
        "ΔT (speed/prelod)": ("_delta_T",   +50, -50),
        "ANSYS deflection":  ("_ansys",     +50, -50),
        "pos_tol_front":     ("pos_tol_front",+80,-50),
        "pos_tol_rear":      ("pos_tol_rear",+80,-50),
        "Straightness":      ("_straight",  +100,-50),
    }

    deltas_hi = []
    deltas_lo = []
    labels    = list(params.keys())

    for lbl, (key, pct_hi, pct_lo) in params.items():
        results = []
        for pct in [pct_hi, pct_lo]:
            vd_mod = dict(var_dict)
            if key.startswith("_"):
                # Special: modify analyser-internal param
                if key == "_bearing_ir":
                    frac   = 1.0 + pct / 100
                    temp_a = ShaftRunoutAnalyser(
                        precision_class     = analyser.prec_class,
                        straightness_grade  = analyser.straightness_grade,
                        tir_geometric_limit = analyser.tir_geom_limit,
                        tir_loaded_limit    = analyser.tir_loaded_limit,
                        cte_shaft           = analyser.cte_shaft,
                        preload_class       = analyser.preload_class,
                    )
                    temp_a.delta_ir = analyser.delta_ir * frac
                    bd = temp_a.analyse(vd_mod, z_front, z_rear, delta_nose, Fr_N, n_rpm)
                elif key == "_delta_T":
                    dT_mod = bd_nom.delta_T_C * (1 + pct/100)
                    bd = analyser.analyse(vd_mod, z_front, z_rear, delta_nose, Fr_N, n_rpm,
                                         delta_T_override=dT_mod)
                elif key == "_ansys":
                    bd = analyser.analyse(vd_mod, z_front, z_rear,
                                         delta_nose * (1+pct/100), Fr_N, n_rpm)
                elif key == "_straight":
                    temp_a = ShaftRunoutAnalyser(
                        precision_class=analyser.prec_class,
                        straightness_grade=analyser.straightness_grade,
                        tir_geometric_limit=analyser.tir_geom_limit,
                        tir_loaded_limit=analyser.tir_loaded_limit,
                    )
                    temp_a.t_s = analyser.t_s * (1 + pct/100)
                    bd = temp_a.analyse(vd_mod, z_front, z_rear, delta_nose, Fr_N, n_rpm)
                else:
                    bd = analyser.analyse(vd_mod, z_front, z_rear, delta_nose, Fr_N, n_rpm)
            else:
                vd_mod[key] = var_dict[key] * (1 + pct / 100)
                bd = analyser.analyse(vd_mod, z_front, z_rear, delta_nose, Fr_N, n_rpm)
            results.append(bd.TIR_rss_um - nom_tir)

        deltas_hi.append(results[0])
        deltas_lo.append(results[1])

    # Sort by max absolute swing
    swings = [abs(h) + abs(l) for h, l in zip(deltas_hi, deltas_lo)]
    order  = sorted(range(len(labels)), key=lambda i: swings[i], reverse=True)
    labels = [labels[i] for i in order]
    d_hi   = [deltas_hi[i] for i in order]
    d_lo   = [deltas_lo[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=NAVY)
    ax.set_facecolor("#112233")
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, d_hi, color=TEAL,  alpha=0.85, label="+perturbation", height=0.4,
            left=0, edgecolor=NAVY, linewidth=0.4)
    ax.barh(y_pos, d_lo, color=CORAL, alpha=0.85, label="−perturbation", height=0.4,
            left=0, edgecolor=NAVY, linewidth=0.4)
    ax.axvline(0, color="white", lw=0.8)
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9, color="white")
    ax.set_xlabel("ΔTIR (RSS) from nominal [μm]", color="white")
    ax.set_title(f"Fig 3 — TIR Sensitivity Tornado  (nominal TIR = {nom_tir:.2f} μm)",
                 color="white")
    ax.tick_params(colors="white")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3, color=GRAY)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from design_variables import DesignSpace, SpindleBearingArrangement
    import numpy as np

    ds   = DesignSpace()
    arr  = SpindleBearingArrangement.default_lathe()
    nom  = ds.decode_vector(ds.get_nominal())
    Fr_N = float(np.sqrt(nom["Ft"]**2 + nom["Fr"]**2))

    z_front, z_rear = get_bearing_positions_from_design(nom, arr)

    analyser = ShaftRunoutAnalyser(
        precision_class="P5", straightness_grade="precision",
        tir_geometric_limit=10.0, tir_loaded_limit=15.0,
        preload_class="MA", lubrication="grease",
    )

    print("=== PRE-FEA (no ANSYS) ===")
    bd0  = analyser.analyse(nom, z_front, z_rear, delta_nose_ansys_um=0, Fr_N=Fr_N, n_rpm=4000)
    con0 = analyser.check_constraints(bd0)
    ShaftRunoutAnalyser.print_report(bd0, con0)

    print("=== POST-FEA (ANSYS δ=35 μm, 4000 RPM) ===")
    bd35  = analyser.analyse(nom, z_front, z_rear, delta_nose_ansys_um=35, Fr_N=Fr_N, n_rpm=4000)
    con35 = analyser.check_constraints(bd35)
    ShaftRunoutAnalyser.print_report(bd35, con35)

    print("=== HIGH SPEED (6000 RPM) ===")
    bd6k  = analyser.analyse(nom, z_front, z_rear, delta_nose_ansys_um=35, Fr_N=Fr_N, n_rpm=6000)
    ShaftRunoutAnalyser.print_report(bd6k, analyser.check_constraints(bd6k))

    # Physics checks
    assert bd35.TIR_thermal_um > bd0.TIR_thermal_um * 0.9, "Thermal should be same (same speed)"
    assert bd6k.TIR_thermal_um > bd35.TIR_thermal_um, "Higher speed → more thermal runout"
    print("✅  Thermal runout increases with speed")
    assert bd35.TIR_loaded_um > bd0.TIR_geometric_um, "ANSYS overlay increases TIR"
    print("✅  ANSYS overlay increases loaded TIR")
    assert bd35.TIR_rss_um < bd35.TIR_linear_um, "RSS < linear"
    print("✅  RSS < linear worst-case")

    # Positional tolerance contribution is non-zero
    assert bd0.TIR_pos_front_um > 0
    assert bd0.TIR_pos_rear_um  > 0
    print(f"✅  Pos.tol front = {bd0.TIR_pos_front_um:.2f} μm")
    print(f"✅  Pos.tol rear  = {bd0.TIR_pos_rear_um:.2f} μm")

    # Plots
    print("\nGenerating runout plots...")
    import os; os.makedirs("/tmp/spindle_plots", exist_ok=True)
    plot_runout_breakdown(bd35, "/tmp/spindle_plots/09a_runout_breakdown.png")
    plot_runout_vs_speed(nom, z_front, z_rear, analyser,
                         delta_nose_um=35, Fr_N=Fr_N,
                         save_path="/tmp/spindle_plots/09b_runout_vs_speed.png")
    plot_tir_sensitivity(nom, z_front, z_rear, analyser,
                         n_rpm=4000, delta_nose=35, Fr_N=Fr_N,
                         save_path="/tmp/spindle_plots/09c_tir_sensitivity.png")
    print("✅  All 3 runout plots generated\n")
    print("✅  Module 09 v3 — all checks passed")
