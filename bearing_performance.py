#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  Bearing Performance Calculator v3
================================================================================

  Scope (post-refactor):
      This module handles ONLY bearing-mechanics quantities that are computed
      independently of ANSYS:

        1. SKF catalog bearing resolution (snap-to-nearest)
        2. Force distribution across stations (static equilibrium)
        3. Pair stiffness — K_radial, K_axial per station
        4. Preload force & spacer delta
        5. Speed & ndm checks
        6. ISO 281 L10 life (per bearing + system combined)
        7. ANSYS COMBIN14 spring table export

  Quantities moved to dedicated modules:
        Runout      → 09_shaft_runout.py    (analytical + ANSYS overlay)
        Eccentricity→ 10_rotor_eccentricity.py  (CG-based, ISO 1940-1)
        Vibration   → ANSYS (harmonic / transient analysis)

  Standards:
        ISO 281:2007        Dynamic bearing load ratings and rating life
        SKF General Catalogue 6000/EN  (stiffness empirical fits)
================================================================================
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from plot_theme import apply_paper_theme, C, savefig_paper

from design_variables import (
    DesignSpace,
    SpindleBearingArrangement,
    BearingStation,
    BearingRecord,
    SKFBearing,
    SKFCRBBearing,
    SKF_ACBB_CATALOG,
    SKF_CRB_CATALOG,
)

NDM_GREASE_LIMIT = 500_000.0   # mm·RPM
NDM_OIL_LIMIT    = 1_000_000.0


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StationState:
    """Per-station resolved bearing state at one operating point."""
    station:             BearingStation
    bearing:             BearingRecord
    n_bearings:          int
    z_positions_mm:      List[float]
    radial_load_N:       float   # total for this station
    axial_load_N:        float   # non-zero only for front (locating) station
    load_per_bearing_N:  float
    K_radial_pair_N_mm:  float   # N/mm (pair stiffness)
    K_axial_pair_N_mm:   float   # N/mm (0 for CRB stations)
    F_preload_N:         float   # 0 for CRB
    delta_spacer_mm:     float   # spacer length difference for axial preload
    ndm:                 float   # pitch-diameter × speed [mm·RPM]
    speed_ok:            bool
    speed_warning:       str
    P_equiv_N:           float   # ISO 281 equiv. load (per bearing)
    L10_single_hours:    float
    L10_station_hours:   float

    @property
    def bearing_type(self) -> str:
        return self.bearing.bearing_type


@dataclass
class BearingSystemState:
    """Complete multi-station bearing state."""
    n_rpm:            float
    stations:         List[StationState]
    L10_system_hours: float
    L10_target_hours: float
    all_speeds_ok:    bool


@dataclass
class BearingConstraints:
    """
    Signed constraint array.  g ≤ 0 → satisfied.

    Dimensions:
        g_speed     : (n_stations,)
        g_ndm       : (n_stations,)
        g_l10       : scalar
        g_preload_lo: scalar   (front ACBB only)
        g_preload_hi: scalar
    """
    g_speed:      np.ndarray
    g_ndm:        np.ndarray
    g_l10:        float
    g_preload_lo: float
    g_preload_hi: float

    @property
    def as_array(self) -> np.ndarray:
        return np.concatenate([
            self.g_speed, self.g_ndm,
            [self.g_l10, self.g_preload_lo, self.g_preload_hi],
        ])

    @property
    def all_satisfied(self) -> bool:
        return bool(np.all(self.as_array <= 0.0))

    def violated_names(self) -> List[str]:
        ns  = len(self.g_speed)
        nms = (
            [f"speed_st{i+1}" for i in range(ns)]
          + [f"ndm_st{i+1}"   for i in range(ns)]
          + ["L10_system", "preload_lo", "preload_hi"]
        )
        return [n for n, g in zip(nms, self.as_array) if g > 0.0]

    def penalty(self, weight: float = 1e4) -> float:
        return float(weight * np.sum(np.maximum(self.as_array, 0.0) ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# Calculator
# ─────────────────────────────────────────────────────────────────────────────

class BearingPerformanceCalculator:
    """
    Multi-station bearing mechanics (no runout / eccentricity / vibration).

    Typical usage
    -------------
    calc  = BearingPerformanceCalculator(design_space, arrangement)
    state = calc.evaluate(x, n_rpm=4000)
    con   = calc.check_constraints(state)
    tbl   = calc.stiffness_for_ansys(state)     # COMBIN14 input table
    """

    def __init__(
        self,
        design_space:      DesignSpace,
        arrangement:       Optional[SpindleBearingArrangement] = None,
        n_nom_rpm:         float = 4_000.0,
        n_max_rpm:         float = 6_000.0,
        l10_target_hours:  float = 20_000.0,
        preload_min_N:     float = 100.0,
        preload_max_N:     float = 4_000.0,
    ):
        self.ds           = design_space
        self.arr          = arrangement or SpindleBearingArrangement.default_lathe()
        self.n_nom        = n_nom_rpm
        self.n_max        = n_max_rpm
        self.l10_target   = l10_target_hours
        self.preload_min  = preload_min_N
        self.preload_max  = preload_max_N

    # ─────────────────────────────────────────────────────────────────────────
    def evaluate(
        self,
        x:     np.ndarray,
        n_rpm: Optional[float] = None,
        Fr_N:  Optional[float] = None,
        Fa_N:  Optional[float] = None,
    ) -> BearingSystemState:
        n = n_rpm if n_rpm is not None else self.n_nom
        v = self.ds.decode_vector(x)

        if Fr_N is None:
            Fr_N = float(np.sqrt(v["Ft"]**2 + v["Fr"]**2))
        if Fa_N is None:
            Fa_N = float(v["Ff"])

        # Resolve all stations to catalog bearings
        resolved   = self.ds.resolve_arrangement(x, n, arrangement=self.arr)
        L1, L2     = v["L1"], v["L2"]
        z_stations = self._station_z_coords(L1, L2)

        # Static force distribution
        radial_rxns, axial_front = self._force_distribution(Fr_N, Fa_N, z_stations)

        station_states: List[StationState] = []
        for k, (st, brg, speed_ok, warn) in enumerate(resolved):
            z_pos       = z_stations[k]
            ndm_val     = brg.ndm
            K_r, K_a    = self._pair_stiffness(st, brg)
            R_sta       = radial_rxns[k]
            load_per    = abs(R_sta) / st.n_bearings
            Fa_this     = axial_front if st.role == "front" else 0.0
            P_eq, L10s  = self._l10_single(brg, load_per, Fa_this, n)
            F_pre       = brg.F_preload_MA_N if isinstance(brg, SKFBearing) else 0.0
            d_spacer    = F_pre / max(K_a, 1.0) if K_a > 0 else 0.0

            station_states.append(StationState(
                station            = st,
                bearing            = brg,
                n_bearings         = st.n_bearings,
                z_positions_mm     = z_pos,
                radial_load_N      = abs(R_sta),
                axial_load_N       = Fa_this,
                load_per_bearing_N = load_per,
                K_radial_pair_N_mm = K_r,
                K_axial_pair_N_mm  = K_a,
                F_preload_N        = F_pre,
                delta_spacer_mm    = d_spacer,
                ndm                = ndm_val,
                speed_ok           = speed_ok,
                speed_warning      = warn,
                P_equiv_N          = P_eq,
                L10_single_hours   = L10s,
                L10_station_hours  = L10s,
            ))

        L10_sys = self._system_l10(station_states)

        return BearingSystemState(
            n_rpm            = n,
            stations         = station_states,
            L10_system_hours = L10_sys,
            L10_target_hours = self.l10_target,
            all_speeds_ok    = all(ss.speed_ok for ss in station_states),
        )

    # ─────────────────────────────────────────────────────────────────────────
    def check_constraints(self, state: BearingSystemState) -> BearingConstraints:
        ns  = len(state.stations)
        lub = self.arr.lubrication
        ndm_lim = NDM_GREASE_LIMIT if lub == "grease" else NDM_OIL_LIMIT

        g_speed = np.zeros(ns)
        g_ndm   = np.zeros(ns)
        for k, ss in enumerate(state.stations):
            n_lim      = ss.bearing.n_grease if lub == "grease" else ss.bearing.n_oil
            g_speed[k] = (state.n_rpm - n_lim) / n_lim
            g_ndm[k]   = (ss.ndm - ndm_lim)    / ndm_lim

        g_l10  = (self.l10_target - state.L10_system_hours) / self.l10_target

        front_ss = next((ss for ss in state.stations if ss.station.role == "front"), None)
        F_pre    = front_ss.F_preload_N if front_ss else self.preload_min
        g_lo     = (self.preload_min - F_pre) / max(self.preload_min, 1.0)
        g_hi     = (F_pre - self.preload_max) / max(self.preload_max, 1.0)

        return BearingConstraints(
            g_speed=g_speed, g_ndm=g_ndm,
            g_l10=g_l10, g_preload_lo=g_lo, g_preload_hi=g_hi,
        )

    # ─────────────────────────────────────────────────────────────────────────
    def stiffness_for_ansys(self, state: BearingSystemState) -> List[Dict]:
        """
        ANSYS COMBIN14 spring table: one row per bearing position.

        Each row:
            z_mm       — axial position from spindle nose [mm]
            K_radial   — radial spring stiffness [N/mm]
            K_axial    — axial spring stiffness [N/mm]  (0 for CRB)
            bearing    — SKF designation
            role       — "front" | "rear"
        """
        rows = []
        for ss in state.stations:
            n_pos  = len(ss.z_positions_mm)
            K_r_ea = ss.K_radial_pair_N_mm / max(n_pos, 1)
            K_a_ea = ss.K_axial_pair_N_mm  / max(n_pos, 1)
            for z in ss.z_positions_mm:
                rows.append({
                    "z_mm":    round(z, 2),
                    "K_radial": round(K_r_ea, 0),
                    "K_axial":  round(K_a_ea, 0) if ss.station.role == "front" else 0.0,
                    "bearing":  ss.bearing.designation,
                    "role":     ss.station.role,
                    "bearing_type": ss.bearing_type,
                })
        return sorted(rows, key=lambda r: r["z_mm"])

    # ─────────────────────────────────────────────────────────────────────────
    # Internal mechanics
    # ─────────────────────────────────────────────────────────────────────────
    def _station_z_coords(self, L1: float, L2: float) -> List[List[float]]:
        return [st.z_positions_mm(L1, L2) for st in self.arr.stations]

    def _force_distribution(
        self,
        Fr_N: float,
        Fa_N: float,
        z_stations: List[List[float]],
    ) -> Tuple[List[float], float]:
        """
        Static equilibrium for propped-cantilever spindle.

        Two-station case (front + rear):
            Moment equilibrium about rear centroid z_r:
                R_front = Fr × z_r / (z_r − z_f)
                R_rear  = Fr − R_front

        Three-station case:
            R_front vs. weighted centroid of remaining stations.
            Remaining stations share R_rear proportional to their K_radial.
        """
        z_cents = [float(np.mean(zs)) for zs in z_stations]
        n_st    = len(z_cents)

        if n_st == 1:
            return [Fr_N], Fa_N

        if n_st == 2:
            z_f, z_r = z_cents
            span = max(z_r - z_f, 1.0)
            R_f  = Fr_N * z_r / span
            R_r  = Fr_N - R_f
            return [R_f, R_r], Fa_N

        # ≥3 stations
        z_rear_c = float(np.mean(z_cents[1:]))
        span     = max(z_rear_c - z_cents[0], 1.0)
        R_f      = Fr_N * z_rear_c / span
        R_r_tot  = Fr_N - R_f

        rear_K = []
        for k in range(1, n_st):
            st  = self.arr.stations[k]
            brg = (SKF_ACBB_CATALOG[len(SKF_ACBB_CATALOG)//2]
                   if st.bearing_type == "ACBB" else
                   SKF_CRB_CATALOG[len(SKF_CRB_CATALOG)//2])
            rear_K.append(brg.radial_stiffness_single_N_mm * st.stiffness_factor)

        K_tot     = max(sum(rear_K), 1.0)
        reactions = [R_f] + [R_r_tot * K_k / K_tot for K_k in rear_K]
        return reactions, Fa_N

    def _pair_stiffness(self, st: BearingStation, brg: BearingRecord) -> Tuple[float, float]:
        """
        Empirical stiffness fitted to SKF spindle catalogue (P5, MA preload):

        ACBB:
            K_r_single [N/μm] = 5.5 × d^0.75   (±15 %)
            K_a_single [N/μm] = K_r × tan²(α)  α = 25°
        CRB:
            K_r_single [N/μm] = 8.0 × d^0.80   (line contact, ±20 %)
            K_a = 0   (NU type, free axially)

        Pair factor f_pair:
            single       : 1.0
            DB / DF      : 1.7
            spacer_pair  : 2.0
            DT           : 1.5
        """
        K_r_s = brg.radial_stiffness_single_N_mm
        f     = st.stiffness_factor
        K_r   = K_r_s * f

        if isinstance(brg, SKFBearing):
            alpha = math.radians(brg.contact_angle_deg)
            K_a   = K_r_s * math.tan(alpha) ** 2 * f
        else:
            K_a   = 0.0

        return K_r, K_a

    def _l10_single(
        self,
        brg:   BearingRecord,
        Fr_N:  float,
        Fa_N:  float,
        n_rpm: float,
    ) -> Tuple[float, float]:
        """
        ISO 281 equivalent dynamic load and basic rating life.

        ACBB (25°, single row):
            Ratio factor e = 0.68  (from SKF catalogue, 25° contact angle)
            If Fa/Fr ≤ e:  P = Fr
            If Fa/Fr > e:  P = 0.41 Fr + 0.87 Fa   (X2, Y2 factors)
            Exponent p = 3 (ball bearing)

        CRB (NU type):
            P = Fr   (pure radial; no axial capacity)
            Exponent p = 10/3 (roller bearing)

        Life equation:
            L10 [revolutions] = (C_r / P)^p × 10^6
            L10h [hours]      = L10 / (60 × n)
        """
        C_r = brg.C_r

        if isinstance(brg, SKFBearing):
            e_r, X2, Y2 = 0.68, 0.41, 0.87
            P = X2 * Fr_N + Y2 * Fa_N if Fa_N / max(Fr_N, 1.0) > e_r else Fr_N
            p = 3.0
        else:
            P = Fr_N
            p = 10.0 / 3.0

        P        = max(P, 0.01 * C_r)
        L10_rev  = (C_r / P) ** p * 1.0e6
        L10h     = L10_rev / (60.0 * n_rpm) if n_rpm > 0 else 0.0
        # NOTE (P2): ISO 281:2007 Annex C defines MODIFIED rating life L10m:
        #   L10m = a1 × aISO × L10
        # where a_ISO accounts for lubrication (κ=ν/ν₁), contamination (eC),
        # and fatigue limit load. For well-lubricated spindle bearings (κ≥1,
        # clean oil-air), a_ISO can be 3–10×, giving L10m >> L10.
        # This code reports basic L10 (a_ISO = 1.0 — conservative).
        return float(P), float(L10h)

    def _system_l10(self, sst: List[StationState]) -> float:
        """
        ISO 281 combined system life for n independent bearings.

            1 / L10_sys^e = Σ_i (1 / L10_i^e)

        e = 10/9 (ball)  or  9/8 (roller).
        Mix: use ball exponent (conservative).
        """
        e   = 10.0 / 9.0
        inv = 0.0
        for ss in sst:
            e_k = 10.0 / 9.0 if isinstance(ss.bearing, SKFBearing) else 9.0 / 8.0
            L10 = max(ss.L10_station_hours, 1.0)
            for _ in range(ss.n_bearings):
                inv += (1.0 / L10) ** e_k
        return float((1.0 / inv) ** (1.0 / e)) if inv > 0 else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def print_report(state: BearingSystemState, con: BearingConstraints) -> None:
        sep = "─" * 70
        print(f"\n{'═'*70}")
        print(f"  Bearing System  @ {state.n_rpm:.0f} RPM")
        print(f"{'═'*70}")
        for k, ss in enumerate(state.stations):
            b = ss.bearing
            print(f"\n  Station {k+1}: {ss.station.role.upper()} | "
                  f"{ss.n_bearings}× {ss.bearing_type} [{ss.station.sub_arrangement}]")
            print(f"  {sep}")
            print(f"    {b.designation:<20}  d={b.d}mm  D={b.D}mm  C_r={b.C_r/1e3:.1f}kN")
            print(f"    z = {[round(z,1) for z in ss.z_positions_mm]} mm")
            print(f"    R_radial  = {ss.radial_load_N:>8.1f} N  ({ss.load_per_bearing_N:.1f} N/brg)")
            if ss.axial_load_N:
                print(f"    R_axial   = {ss.axial_load_N:>8.1f} N")
            print(f"    K_r (pair)= {ss.K_radial_pair_N_mm/1e3:>8.1f} N/μm")
            print(f"    K_a (pair)= {ss.K_axial_pair_N_mm/1e3:>8.1f} N/μm")
            if ss.F_preload_N:
                print(f"    F_preload = {ss.F_preload_N:>8.0f} N  "
                      f"(Δspacer={ss.delta_spacer_mm*1e3:.2f} μm)")
            print(f"    ndm = {ss.ndm/1e3:.0f} ×10³ mm·RPM")
            print(f"    P   = {ss.P_equiv_N:>8.0f} N   L10 = {ss.L10_station_hours:>10,.0f} h")
            if ss.speed_warning:
                print(f"    ⚠️  {ss.speed_warning}")
        print(f"\n  {sep}")
        print(f"  System L10 = {state.L10_system_hours:>10,.0f} h  "
              f"(target {state.L10_target_hours:.0f} h — "
              f"{'OK ✅' if state.L10_system_hours >= state.L10_target_hours else '❌ LOW'})")
        print(f"\n  Constraints:")
        for nm, g in zip(con.violated_names() + [""], con.as_array):
            pass  # just use the array directly
        all_nms = (
            [f"speed_st{i+1}" for i in range(len(state.stations))]
          + [f"ndm_st{i+1}"   for i in range(len(state.stations))]
          + ["L10_system", "preload_lo", "preload_hi"]
        )
        for nm, gi in zip(all_nms, con.as_array):
            tag = "✅" if gi <= 0 else "❌"
            print(f"    {tag} {nm:<18} g = {gi:+.4f}")
        print(f"\n  {'ALL OK ✅' if con.all_satisfied else 'VIOLATIONS: ' + str(con.violated_names())}")
        print(f"{'═'*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_bearing_performance(
    calc:      "BearingPerformanceCalculator",
    ds:        "DesignSpace",
    x_nominal: np.ndarray,
    speeds:    Optional[List[float]] = None,
    save_dir:  str = ".",
) -> None:
    """
    Three bearing-performance plots:

    Fig 08a — L10 system life vs. operating speed
    Fig 08b — Bearing stiffness (K_radial, K_axial) per station @ nominal
    Fig 08c — Constraint bar chart @ nominal 4,000 RPM
    """
    import matplotlib.pyplot as plt
    import os

    NAVY=C.NAVY; TEAL=C.TEAL; CORAL=C.RED; GOLD=C.ORANGE
    MINT=C.GREEN; GRAY=C.GRAY; PURPLE=C.PURPLE
    os.makedirs(save_dir, exist_ok=True)

    apply_paper_theme()

    if speeds is None:
        speeds = [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]

    # ── Fig 08a: L10 vs speed ─────────────────────────────────────────────
    l10_vals = []
    for n in speeds:
        s = calc.evaluate(x_nominal, n_rpm=float(n))
        l10_vals.append(s.L10_system_hours)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=C.BG)
    ax.set_facecolor(C.BG)
    ax.plot(speeds, [v/1000 for v in l10_vals], color=TEAL, lw=2, marker="o", ms=5)
    ax.axhline(calc.l10_target/1000, color=GOLD, lw=1.5, linestyle="--",
               label=f"Target L10 = {calc.l10_target/1000:.0f} kh")
    ax.axvline(4000, color=GRAY, lw=0.8, linestyle=":", alpha=0.7, label="4,000 RPM nominal")
    ax.axvline(6000, color=CORAL, lw=0.8, linestyle=":", alpha=0.7, label="6,000 RPM max")
    ax.set_xlabel("Operating speed [RPM]")
    ax.set_ylabel("System L10 life [×10³ hours]")
    ax.set_title("Fig 08a — ISO 281 System L10 Life vs. Speed", pad=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(save_dir, "08a_l10_vs_speed.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=C.BG)
    plt.close(fig); print(f"  Saved → {p}")

    # ── Fig 08b: Stiffness per station ────────────────────────────────────
    s4     = calc.evaluate(x_nominal, n_rpm=4000)
    st_labels = [
        f"St{k+1} {ss.station.role} {ss.bearing_type}"
        for k, ss in enumerate(s4.stations)
    ]
    kr_vals = [ss.K_radial_pair_N_mm / 1000 for ss in s4.stations]   # N/μm
    ka_vals = [ss.K_axial_pair_N_mm  / 1000 for ss in s4.stations]

    x_pos = np.arange(len(st_labels))
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=C.BG)
    ax.set_facecolor(C.BG)
    bars_r = ax.bar(x_pos - 0.2, kr_vals, 0.35, label="K_radial [N/μm]",
                    color=TEAL, edgecolor=NAVY, linewidth=0.5)
    bars_a = ax.bar(x_pos + 0.2, ka_vals, 0.35, label="K_axial [N/μm]",
                    color=CORAL, edgecolor=NAVY, linewidth=0.5)
    for bar in list(bars_r) + list(bars_a):
        h = bar.get_height()
        if h > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=8, color="white")
    ax.set_xticks(x_pos); ax.set_xticklabels(st_labels, fontsize=9)
    ax.set_ylabel("Stiffness [N/μm]")
    ax.set_title("Fig 08b — Bearing Pair Stiffness per Station @ 4,000 RPM", pad=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(save_dir, "08b_stiffness_per_station.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=C.BG)
    plt.close(fig); print(f"  Saved → {p}")

    # ── Fig 08c: Constraint satisfaction bar chart ────────────────────────
    c4 = calc.check_constraints(s4)
    g_arr  = c4.as_array
    ns = len(s4.stations)
    c_names = (
        [f"speed st{i+1}" for i in range(ns)]
      + [f"ndm st{i+1}"   for i in range(ns)]
      + ["L10 sys", "preload lo", "preload hi"]
    )
    colours_c = [CORAL if gi > 0 else TEAL for gi in g_arr]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=C.BG)
    ax.set_facecolor(C.BG)
    ax.barh(c_names, g_arr, color=colours_c, edgecolor=NAVY, linewidth=0.4)
    ax.axvline(0, color=GOLD, lw=1.5, linestyle="--")
    ax.set_xlabel("Constraint value g  (g ≤ 0 = satisfied)")
    ax.set_title("Fig 08c — Bearing Constraint Values @ 4,000 RPM  (g ≤ 0 = ✅)", pad=10)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(save_dir, "08c_constraints.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=C.BG)
    plt.close(fig); print(f"  Saved → {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── Also run the manufacturer comparison plot ───────────────────────────
    try:
        from bearing_catalog import compare_manufacturers
        import matplotlib.pyplot as plt
        from plot_theme import apply_paper_theme, C as PT, savefig_paper
        apply_paper_theme()

        bore_test = 110.0   # typical CNC lathe front spindle bore
        rows_acbb = compare_manufacturers(bore_test, "ACBB", n_rpm=4000.0)
        rows_crb  = compare_manufacturers(bore_test, "CRB",  n_rpm=4000.0)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=PT.BG)
        colours   = [PT.BLUE, PT.RED, PT.ORANGE, PT.GREEN, PT.PURPLE, PT.GRAY]

        for ax, rows, title, label in [
            (axes[0], rows_acbb, f"ACBB  Ø{bore_test:.0f}mm bore",  "C_r [kN]"),
            (axes[1], rows_crb,  f"CRB   Ø{bore_test:.0f}mm bore", "C_r [kN]"),
        ]:
            ax.set_facecolor(PT.BG)
            mfrs = [r["manufacturer"] for r in rows]
            vals = [r["C_r_kN"]       for r in rows]
            bars = ax.barh(mfrs, vals,
                           color=[colours[i % len(colours)] for i in range(len(mfrs))],
                           edgecolor=PT.NAVY, linewidth=0.6)
            for bar, val in zip(bars, vals):
                ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f} kN", va="center", fontsize=8.5, color=PT.TEXT)
            ax.set_xlabel(label); ax.set_title(title)
            # Speed-limit indicators
            for i, r in enumerate(rows):
                clr = PT.GREEN if r["speed_ok"] else PT.RED
                ax.text(0.98, i, "✅" if r["speed_ok"] else "❌",
                        transform=ax.get_yaxis_transform(),
                        ha="right", va="center", fontsize=11, color=clr)

        fig.suptitle(f"Manufacturer Comparison — n = 4000 rpm\n"
                     f"(sorted by C_r; ✅/❌ = grease speed OK at 4000 rpm)",
                     fontweight="bold")
        plt.tight_layout()
        savefig_paper(fig, "/tmp/08d_manufacturer_comparison.png")
        plt.close(fig)
        print("Manufacturer comparison plot: /tmp/08d_manufacturer_comparison.png")
    except Exception as e:
        print(f"Manufacturer comparison plot skipped: {e}")
    import sys; sys.path.insert(0, ".")
    from design_variables import DesignSpace, SpindleBearingArrangement
    import os

    ds   = DesignSpace()
    arr  = SpindleBearingArrangement.default_lathe()
    calc = BearingPerformanceCalculator(ds, arr, n_nom_rpm=4000, n_max_rpm=6000)
    nom  = ds.get_nominal()

    s4 = calc.evaluate(nom, n_rpm=4000)
    c4 = calc.check_constraints(s4)
    BearingPerformanceCalculator.print_report(s4, c4)

    s6 = calc.evaluate(nom, n_rpm=6000)
    c6 = calc.check_constraints(s6)
    BearingPerformanceCalculator.print_report(s6, c6)

    springs = calc.stiffness_for_ansys(s4)
    print("ANSYS COMBIN14 spring table:")
    for r in springs:
        print(f"  z={r['z_mm']:>7.1f}mm  K_r={r['K_radial']:>8.0f}  "
              f"K_a={r['K_axial']:>8.0f}  {r['role']}  {r['bearing']}")

    assert s6.L10_system_hours < s4.L10_system_hours
    print("\n✅  L10 decreases with speed")

    os.makedirs("/tmp/spindle_plots", exist_ok=True)
    print("\nGenerating bearing performance plots...")
    plot_bearing_performance(calc, ds, nom,
                             speeds=[1000,2000,3000,4000,5000,6000],
                             save_dir="/tmp/spindle_plots")
    print("✅  Module 08 — all checks + plots done")
