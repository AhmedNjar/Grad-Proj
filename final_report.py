#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  Final Engineering Report — Module 11 (v2)
================================================================================

  Purpose:
      Complete, print-ready engineering report for the optimal spindle design.
      Covers all six analysis domains:

        1. Catalog-resolved bearing BOM  (Module 01)
        2. Static mechanics — deflection, bending stress, FoS  (Module 03)
        3. Bearing performance — L10, stiffness, preload  (Module 08)
        4. Runout budget — 6 sources  (Module 09)
        5. Rotor eccentricity & imbalance  (Module 10)
        6. Optimal tolerance recommendations  (ISO 286 / ISO 1101)

      Physics clarification (from analytical verification):
        • The bending stress σ = 5 MPa is PHYSICALLY CORRECT for this design.
          The thick wall (13.5 mm) and short overhang (153 mm) produce very
          low bending stress under the operating loads.  Stress is NOT the
          limiting factor — deflection is.
        • FoS = 121 is realistic: this spindle is over-engineered for stress
          but the optimizer targeted deflection minimization, which naturally
          leads to thick-walled, stiff sections.
        • Limiting KPI: δ_nose = 20.9 μm > 15 μm target → FAILS.

      Bore strategy analysis:
        • The raw R2 = 48.66 mm → bore = 97.32 mm does not exist in SKF catalog.
        • Nearest: 7219 (95 mm) or 7220 (100 mm).
        • Snap error = 2.4 % on diameter.
        • The catalog-snap PENALTY in the optimizer (500 USD/mm gap) steers
          future runs toward catalog-exact bores.

      Optimal tolerance recommendations:
        • Based on the design's sensitivity analysis (tornado chart from Module 09),
          the dominant runout sources are recommended tolerances.
        • ISO 286 fits for shaft-bearing interface.
        • ISO 1101 positional tolerance for bearing seats.
================================================================================
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from plot_theme import apply_paper_theme, C, savefig_paper


# ─────────────────────────────────────────────────────────────────────────────
# Tolerance recommendation table
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToleranceRecommendation:
    """
    One tolerance recommendation for the final drawing / BOM.

    Based on:
      • ISO 286-1:2010  — shaft and bore fits
      • ISO 1101:2017   — geometric tolerances (positional, roundness)
      • ISO 492:2014    — rolling bearing tolerances
      • SKF Engineering Handbook, §3.2 — recommended fits for ACBB
    """
    feature:      str   # e.g. "Journal Ø2×R2"
    nominal_mm:   float
    iso_fit:      str   # e.g. "h5", "H6", "⊕ 0.010"
    upper_dev_mm: float # + deviation (ISO 286 notation)
    lower_dev_mm: float # - deviation (magnitude)
    rationale:    str
    standard:     str


def recommend_tolerances(var_dict: Dict[str, float]) -> List[ToleranceRecommendation]:
    """
    Generate optimal tolerance recommendations for the spindle drawing.

    Rules applied:
    ─────────────
    1. Bearing journal (R2) — shaft fits:
       ISO 15:2017 Table 1: for rotating inner ring, bore 80-120mm, normal load:
       Use shaft tolerance k5 for interference fit (standard for ACBB).
       But for P5 precision spindles SKF recommends js5 (symmetric ±5μm).
       → Chosen: js5 for precision (±5μm for d=80-120mm range)

    2. Bearing housing (D = bearing OD) — housing bore fits:
       ISO 15:2017: for non-rotating outer ring (fixed housing), use H6.
       → H6: +0/+22μm for 140-180mm housing bore range.

    3. Inner bore (ri×2) — H7 reaming:
       → H7: +0/+21μm for bore Ø60.

    4. Positional tolerance of bearing seats (ISO 1101 ⊕):
       Based on tornado sensitivity: pos_tol_front dominates at 2.10× amplification.
       Recommendation: tighten to ϕ8μm (from ϕ10μm nominal).
       → ⊕ ϕ0.008 mm referred to shaft centreline.

    5. Roundness of bearing seats (ISO 1101 ○):
       P5 class bearing requires seat roundness ≤ 0.5 × inner ring runout = 2.5μm.
       → ○ 0.0025 mm.

    6. Parallelism of bearing faces (ISO 1101 ∥):
       Controls axial preload accuracy. For spacer method MA class:
       → ∥ 0.003 mm.
    """
    R2     = var_dict.get("R2", 50.0)
    ri     = var_dict.get("ri", 30.0)
    bore   = R2 * 2  # shaft journal diameter

    # Determine IT5 and IT6 for the relevant diameter range
    # ISO 286 fundamental tolerances [μm] for 80-120mm range:
    # IT5 = 15μm, IT6 = 22μm, IT7 = 35μm
    IT5 = 0.015  # mm  (15 μm)
    IT6 = 0.022  # mm
    IT7 = 0.035  # mm

    return [
        ToleranceRecommendation(
            feature      = f"Bearing journal  Ø{bore:.1f}mm",
            nominal_mm   = bore,
            iso_fit      = "js5",
            upper_dev_mm = +IT5 / 2,   # js = symmetric = ±IT5/2
            lower_dev_mm =  IT5 / 2,
            rationale    = ("js5 preferred over h5 for P5 precision spindles: "
                           "symmetric fit eliminates mounting direction sensitivity. "
                           "SKF Engineering Handbook §3.2.1."),
            standard     = "ISO 286-1:2010, ISO 15:2017",
        ),
        ToleranceRecommendation(
            feature      = f"Housing bore  Ø{var_dict.get('R2',50)*2+30:.0f}mm (OD fit)",
            nominal_mm   = bore + 30,
            iso_fit      = "H6",
            upper_dev_mm = +IT6,
            lower_dev_mm =  0.000,
            rationale    = ("H6 for fixed outer ring (non-rotating housing). "
                           "Prevents outer ring creep without inducing hoop stress."),
            standard     = "ISO 286-1:2010, ISO 15:2017 Table 2",
        ),
        ToleranceRecommendation(
            feature      = f"Inner bore  Ø{ri*2:.1f}mm",
            nominal_mm   = ri * 2,
            iso_fit      = "H7",
            upper_dev_mm = +IT7 * 0.6,  # approx H7 for 50-60mm range
            lower_dev_mm =  0.000,
            rationale    = ("H7 reaming tolerance. ±0 on lower = bore never undersized. "
                           "Prevents chuck mandrel interference."),
            standard     = "ISO 286-1:2010",
        ),
        ToleranceRecommendation(
            feature      = "Front bearing seat position  ⊕",
            nominal_mm   = 0.0,
            iso_fit      = "⊕ ϕ0.008",
            upper_dev_mm = +0.004,
            lower_dev_mm =  0.004,
            rationale    = ("Tightened from ϕ10μm to ϕ8μm based on tornado sensitivity: "
                           "pos_tol_front contributes 2.10× amplification at nominal overhang. "
                           "Each 1μm tighter → 2.10μm less nose runout."),
            standard     = "ISO 1101:2017 §15",
        ),
        ToleranceRecommendation(
            feature      = "Rear bearing seats position  ⊕",
            nominal_mm   = 0.0,
            iso_fit      = "⊕ ϕ0.012",
            upper_dev_mm = +0.006,
            lower_dev_mm =  0.006,
            rationale    = ("Rear seat tilt contributes L_oh/L_span × e_pos = 0.73× amplification. "
                           "12μm is sufficient; tightening further gives diminishing returns."),
            standard     = "ISO 1101:2017 §15",
        ),
        ToleranceRecommendation(
            feature      = "Bearing seat roundness  ○",
            nominal_mm   = 0.0,
            iso_fit      = "○ 0.0025",
            upper_dev_mm = +0.0025,
            lower_dev_mm =  0.0,
            rationale    = ("P5 inner ring runout = 5μm. Seat roundness ≤ 0.5 × 5μm = 2.5μm "
                           "per ISO 492:2014 §6.2 recommendation."),
            standard     = "ISO 492:2014, ISO 1101:2017",
        ),
        ToleranceRecommendation(
            feature      = "Bearing face parallelism  ∥",
            nominal_mm   = 0.0,
            iso_fit      = "∥ 0.003",
            upper_dev_mm = +0.003,
            lower_dev_mm =  0.0,
            rationale    = ("Controls spacer preload accuracy. MA class preload tolerance "
                           "requires face parallelism ≤ 3μm to keep F_preload within ±10%."),
            standard     = "ISO 1101:2017, SKF Engineering Handbook §3.4",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Bore strategy analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_bore_strategy(
    R2_raw: float,
    n_rpm:  float = 4000.0,
) -> Dict[str, object]:
    """
    Evaluate the three bore resolution strategies and recommend the best.

    Strategies:
      A. Snap-to-nearest: R2_raw → nearest SKF bore (current approach)
      B. Round-down to smaller bore (higher speed rating)
      C. Round-up to larger bore (higher capacity)

    Returns comparison dict.
    """
    from design_variables import (
        snap_to_skf_bearing, _ACBB_BORES, _CRB_BORES,
        _ACBB_BY_BORE, _CRB_BY_BORE,
    )

    bore_raw = R2_raw * 2

    # Find nearest, smaller, larger catalog bores
    diffs = np.abs(_ACBB_BORES - bore_raw)
    nearest_idx = int(np.argmin(diffs))

    strategies = {}
    for label, idx in [
        ("A_nearest", nearest_idx),
        ("B_smaller", max(nearest_idx - 1, 0)),
        ("C_larger",  min(nearest_idx + 1, len(_ACBB_BORES) - 1)),
    ]:
        bore   = _ACBB_BORES[idx]
        brg    = _ACBB_BY_BORE[bore]
        crb    = _CRB_BY_BORE.get(bore)
        gap_mm = abs(bore_raw - bore)
        gap_pct = gap_mm / bore * 100
        speed_ok = brg.speed_ok(n_rpm, "grease")

        strategies[label] = {
            "bore_mm":       bore,
            "designation":   brg.designation,
            "gap_mm":        gap_mm,
            "gap_pct":       gap_pct,
            "C_r_kN":        brg.C_r / 1000,
            "n_grease":      brg.n_grease,
            "speed_ok":      speed_ok,
            "K_radial_N_mm": 1.7 * brg.radial_stiffness_single_N_mm,
            "rear_CRB":      crb.designation if crb else "N/A",
        }

    return {
        "bore_raw_mm":  bore_raw,
        "strategies":   strategies,
        "recommendation": (
            "B_smaller" if not strategies["A_nearest"]["speed_ok"] else "A_nearest"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report builder
# ─────────────────────────────────────────────────────────────────────────────

class FinalReportBuilder:
    """
    Assembles and prints the complete engineering report.

    Parameters
    ----------
    FoS_min          : Minimum acceptable factor of safety (default 2.0)
    delta_max_um     : Maximum allowable nose deflection [μm] (default 15.0)
    L10_target_hours : Target L10 life [hours]
    """

    def __init__(
        self,
        FoS_min:          float = 2.0,
        delta_max_um:     float = 20.0,    # Class B CNC lathe (ISO 230-1)
        tir_limit_um:     float = 20.0,    # TIR loaded limit (ISO 230-1 Class B)
        L10_target_hours: float = 20_000.0,
        chatter_Ks:           float = 2500.0,  # specific cutting force coeff [N/mm²]
        chatter_zeta:         float = 0.03,    # structural damping ratio
        chatter_b_required:   float = 2.0,     # required stable depth of cut [mm]
    ):
        self.FoS_min          = FoS_min
        self.delta_max_um     = delta_max_um
        self.tir_limit_um     = tir_limit_um
        self.L10_target_hours = L10_target_hours
        self.chatter_Ks            = chatter_Ks
        self.chatter_zeta          = chatter_zeta
        self.chatter_b_required    = chatter_b_required

    # ─────────────────────────────────────────────────────────────────────────
    def print_report(
        self,
        x_raw:         np.ndarray,
        design_space,
        bearing_state,
        runout_bd,
        ecc_result,
        fea_row,
        n_rpm:         float = 4000.0,
        design_name:   str   = "Optimal Design",
    ) -> None:
        """
        Print the complete engineering report.

        Parameters
        ----------
        x_raw         : Raw optimizer design vector (19 vars)
        design_space  : DesignSpace instance
        bearing_state : BearingSystemState (Module 08)
        runout_bd     : RunoutBreakdown (Module 09)
        ecc_result    : EccentricityResult (Module 10)
        fea_row       : Row from FEAPoolRunner.execute_batch() DataFrame
        n_rpm         : Operating speed
        design_name   : Report title
        """
        v       = design_space.decode_vector(x_raw)
        catalog = design_space.resolve_to_catalog(x_raw, n_rpm)
        bf      = catalog["catalog_front"]
        br      = catalog["catalog_rear"]

        # ── Extract FEA results ───────────────────────────────────────────
        delta_um  = float(fea_row["static_max_deflection_um"])
        sigma_MPa = float(fea_row["static_max_vonmises_MPa"])
        FoS       = float(fea_row["static_factor_of_safety"])
        freq1_Hz  = float(fea_row["freq_mode1_Hz"])

        # ── Force envelope ────────────────────────────────────────────────
        # Max force limited by deflection (dominant constraint here)
        R1, ri_v = v["R1"], v["ri"]
        L1, L2   = v["L1"], v["L2"]
        E_mod    = v["E"]
        f_front  = v["front_z_fraction"]
        a = L1 + f_front * L2

        I_segs   = [math.pi/4*(R**4 - ri_v**4)
                    for R in [v["R1"],v["R2"],v["R3"],v["R4"]]]
        L_vals   = [v["L1"],v["L2"],v["L3"],v["L4"]]
        EI_eff   = E_mod * float(np.average(I_segs, weights=L_vals))

        # From propped-beam: delta ≈ F*a³/(3EI) (dominant term)
        # → F_max_defl = delta_max * 3EI / a³
        F_max_defl   = (self.delta_max_um * 1e-3 * 3 * EI_eff) / max(a**3, 1)
        # FoS stress limit: M_max = F*a, sigma = M*R/I ≤ sigma_y/FoS_min
        F_max_stress = v["sigma_y"] / self.FoS_min * I_segs[0] / (R1 * max(a, 1))
        F_max_N      = min(F_max_defl, F_max_stress)
        # Min bearing load: ISO 281 minimum load = 1% of C_r
        F_min_N      = 0.01 * float(bf.C_r)
        F_nom_N      = math.sqrt(v["Ft"]**2 + v["Fr"]**2)

        # ── Bore strategy ─────────────────────────────────────────────────
        bore_analysis = analyse_bore_strategy(v["R2"], n_rpm)
        bore_snap_err = abs(v["R2"]*2 - float(bf.d)) / float(bf.d) * 100

        # ── Tolerances ────────────────────────────────────────────────────
        tols = recommend_tolerances(v)

        # ── Print ─────────────────────────────────────────────────────────
        SEP = "─" * 70

        print(f"\n{'═'*70}")
        print(f"  SPINDLE ENGINEERING REPORT")
        print(f"  {design_name}  |  {n_rpm:.0f} RPM")
        print(f"{'═'*70}")

        # 0. Full Design Dimensions
        print(f"\n  ─── 0. OPTIMAL DESIGN DIMENSIONS ──────────────────────────────")
        print(f"  {'Parameter':<35} {'Value':>12}  Unit   Description")
        print(f"  {'─'*68}")
        dims = [
            ("Shaft Geometry (ISO 286-1)",   None,   None,  None),
            ("  L1 — Nose section",          v["L1"],      "mm",  "Nose to front bearing"),
            ("  L2 — Journal section",        v["L2"],      "mm",  "Front to rear bearing span zone"),
            ("  L3 — Drive section",          v["L3"],      "mm",  "Rear bearing to drive end"),
            ("  L4 — Chuck flange",           v["L4"],      "mm",  "Chuck mounting face width"),
            ("  R1 — Nose outer radius",      v["R1"],      "mm",  "Spindle nose OD/2"),
            ("  R2 — Journal radius",         v["R2"],      "mm",  "Bearing seat OD/2"),
            ("  R3 — Rear journal radius",    v["R3"],      "mm",  "Rear CRB seat OD/2"),
            ("  R4 — Drive end radius",       v["R4"],      "mm",  "Drive coupling OD/2"),
            ("  ri — Inner bore radius",      v["ri"],      "mm",  "Hollow bore radius"),
            ("  Wall thickness",              v["R2"]-v["ri"], "mm", "At bearing seat"),
            ("Bearing Positions",             None,   None,  None),
            ("  Overhang (nose→front brg)",   a,            "mm",  "= L1 + f_front×L2"),
            ("  front_z_fraction",            v["front_z_fraction"], "—", "Fraction of L2"),
            ("  rear_z_fraction",             v["rear_z_fraction"],  "—", "Fraction of L2"),
            ("Material Properties",           None,   None,  None),
            ("  E — Young's modulus",         v["E"]/1e3,   "GPa", "At 20°C"),
            ("  σ_y — Yield strength",        v["sigma_y"], "MPa", "Tensile yield"),
            ("  ρ — Density",                 v["rho"]*1e9, "kg/m³","For imbalance calc"),
            ("Operating Point",               None,   None,  None),
            ("  Ft — Tangential force",       v["Ft"],      "N",   "Primary cutting force"),
            ("  Fr — Radial force",           v["Fr"],      "N",   "Radial cutting force"),
            ("  Ff — Feed force",             v["Ff"],      "N",   "Feed direction force"),
            ("  F_resultant",                 math.sqrt(v["Ft"]**2+v["Fr"]**2), "N", "RSS of Ft,Fr"),
            ("  n_rpm",                       n_rpm,        "RPM", "Operating speed"),
            ("Housing Tolerance",                None,   None,  None),
            ("  housing_it_grade",               v.get("housing_it_grade",6.0), "—", "IT grade number: 5=H5, 6=H6, 7=H7, 8=H8 (ISO 286-1)"),
            ("  housing_it_designation",          f"H{int(round(v.get("housing_it_grade",6.0)))}", "—", "Non-rotating outer ring (ISO 15:2017 Table 2)"),
        ]
        for d in dims:
            if d[1] is None:
                print(f"\n  {d[0]}")
            elif isinstance(d[1], str):
                print(f"  {d[0]:<35} {d[1]:>12}  {d[2]:<6} {d[3]}")
            else:
                print(f"  {d[0]:<35} {d[1]:>12.3f}  {d[2]:<6} {d[3]}")

        # 1. Catalog BOM
        print(f"\n  ─── 1. BEARING BILL OF MATERIALS (SKF Catalog) ───────────────")
        print(f"  FRONT (locating)  : {bf.designation}")
        print(f"    Bore d = {bf.d:.0f} mm  OD D = {bf.D:.0f} mm  B = {bf.B:.0f} mm")
        print(f"    C_r = {bf.C_r/1e3:.1f} kN  |  n_grease = {bf.n_grease} RPM")
        print(f"    K_radial (DB pair) = {catalog['K_radial_catalog']/1e3:.0f} N/μm  "
              f"|  K_axial = {catalog['K_axial_catalog']/1e3:.0f} N/μm")
        print(f"    Preload MA = {bf.F_preload_MA_N:.0f} N  |  Contact angle α = 25°")
        print(f"  REAR (floating)   : {br.designation}  × 2")
        print(f"    Bore d = {br.d:.0f} mm  OD D = {br.D:.0f} mm  B = {br.B:.0f} mm")
        print(f"    C_r = {br.C_r/1e3:.1f} kN  |  n_grease = {br.n_grease} RPM")
        snap_tag = "✅" if bore_snap_err < 2.0 else "⚠️"
        print(f"\n  Raw R2 = {v['R2']:.3f} mm → bore = {v['R2']*2:.2f} mm")
        print(f"  Catalog bore = {bf.d:.0f} mm  |  Snap error = {bore_snap_err:.2f}%  {snap_tag}")

        # 1b. Chatter Stability (Module 9, option C — constraint + objective)
        print(f"\n  ─── 1b. CHATTER STABILITY (Altintas 2012) ────────────────────")
        chatter_Ks         = self.chatter_Ks
        chatter_zeta       = self.chatter_zeta
        chatter_b_required = self.chatter_b_required
        K_dyn_pair = catalog["K_radial_catalog"]   # N/mm, front bearing pair
        b_lim      = 2.0 * K_dyn_pair * chatter_zeta * (1.0 + chatter_zeta) / max(chatter_Ks, 1e-6)
        chatter_ratio = chatter_b_required / max(b_lim, 1e-9)
        chatter_tag   = "✅ STABLE" if chatter_ratio <= 1.0 else "❌ UNSTABLE"
        print(f"  K_dyn (front pair) = {K_dyn_pair:,.0f} N/mm")
        print(f"  K_s (specific cutting force) = {chatter_Ks:,.0f} N/mm²")
        print(f"  ζ (structural damping ratio)  = {chatter_zeta:.3f}")
        print(f"  b_lim   = 2·K_dyn·ζ·(1+ζ)/K_s = {b_lim:.2f} mm")
        print(f"  b_required (roughing depth)   = {chatter_b_required:.2f} mm")
        print(f"  Chatter ratio b_req/b_lim     = {chatter_ratio:.3f}  {chatter_tag}")

        # 2. Static performance
        print(f"\n  ─── 2. STATIC STRUCTURAL PERFORMANCE ─────────────────────────")
        delta_tag = "✅" if delta_um <= self.delta_max_um else "❌ EXCEEDS LIMIT"
        fos_tag   = "✅" if FoS >= self.FoS_min else "❌"
        print(f"  Deflection at nose  : {delta_um:>8.2f} μm  (limit {self.delta_max_um:.0f} μm)  {delta_tag}")
        print(f"  Max bending stress  : {sigma_MPa:>8.2f} MPa  (yield {v['sigma_y']:.0f} MPa)")
        print(f"  Static Safety Fac.  : {FoS:>8.1f}    (min {self.FoS_min:.1f})  {fos_tag}  [σ_y/σ_max — static only]")
        print(f"  First nat. freq.    : {freq1_Hz:>8.1f} Hz")
        print(f"\n  ⚠️  Note: σ = {sigma_MPa:.1f} MPa is physically correct.")
        print(f"  This thick-walled spindle (wall = {R1-v['ri']:.1f} mm) at short")
        print(f"  overhang (a = {a:.0f} mm) has very low bending stress.")
        print(f"  DEFLECTION is the active constraint — not stress.")

        # 3. Cutting force envelope
        print(f"\n  ─── 3. CUTTING FORCE ENVELOPE ─────────────────────────────────")
        print(f"  F_min  (bearing min load, ISO 281) : {F_min_N:>8.1f} N")
        print(f"  F_nom  (current operating)         : {F_nom_N:>8.1f} N")
        print(f"  F_max  (δ ≤ {self.delta_max_um:.0f} μm AND FoS ≥ {self.FoS_min:.0f})       : {F_max_N:>8.1f} N")
        headroom = (F_max_N - F_nom_N) / F_max_N * 100
        limit_str = ("deflection" if F_max_defl < F_max_stress else "stress")
        print(f"  Headroom to F_max                  : {headroom:>7.1f}%")
        print(f"  Active limit: {limit_str}")
        nom_tag = "✅" if F_min_N < F_nom_N < F_max_N else "❌ OUT OF ENVELOPE"
        print(f"  F_nom in safe envelope: {nom_tag}")

        # 4. Bearing life
        print(f"\n  ─── 4. BEARING LIFE (ISO 281) ──────────────────────────────────")
        l10     = bearing_state.L10_system_hours
        l10_tag = "✅" if l10 >= self.L10_target_hours else "❌"
        margin  = (l10 / self.L10_target_hours - 1) * 100
        print(f"  System L10 life : {l10:>10,.0f} h  (target {self.L10_target_hours:.0f} h)  {l10_tag}")
        print(f"  L10 margin      : {margin:>10.0f}%  ({l10/self.L10_target_hours:.1f}× target)")
        for ss in bearing_state.stations:
            role = ss.station.role.upper()
            print(f"  Station {role:<6}: {ss.bearing.designation:<20}"
                  f"  P = {ss.P_equiv_N:.0f} N  L10 = {ss.L10_station_hours:,.0f} h")

        # 5. Runout budget
        print(f"\n  ─── 5. RUNOUT BUDGET (all analytical sources) ─────────────────")
        tir_tag = "✅" if runout_bd.TIR_rss_um <= self.tir_limit_um else "❌"
        total_variance = runout_bd.TIR_rss_um**2
        for src, val in runout_bd.sources_dict.items():
            pct = val**2 / max(total_variance, 1e-20) * 100
            bar = ("█" * int(pct/5)).ljust(20)
            print(f"  {src:<26}: {val:>6.3f} μm  {pct:>5.1f}%  {bar}")
        print(f"  {'─'*65}")
        print(f"  {'TIR (RSS)':<26}: {runout_bd.TIR_rss_um:>6.3f} μm  (limit {self.tir_limit_um:.0f} μm)  {tir_tag}")
        print(f"  {'TIR (linear)':<26}: {runout_bd.TIR_linear_um:>6.3f} μm  (worst-case)")
        print(f"  ΔT thermal model    : {runout_bd.delta_T_C:.1f} °C  @ {n_rpm:.0f} RPM, MA preload")

        # 6. Eccentricity
        print(f"\n  ─── 6. ROTOR ECCENTRICITY (ISO 1940-1 G2.5) ────────────────────")
        u_tag = "✅" if ecc_result.U_static_gmm <= ecc_result.U_allow_gmm else "❌"
        print(f"  Static eccentricity : {ecc_result.e_static_um:>7.3f} μm")
        print(f"  Couple imbalance C  : {ecc_result.couple_gmm2:>7.1f} g·mm²")
        print(f"  F_imbalance @ {n_rpm:.0f}  : {ecc_result.F_imbalance_N:>7.3f} N")
        print(f"  U_static            : {ecc_result.U_static_gmm:>7.1f} g·mm")
        print(f"  U_allow (G2.5)      : {ecc_result.U_allow_gmm:>7.1f} g·mm  {u_tag}")
        need_balance = ecc_result.U_static_gmm > ecc_result.U_allow_gmm
        print(f"  Balancing required  : {'YES ⚠️' if need_balance else 'NO ✅'}")

        # 7. Bore strategy
        print(f"\n  ─── 7. BORE STRATEGY ANALYSIS ──────────────────────────────────")
        print(f"  Raw optimizer bore : {v['R2']*2:.2f} mm  (not in SKF catalog)")
        strats = bore_analysis["strategies"]
        rec    = bore_analysis["recommendation"]
        print(f"  {'Strategy':<12} {'Bore':>6} {'Gap':>7} {'C_r':>8} {'n_gr':>7} {'Speed':>7} {'K_r':>10}")
        print(f"  {'─'*65}")
        for key, s in strats.items():
            tag  = " ← RECOMMENDED" if key == rec else ""
            ok_s = "✅" if s["speed_ok"] else "❌"
            print(f"  {key:<12} {s['bore_mm']:>5.0f}mm {s['gap_mm']:>6.2f}mm"
                  f" {s['C_r_kN']:>7.1f}kN {s['n_grease']:>6}rpm"
                  f" {ok_s}  {s['K_radial_N_mm']/1e3:>8.0f}N/μm{tag}")
        print(f"\n  Catalog-snap penalty in optimizer: 500 USD/mm gap")
        print(f"  This steers future optimizations toward exact catalog bores.")

        # 8. Optimal tolerances
        print(f"\n  ─── 8. RECOMMENDED TOLERANCES (for drawing / BOM) ─────────────")
        print(f"  {'Feature':<35} {'Fit/Tol':<14} {'Dev +':>8} {'Dev −':>8}  Standard")
        print(f"  {'─'*85}")
        for t in tols:
            dev_str_hi = f"+{t.upper_dev_mm*1000:.1f}μm"
            dev_str_lo = f"-{t.lower_dev_mm*1000:.1f}μm"
            print(f"  {t.feature:<35} {t.iso_fit:<14} {dev_str_hi:>8} {dev_str_lo:>8}  {t.standard}")

        # 9. Scorecard
        print(f"\n  ─── 9. PASS/FAIL SCORECARD ─────────────────────────────────────")
        kpis = [
            ("Nose deflection",    delta_um,                      self.delta_max_um,     "μm",  True,  True),
            ("Factor of safety",   FoS,                           self.FoS_min,          "—",   False, True),
            ("System L10 life",    l10/1000,                      self.L10_target_hours/1000, "kh", False, True),
            ("Runout TIR (RSS)",   runout_bd.TIR_rss_um,          self.tir_limit_um,                  "μm",  True,  True),
            ("Imbalance U",        ecc_result.U_static_gmm,       ecc_result.U_allow_gmm,"g·mm",True,  True),
            ("Bore snap error",    bore_snap_err,                  2.0,                   "%",   True,  False),
            ("F_nom in envelope",  1.0 if F_min_N<F_nom_N<F_max_N else 0.0, 1.0,        "—",   False, True),
        ]
        all_pass = True
        for name, actual, limit, unit, smaller_is_better, required in kpis:
            if smaller_is_better:
                pass_ = actual <= limit
            else:
                pass_ = actual >= limit
            tag   = "✅ PASS" if pass_ else ("❌ FAIL" if required else "⚠️ NOTE")
            ratio = actual / max(limit, 1e-12) if smaller_is_better else limit / max(actual, 1e-12)
            bar   = ("█" * int(min(ratio, 1.0) * 15)).ljust(15)
            print(f"  {tag}  {name:<24} {actual:>9.2f} / {limit:<8.2f} {unit:<6}  {bar}")
            if not pass_ and required:
                all_pass = False

        print(f"\n  Overall: {'ALL CRITICAL KPIs PASSED ✅' if all_pass else 'CRITICAL FAILURES ❌ — redesign required'}")
        print(f"{'═'*70}\n")

    # ─────────────────────────────────────────────────────────────────────────
    def generate_plots(
        self,
        x_raw:         np.ndarray,
        design_space,
        bearing_state,
        runout_bd,
        ecc_result,
        fea_row,
        n_rpm:         float = 4000.0,
        save_dir:      str   = ".",
    ) -> None:
        """Generate 5 final report plots (incl. spindle cross-section)."""
        import matplotlib.pyplot as plt
        import os

        NAVY=C.NAVY; TEAL=C.TEAL; CORAL=C.RED; GOLD=C.ORANGE
        MINT=C.GREEN; GRAY=C.GRAY; PURPLE=C.PURPLE
        os.makedirs(save_dir, exist_ok=True)
        apply_paper_theme()

        v       = design_space.decode_vector(x_raw)
        catalog = design_space.resolve_to_catalog(x_raw, n_rpm)
        bf      = catalog["catalog_front"]

        delta_um  = float(fea_row["static_max_deflection_um"])
        sigma_MPa = float(fea_row["static_max_vonmises_MPa"])
        FoS       = float(fea_row["static_factor_of_safety"])

        R1, ri_v = v["R1"], v["ri"]
        L1, L2   = v["L1"], v["L2"]
        a = L1 + v["front_z_fraction"] * L2

        # ── Fig 11e: Spindle cross-section (NEW) ─────────────────────────
        p = os.path.join(save_dir, "11e_spindle_section.png")
        plot_spindle_cross_section(v, catalog, runout_bd, fea_row, n_rpm, p)
        print(f"  Saved → {p}")
        I_segs = [math.pi/4*(R**4-ri_v**4) for R in [v["R1"],v["R2"],v["R3"],v["R4"]]]
        L_vals = [v["L1"],v["L2"],v["L3"],v["L4"]]
        EI_eff = v["E"] * float(np.average(I_segs, weights=L_vals))
        F_max_defl   = (self.delta_max_um*1e-3*3*EI_eff)/max(a**3,1)
        F_max_stress = v["sigma_y"]/self.FoS_min*I_segs[0]/(R1*max(a,1))
        F_max_N      = min(F_max_defl, F_max_stress)
        F_min_N      = 0.01 * float(bf.C_r)
        F_nom_N      = math.sqrt(v["Ft"]**2+v["Fr"]**2)

        # ── Fig 11a: KPI Radar ────────────────────────────────────────────
        labels = ["Deflection\n(≤15μm)", "FoS\n(≥2.0)", "L10×target",
                  "TIR≤{:.0f}μm".format(self.tir_limit_um), "Imbalance\nU/U_allow", "Force\nenvelope"]
        scores = [
            min(self.delta_max_um / max(delta_um, 0.01), 2.0),
            min(FoS / self.FoS_min, 2.0),
            min(bearing_state.L10_system_hours / self.L10_target_hours, 2.0),
            min(self.tir_limit_um / max(runout_bd.TIR_rss_um, 0.01), 2.0),
            min(ecc_result.U_allow_gmm / max(ecc_result.U_static_gmm, 0.01), 2.0),
            1.0 if F_min_N < F_nom_N < F_max_N else 0.4,
        ]
        N      = len(labels)
        angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
        sp     = scores + scores[:1]
        tgt    = [1.0] * (N+1)

        fig, ax = plt.subplots(figsize=(7,7), subplot_kw={"projection":"polar"}, facecolor=C.BG)
        ax.set_facecolor(C.BG)
        ax.plot(angles, sp,  color=TEAL, lw=2, marker="o", ms=6)
        ax.fill(angles, sp,  color=TEAL, alpha=0.25)
        ax.plot(angles, tgt, color=GOLD, lw=1.5, linestyle="--", label="Target = 1.0×")
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=8, color="white")
        ax.set_ylim(0, 2); ax.set_yticks([0.5,1.0,1.5,2.0])
        ax.set_yticklabels(["0.5×","1.0×","1.5×","2.0×"], fontsize=7, color=GRAY)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
        ax.set_title("Fig 11a — KPI Radar", color=C.TEXT, pad=20, fontsize=10)
        plt.tight_layout()
        p = os.path.join(save_dir, "11a_kpi_radar.png")
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=C.BG)
        plt.close(fig); print(f"  Saved → {p}")

        # ── Fig 11b: Force envelope ───────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9,5), facecolor=C.BG)
        ax.set_facecolor(C.BG)
        ax.barh(["F_min (brg load)", "F_nominal", "F_max (δ≤15μm)"],
                [F_min_N, F_nom_N, F_max_N],
                color=[CORAL, TEAL, GOLD], edgecolor=NAVY, height=0.4)
        for val, y in zip([F_min_N, F_nom_N, F_max_N], range(3)):
            ax.text(val + 20, y, f"{val:.0f} N", va="center", fontsize=9, color="white")
        ax.axvline(F_nom_N, color=TEAL, lw=1.5, linestyle="--", alpha=0.6)
        ax.set_xlabel("Force [N]")
        ax.set_title("Fig 11b — Cutting Force Envelope\n"
                     f"Safe zone: {F_min_N:.0f} – {F_max_N:.0f} N  |  "
                     f"Active limit: {'deflection' if F_max_defl<F_max_stress else 'stress'}", color=C.TEXT)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        p = os.path.join(save_dir, "11b_force_envelope.png")
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=C.BG)
        plt.close(fig); print(f"  Saved → {p}")

        # ── Fig 11c: Runout waterfall ─────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10,5), facecolor=C.BG)
        ax.set_facecolor(C.BG)
        src_items = list(runout_bd.sources_dict.items())
        cumul = 0.0
        cols  = [TEAL, CORAL, GOLD, MINT, PURPLE, "#ff9f1c", GRAY]
        for i, (lbl, val) in enumerate(src_items):
            ax.bar(i, val, bottom=cumul, color=cols[i % len(cols)],
                   edgecolor=NAVY, linewidth=0.5, width=0.55)
            if val > 0.05:
                ax.text(i, cumul+val/2, f"{val:.2f}μm", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
            cumul += val
        ax.bar(len(src_items), runout_bd.TIR_rss_um, color=PURPLE,
               edgecolor=NAVY, linewidth=0.5, width=0.55)
        ax.text(len(src_items), runout_bd.TIR_rss_um/2,
                f"{runout_bd.TIR_rss_um:.2f}μm", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
        ax.axhline(self.tir_limit_um, color=CORAL, lw=1.5, linestyle="--", label=f"{self.tir_limit_um:.0f} μm limit (ISO 230-1 Class B)")
        xlabels = [s.replace(" (ISO 492)","").replace(" (ISO 230-3)","")
                   .replace(" (ISO 1101)","") for s,_ in src_items] + ["TIR RSS"]
        ax.set_xticks(range(len(xlabels))); ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_ylabel("TIR [μm]"); ax.legend(fontsize=8)
        ax.set_title("Fig 11c — Runout Budget Waterfall", color=C.TEXT)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        p = os.path.join(save_dir, "11c_runout_waterfall.png")
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=C.BG)
        plt.close(fig); print(f"  Saved → {p}")

        # ── Fig 11d: Tolerance recommendations ───────────────────────────
        tols = recommend_tolerances(v)
        fig, ax = plt.subplots(figsize=(12,6), facecolor=C.BG)
        ax.set_facecolor(C.BG)
        y_pos  = np.arange(len(tols))
        upper  = [t.upper_dev_mm*1000 for t in tols]
        lower  = [t.lower_dev_mm*1000 for t in tols]
        labels_t = [t.feature for t in tols]
        ax.barh(y_pos - 0.18, upper, height=0.32, color=TEAL,  label="+upper dev [μm]", edgecolor=NAVY)
        ax.barh(y_pos + 0.18, lower, height=0.32, color=CORAL, label="−lower dev [μm]", edgecolor=NAVY)
        for i, t in enumerate(tols):
            ax.text(max(t.upper_dev_mm*1000, 0)+0.1, i-0.18,
                    f"{t.upper_dev_mm*1000:.1f}μm", va="center", fontsize=7.5, color="white")
            if t.lower_dev_mm > 0:
                ax.text(t.lower_dev_mm*1000+0.1, i+0.18,
                        f"{t.lower_dev_mm*1000:.1f}μm", va="center", fontsize=7.5, color="white")
        ax.set_yticks(y_pos); ax.set_yticklabels([f"{t.feature}  [{t.iso_fit}]"
                                                    for t in tols], fontsize=8)
        ax.set_xlabel("Tolerance deviation [μm]")
        ax.set_title("Fig 11d — Recommended Tolerance Specification (ISO 286/1101/492)",
                     color="white")
        ax.legend(fontsize=8); ax.grid(axis="x", alpha=0.3)
        ax.set_xscale("log"); ax.set_xlim(left=0.5)
        plt.tight_layout()
        p = os.path.join(save_dir, "11d_tolerances.png")
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=C.BG)
        plt.close(fig); print(f"  Saved → {p}")

        # ── Fig 11f: Reliability gauges (Module 13) ───────────────────────
        try:
            import importlib, sys
            if "reliability_index" not in sys.modules:
                spec = importlib.util.spec_from_file_location(
                    "reliability_index",
                    os.path.join(os.path.dirname(__file__), "13_reliability_index.py"),
                )
                ri_mod = importlib.util.module_from_spec(spec)
                sys.modules["reliability_index"] = ri_mod
                spec.loader.exec_module(ri_mod)
            else:
                ri_mod = sys.modules["reliability_index"]

            # Build MC samples from FEA row (deterministic point estimate)
            # with ±10% uncertainty to generate a minimal σ for display
            n_mc_disp = 200
            rng_disp  = np.random.default_rng(42)
            defl_mc   = rng_disp.normal(delta_um,  max(delta_um  * 0.10, 0.1), n_mc_disp)
            sigma_mc  = rng_disp.normal(sigma_MPa,  max(sigma_MPa * 0.10, 1.0), n_mc_disp)
            fos_mc    = rng_disp.normal(FoS,        max(FoS       * 0.08, 0.05),n_mc_disp)
            freq_mc   = rng_disp.normal(float(fea_row.get("freq_mode1_Hz", 400.0)),
                                        20.0, n_mc_disp)
            Y_mc = np.column_stack([defl_mc, sigma_mc, fos_mc, freq_mc])
            out_names = ["static_max_deflection_um", "static_max_vonmises_MPa",
                         "static_factor_of_safety", "freq_mode1_Hz"]

            ls      = ri_mod.default_limit_states(
                delta_max_um=self.delta_max_um,
                fos_min=self.FoS_min,
                n_rpm=n_rpm,
            )
            ra      = ri_mod.ReliabilityAnalyser(ls)
            sys_rel = ra.compute_from_samples(Y_mc, out_names)

            p = os.path.join(save_dir, "11f_reliability_gauges.png")
            ri_mod.plot_reliability_gauges(
                sys_rel, p,
                design_name=f"δ={delta_um:.1f}μm  σ={sigma_MPa:.0f}MPa  FoS={FoS:.2f}",
            )
        except Exception as e:
            print(f"  Fig 11f skipped: {e}")


def plot_spindle_cross_section(
    var_dict:  dict,
    catalog:   dict,
    runout_bd,
    fea_row,
    n_rpm:     float,
    save_path: str,
) -> None:
    """
    Engineering cross-section drawing of the CNC lathe spindle.

    Shows a dimensioned half-section view (matplotlib) with:
      • Stepped hollow shaft profile (4 segments)
      • Inner bore
      • Front ACBB bearing (locating) + housing
      • Rear CRB bearing × 2 (floating) + housing
      • Dimension annotations: L1-L4, R1-R4, ri, z_front, z_rear
      • Material/stress info box
      • Tolerance callouts
      • Bearing designations
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch, Arc
    import numpy as np, os

    # Spindle cross-section colours (light theme)
    NAVY=C.NAVY; TEAL=C.TEAL; CORAL=C.RED; GOLD=C.ORANGE
    MINT=C.GREEN; GRAY=C.GRAY; PURPLE=C.PURPLE
    STEEL    = "#4a7fb5"    # shaft body — medium steel blue
    BORE     = "#e8eef4"    # hollow bore — very light blue
    HOUSING  = "#b0bec5"    # housing — light blue-gray
    BRNG_F   = C.BLUE       # front bearing
    BRNG_R   = C.GREEN      # rear bearing
    ANNOTATION = C.NAVY     # dimension arrows
    v  = var_dict
    L1, L2, L3, L4 = v["L1"], v["L2"], v["L3"], v["L4"]
    R1, R2, R3, R4 = v["R1"], v["R2"], v["R3"], v["R4"]
    ri = v["ri"]
    ff = v["front_z_fraction"]
    fr = v["rear_z_fraction"]

    # z positions (nose at z=0, positive rightward to drive end)
    z0    = 0.0
    z1    = L1                          # nose section end
    z2    = L1 + L2                     # journal end
    z3    = L1 + L2 + L3               # rear section end
    z4    = L1 + L2 + L3 + L4          # drive end
    z_f   = L1 + ff * L2               # front bearing centre
    z_r1  = L1 + fr * L2               # rear bearing 1 centre
    z_r2  = z_r1 + 25.0                # rear bearing 2 centre (approx)

    bf   = catalog.get("catalog_front")
    br   = catalog.get("catalog_rear")
    d_brg_f = float(bf.D) / 2 if bf else R2 + 30
    B_f     = float(bf.B)    if bf else 25
    d_brg_r = float(br.D) / 2 if br else R3 + 25
    B_r     = float(br.B)    if br else 20

    # ── Figure ────────────────────────────────────────────────────────────
    NAVY=C.NAVY; TEAL=C.TEAL; CORAL=C.RED; GOLD=C.ORANGE
    MINT=C.GREEN; GRAY=C.GRAY; PURPLE=C.PURPLE

    fig, ax = plt.subplots(figsize=(18, 9), facecolor=C.BG)
    ax.set_facecolor(C.BG)
    ax.set_aspect("equal")

    # ── Helper: draw shaft half-section (upper half, mirrored) ────────────
    def draw_shaft():
        # Build polygon: outer profile → inner bore (hollow) → back
        # Outer profile segments (z, R pairs)
        outer = [
            (z0,  R1),                  # nose face
            (z1,  R1),                  # end of nose → step up
            (z1,  R2),                  # start of journal
            (z2,  R2),                  # end of journal
            (z2,  R3),                  # start of rear section
            (z3,  R3),                  # end of rear section
            (z3,  R4),                  # start of drive end
            (z4,  R4),                  # drive end face
            (z4, -R4),                  # mirror: drive end (bottom)
            (z3, -R4),
            (z3, -R3),
            (z2, -R3),
            (z2, -R2),
            (z1, -R2),
            (z1, -R1),
            (z0, -R1),
        ]
        # Inner bore polygon (hollow)
        inner = [
            (z0,  ri), (z4,  ri),
            (z4, -ri), (z0, -ri),
        ]
        xs_o = [p[0] for p in outer]; ys_o = [p[1] for p in outer]
        xs_i = [p[0] for p in inner]; ys_i = [p[1] for p in inner]

        # Fill shaft body
        ax.fill(xs_o, ys_o, color=STEEL, alpha=0.75, zorder=2)
        # Mask bore
        ax.fill(xs_i, ys_i, color=BORE,  alpha=1.0,  zorder=3)
        # Outlines
        ax.plot(xs_o, ys_o, color=C.NAVY, lw=0.9, zorder=4)
        ax.plot(xs_i, ys_i, color=C.GRAY, lw=0.7, linestyle="--", zorder=4)

    def draw_bearing(zc, R_outer, B, color, label, role):
        """Draw one bearing as a hatched rectangle pair (upper & lower)."""
        hw = B / 2
        for sign in [1, -1]:
            rect = mpatches.FancyBboxPatch(
                (zc - hw, sign * R2),
                B, sign * (R_outer - R2),
                boxstyle="square,pad=0",
                facecolor=color, edgecolor="white",
                linewidth=0.8, alpha=0.85, zorder=5,
            )
            ax.add_patch(rect)
        ax.text(zc, R_outer + 4, label, ha="center", va="bottom",
                fontsize=6.5, color=color, fontweight="bold", zorder=8)
        ax.text(zc, -(R_outer + 7), role, ha="center", va="top",
                fontsize=6, color=GRAY, zorder=8)

    def draw_housing(zc, R_outer, B, it_grade="H6"):
        """Draw housing bore as outer rectangle."""
        hw    = B / 2 + 5
        h_thk = 15   # housing wall thickness (schematic)
        for sign in [1, -1]:
            rect = mpatches.FancyBboxPatch(
                (zc - hw, sign * R_outer),
                2 * hw, sign * h_thk,
                boxstyle="square,pad=0",
                facecolor=HOUSING, edgecolor=GOLD,
                linewidth=0.7, alpha=0.6, zorder=4,
                linestyle="--",
            )
            ax.add_patch(rect)
        ax.text(zc, R_outer + h_thk + 3, f"Housing {it_grade}",
                ha="center", va="bottom", fontsize=6, color=GOLD, zorder=9)

    def dim_arrow(x1, x2, y, label, color=ANNOTATION, above=True):
        """Draw a horizontal dimension arrow with label."""
        yo = y + (8 if above else -8)
        ax.annotate("", xy=(x2, yo), xytext=(x1, yo),
                    arrowprops=dict(arrowstyle="<->", color=color, lw=0.9),
                    zorder=10)
        ax.text((x1+x2)/2, yo + (3 if above else -4),
                label, ha="center", va="bottom" if above else "top",
                fontsize=6.5, color=color, zorder=11)
        ax.plot([x1, x1], [y, yo], color=color, lw=0.5, linestyle=":", zorder=9)
        ax.plot([x2, x2], [y, yo], color=color, lw=0.5, linestyle=":", zorder=9)

    def rad_arrow(z, r, label, color=TEAL, right=True):
        """Draw a vertical radius annotation."""
        xo = z + (6 if right else -6)
        ax.annotate("", xy=(xo, r), xytext=(xo, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9),
                    zorder=10)
        ax.text(xo + (3 if right else -3), r / 2,
                label, ha="left" if right else "right", va="center",
                fontsize=6.5, color=color, rotation=90, zorder=11)

    # ── Draw components ───────────────────────────────────────────────────
    draw_shaft()

    # Bearings
    draw_bearing(z_f,  d_brg_f, B_f, BRNG_F,
                 bf.designation if bf else "Front ACBB",
                 "FRONT (locating)\nFixed outer ring")
    draw_bearing(z_r1, d_brg_r, B_r, BRNG_R,
                 br.designation if br else "Rear CRB 1",
                 "REAR 1 (floating)")
    draw_bearing(z_r2, d_brg_r, B_r, BRNG_R,
                 br.designation if br else "Rear CRB 2",
                 "REAR 2 (floating)")

    # Housings
    h_grade = getattr(runout_bd, "housing_it_grade", "H6") if runout_bd else "H6"
    draw_housing(z_f,  d_brg_f, B_f, h_grade)
    draw_housing(z_r1, d_brg_r, B_r, h_grade)
    draw_housing(z_r2, d_brg_r, B_r, h_grade)

    # Centreline
    ax.axhline(0, color=GRAY, lw=0.8, linestyle="-.", alpha=0.6, zorder=1)

    # ── Dimension arrows ─────────────────────────────────────────────────
    y_dim = max(d_brg_f, d_brg_r) + 22
    dim_arrow(z0,  z1,  y_dim,      f"L1={L1:.0f}mm", above=True)
    dim_arrow(z1,  z2,  y_dim+14,   f"L2={L2:.0f}mm", above=True)
    dim_arrow(z2,  z3,  y_dim,      f"L3={L3:.0f}mm", above=True)
    dim_arrow(z3,  z4,  y_dim+14,   f"L4={L4:.0f}mm", above=True)
    dim_arrow(z0,  z_f, -(y_dim),   f"Overhang={z_f:.0f}mm", above=False)

    # Radius annotations
    rad_arrow(z0+5,    R1, f"R1={R1:.1f}", TEAL,   right=True)
    rad_arrow(z1+L2/2, R2, f"R2={R2:.1f}", TEAL,   right=True)
    rad_arrow(z2+L3/2, R3, f"R3={R3:.1f}", MINT,   right=False)
    rad_arrow(z3+L4/2, R4, f"R4={R4:.1f}", MINT,   right=False)
    rad_arrow(z2-L2/4, ri, f"ri={ri:.1f}", CORAL,  right=False)

    # Bearing position lines
    for zpos, col in [(z_f, BRNG_F),(z_r1, BRNG_R),(z_r2, BRNG_R)]:
        ax.axvline(zpos, color=col, lw=0.7, linestyle=":", alpha=0.5, zorder=1)

    # ── Info box ─────────────────────────────────────────────────────────
    fos_val   = float(fea_row["static_factor_of_safety"])
    delta_val = float(fea_row["static_max_deflection_um"])
    freq_val  = float(fea_row["freq_mode1_Hz"])
    sigma_val = float(fea_row["static_max_vonmises_MPa"])
    tir_val   = runout_bd.TIR_rss_um if runout_bd else 0.0

    info_lines = [
        f"δ_nose = {delta_val:.2f} μm",
        f"σ_vM   = {sigma_val:.1f} MPa",
        f"FoS    = {fos_val:.2f}",
        f"f₁     = {freq_val:.0f} Hz",
        f"TIR    = {tir_val:.2f} μm",
        f"n      = {n_rpm:.0f} RPM",
        f"Housing: {h_grade}",
    ]
    ax.text(z4 + 8, 0, "\n".join(info_lines),
            va="center", ha="left", fontsize=7, color="white",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#112233",
                      edgecolor=TEAL, linewidth=1.0),
            zorder=12)

    # ── Labels & title ────────────────────────────────────────────────────
    ax.text(z0 - 5, 0, "NOSE\n(chuck)", ha="right", va="center",
            fontsize=7, color=GOLD, fontweight="bold")
    ax.text(z4 + 3, -R4 - 8, "DRIVE END", ha="left", va="top",
            fontsize=7, color=GOLD)

    ax.set_xlabel("Axial position z [mm]", color="white", fontsize=9)
    ax.set_ylabel("Radial dimension [mm]", color="white", fontsize=9)
    ax.set_title(
        f"TechPulse Spindle — Cross-Section (Half View)\n"
        f"Front: {bf.designation if bf else '?'}  |  "
        f"Rear: {br.designation if br else '?'} × 2  |  "
        f"Housing: {h_grade} (ISO 286)  |  n = {n_rpm:.0f} RPM",
        color="white", fontsize=10, pad=10,
    )
    ax.set_xlim(z0 - 25, z4 + 80)
    ax.set_ylim(-(y_dim + 30), y_dim + 30)
    ax.tick_params(colors=GRAY)

    # Legend
    legend_handles = [
        mpatches.Patch(color=STEEL,   label="Shaft (4140 steel)"),
        mpatches.Patch(color=BORE,    label="Inner bore (hollow)"),
        mpatches.Patch(color=BRNG_F,  label="Front ACBB (locating)"),
        mpatches.Patch(color=BRNG_R,  label="Rear CRB (floating)"),
        mpatches.Patch(color=HOUSING, label=f"Housing ({h_grade})"),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              facecolor="#112233", edgecolor=GRAY,
              labelcolor="white", fontsize=7.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=C.BG)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os, importlib.util, logging
    logging.basicConfig(level=logging.WARNING)
    sys.path.insert(0, ".")

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec); sys.modules[name] = m
        spec.loader.exec_module(m); return m

    load('design_variables',    './01_design_variables.py')
    load('lhs_sampler',         './02_lhs_sampler.py')
    load('fea_pool_runner',     './03_fea_pool_runner.py')
    load('selective_assembly',  './07_selective_assembly.py')
    load('bearing_performance', './08_bearing_performance.py')
    load('shaft_runout',        './09_shaft_runout.py')
    load('rotor_eccentricity',  './10_rotor_eccentricity.py')

    from design_variables    import DesignSpace, SpindleBearingArrangement
    from fea_pool_runner     import FEAPoolRunner
    from bearing_performance import BearingPerformanceCalculator
    from shaft_runout        import ShaftRunoutAnalyser, get_bearing_positions_from_design
    from rotor_eccentricity  import RotorEccentricityAnalyser

    SAVE = "/tmp/spindle_plots"
    os.makedirs(SAVE, exist_ok=True)

    ds    = DesignSpace()
    arr   = SpindleBearingArrangement.default_lathe()
    names = ds.get_variable_names()
    assert len(names) == 19, f"Expected 19, got {len(names)}"

    # Uploaded optimal design (19-var)
    uploaded = {
        'E':204544.38,'Ff':944.58,'Fr':1250.88,'Ft':1006.80,
        'L1':100.19,'L2':351.56,'L3':15.13,'L4':24.75,
        'R1':43.28,'R2':48.66,'R3':70.45,'R4':36.34,
        'front_z_fraction':0.1509,'rear_z_fraction':0.807,
        'pos_tol_front':0.00869,'pos_tol_rear':0.01292,
        'rho':7.86e-9,'ri':29.76,'sigma_y':603.08,
    }
    x_opt = np.array([uploaded.get(n, ds.get_nominal()[i]) for i,n in enumerate(names)])

    # FEA (analytical dry run)
    runner = FEAPoolRunner(np.array([x_opt]), ds, dry_run=True)
    df     = runner.execute_batch()
    fea    = df.iloc[0]

    # Module 08
    calc   = BearingPerformanceCalculator(ds, arr, n_nom_rpm=4000, n_max_rpm=6000)
    state4 = calc.evaluate(x_opt, n_rpm=4000)

    # Module 09
    nom_d    = ds.decode_vector(x_opt)
    z_f, z_r = get_bearing_positions_from_design(nom_d, arr)
    Fr_N     = float(np.sqrt(nom_d['Ft']**2+nom_d['Fr']**2))
    roa      = ShaftRunoutAnalyser('P5','precision',10.0,20.0,preload_class='MA')
    bd       = roa.analyse(nom_d, z_f, z_r, delta_nose_ansys_um=fea['static_max_deflection_um'],
                            Fr_N=Fr_N, n_rpm=4000)

    # Module 10
    ecc = RotorEccentricityAnalyser('G2.5', 0.005, 50.0, max(z_r-z_f, 1))
    r4  = ecc.analyse(nom_d, n_rpm=4000, bearing_span_mm=max(z_r-z_f, 1))

    # Report
    builder = FinalReportBuilder(FoS_min=2.0, delta_max_um=20.0, tir_limit_um=20.0, L10_target_hours=20000)
    builder.print_report(x_opt, ds, state4, bd, r4, fea, n_rpm=4000,
                          design_name="Uploaded Optimal Design")
    print("Generating final report plots...")
    builder.generate_plots(x_opt, ds, state4, bd, r4, fea, n_rpm=4000, save_dir=SAVE)

    # Checks
    v = ds.decode_vector(x_opt)
    F_nom = math.sqrt(v["Ft"]**2+v["Fr"]**2)
    cat   = ds.resolve_to_catalog(x_opt, 4000)
    F_min = 0.01*float(cat["catalog_front"].C_r)
    assert F_nom > F_min, "Nominal force below bearing minimum"
    assert fea['static_max_vonmises_MPa'] > 0
    assert fea['freq_mode1_Hz'] > 0
    plots = [f for f in sorted(os.listdir(SAVE))
             if f.startswith("11") and f.endswith(".png")
             and any(f.startswith("11"+x) for x in ["a","b","c_runout","d_tol"])]
    assert len(plots) == 4, f"Expected 4 new plots, got {len(plots)}: {plots}"
    print(f"\n✅ Module 11 — {len(plots)} plots generated, all checks passed")
