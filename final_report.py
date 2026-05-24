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
    tir_limit_um     : Maximum allowable total indicator reading [μm] (default 20.0)
    L10_target_hours : Target L10 life [hours]
    """

    def __init__(
        self,
        FoS_min:          float = 2.0,
        delta_max_um:     float = 20.0,    # Class B CNC lathe (ISO 230-1)
        tir_limit_um:     float = 20.0,    # TIR loaded limit (ISO 230-1 Class B)
        L10_target_hours: float = 20_000.0,
    ):
        self.FoS_min          = FoS_min
        self.delta_max_um     = delta_max_um
        self.tir_limit_um     = tir_limit_um
        self.L10_target_hours = L10_target_hours

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

        # 2. Static performance
        print(f"\n  ─── 2. STATIC STRUCTURAL PERFORMANCE ─────────────────────────")
        delta_tag = "✅" if delta_um <= self.delta_max_um else "❌ EXCEEDS LIMIT"
        fos_tag   = "✅" if FoS >= self.FoS_min else "❌"
        print(f"  Deflection at nose  : {delta_um:>8.2f} μm  (limit {self.delta_max_um:.0f} μm)  {delta_tag}")
        print(f"  Max bending stress  : {sigma_MPa:>8.2f} MPa  (yield {v['sigma_y']:.0f} MPa)")
        print(f"  Factor of Safety    : {FoS:>8.1f}    (min {self.FoS_min:.1f})  {fos_tag}")
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
        """Generate 4 final report plots."""
        import matplotlib.pyplot as plt
        import os

        NAVY="#0d1b2a"; TEAL="#00b4d8"; CORAL="#e63946"; GOLD="#ffd166"
        MINT="#06d6a0"; GRAY="#8d99ae"; PURPLE="#7400b8"
        os.makedirs(save_dir, exist_ok=True)
        plt.rcParams.update({
            "figure.facecolor": NAVY, "axes.facecolor": "#112233",
            "axes.edgecolor": GRAY, "axes.labelcolor": "white",
            "xtick.color": GRAY, "ytick.color": GRAY, "text.color": "white",
            "grid.color": "#2d4060", "grid.alpha": 0.5, "font.size": 9,
        })

        v       = design_space.decode_vector(x_raw)
        catalog = design_space.resolve_to_catalog(x_raw, n_rpm)
        bf      = catalog["catalog_front"]

        delta_um  = float(fea_row["static_max_deflection_um"])
        sigma_MPa = float(fea_row["static_max_vonmises_MPa"])
        FoS       = float(fea_row["static_factor_of_safety"])

        R1, ri_v = v["R1"], v["ri"]
        L1, L2   = v["L1"], v["L2"]
        a = L1 + v["front_z_fraction"] * L2
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
                  "TIR≤15μm", "Imbalance\nU/U_allow", "Force\nenvelope"]
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

        fig, ax = plt.subplots(figsize=(7,7), subplot_kw={"projection":"polar"}, facecolor=NAVY)
        ax.set_facecolor("#112233")
        ax.plot(angles, sp,  color=TEAL, lw=2, marker="o", ms=6)
        ax.fill(angles, sp,  color=TEAL, alpha=0.25)
        ax.plot(angles, tgt, color=GOLD, lw=1.5, linestyle="--", label="Target = 1.0×")
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=8, color="white")
        ax.set_ylim(0, 2); ax.set_yticks([0.5,1.0,1.5,2.0])
        ax.set_yticklabels(["0.5×","1.0×","1.5×","2.0×"], fontsize=7, color=GRAY)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
        ax.set_title("Fig 11a — KPI Radar", color="white", pad=20, fontsize=10)
        plt.tight_layout()
        p = os.path.join(save_dir, "11a_kpi_radar.png")
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=NAVY)
        plt.close(fig); print(f"  Saved → {p}")

        # ── Fig 11b: Force envelope ───────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9,5), facecolor=NAVY)
        ax.set_facecolor("#112233")
        ax.barh(["F_min (brg load)", "F_nominal", "F_max (δ≤15μm)"],
                [F_min_N, F_nom_N, F_max_N],
                color=[CORAL, TEAL, GOLD], edgecolor=NAVY, height=0.4)
        for val, y in zip([F_min_N, F_nom_N, F_max_N], range(3)):
            ax.text(val + 20, y, f"{val:.0f} N", va="center", fontsize=9, color="white")
        ax.axvline(F_nom_N, color=TEAL, lw=1.5, linestyle="--", alpha=0.6)
        ax.set_xlabel("Force [N]")
        ax.set_title("Fig 11b — Cutting Force Envelope\n"
                     f"Safe zone: {F_min_N:.0f} – {F_max_N:.0f} N  |  "
                     f"Active limit: {'deflection' if F_max_defl<F_max_stress else 'stress'}",
                     color="white")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        p = os.path.join(save_dir, "11b_force_envelope.png")
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=NAVY)
        plt.close(fig); print(f"  Saved → {p}")

        # ── Fig 11c: Runout waterfall ─────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10,5), facecolor=NAVY)
        ax.set_facecolor("#112233")
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
        ax.set_title("Fig 11c — Runout Budget Waterfall", color="white")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        p = os.path.join(save_dir, "11c_runout_waterfall.png")
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=NAVY)
        plt.close(fig); print(f"  Saved → {p}")

        # ── Fig 11d: Tolerance recommendations ───────────────────────────
        tols = recommend_tolerances(v)
        fig, ax = plt.subplots(figsize=(12,6), facecolor=NAVY)
        ax.set_facecolor("#112233")
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
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=NAVY)
        plt.close(fig); print(f"  Saved → {p}")


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
