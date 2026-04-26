#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  Rotor Eccentricity Analysis — Module 10  (v2 — peer-review corrections)
================================================================================

  Purpose:
      Compute the rotor Centre of Gravity (CG), static and DYNAMIC (couple)
      imbalance for the stepped hollow spindle, and export the correct ANSYS
      MAPDL harmonic-analysis force block.

  Corrections applied vs. v1  (source: expert peer review):
  ──────────────────────────────────────────────────────────
  FIX-A  Unit clarity in U_allow
      The code calculation WAS numerically correct:
          U_allow [g·mm] = 1000 × m [kg] × G [mm/s] / ω [rad/s]
      However the inline comment wrote "U_allow = m × G/ω [g·mm]"
      without the ×1000, which was misleading.  Comment now shows the
      full formula explicitly.

  FIX-B  Couple (dynamic) imbalance added
      v1 computed only the STATIC imbalance — the net CG offset e_CG from
      the spin axis.  Even when e_CG = 0 (CG exactly on axis), individual
      segment eccentricities e_i at different axial positions z_i create a
      bending MOMENT (couple) that excites conical whirl modes.

      Couple imbalance magnitude:
          C = Σ_i  U_i × (z_{CG,i} − z_{CG,system})
            = Σ_i  (m_i × e_i) × Δz_i    [g·mm²]

      Couple imbalance force pair at axial distance d_span:
          F_couple = C / d_span  [N]   (symmetric ± pair about system CG)

      Per-segment forces (the rigorous ANSYS excitation approach):
          F_i = m_i [kg] × e_i [m] × ω²   [N]
      Applied individually at each z_{CG,i} rather than lumped at z_CG.
      This automatically captures both static and couple imbalance, exciting
      cylindrical AND conical bending modes.

  FIX-C  ANSYS rotating load corrected
      v1 applied only:
          F, N_CG, FX, F_imbal        ← oscillating force in X only
      This simulates a 1D oscillation, not a rotating vector.

      Correct definition of a synchronous rotating force in ANSYS Harmonic:
          Real component  →  F, node, FX,  F_imbal,      0
          Imaginary part  →  F, node, FY,  0,        −F_imbal
      The negative sign on FY gives FORWARD whirl (consistent with +Z spin
      axis per the right-hand rule).  Use +F_imbal on FY for backward whirl.

      For the per-segment approach, this is repeated for every segment node.

  Physical Model  (unchanged from v1):
  ──────────────────────────────────────
      Radial CG offset of a hollow annular segment due to bore eccentricity ε:
          e_i = ε × r_i² / (R_i² − r_i²)
      The sign of e_i is OPPOSITE to ε (missing-material pulls CG toward
      the thicker wall).  In worst-case analysis sign is irrelevant;
      in Monte Carlo angular analysis the 180° phase matters.

  Standards:
  ──────────
      ISO 1940-1:2003   — Balance quality requirements (G grades, U_allow)
      ISO 1940-2:1997   — Balance errors and tolerances
================================================================================
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

RPM_TO_RADS   = 2.0 * math.pi / 60.0
G_GRADE_MM_S  = {"G1": 1.0, "G2.5": 2.5, "G6.3": 6.3, "G16": 16.0}


# ─────────────────────────────────────────────────────────────────────────────
# Segment-level container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ShaftSegment:
    """
    One hollow-cylinder segment of the stepped spindle shaft.

    Attributes
    ----------
    name         : "nose" | "journal" | "flange" | "tail"
    z_start      : Axial start from spindle nose [mm]
    length       : Segment length L_i [mm]
    R_outer      : Outer radius R_i [mm]
    R_inner      : Inner bore radius r_i [mm]
    rho          : Density [ton/mm³]
    bore_offset  : Radial offset of bore axis from outer-surface axis ε_i [mm]
                   Sign: positive = bore shifted in +Y direction.
                   CG shift is in −Y direction (see radial_cg_offset_mm).
    """
    name:        str
    z_start:     float
    length:      float
    R_outer:     float
    R_inner:     float
    rho:         float
    bore_offset: float = 0.0

    @property
    def z_end(self) -> float:
        return self.z_start + self.length

    @property
    def z_cg(self) -> float:
        return self.z_start + self.length / 2.0

    @property
    def cross_area(self) -> float:
        return math.pi * (self.R_outer**2 - self.R_inner**2)

    @property
    def volume(self) -> float:
        return self.cross_area * self.length

    @property
    def mass_ton(self) -> float:
        return self.rho * self.volume

    @property
    def mass_kg(self) -> float:
        return self.mass_ton * 1000.0

    @property
    def radial_cg_offset_mm(self) -> float:
        """
        Radial CG shift of the hollow segment due to bore eccentricity ε.

        Derivation — "missing mass" principle:
            Hollow = Full disc − Bore disc
            CG of full disc = on spin axis → contribution 0
            CG of bore disc  = ε from spin axis  (mass = ρ π r² L)

            m_hollow × e_cg_hollow = −m_bore × ε
            ∴ e_cg = −ε × r² / (R² − r²)

        Magnitude returned (sign analysis handled by caller):
            |e_cg_i| = |ε_i| × r_i² / (R_i² − r_i²)

        Note: The CG shift is in the OPPOSITE direction of ε (negative sign).
              In worst-case linear summation the sign is ignored.
              In angular Monte Carlo analysis include the minus sign.
        """
        denom = self.R_outer**2 - self.R_inner**2
        if denom <= 0:
            return 0.0
        return abs(self.bore_offset) * self.R_inner**2 / denom


# ─────────────────────────────────────────────────────────────────────────────
# Per-segment imbalance force (FIX-B)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SegmentImbalanceForce:
    """
    Individual imbalance force contribution from one shaft segment.

    Applied at z_cg_mm in the ANSYS harmonic model as a rotating vector.
    """
    segment_name:  str
    z_cg_mm:       float    # Axial position of segment CG [mm]
    m_kg:          float    # Segment mass [kg]
    e_mm:          float    # Segment eccentricity [mm]
    F_N:           float    # F_i = m_i × e_i × ω² [N]
    delta_z_mm:    float    # z_cg_i − z_CG_system [mm]  (for couple calculation)


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EccentricityResult:
    """
    Complete static + dynamic imbalance results for the rotor system.

    Static imbalance  : net CG offset → excites cylindrical (translational) whirl
    Couple imbalance  : moment from axial distribution → excites conical whirl
    """
    # ── System CG ─────────────────────────────────────────────────────────
    z_cg_mm:           float    # Axial CG position from nose [mm]
    e_static_mm:       float    # Static eccentricity worst-case [mm]
    e_static_rss_mm:   float    # Static eccentricity RSS [mm]

    # ── Static imbalance ──────────────────────────────────────────────────
    U_static_gmm:      float    # U = m_total × e_CG [g·mm]
    U_allow_gmm:       float    # ISO 1940-1 allowable U [g·mm]
    U_excess_factor:   float    # U_static / U_allow

    # ── Couple (dynamic) imbalance (FIX-B) ────────────────────────────────
    couple_gmm2:       float    # C = Σ U_i × Δz_i  [g·mm²]
    F_couple_N:        float    # Couple force magnitude at bearing span ends [N]
    bearing_span_mm:   float    # Span used for F_couple = C / span [mm]

    # ── Per-segment forces (FIX-B) ────────────────────────────────────────
    segment_forces:    List[SegmentImbalanceForce]

    # ── System imbalance force ────────────────────────────────────────────
    n_rpm:             float
    F_imbalance_N:     float    # System CG: F = m_total × e_static × ω² [N]
    F_imbalance_rss_N: float

    # ── Mass and grade ────────────────────────────────────────────────────
    segments:          List[ShaftSegment]
    m_total_kg:        float
    balance_grade:     str

    # ── ANSYS export ──────────────────────────────────────────────────────
    ansys_force_node_z_mm: float  # z_CG for reference (forces now per-segment)
    ansys_F_imbal_N:       float  # Largest single-segment force (for reference)

    @property
    def needs_balancing(self) -> bool:
        return self.U_static_gmm > self.U_allow_gmm

    @property
    def e_static_um(self) -> float:
        return self.e_static_mm * 1000.0

    @property
    def e_static_rss_um(self) -> float:
        return self.e_static_rss_mm * 1000.0


@dataclass
class EccentricityConstraints:
    """Signed constraints.  g ≤ 0 = satisfied."""
    g_static_imbalance: float    # (U_static − U_allow) / U_allow
    g_couple_imbalance: float    # (C − C_allow) / C_allow
    g_force:            float    # (F_max_segment − F_limit) / F_limit
    C_allow_gmm2:       float    # Reference couple imbalance limit

    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.g_static_imbalance, self.g_couple_imbalance, self.g_force])

    @property
    def all_satisfied(self) -> bool:
        return bool(np.all(self.as_array <= 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Main analyser
# ─────────────────────────────────────────────────────────────────────────────

class RotorEccentricityAnalyser:
    """
    Computes static + couple imbalance for the stepped hollow spindle.

    Parameters
    ----------
    balance_grade     : ISO 1940-1 grade string.  "G2.5" for machine tool spindles.
    bore_offset_mm    : Uniform bore eccentricity per segment [mm].
                        Use ½ × bore grinding tolerance (typ. 5 μm for P5 boring).
    F_imbal_limit_N   : Maximum allowable per-segment imbalance force [N].
    bearing_span_mm   : Axial span between front and rear bearing centroids [mm].
                        Used to convert couple imbalance C to force pair.
                        Default 200 mm; overridden by caller when available.
    """

    def __init__(
        self,
        balance_grade:    str   = "G2.5",
        bore_offset_mm:   float = 0.005,
        F_imbal_limit_N:  float = 50.0,
        bearing_span_mm:  float = 200.0,
    ):
        self.grade          = balance_grade
        self.G              = G_GRADE_MM_S.get(balance_grade, 2.5)
        self.bore_offset    = bore_offset_mm
        self.F_limit        = F_imbal_limit_N
        self.bearing_span   = bearing_span_mm

    # ─────────────────────────────────────────────────────────────────────────
    def build_segments(self, var_dict: Dict[str, float]) -> List[ShaftSegment]:
        """Build 4 ShaftSegment objects from decoded design vector."""
        L1, L2, L3, L4 = var_dict["L1"], var_dict["L2"], var_dict["L3"], var_dict["L4"]
        R1, R2, R3, R4 = var_dict["R1"], var_dict["R2"], var_dict["R3"], var_dict["R4"]
        ri, rho         = var_dict["ri"], var_dict["rho"]
        z0 = 0.0; z1 = z0+L1; z2 = z1+L2; z3 = z2+L3
        return [
            ShaftSegment("nose",    z0, L1, R1, ri, rho, self.bore_offset),
            ShaftSegment("journal", z1, L2, R2, ri, rho, self.bore_offset),
            ShaftSegment("flange",  z2, L3, R3, ri, rho, self.bore_offset),
            ShaftSegment("tail",    z3, L4, R4, ri, rho, self.bore_offset),
        ]

    # ─────────────────────────────────────────────────────────────────────────
    def analyse(
        self,
        var_dict:            Dict[str, float],
        n_rpm:               float,
        custom_bore_offsets: Optional[List[float]] = None,
        bearing_span_mm:     Optional[float] = None,
    ) -> EccentricityResult:
        """
        Compute full static + couple imbalance analysis.

        Parameters
        ----------
        var_dict             : Decoded design vector
        n_rpm                : Operating speed [RPM]
        custom_bore_offsets  : Optional list of 4 bore eccentricities [mm]
                               for each segment (nose, journal, flange, tail).
        bearing_span_mm      : Front-to-rear bearing span [mm].
                               Overrides constructor default.

        Returns
        -------
        EccentricityResult
        """
        segments = self.build_segments(var_dict)
        if custom_bore_offsets is not None:
            for seg, eps in zip(segments, custom_bore_offsets):
                seg.bore_offset = eps

        span  = bearing_span_mm if bearing_span_mm is not None else self.bearing_span
        omega = n_rpm * RPM_TO_RADS

        # ── System CG (axial) ─────────────────────────────────────────────
        m_total_kg = sum(s.mass_kg for s in segments)
        z_cg       = sum(s.mass_kg * s.z_cg for s in segments) / max(m_total_kg, 1e-12)

        # ── Radial eccentricity ───────────────────────────────────────────
        e_contribs  = [s.mass_kg * s.radial_cg_offset_mm for s in segments]
        e_static_mm = sum(e_contribs) / max(m_total_kg, 1e-12)
        e_rss_mm    = math.sqrt(sum(ec**2 for ec in e_contribs)) / max(m_total_kg, 1e-12)

        # ── Static imbalance ──────────────────────────────────────────────
        # FIX-A — clear formula: U [g·mm] = 1000 × m[kg] × e[mm]
        # Units: 1000 [g/kg] × m[kg] × e[mm] = g·mm  ✓
        U_static_gmm = 1000.0 * m_total_kg * e_static_mm    # g·mm

        # ISO 1940-1 allowable:
        #   e_allow [mm]  = G [mm/s] / ω [rad/s]
        #   U_allow [g·mm] = 1000 [g/kg] × m [kg] × e_allow [mm]
        # FIX-A — comment now matches the actual computation exactly
        e_allow_mm   = self.G / omega if omega > 0 else 0.0
        U_allow_gmm  = 1000.0 * m_total_kg * e_allow_mm      # g·mm  ✓

        excess_factor = U_static_gmm / max(U_allow_gmm, 1e-12)

        # ── System imbalance force (from net CG offset) ───────────────────
        # F [N] = m [kg] × e [m] × ω² [rad²/s²]
        # Note: e converted mm → m via × 1e-3
        F_worst = m_total_kg * (e_static_mm * 1e-3) * omega**2
        F_rss   = m_total_kg * (e_rss_mm   * 1e-3) * omega**2

        # ── Per-segment forces — FIX-B ────────────────────────────────────
        # Apply individual F_i at each segment's CG rather than lumping at z_CG.
        # This automatically captures BOTH static and couple imbalance in ANSYS.
        #
        # F_i [N] = m_i [kg] × e_i [m] × ω²
        seg_forces: List[SegmentImbalanceForce] = []
        for s, e_c in zip(segments, e_contribs):
            e_i_mm = s.radial_cg_offset_mm          # magnitude [mm]
            F_i    = s.mass_kg * (e_i_mm * 1e-3) * omega**2  # [N]
            delta_z = s.z_cg - z_cg                  # [mm] axial offset from system CG
            seg_forces.append(SegmentImbalanceForce(
                segment_name  = s.name,
                z_cg_mm       = s.z_cg,
                m_kg          = s.mass_kg,
                e_mm          = e_i_mm,
                F_N           = F_i,
                delta_z_mm    = delta_z,
            ))

        # ── Couple (dynamic) imbalance — FIX-B ───────────────────────────
        # C [g·mm²] = Σ_i  U_i × Δz_i
        #           = Σ_i  (1000 × m_i × e_i) × (z_{CG,i} − z_{CG,sys})
        # Units: g·mm × mm = g·mm²
        couple_gmm2 = sum(
            1000.0 * sf.m_kg * sf.e_mm * sf.delta_z_mm
            for sf in seg_forces
        )
        # Equivalent force pair at bearing span ends
        F_couple_N = abs(couple_gmm2 * 1e-3) / max(span, 1.0)
        # (1e-3: g·mm² → kg·mm² ; then / span [mm] = kg·mm → N when × ω²...
        #  Actually: C [g·mm²] ÷ span [mm] = [g·mm]; force pair = C/(1000×span) × ω² × m
        #  Simpler physical interpretation: F_couple = |C [kg·m²]| / span [m] — see derivation)
        # Correct formula:
        # C_SI [kg·m²] = couple_gmm2 × 1e-3 × 1e-6 (g→kg, mm²→m²) = couple_gmm2 × 1e-9
        # F_couple [N] = C_SI × ω² / (span [m])
        C_SI_kgm2  = abs(couple_gmm2) * 1e-9           # kg·m²
        F_couple_N = C_SI_kgm2 * omega**2 / max(span * 1e-3, 1e-6)  # N

        # Largest per-segment force (for constraint check)
        F_max_seg = max((sf.F_N for sf in seg_forces), default=0.0)

        return EccentricityResult(
            z_cg_mm            = z_cg,
            e_static_mm        = e_static_mm,
            e_static_rss_mm    = e_rss_mm,
            U_static_gmm       = U_static_gmm,
            U_allow_gmm        = U_allow_gmm,
            U_excess_factor    = excess_factor,
            couple_gmm2        = couple_gmm2,
            F_couple_N         = F_couple_N,
            bearing_span_mm    = span,
            segment_forces     = seg_forces,
            n_rpm              = n_rpm,
            F_imbalance_N      = F_worst,
            F_imbalance_rss_N  = F_rss,
            segments           = segments,
            m_total_kg         = m_total_kg,
            balance_grade      = self.grade,
            ansys_force_node_z_mm = z_cg,
            ansys_F_imbal_N       = F_max_seg,
        )

    # ─────────────────────────────────────────────────────────────────────────
    def check_constraints(
        self,
        result: EccentricityResult,
    ) -> EccentricityConstraints:
        """
        Build signed constraint vector (g ≤ 0 = satisfied).

        Three constraints:
            1. Static imbalance U ≤ U_allow   (ISO 1940-1)
            2. Couple imbalance C ≤ C_allow   (ISO 1940-2, couple quality grade)
            3. Max per-segment force ≤ F_limit
        """
        g_static = (result.U_static_gmm - result.U_allow_gmm) / max(result.U_allow_gmm, 1e-12)

        # Couple allowable: ISO 1940-2 suggests C_allow ≈ U_allow × d_span / 2
        # where d_span is the bearing span.  This is an approximate guideline.
        C_allow_gmm2 = result.U_allow_gmm * result.bearing_span_mm / 2.0
        g_couple = (abs(result.couple_gmm2) - C_allow_gmm2) / max(C_allow_gmm2, 1e-12)

        g_force  = (result.ansys_F_imbal_N - self.F_limit) / max(self.F_limit, 1e-12)

        return EccentricityConstraints(
            g_static_imbalance = g_static,
            g_couple_imbalance = g_couple,
            g_force            = g_force,
            C_allow_gmm2       = C_allow_gmm2,
        )

    # ─────────────────────────────────────────────────────────────────────────
    def ansys_input_block(self, result: EccentricityResult) -> str:
        """
        Generate ANSYS MAPDL commands to apply rotating imbalance forces.

        FIX-B: Forces applied at each SEGMENT CG individually, not lumped at
        the system CG.  This excites both cylindrical and conical bending modes.

        FIX-C: Correct synchronous rotating vector definition for ANSYS
        Harmonic Response Analysis:
            Real  (X):  F_node, FX,  F_i,    0
            Imag  (Y):  F_node, FY,  0,   −F_i    ← negative = forward whirl
                                                      (+Z spin axis, right-hand rule)
        ANSYS interprets (real_X, imag_Y) = (F, −F) as a counter-clockwise
        rotating force vector in the XY plane:
            F_x(t) =  F_i × cos(ωt)
            F_y(t) =  F_i × sin(ωt)   [forward whirl]
        """
        lines = [
            "! ══════════════════════════════════════════════════════════════",
            f"! Rotating Imbalance Force Block  (ISO 1940-1 Grade {result.balance_grade})",
            "! ══════════════════════════════════════════════════════════════",
            f"! m_total        = {result.m_total_kg:.4f} kg",
            f"! e_CG (worst)   = {result.e_static_um:.2f} μm",
            f"! U_static       = {result.U_static_gmm:.1f} g·mm",
            f"! U_allow        = {result.U_allow_gmm:.1f} g·mm",
            f"! Couple C       = {result.couple_gmm2:.1f} g·mm²",
            f"! F_couple       = {result.F_couple_N:.2f} N  (at span ends)",
            f"! Speed          = {result.n_rpm:.0f} RPM",
            "! ",
            "! Per-segment application (FIX-B): captures static + couple imbalance",
            "! Rotating vector (FIX-C): real FX + imaginary FY = forward whirl",
            "! ",
            "/SOLU",
        ]

        for sf in result.segment_forces:
            if sf.F_N < 1e-8:
                continue   # skip negligible forces
            lines += [
                f"! ── Segment: {sf.segment_name}  z={sf.z_cg_mm:.1f}mm  "
                f"e={sf.e_mm*1000:.2f}μm  F={sf.F_N:.4f}N ──",
                f"NSEL, S, LOC, Z, {sf.z_cg_mm:.3f}",
                f"*GET, N_{sf.segment_name.upper()}, NODE, 0, NUM, MIN",
                f"F, N_{sf.segment_name.upper()}, FX,  {sf.F_N:.6f},  0.0",
                f"F, N_{sf.segment_name.upper()}, FY,  0.0, {-sf.F_N:.6f}",
                f"!   ^ Real X           ^ Imag Y = −F  (forward whirl, +Z spin axis)",
                "NSEL, ALL",
                "",
            ]

        lines += [
            "ALLSEL",
            "! ══════════════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def print_report(result: EccentricityResult, con: EccentricityConstraints) -> None:
        print(f"\n{'═'*72}")
        print(f"  Rotor Eccentricity Analysis  (ISO 1940-1, Grade {result.balance_grade})")
        print(f"{'═'*72}")

        # Segment table
        print(f"\n  {'Segment':<10} {'m [kg]':>8} {'z_CG [mm]':>10} "
              f"{'Δz [mm]':>9} {'ε_bore [μm]':>11} {'e_CG [μm]':>10} {'F_i [N]':>9}")
        print(f"  {'─'*72}")
        for sf in result.segment_forces:
            s = next(s for s in result.segments if s.name == sf.segment_name)
            print(f"  {sf.segment_name:<10} {sf.m_kg:>8.4f} {sf.z_cg_mm:>10.1f} "
                  f"{sf.delta_z_mm:>+9.1f} {s.bore_offset*1000:>11.1f} "
                  f"{sf.e_mm*1000:>10.3f} {sf.F_N:>9.4f}")
        print(f"  {'─'*72}")
        print(f"  {'TOTAL':<10} {result.m_total_kg:>8.4f}")

        print(f"\n  System CG (axial)      = {result.z_cg_mm:>8.1f} mm  from nose")
        print(f"  e_static (worst-case)  = {result.e_static_um:>8.2f} μm")
        print(f"  e_static (RSS)         = {result.e_static_rss_um:>8.2f} μm")

        print(f"\n  Static Imbalance:")
        print(f"    U_static   = {result.U_static_gmm:>10.2f} g·mm")
        print(f"    U_allow    = {result.U_allow_gmm:>10.2f} g·mm  "
              f"(ISO 1940-1 G{result.balance_grade}, {result.n_rpm:.0f} RPM)")
        print(f"    Excess ×   = {result.U_excess_factor:>10.3f}  "
              f"{'→ BALANCE REQUIRED ❌' if result.needs_balancing else '→ OK ✅'}")

        print(f"\n  Couple (Dynamic) Imbalance:")  # FIX-B
        print(f"    C          = {result.couple_gmm2:>10.1f} g·mm²")
        print(f"    C_allow    = {con.C_allow_gmm2:>10.1f} g·mm²  "
              f"(≈ U_allow × span/2)")
        print(f"    F_couple   = {result.F_couple_N:>10.3f} N  "
              f"(at bearing span {result.bearing_span_mm:.0f} mm)")

        print(f"\n  System Force @ {result.n_rpm:.0f} RPM:")
        print(f"    F (worst)  = {result.F_imbalance_N:>10.4f} N  (from e_static)")
        print(f"    F (RSS)    = {result.F_imbalance_rss_N:>10.4f} N")
        print(f"    F_max_seg  = {result.ansys_F_imbal_N:>10.4f} N  (largest segment)")

        print(f"\n  Constraints:")
        g_vals = con.as_array
        for nm, gi in zip(["static_imbalance", "couple_imbalance", "force_limit"], g_vals):
            print(f"    {'✅' if gi <= 0 else '❌'}  {nm:<20}  g = {gi:+.4f}")
        print(f"  {'ALL OK ✅' if con.all_satisfied else 'VIOLATIONS ❌'}")
        print(f"{'═'*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_eccentricity(
    analyser:  "RotorEccentricityAnalyser",
    var_dict:  Dict[str, float],
    speeds:    Optional[List[float]] = None,
    save_dir:  str = ".",
) -> None:
    """
    Three eccentricity plots:

    Fig 10a — U_static vs. U_allow vs. speed  (ISO 1940-1)
    Fig 10b — Per-segment imbalance force F_i @ 4,000 RPM  (waterfall bar)
    Fig 10c — F_imbalance (total + per-segment max) vs. speed
    """
    import matplotlib.pyplot as plt
    import os

    NAVY="#0d1b2a"; TEAL="#00b4d8"; CORAL="#e63946"
    GOLD="#ffd166"; GRAY="#8d99ae"; MINT="#06d6a0"; PURPLE="#7400b8"
    SEG_COLS = [TEAL, MINT, GOLD, CORAL]
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": NAVY, "axes.facecolor": "#112233",
        "axes.edgecolor": GRAY, "axes.labelcolor": "white",
        "xtick.color": GRAY, "ytick.color": GRAY,
        "text.color": "white", "grid.color": "#2d4060",
        "grid.alpha": 0.5, "font.size": 9,
    })

    if speeds is None:
        speeds = list(range(500, 6500, 250))

    # precompute over speed range
    U_statics, U_allows, F_totals, F_max_segs = [], [], [], []
    for n in speeds:
        r = analyser.analyse(var_dict, n_rpm=n)
        U_statics.append(r.U_static_gmm)
        U_allows.append(r.U_allow_gmm)
        F_totals.append(r.F_imbalance_N)
        F_max_segs.append(r.ansys_F_imbal_N)

    # ── Fig 10a: U_static vs U_allow ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=NAVY)
    ax.set_facecolor("#112233")
    ax.plot(speeds, U_statics, color=CORAL, lw=2, label="U_static (worst-case)")
    ax.plot(speeds, U_allows,  color=GOLD,  lw=2, linestyle="--",
            label="U_allow (ISO 1940-1 G2.5)")
    ax.fill_between(speeds, U_statics, U_allows,
                    where=[u > a for u, a in zip(U_statics, U_allows)],
                    color=CORAL, alpha=0.25, label="Balancing required zone")
    ax.axvline(4000, color=GRAY, lw=0.8, linestyle=":", alpha=0.7, label="4,000 RPM")
    ax.axvline(6000, color=GRAY, lw=0.6, linestyle=":", alpha=0.5, label="6,000 RPM")
    ax.set_xlabel("Operating speed [RPM]")
    ax.set_ylabel("Imbalance [g·mm]")
    ax.set_title("Fig 10a — Static Imbalance U vs. ISO 1940-1 G2.5 Allowable", pad=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(save_dir, "10a_imbalance_vs_speed.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig); print(f"  Saved → {p}")

    # ── Fig 10b: Per-segment F_i @ 4,000 RPM ─────────────────────────────
    r4 = analyser.analyse(var_dict, n_rpm=4000)
    seg_names = [sf.segment_name for sf in r4.segment_forces]
    seg_F     = [sf.F_N        for sf in r4.segment_forces]
    seg_e     = [sf.e_mm*1000  for sf in r4.segment_forces]  # μm

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=NAVY)
    for ax in (ax1, ax2):
        ax.set_facecolor("#112233")

    bars = ax1.bar(seg_names, seg_F, color=SEG_COLS[:len(seg_names)],
                   edgecolor=NAVY, linewidth=0.5)
    for bar, val in zip(bars, seg_F):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f"{val:.4f} N", ha="center", va="bottom", fontsize=8, color="white")
    ax1.axhline(analyser.F_limit, color=GOLD, lw=1.5, linestyle="--",
                label=f"F_limit = {analyser.F_limit} N")
    ax1.set_ylabel("Imbalance force F_i [N]")
    ax1.set_title("Per-Segment F_i @ 4,000 RPM", pad=8)
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)

    ax2.bar(seg_names, seg_e, color=SEG_COLS[:len(seg_names)],
            edgecolor=NAVY, linewidth=0.5)
    ax2.set_ylabel("Segment eccentricity e_i [μm]")
    ax2.set_title("Per-Segment CG Eccentricity", pad=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Fig 10b — Per-Segment Imbalance Breakdown (ISO 1940-1 G2.5)",
                 color="white", y=1.02)
    plt.tight_layout()
    p = os.path.join(save_dir, "10b_per_segment_imbalance.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig); print(f"  Saved → {p}")

    # ── Fig 10c: F_imbalance vs speed ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=NAVY)
    ax.set_facecolor("#112233")
    ax.plot(speeds, F_totals,   color=TEAL, lw=2, label="F total (from e_static)")
    ax.plot(speeds, F_max_segs, color=MINT, lw=1.5, linestyle="--",
            label="F max per segment")
    ax.axhline(analyser.F_limit, color=GOLD, lw=1.5, linestyle="--",
               label=f"F_limit = {analyser.F_limit} N")
    ax.axvline(4000, color=GRAY, lw=0.8, linestyle=":", alpha=0.7)
    ax.axvline(6000, color=GRAY, lw=0.6, linestyle=":", alpha=0.5)
    ax.set_xlabel("Operating speed [RPM]")
    ax.set_ylabel("Imbalance force [N]")
    ax.set_title("Fig 10c — Rotating Imbalance Force vs. Speed  (F ∝ ω²)", pad=10)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(save_dir, "10c_imbalance_force_vs_speed.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig); print(f"  Saved → {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from design_variables import DesignSpace

    ds  = DesignSpace()
    nom = ds.decode_vector(ds.get_nominal())

    analyser = RotorEccentricityAnalyser(
        balance_grade   = "G2.5",
        bore_offset_mm  = 0.005,
        F_imbal_limit_N = 50.0,
        bearing_span_mm = 200.0,
    )

    r4 = analyser.analyse(nom, n_rpm=4000, bearing_span_mm=200.0)
    c4 = analyser.check_constraints(r4)
    RotorEccentricityAnalyser.print_report(r4, c4)

    r6 = analyser.analyse(nom, n_rpm=6000, bearing_span_mm=200.0)
    RotorEccentricityAnalyser.print_report(r6, analyser.check_constraints(r6))

    # ── Physics checks ────────────────────────────────────────────────────
    print("── Physics checks ──")

    # FIX-A verification: U_allow formula
    omega4 = 4000 * 2 * math.pi / 60
    e_allow_expected = 2.5 / omega4          # mm
    U_allow_expected = 1000 * r4.m_total_kg * e_allow_expected
    assert abs(r4.U_allow_gmm - U_allow_expected) < 0.01, \
        f"U_allow mismatch: {r4.U_allow_gmm:.2f} vs {U_allow_expected:.2f}"
    print(f"✅  FIX-A: U_allow = 1000 × m × G/ω = {r4.U_allow_gmm:.1f} g·mm  ✓")

    # FIX-B verification: 4 per-segment forces (not 1 lumped)
    assert len(r4.segment_forces) == 4, \
        f"Expected 4 segment forces, got {len(r4.segment_forces)}"
    print(f"✅  FIX-B: {len(r4.segment_forces)} per-segment forces (not 1 lumped)")

    # Couple imbalance exists and is non-zero (segments at different z → moment)
    assert abs(r4.couple_gmm2) > 0.0
    print(f"✅  FIX-B: Couple C = {r4.couple_gmm2:.2f} g·mm² ≠ 0")

    # Higher speed → higher imbalance forces
    assert r6.F_imbalance_N > r4.F_imbalance_N
    print("✅  F_imbal increases with speed (F ∝ ω²)")

    # Higher speed → smaller U_allow (tighter balance requirement)
    assert r6.U_allow_gmm < r4.U_allow_gmm
    print("✅  U_allow tightens at higher speed")

    # Larger bore offset → larger eccentricity
    a_loose = RotorEccentricityAnalyser(bore_offset_mm=0.020)
    a_tight = RotorEccentricityAnalyser(bore_offset_mm=0.001)
    assert a_loose.analyse(nom, 4000).e_static_mm > a_tight.analyse(nom, 4000).e_static_mm
    print("✅  Larger bore offset → larger eccentricity")

    # FIX-C verification: ANSYS block contains FX real + FY imaginary
    blk = analyser.ansys_input_block(r4)
    assert "FX" in blk and "FY" in blk
    assert "forward whirl" in blk
    print("✅  FIX-C: ANSYS block has FX (real) + FY (imag) rotating vector")

    # Plots
    import os
    os.makedirs("/tmp/spindle_plots", exist_ok=True)
    print("\nGenerating eccentricity plots...")
    plot_eccentricity(analyser, nom, speeds=list(range(500, 6500, 500)),
                      save_dir="/tmp/spindle_plots")
    print("✅  Module 10 v2 — all checks + plots done\n")
