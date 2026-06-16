#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  bearing_catalog.py  —  Multi-Manufacturer Bearing Catalog
================================================================================

  Provides bearing data for six major manufacturers used in CNC lathe spindle
  assemblies:

      SKF  (Sweden)        — most widely referenced in machine-tool literature
      FAG / Schaeffler     — common in European CNC (Schaublin, DMG Mori, etc.)
      NSK  (Japan)         — dominant in Japanese CNC (Fanuc, Makino, etc.)
      NTN  (Japan)         — popular alternative to NSK, strong precision range
      JTEKT / Koyo (Japan) — OEM supplier to Toyota/Mazda machine tools
      Timken (USA)         — strong for tapered-roller; angular-contact line used
                             where high-thrust-load capacity is required

  ANGULAR-CONTACT BALL BEARINGS (ACBB)
  ─────────────────────────────────────
  All ACBB entries are for the standard spindle-grade (15° / 25° contact
  angle) single-row angular-contact series unless otherwise noted.  Speed
  ratings given are for grease lubrication with a normal cage; oil-jet or
  air-oil ratings are 30–40% higher.

  Data sources (approximate catalog values — verify against current mfr. PDFs
  before use in a final design):
      SKF:    Rolling Bearings Catalogue (PUB BU/P1 15000/2/EN, 2022 edition)
      FAG:    FAG Super Precision Bearings Catalogue (WL 82 102/2 EA, 2021)
      NSK:    Super-Precision Bearings Catalogue (CAT.No.E7004g, 2022)
      NTN:    Super Precision Bearings Catalogue (3220-X/E, 2021)
      JTEKT:  Precision Machine Tool Bearings Catalogue (B2008E, 2021)
      Timken: Single Row Angular Contact Bearings (10675, 2022)

  CYLINDRICAL ROLLER BEARINGS (CRB)
  ──────────────────────────────────
  NU/NJ/N-type CRBs used for rear/secondary bearing stations where axial
  displacement must be accommodated.  All entries are NU2xxx series (or
  nearest equivalent designation per manufacturer).

  STIFFNESS MODEL
  ───────────────
  Individual-bearing radial stiffness is estimated from Palmgren's formula:
      ACBB: k_single ≈ 5.5 × d^0.75 × 1000  [N/mm]   (contact angle 25°)
      CRB:  k_single ≈ 8.0 × d^0.80 × 1000  [N/mm]
  For a back-to-back duplex ACBB pair: k_pair = 1.7 × k_single.
  These are approximations; catalogue suppliers publish more detailed
  stiffness vs. preload curves.

  UNITS
  ─────
      C_r, C_0r  : N (Newtons)
      d, D, B    : mm
      n_grease, n_oil : rpm
      mass_kg    : kg
================================================================================
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Universal Bearing Dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Bearing:
    """
    Manufacturer-agnostic bearing record.

    Works for both angular-contact ball bearings (ACBB) and cylindrical
    roller bearings (CRB); the bearing_type field selects the stiffness
    model branch.

    Parameters
    ----------
    manufacturer    : e.g. "SKF", "FAG", "NSK", "NTN", "JTEKT", "Timken"
    designation     : manufacturer part number, e.g. "7209 BECBP" / "B7209-C-T-P4S"
    d               : bore diameter [mm]
    D               : outer diameter [mm]
    B               : width [mm]
    C_r             : basic dynamic radial load rating [N]
    C_0r            : basic static radial load rating [N]
    n_grease        : reference speed (grease lubrication) [rpm]
    n_oil           : limiting speed (oil lubrication) [rpm]
    mass_kg         : bearing mass [kg]
    contact_angle_deg : nominal contact angle [°] (ACBB: 15 or 25; CRB: 0)
    precision_class : ISO tolerance class — "P5" (normal), "P4", "P2"
                      ABEC equivalents: P5=ABEC-9, P4=ABEC-7, P2=ABEC-5
    F_preload_MA_N  : matched-pair medium-preload force [N] (ACBB only)
    bearing_type    : "ACBB" or "CRB"
    """
    manufacturer:       str
    designation:        str
    d:                  float      # bore [mm]
    D:                  float      # OD [mm]
    B:                  float      # width [mm]
    C_r:                float      # dynamic rating [N]
    C_0r:               float      # static rating [N]
    n_grease:           int        # grease speed [rpm]
    n_oil:              int        # oil speed [rpm]
    mass_kg:            float
    contact_angle_deg:  float = 25.0
    precision_class:    str   = "P5"
    F_preload_MA_N:     float = 0.0
    preload_class:      str   = "B"      # "A"=light, "B"=medium, "C"=heavy
    bearing_type:       str   = "ACBB"

    # ── Derived properties ────────────────────────────────────────────────
    @property
    def d_m(self) -> float:
        """Pitch diameter [mm]."""
        return (self.d + self.D) / 2.0

    @property
    def radius_bore(self) -> float:
        return self.d / 2.0

    # Published preload-stiffness multipliers (vs medium preload = 1.0)
    # Source: SKF/FAG technical guides; Palmgren (1959) bearing approximation
    # Measured range: light preload ~0.55×, medium ~1.0×, heavy ~1.40×
    _PRELOAD_FACTOR: "ClassVar[Dict[str, float]]" = {
        "A": 0.55,   # light  — minimal thermal runout, highest speed rating
        "B": 1.00,   # medium — balanced rigidity / speed (default for lathe spindle)
        "C": 1.40,   # heavy  — maximum stiffness, lowest speed, highest heat generation
    }

    @property
    def radial_stiffness_single_N_mm(self) -> float:
        """
        Approximate radial stiffness [N/mm] for a single bearing.

        Uses Palmgren d^0.75 base formula scaled by preload class:
            ACBB: k ≈ 5.5 × d^0.75 × 1000 × preload_factor
            CRB:  k ≈ 8.0 × d^0.80 × 1000            (preload not applied to CRB)

        Palmgren base formula is calibrated to medium preload (class B).
        Published stiffness data for 7222 ACBB (d=110mm):
            Light  (A): ~100 N/µm  →  preload_factor ≈ 0.55
            Medium (B): ~180 N/µm  →  preload_factor ≈ 1.00  (Palmgren matches)
            Heavy  (C): ~250 N/µm  →  preload_factor ≈ 1.40
        Source: FAG WL82102 / SKF Bearing Maintenance Guide.

        Note: Stiffness is ALSO load-dependent (Hertzian contact: k ∝ Q^(1/3)).
        The preload-class factor captures the dominant first-order effect.
        For precision analysis, use the Jones-Harris Hertz model.
        """
        pf = self._PRELOAD_FACTOR.get(self.preload_class, 1.0)
        if self.bearing_type == "CRB":
            return 8.0 * (self.d ** 0.80) * 1000.0   # CRB: preload via lock-nut
        else:
            # Bore-size correction: Palmgren overestimates for d<75mm
            # where fewer/smaller balls reduce stiffness below the d^0.75 trend.
            # Calibrated against published data (FAG WL82102, SKF catalog):
            #   d=40mm: raw 87kN/mm → correction 0.753 → 66kN/mm (pub 65kN/mm, err 1.5%)
            #   d=80mm: raw 147kN/mm → correction 1.0 → 147kN/mm (pub 130kN/mm, err 13%)
            #   d=110mm: correction 1.0 → 187kN/mm (pub 180kN/mm, err 4%)
            f_bore = min(1.0, (self.d / 75.0) ** 0.45)
            return 5.5 * (self.d ** 0.75) * 1000.0 * pf * f_bore

    @property
    def axial_stiffness_single_N_mm(self) -> float:
        """Approximate axial stiffness [N/mm] using thrust factor."""
        alpha = math.radians(self.contact_angle_deg)
        return self.radial_stiffness_single_N_mm * math.tan(alpha) ** 2

    @property
    def ndm(self) -> float:
        """Speed parameter n·d_m [mm/min] at grease speed limit."""
        return self.d_m * self.n_grease

    def speed_ok(self, n_rpm: float, lubrication: str = "grease") -> bool:
        limit = self.n_grease if lubrication == "grease" else self.n_oil
        return n_rpm <= limit

    def __str__(self) -> str:
        return (f"{self.manufacturer} {self.designation}  "
                f"d={self.d:.0f}mm  D={self.D:.0f}mm  B={self.B:.0f}mm  "
                f"C_r={self.C_r/1000:.1f}kN  "
                f"n_gr={self.n_grease:,}rpm  [{self.precision_class}]")


# ─────────────────────────────────────────────────────────────────────────────
# SKF — Angular Contact Ball Bearings (7xxx BECBP series, P5, 25° contact)
# Source: SKF Rolling Bearings Catalogue PUB BU/P1 15000/2/EN
# ─────────────────────────────────────────────────────────────────────────────
_SKF_ACBB: List[Bearing] = [
    Bearing("SKF","7206 BECBP", 30, 62, 16,  19900,  13700, 18000, 24000, 0.12, 25, "P5",  150),
    Bearing("SKF","7207 BECBP", 35, 72, 17,  25700,  17900, 16000, 21000, 0.17, 25, "P5",  190),
    Bearing("SKF","7208 BECBP", 40, 80, 18,  30500,  21600, 14000, 19000, 0.23, 25, "P5",  230),
    Bearing("SKF","7209 BECBP", 45, 85, 19,  33200,  23600, 12000, 16000, 0.27, 25, "P5",  270),
    Bearing("SKF","7210 BECBP", 50, 90, 20,  35500,  25500, 11000, 15000, 0.31, 25, "P5",  320),
    Bearing("SKF","7211 BECBP", 55,100, 21,  43000,  32000,  9500, 13000, 0.42, 25, "P5",  390),
    Bearing("SKF","7212 BECBP", 60,110, 22,  51500,  38500,  8500, 11000, 0.55, 25, "P5",  470),
    Bearing("SKF","7213 BECBP", 65,120, 23,  61000,  46500,  7500, 10000, 0.70, 25, "P5",  560),
    Bearing("SKF","7214 BECBP", 70,125, 24,  63000,  49000,  6700,  9000, 0.75, 25, "P5",  630),
    Bearing("SKF","7215 BECBP", 75,130, 25,  66500,  53000,  6300,  8500, 0.85, 25, "P5",  710),
    Bearing("SKF","7216 BECBP", 80,140, 26,  73500,  60000,  5600,  7500, 1.05, 25, "P5",  800),
    Bearing("SKF","7217 BECBP", 85,150, 28,  87000,  73500,  5300,  7000, 1.35, 25, "P5",  900),
    Bearing("SKF","7218 BECBP", 90,160, 30,  98000,  80000,  4800,  6300, 1.65, 25, "P5", 1000),
    Bearing("SKF","7219 BECBP", 95,170, 32, 110000,  90000,  4500,  6000, 2.05, 25, "P5", 1100),
    Bearing("SKF","7220 BECBP",100,180, 34, 118000,  98000,  4300,  5600, 2.50, 25, "P5", 1200),
    Bearing("SKF","7221 BECBP",105,190, 36, 130000, 108000,  4000,  5300, 3.00, 25, "P5", 1350),
    Bearing("SKF","7222 BECBP",110,200, 38, 143000, 118000,  3800,  5000, 3.60, 25, "P5", 1500),
    Bearing("SKF","7224 BECBP",120,215, 40, 153000, 130000,  3400,  4500, 4.25, 25, "P5", 1700),
]

# ─────────────────────────────────────────────────────────────────────────────
# FAG / Schaeffler — B7xxx-C-T-P4S Super-Precision ACBB
# Contact angle 15° (code C), TN9 cage, P4 precision class
# Source: FAG Super Precision Bearings WL 82 102/2 EA
# Note: 15° contact angle → higher speed rating but lower axial load capacity
#       than SKF's 25° series; preferred for very-high-speed light-to-medium
#       load spindle front positions
# ─────────────────────────────────────────────────────────────────────────────
_FAG_ACBB: List[Bearing] = [
    Bearing("FAG","B7206-C-T-P4S", 30, 62, 16,  17900,  11600, 20000, 28000, 0.12, 15, "P4",  140),
    Bearing("FAG","B7207-C-T-P4S", 35, 72, 17,  22600,  15100, 18000, 24000, 0.17, 15, "P4",  175),
    Bearing("FAG","B7208-C-T-P4S", 40, 80, 18,  27600,  18500, 16000, 22000, 0.23, 15, "P4",  215),
    Bearing("FAG","B7209-C-T-P4S", 45, 85, 19,  28800,  19800, 14000, 19000, 0.27, 15, "P4",  250),
    Bearing("FAG","B7210-C-T-P4S", 50, 90, 20,  33500,  23200, 13000, 18000, 0.31, 15, "P4",  295),
    Bearing("FAG","B7211-C-T-P4S", 55,100, 21,  39400,  28600, 11000, 15000, 0.42, 15, "P4",  360),
    Bearing("FAG","B7212-C-T-P4S", 60,110, 22,  46200,  34000,  9500, 13000, 0.55, 15, "P4",  430),
    Bearing("FAG","B7213-C-T-P4S", 65,120, 23,  53000,  40500,  8500, 12000, 0.70, 15, "P4",  505),
    Bearing("FAG","B7214-C-T-P4S", 70,125, 24,  62500,  48500,  7500, 10000, 0.75, 15, "P4",  580),
    Bearing("FAG","B7215-C-T-P4S", 75,130, 25,  65500,  52500,  7000,  9500, 0.85, 15, "P4",  650),
    Bearing("FAG","B7216-C-T-P4S", 80,140, 26,  76500,  63000,  6300,  8500, 1.05, 15, "P4",  730),
    Bearing("FAG","B7217-C-T-P4S", 85,150, 28,  85000,  71500,  5600,  7500, 1.35, 15, "P4",  820),
    Bearing("FAG","B7218-C-T-P4S", 90,160, 30,  98500,  83500,  5300,  7000, 1.65, 15, "P4",  920),
    Bearing("FAG","B7219-C-T-P4S", 95,170, 32, 112000,  96500,  4800,  6300, 2.05, 15, "P4", 1020),
    Bearing("FAG","B7220-C-T-P4S",100,180, 34, 125000, 109000,  4500,  6000, 2.50, 15, "P4", 1130),
    Bearing("FAG","B7221-C-T-P4S",105,190, 36, 136000, 120000,  4300,  5600, 3.00, 15, "P4", 1260),
    Bearing("FAG","B7222-C-T-P4S",110,200, 38, 148000, 132000,  4000,  5300, 3.60, 15, "P4", 1390),
    Bearing("FAG","B7224-C-T-P4S",120,215, 40, 163000, 152000,  3600,  4800, 4.25, 15, "P4", 1580),
]

# ─────────────────────────────────────────────────────────────────────────────
# NSK — 7xxx CTYNDBL Super-Precision ACBB
# Contact angle 15° (C suffix), PEEK cage (Y), sealed (D), back-to-back (BL)
# Source: NSK Super-Precision Bearings CAT.No.E7004g
# Note: NSK NSKHPS (High-Performance Standard) bearings designed for
#       spindle motor direct-drive at extreme speeds; n_oil ratings reflect
#       this focus on high-speed capability
# ─────────────────────────────────────────────────────────────────────────────
_NSK_ACBB: List[Bearing] = [
    Bearing("NSK","7206CTYNDBL",  30, 62, 16,  17200,  11200, 22000, 30000, 0.12, 15, "P4",  135),
    Bearing("NSK","7207CTYNDBL",  35, 72, 17,  21800,  14600, 19000, 26000, 0.17, 15, "P4",  170),
    Bearing("NSK","7208CTYNDBL",  40, 80, 18,  26500,  18000, 17000, 23000, 0.23, 15, "P4",  205),
    Bearing("NSK","7209CTYNDBL",  45, 85, 19,  29300,  20200, 15000, 20000, 0.27, 15, "P4",  240),
    Bearing("NSK","7210CTYNDBL",  50, 90, 20,  33000,  23000, 13000, 18000, 0.31, 15, "P4",  280),
    Bearing("NSK","7211CTYNDBL",  55,100, 21,  40500,  29500, 11000, 15000, 0.42, 15, "P4",  345),
    Bearing("NSK","7212CTYNDBL",  60,110, 22,  47500,  35500,  9500, 13000, 0.55, 15, "P4",  410),
    Bearing("NSK","7213CTYNDBL",  65,120, 23,  56500,  43500,  8500, 11500, 0.70, 15, "P4",  490),
    Bearing("NSK","7214CTYNDBL",  70,125, 24,  61500,  48000,  7500, 10500, 0.75, 15, "P4",  565),
    Bearing("NSK","7215CTYNDBL",  75,130, 25,  65500,  52500,  7000,  9500, 0.85, 15, "P4",  635),
    Bearing("NSK","7216CTYNDBL",  80,140, 26,  75000,  61000,  6300,  8500, 1.05, 15, "P4",  715),
    Bearing("NSK","7217CTYNDBL",  85,150, 28,  88000,  73000,  5600,  7500, 1.35, 15, "P4",  810),
    Bearing("NSK","7218CTYNDBL",  90,160, 30, 100000,  84000,  5300,  7000, 1.65, 15, "P4",  905),
    Bearing("NSK","7219CTYNDBL",  95,170, 32, 112000,  96000,  4800,  6300, 2.05, 15, "P4", 1010),
    Bearing("NSK","7220CTYNDBL", 100,180, 34, 122000, 107000,  4500,  6000, 2.50, 15, "P4", 1120),
    Bearing("NSK","7221CTYNDBL", 105,190, 36, 135000, 119000,  4200,  5600, 3.00, 15, "P4", 1240),
    Bearing("NSK","7222CTYNDBL", 110,200, 38, 146000, 131000,  4000,  5300, 3.60, 15, "P4", 1360),
    Bearing("NSK","7224CTYNDBL", 120,215, 40, 163000, 150000,  3600,  4800, 4.25, 15, "P4", 1560),
]

# ─────────────────────────────────────────────────────────────────────────────
# NTN — 7xxxT1G/GN Angular Contact Ball Bearings
# Contact angle 25° (standard) with high-precision cage
# Source: NTN Super Precision Bearings catalogue 3220-X/E
# ─────────────────────────────────────────────────────────────────────────────
_NTN_ACBB: List[Bearing] = [
    Bearing("NTN","7206T1G",  30, 62, 16,  18900,  13000, 17000, 23000, 0.12, 25, "P5",  145),
    Bearing("NTN","7207T1G",  35, 72, 17,  24100,  17000, 15000, 20000, 0.17, 25, "P5",  180),
    Bearing("NTN","7208T1G",  40, 80, 18,  29000,  21000, 13000, 18000, 0.23, 25, "P5",  220),
    Bearing("NTN","7209T1G",  45, 85, 19,  31500,  22800, 11000, 15000, 0.27, 25, "P5",  255),
    Bearing("NTN","7210T1G",  50, 90, 20,  34000,  24500, 10500, 14000, 0.31, 25, "P5",  300),
    Bearing("NTN","7211T1G",  55,100, 21,  41500,  31000,  9000, 12000, 0.42, 25, "P5",  370),
    Bearing("NTN","7212T1G",  60,110, 22,  49500,  37500,  8000, 10500, 0.55, 25, "P5",  445),
    Bearing("NTN","7213T1G",  65,120, 23,  59000,  45500,  7200,  9500, 0.70, 25, "P5",  530),
    Bearing("NTN","7214T1G",  70,125, 24,  61000,  47800,  6400,  8500, 0.75, 25, "P5",  605),
    Bearing("NTN","7215T1G",  75,130, 25,  64500,  51500,  6000,  8000, 0.85, 25, "P5",  685),
    Bearing("NTN","7216T1G",  80,140, 26,  72000,  58500,  5300,  7000, 1.05, 25, "P5",  770),
    Bearing("NTN","7217T1G",  85,150, 28,  85000,  72000,  5000,  6700, 1.35, 25, "P5",  870),
    Bearing("NTN","7218T1G",  90,160, 30,  96000,  79000,  4600,  6000, 1.65, 25, "P5",  965),
    Bearing("NTN","7219T1G",  95,170, 32, 108000,  90000,  4300,  5600, 2.05, 25, "P5", 1060),
    Bearing("NTN","7220T1G", 100,180, 34, 116000,  97000,  4000,  5300, 2.50, 25, "P5", 1165),
    Bearing("NTN","7221T1G", 105,190, 36, 128000, 107000,  3800,  5000, 3.00, 25, "P5", 1290),
    Bearing("NTN","7222T1G", 110,200, 38, 140000, 117000,  3600,  4800, 3.60, 25, "P5", 1420),
    Bearing("NTN","7224T1G", 120,215, 40, 152000, 130000,  3200,  4300, 4.25, 25, "P5", 1620),
]

# ─────────────────────────────────────────────────────────────────────────────
# JTEKT / Koyo — 7xxx precision series (machine tool spindle range)
# Contact angle 25°, pressed-steel or polyamide cage, P5 precision
# Source: JTEKT Precision Machine Tool Bearings B2008E
# Note: JTEKT supplies direct to machine-tool OEMs (Toyota Industries, Okuma)
#       and often preferred where Japanese-OEM compatibility is required
# ─────────────────────────────────────────────────────────────────────────────
_JTEKT_ACBB: List[Bearing] = [
    Bearing("JTEKT","7206B",  30, 62, 16,  19000,  13200, 17000, 22000, 0.12, 25, "P5",  148),
    Bearing("JTEKT","7207B",  35, 72, 17,  24600,  17300, 15000, 20000, 0.17, 25, "P5",  185),
    Bearing("JTEKT","7208B",  40, 80, 18,  29500,  21100, 13000, 18000, 0.23, 25, "P5",  228),
    Bearing("JTEKT","7209B",  45, 85, 19,  32200,  23000, 11500, 15500, 0.27, 25, "P5",  265),
    Bearing("JTEKT","7210B",  50, 90, 20,  34800,  25000, 10500, 14000, 0.31, 25, "P5",  310),
    Bearing("JTEKT","7211B",  55,100, 21,  42500,  31500,  9000, 12000, 0.42, 25, "P5",  380),
    Bearing("JTEKT","7212B",  60,110, 22,  50500,  37800,  8000, 10500, 0.55, 25, "P5",  455),
    Bearing("JTEKT","7213B",  65,120, 23,  60500,  46000,  7200,  9500, 0.70, 25, "P5",  545),
    Bearing("JTEKT","7214B",  70,125, 24,  62000,  48500,  6500,  8500, 0.75, 25, "P5",  620),
    Bearing("JTEKT","7215B",  75,130, 25,  65500,  52000,  6000,  8000, 0.85, 25, "P5",  700),
    Bearing("JTEKT","7216B",  80,140, 26,  74000,  60500,  5300,  7000, 1.05, 25, "P5",  785),
    Bearing("JTEKT","7217B",  85,150, 28,  87000,  73500,  5000,  6500, 1.35, 25, "P5",  885),
    Bearing("JTEKT","7218B",  90,160, 30,  98500,  81000,  4600,  6000, 1.65, 25, "P5",  980),
    Bearing("JTEKT","7219B",  95,170, 32, 110000,  91500,  4300,  5600, 2.05, 25, "P5", 1080),
    Bearing("JTEKT","7220B", 100,180, 34, 119000,  99000,  4000,  5300, 2.50, 25, "P5", 1190),
    Bearing("JTEKT","7221B", 105,190, 36, 131000, 109000,  3800,  5000, 3.00, 25, "P5", 1315),
    Bearing("JTEKT","7222B", 110,200, 38, 144000, 120000,  3600,  4800, 3.60, 25, "P5", 1445),
    Bearing("JTEKT","7224B", 120,215, 40, 155000, 133000,  3200,  4300, 4.25, 25, "P5", 1640),
]

# ─────────────────────────────────────────────────────────────────────────────
# Timken — Angular Contact Ball Bearings
# Contact angle 25°, P5 class; Timken's primary strength is tapered roller
# but their ACBBs are chosen when combined radial+axial capacity is critical
# Source: Timken Single Row Angular Contact Bearings 10675
# Note: Timken ACBB C_r values are conservative (they assume the more heavily
#       loaded of a pair carries all the thrust), making them suitable for
#       worst-case life calculations
# ─────────────────────────────────────────────────────────────────────────────
_TIMKEN_ACBB: List[Bearing] = [
    Bearing("Timken","206WD",  30, 62, 16,  18200,  12400, 16000, 22000, 0.12, 25, "P5",  140),
    Bearing("Timken","207WD",  35, 72, 17,  24000,  16700, 14000, 19000, 0.17, 25, "P5",  178),
    Bearing("Timken","208WD",  40, 80, 18,  28400,  20000, 12500, 17000, 0.23, 25, "P5",  215),
    Bearing("Timken","209WD",  45, 85, 19,  31200,  22300, 11000, 15000, 0.27, 25, "P5",  250),
    Bearing("Timken","210WD",  50, 90, 20,  33800,  24300, 10000, 14000, 0.31, 25, "P5",  290),
    Bearing("Timken","211WD",  55,100, 21,  41000,  30000,  8500, 11500, 0.42, 25, "P5",  360),
    Bearing("Timken","212WD",  60,110, 22,  49000,  36600,  7500, 10000, 0.55, 25, "P5",  430),
    Bearing("Timken","213WD",  65,120, 23,  58500,  44500,  6700,  9000, 0.70, 25, "P5",  515),
    Bearing("Timken","214WD",  70,125, 24,  60500,  47000,  6000,  8000, 0.75, 25, "P5",  590),
    Bearing("Timken","215WD",  75,130, 25,  64000,  51000,  5600,  7500, 0.85, 25, "P5",  665),
    Bearing("Timken","216WD",  80,140, 26,  71500,  58500,  5000,  6700, 1.05, 25, "P5",  750),
    Bearing("Timken","217WD",  85,150, 28,  85000,  71500,  4800,  6300, 1.35, 25, "P5",  850),
    Bearing("Timken","218WD",  90,160, 30,  95000,  78500,  4300,  5600, 1.65, 25, "P5",  945),
    Bearing("Timken","219WD",  95,170, 32, 107000,  88500,  4000,  5300, 2.05, 25, "P5", 1045),
    Bearing("Timken","220WD", 100,180, 34, 115000,  95500,  3800,  5000, 2.50, 25, "P5", 1150),
    Bearing("Timken","221WD", 105,190, 36, 127000, 105000,  3600,  4800, 3.00, 25, "P5", 1275),
    Bearing("Timken","222WD", 110,200, 38, 139000, 116000,  3400,  4500, 3.60, 25, "P5", 1400),
    Bearing("Timken","224WD", 120,215, 40, 150000, 128000,  3000,  4000, 4.25, 25, "P5", 1590),
]


# ─────────────────────────────────────────────────────────────────────────────
# CRB Catalogs — NU2xxx series (rear/secondary bearing stations)
# ─────────────────────────────────────────────────────────────────────────────
_SKF_CRB: List[Bearing] = [
    Bearing("SKF","NU2206 ECP", 30, 62, 20,  28600,  24000, 17000, 22000, 0.19, 0, "P5"),
    Bearing("SKF","NU2207 ECP", 35, 72, 23,  37700,  32000, 15000, 19000, 0.28, 0, "P5"),
    Bearing("SKF","NU2208 ECP", 40, 80, 23,  47800,  42000, 13000, 17000, 0.37, 0, "P5"),
    Bearing("SKF","NU2209 ECP", 45, 85, 23,  51800,  46500, 11000, 15000, 0.41, 0, "P5"),
    Bearing("SKF","NU2210 ECP", 50, 90, 23,  55600,  51000, 10000, 14000, 0.46, 0, "P5"),
    Bearing("SKF","NU2211 ECP", 55,100, 25,  68000,  63000,  8500, 11000, 0.65, 0, "P5"),
    Bearing("SKF","NU2212 ECP", 60,110, 28,  86500,  80000,  7500, 10000, 0.88, 0, "P5"),
    Bearing("SKF","NU2213 ECP", 65,120, 31, 108000, 100000,  6700,  9000, 1.20, 0, "P5"),
    Bearing("SKF","NU2214 ECP", 70,125, 31, 110000, 102000,  6000,  8000, 1.25, 0, "P5"),
    Bearing("SKF","NU2215 ECP", 75,130, 31, 114000, 107000,  5600,  7500, 1.30, 0, "P5"),
    Bearing("SKF","NU2216 ECP", 80,140, 33, 137000, 127000,  5000,  6700, 1.70, 0, "P5"),
    Bearing("SKF","NU2217 ECP", 85,150, 36, 163000, 155000,  4800,  6300, 2.25, 0, "P5"),
    Bearing("SKF","NU2218 ECP", 90,160, 40, 190000, 180000,  4300,  5600, 2.90, 0, "P5"),
    Bearing("SKF","NU2219 ECP", 95,170, 43, 216000, 208000,  4000,  5300, 3.60, 0, "P5"),
    Bearing("SKF","NU2220 ECP",100,180, 46, 228000, 224000,  3800,  5000, 4.50, 0, "P5"),
    Bearing("SKF","NU2222 ECP",110,200, 53, 270000, 270000,  3400,  4500, 6.40, 0, "P5"),
    Bearing("SKF","NU2224 ECP",120,215, 58, 305000, 315000,  3000,  4000, 7.90, 0, "P5"),
]

_FAG_CRB: List[Bearing] = [
    Bearing("FAG","NU2206-E-TVP2", 30, 62, 20,  27200,  23500, 18000, 24000, 0.19, 0, "P5"),
    Bearing("FAG","NU2207-E-TVP2", 35, 72, 23,  36000,  31000, 16000, 21000, 0.28, 0, "P5"),
    Bearing("FAG","NU2208-E-TVP2", 40, 80, 23,  46000,  41000, 14000, 18000, 0.37, 0, "P5"),
    Bearing("FAG","NU2209-E-TVP2", 45, 85, 23,  49500,  45000, 12000, 16000, 0.41, 0, "P5"),
    Bearing("FAG","NU2210-E-TVP2", 50, 90, 23,  53500,  49500, 11000, 15000, 0.46, 0, "P5"),
    Bearing("FAG","NU2211-E-TVP2", 55,100, 25,  66000,  61500,  9000, 12000, 0.65, 0, "P5"),
    Bearing("FAG","NU2212-E-TVP2", 60,110, 28,  83000,  77500,  8000, 10500, 0.88, 0, "P5"),
    Bearing("FAG","NU2213-E-TVP2", 65,120, 31, 104000,  97000,  7200,  9500, 1.20, 0, "P5"),
    Bearing("FAG","NU2214-E-TVP2", 70,125, 31, 107000, 100000,  6400,  8500, 1.25, 0, "P5"),
    Bearing("FAG","NU2215-E-TVP2", 75,130, 31, 111000, 105000,  6000,  8000, 1.30, 0, "P5"),
    Bearing("FAG","NU2216-E-TVP2", 80,140, 33, 133000, 124000,  5300,  7000, 1.70, 0, "P5"),
    Bearing("FAG","NU2217-E-TVP2", 85,150, 36, 159000, 152000,  5000,  6500, 2.25, 0, "P5"),
    Bearing("FAG","NU2218-E-TVP2", 90,160, 40, 185000, 177000,  4600,  6000, 2.90, 0, "P5"),
    Bearing("FAG","NU2219-E-TVP2", 95,170, 43, 210000, 204000,  4300,  5600, 3.60, 0, "P5"),
    Bearing("FAG","NU2220-E-TVP2",100,180, 46, 224000, 220000,  4000,  5300, 4.50, 0, "P5"),
    Bearing("FAG","NU2222-E-TVP2",110,200, 53, 263000, 265000,  3600,  4800, 6.40, 0, "P5"),
    Bearing("FAG","NU2224-E-TVP2",120,215, 58, 300000, 310000,  3200,  4200, 7.90, 0, "P5"),
]

_NSK_CRB: List[Bearing] = [
    Bearing("NSK","NU2206EW",  30, 62, 20,  28000,  24200, 18000, 23000, 0.19, 0, "P5"),
    Bearing("NSK","NU2207EW",  35, 72, 23,  37200,  32500, 16000, 21000, 0.28, 0, "P5"),
    Bearing("NSK","NU2208EW",  40, 80, 23,  47000,  42500, 14000, 18000, 0.37, 0, "P5"),
    Bearing("NSK","NU2209EW",  45, 85, 23,  51000,  47000, 12000, 16000, 0.41, 0, "P5"),
    Bearing("NSK","NU2210EW",  50, 90, 23,  55000,  51500, 11000, 15000, 0.46, 0, "P5"),
    Bearing("NSK","NU2211EW",  55,100, 25,  67500,  63500,  9000, 12000, 0.65, 0, "P5"),
    Bearing("NSK","NU2212EW",  60,110, 28,  86000,  81000,  8000, 10500, 0.88, 0, "P5"),
    Bearing("NSK","NU2213EW",  65,120, 31, 107000, 101000,  7200,  9500, 1.20, 0, "P5"),
    Bearing("NSK","NU2214EW",  70,125, 31, 109000, 103000,  6400,  8500, 1.25, 0, "P5"),
    Bearing("NSK","NU2215EW",  75,130, 31, 113000, 108000,  6000,  8000, 1.30, 0, "P5"),
    Bearing("NSK","NU2216EW",  80,140, 33, 135000, 128000,  5300,  7000, 1.70, 0, "P5"),
    Bearing("NSK","NU2217EW",  85,150, 36, 162000, 156000,  5000,  6500, 2.25, 0, "P5"),
    Bearing("NSK","NU2218EW",  90,160, 40, 188000, 181000,  4600,  6000, 2.90, 0, "P5"),
    Bearing("NSK","NU2219EW",  95,170, 43, 215000, 209000,  4300,  5600, 3.60, 0, "P5"),
    Bearing("NSK","NU2220EW", 100,180, 46, 226000, 223000,  4000,  5300, 4.50, 0, "P5"),
    Bearing("NSK","NU2222EW", 110,200, 53, 268000, 268000,  3600,  4800, 6.40, 0, "P5"),
    Bearing("NSK","NU2224EW", 120,215, 58, 303000, 313000,  3200,  4200, 7.90, 0, "P5"),
]

_NTN_CRB: List[Bearing] = [
    Bearing("NTN","NU2206",  30, 62, 20,  27500,  23800, 17000, 22000, 0.19, 0, "P5"),
    Bearing("NTN","NU2207",  35, 72, 23,  36500,  31500, 15000, 19000, 0.28, 0, "P5"),
    Bearing("NTN","NU2208",  40, 80, 23,  46500,  41500, 13000, 17000, 0.37, 0, "P5"),
    Bearing("NTN","NU2209",  45, 85, 23,  50500,  46000, 11000, 15000, 0.41, 0, "P5"),
    Bearing("NTN","NU2210",  50, 90, 23,  54500,  50500, 10000, 14000, 0.46, 0, "P5"),
    Bearing("NTN","NU2211",  55,100, 25,  67000,  62500,  8500, 11500, 0.65, 0, "P5"),
    Bearing("NTN","NU2212",  60,110, 28,  85000,  79500,  7500, 10000, 0.88, 0, "P5"),
    Bearing("NTN","NU2213",  65,120, 31, 106000,  99500,  6700,  9000, 1.20, 0, "P5"),
    Bearing("NTN","NU2214",  70,125, 31, 109000, 101500,  6000,  8000, 1.25, 0, "P5"),
    Bearing("NTN","NU2215",  75,130, 31, 112500, 106500,  5600,  7500, 1.30, 0, "P5"),
    Bearing("NTN","NU2216",  80,140, 33, 135500, 126000,  5000,  6700, 1.70, 0, "P5"),
    Bearing("NTN","NU2217",  85,150, 36, 161000, 153500,  4800,  6300, 2.25, 0, "P5"),
    Bearing("NTN","NU2218",  90,160, 40, 188000, 179000,  4300,  5600, 2.90, 0, "P5"),
    Bearing("NTN","NU2219",  95,170, 43, 213000, 206000,  4000,  5300, 3.60, 0, "P5"),
    Bearing("NTN","NU2220", 100,180, 46, 225500, 222000,  3800,  5000, 4.50, 0, "P5"),
    Bearing("NTN","NU2222", 110,200, 53, 268000, 267000,  3400,  4500, 6.40, 0, "P5"),
    Bearing("NTN","NU2224", 120,215, 58, 302000, 311000,  3000,  4000, 7.90, 0, "P5"),
]

_JTEKT_CRB: List[Bearing] = [
    Bearing("JTEKT","NU2206",  30, 62, 20,  27800,  24100, 17000, 22000, 0.19, 0, "P5"),
    Bearing("JTEKT","NU2207",  35, 72, 23,  37000,  32000, 15000, 20000, 0.28, 0, "P5"),
    Bearing("JTEKT","NU2208",  40, 80, 23,  47000,  42000, 13000, 17000, 0.37, 0, "P5"),
    Bearing("JTEKT","NU2209",  45, 85, 23,  51200,  46800, 11000, 15000, 0.41, 0, "P5"),
    Bearing("JTEKT","NU2210",  50, 90, 23,  55000,  51200, 10000, 13500, 0.46, 0, "P5"),
    Bearing("JTEKT","NU2211",  55,100, 25,  67500,  63000,  8500, 11000, 0.65, 0, "P5"),
    Bearing("JTEKT","NU2212",  60,110, 28,  86000,  80500,  7500, 10000, 0.88, 0, "P5"),
    Bearing("JTEKT","NU2213",  65,120, 31, 107000, 100500,  6700,  9000, 1.20, 0, "P5"),
    Bearing("JTEKT","NU2214",  70,125, 31, 109500, 102000,  6000,  8000, 1.25, 0, "P5"),
    Bearing("JTEKT","NU2215",  75,130, 31, 113500, 107000,  5600,  7500, 1.30, 0, "P5"),
    Bearing("JTEKT","NU2216",  80,140, 33, 136000, 127000,  5000,  6700, 1.70, 0, "P5"),
    Bearing("JTEKT","NU2217",  85,150, 36, 162000, 154500,  4800,  6300, 2.25, 0, "P5"),
    Bearing("JTEKT","NU2218",  90,160, 40, 189000, 180000,  4300,  5600, 2.90, 0, "P5"),
    Bearing("JTEKT","NU2219",  95,170, 43, 215000, 208000,  4000,  5300, 3.60, 0, "P5"),
    Bearing("JTEKT","NU2220", 100,180, 46, 227000, 223000,  3800,  5000, 4.50, 0, "P5"),
    Bearing("JTEKT","NU2222", 110,200, 53, 270000, 269000,  3400,  4500, 6.40, 0, "P5"),
    Bearing("JTEKT","NU2224", 120,215, 58, 304000, 314000,  3000,  4000, 7.90, 0, "P5"),
]

_TIMKEN_CRB: List[Bearing] = [
    Bearing("Timken","NU2206",  30, 62, 20,  26800,  23200, 16000, 21000, 0.19, 0, "P5"),
    Bearing("Timken","NU2207",  35, 72, 23,  35800,  30500, 14000, 18000, 0.28, 0, "P5"),
    Bearing("Timken","NU2208",  40, 80, 23,  45500,  40500, 12500, 16000, 0.37, 0, "P5"),
    Bearing("Timken","NU2209",  45, 85, 23,  49500,  45000, 10500, 14000, 0.41, 0, "P5"),
    Bearing("Timken","NU2210",  50, 90, 23,  53500,  49500,  9500, 13000, 0.46, 0, "P5"),
    Bearing("Timken","NU2211",  55,100, 25,  65500,  61000,  8000, 10500, 0.65, 0, "P5"),
    Bearing("Timken","NU2212",  60,110, 28,  83000,  77500,  7000,  9500, 0.88, 0, "P5"),
    Bearing("Timken","NU2213",  65,120, 31, 104000,  97500,  6300,  8500, 1.20, 0, "P5"),
    Bearing("Timken","NU2214",  70,125, 31, 106000, 100000,  5600,  7500, 1.25, 0, "P5"),
    Bearing("Timken","NU2215",  75,130, 31, 110000, 104500,  5300,  7000, 1.30, 0, "P5"),
    Bearing("Timken","NU2216",  80,140, 33, 132000, 123500,  4800,  6300, 1.70, 0, "P5"),
    Bearing("Timken","NU2217",  85,150, 36, 157000, 150000,  4500,  6000, 2.25, 0, "P5"),
    Bearing("Timken","NU2218",  90,160, 40, 183000, 175000,  4000,  5300, 2.90, 0, "P5"),
    Bearing("Timken","NU2219",  95,170, 43, 208000, 201500,  3800,  5000, 3.60, 0, "P5"),
    Bearing("Timken","NU2220", 100,180, 46, 220000, 218000,  3600,  4800, 4.50, 0, "P5"),
    Bearing("Timken","NU2222", 110,200, 53, 260000, 263000,  3200,  4200, 6.40, 0, "P5"),
    Bearing("Timken","NU2224", 120,215, 58, 295000, 308000,  2800,  3800, 7.90, 0, "P5"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Master catalog structure
# ─────────────────────────────────────────────────────────────────────────────
SUPPORTED_MANUFACTURERS = ["SKF", "FAG", "NSK", "NTN", "JTEKT", "Timken"]

ACBB_CATALOG: Dict[str, List[Bearing]] = {
    "SKF":    _SKF_ACBB,
    "FAG":    _FAG_ACBB,
    "NSK":    _NSK_ACBB,
    "NTN":    _NTN_ACBB,
    "JTEKT":  _JTEKT_ACBB,
    "Timken": _TIMKEN_ACBB,
}

CRB_CATALOG: Dict[str, List[Bearing]] = {
    "SKF":    _SKF_CRB,
    "FAG":    _FAG_CRB,
    "NSK":    _NSK_CRB,
    "NTN":    _NTN_CRB,
    "JTEKT":  _JTEKT_CRB,
    "Timken": _TIMKEN_CRB,
}

# All unique bore sizes available across all manufacturers (ACBB)
ALL_ACBB_BORES: np.ndarray = np.array(sorted({
    b.d for catalog in ACBB_CATALOG.values() for b in catalog
}))

# All unique bore sizes available across all manufacturers (CRB)
ALL_CRB_BORES: np.ndarray = np.array(sorted({
    b.d for catalog in CRB_CATALOG.values() for b in catalog
}))


# ─────────────────────────────────────────────────────────────────────────────
# Lookup functions
# ─────────────────────────────────────────────────────────────────────────────
def get_bearing(
    bore_mm:      float,
    bearing_type: str  = "ACBB",
    manufacturer: Optional[str] = None,
    criterion:    str  = "max_Cr",
) -> Tuple[Bearing, float, str]:
    """
    Return the best-matching bearing for a given bore diameter.

    Parameters
    ----------
    bore_mm       : target bore diameter [mm] (continuous, will be snapped)
    bearing_type  : "ACBB" or "CRB"
    manufacturer  : one of SUPPORTED_MANUFACTURERS, or None to search ALL
    criterion     : "max_Cr"      → maximise C_r (highest load capacity)
                    "max_speed"   → maximise n_grease (best for high-speed)
                    "min_cost"    → placeholder (uses C_r as cost proxy)
                    "best_fit"    → minimise bore snap error

    Returns
    -------
    (Bearing, snap_error_pct, warning_str)
        snap_error_pct : |bore_actual − bore_target| / bore_target × 100
        warning_str    : "" if speed OK, else warning text
    """
    catalog_map = ACBB_CATALOG if bearing_type == "ACBB" else CRB_CATALOG

    # Build candidate list
    if manufacturer is not None:
        if manufacturer not in SUPPORTED_MANUFACTURERS:
            raise ValueError(
                f"Unknown manufacturer '{manufacturer}'. "
                f"Supported: {SUPPORTED_MANUFACTURERS}"
            )
        candidates_by_mfr = {manufacturer: catalog_map[manufacturer]}
    else:
        candidates_by_mfr = catalog_map

    # All bearings from selected manufacturers
    all_candidates: List[Bearing] = [
        b for cat in candidates_by_mfr.values() for b in cat
    ]
    if not all_candidates:
        raise ValueError(f"No {bearing_type} bearings found for manufacturer={manufacturer}")

    # Snap to nearest bore
    all_bores = np.array([b.d for b in all_candidates])
    nearest_bore = all_bores[int(np.argmin(np.abs(all_bores - bore_mm)))]

    # Filter to that bore
    at_bore = [b for b in all_candidates if b.d == nearest_bore]
    if not at_bore:
        raise ValueError(f"No {bearing_type} bearing at bore={nearest_bore}mm")

    # Apply selection criterion
    if criterion == "max_speed":
        best = max(at_bore, key=lambda b: b.n_grease)
    elif criterion == "best_fit":
        # Find the bore that minimises snap error regardless of manufacturer
        best = min(at_bore, key=lambda b: abs(b.d - bore_mm))
    else:  # "max_Cr" (default)
        best = max(at_bore, key=lambda b: b.C_r)

    snap_err = abs(best.d - bore_mm) / max(bore_mm, 1e-6) * 100.0
    warning  = ""
    return best, snap_err, warning


def compare_manufacturers(
    bore_mm:      float,
    bearing_type: str = "ACBB",
    n_rpm:        float = 4000.0,
    lubrication:  str  = "grease",
) -> List[Dict]:
    """
    Compare all manufacturers for a given bore size.

    Returns list of dicts with keys:
        manufacturer, designation, d, C_r_kN, C_0r_kN, n_grease, n_oil,
        speed_ok, precision_class, contact_angle_deg, radial_stiffness_N_mm
    Sorted by C_r descending (highest load capacity first).
    """
    catalog_map = ACBB_CATALOG if bearing_type == "ACBB" else CRB_CATALOG
    all_candidates = [b for cat in catalog_map.values() for b in cat]
    bores = np.array([b.d for b in all_candidates])
    nearest = bores[int(np.argmin(np.abs(bores - bore_mm)))]

    rows = []
    for b in all_candidates:
        if b.d != nearest:
            continue
        rows.append({
            "manufacturer":         b.manufacturer,
            "designation":          b.designation,
            "d_mm":                 b.d,
            "D_mm":                 b.D,
            "B_mm":                 b.B,
            "C_r_kN":               b.C_r / 1000.0,
            "C_0r_kN":              b.C_0r / 1000.0,
            "n_grease":             b.n_grease,
            "n_oil":                b.n_oil,
            "speed_ok":             b.speed_ok(n_rpm, lubrication),
            "precision_class":      b.precision_class,
            "contact_angle_deg":    b.contact_angle_deg,
            "radial_stiffness_N_mm":b.radial_stiffness_single_N_mm,
            "mass_kg":              b.mass_kg,
        })

    rows.sort(key=lambda r: r["C_r_kN"], reverse=True)
    return rows


def print_manufacturer_comparison(
    bore_mm:      float,
    bearing_type: str  = "ACBB",
    n_rpm:        float = 4000.0,
) -> None:
    """Print a formatted manufacturer-comparison table to stdout."""
    rows = compare_manufacturers(bore_mm, bearing_type, n_rpm)
    if not rows:
        print(f"No {bearing_type} bearings found near bore={bore_mm:.1f}mm"); return

    actual_bore = rows[0]["d_mm"]
    print(f"\n{'═'*90}")
    print(f"  {bearing_type} Bearing Comparison  |  bore={actual_bore:.0f}mm  "
          f"|  n={n_rpm:.0f}rpm")
    print(f"{'═'*90}")
    print(f"  {'Mfr':<8} {'Designation':<22} {'Cr':>7} {'C0r':>7} "
          f"{'n_gr':>7} {'n_oil':>7} {'α°':>4} {'Class':<5} "
          f"{'k_N/mm':>8}  Spd")
    print(f"  {'─'*86}")
    for r in rows:
        ok = "✅" if r["speed_ok"] else "❌"
        print(f"  {r['manufacturer']:<8} {r['designation']:<22} "
              f"{r['C_r_kN']:>7.1f} {r['C_0r_kN']:>7.1f} "
              f"{r['n_grease']:>7,} {r['n_oil']:>7,} "
              f"{r['contact_angle_deg']:>4.0f} {r['precision_class']:<5} "
              f"{r['radial_stiffness_N_mm']:>8,.0f}  {ok}")
    print(f"  {'─'*86}")
    print(f"  C_r [kN], C_0r [kN], speeds [rpm], k = radial stiffness [N/mm]")
    print(f"{'═'*90}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Compatibility shim  —  preserves existing snap_to_skf_bearing() behaviour
# ─────────────────────────────────────────────────────────────────────────────
def snap_to_bearing(
    bore_radius_mm: float,
    bearing_type:   str  = "ACBB",
    n_rpm:          float = 4000.0,
    lubrication:    str  = "grease",
    manufacturer:   Optional[str] = None,
    criterion:      str  = "max_Cr",
) -> Tuple[Bearing, bool, str]:
    """
    Extended snap-to-catalog function.

    Parameters
    ----------
    bore_radius_mm : design-variable bore RADIUS [mm] (module 01 convention)
                     Converted internally to bore diameter = 2 × bore_radius_mm.
    manufacturer   : None → search across ALL 6 manufacturers (returns highest C_r)
                     "SKF" / "FAG" / etc. → restrict to that manufacturer only

    Returns
    -------
    (Bearing, speed_ok: bool, warning_str: str)
    """
    bore_mm = bore_radius_mm * 2.0
    bearing, snap_err, _ = get_bearing(bore_mm, bearing_type, manufacturer, criterion)
    speed_ok = bearing.speed_ok(n_rpm, lubrication)
    warning  = ""
    if not speed_ok:
        limit = bearing.n_grease if lubrication == "grease" else bearing.n_oil
        warning = (f"⚠️  {bearing.manufacturer} {bearing.designation}: "
                   f"{lubrication} speed {limit:,} RPM < n={n_rpm:.0f} RPM")
    if snap_err > 5.0:
        warning += (f"  |  Bore snap error {snap_err:.1f}% "
                    f"(target Ø{bore_mm:.1f}mm → actual Ø{bearing.d:.0f}mm)")
    return bearing, speed_ok, warning


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatibility alias (used by 01_design_variables.py)
# ─────────────────────────────────────────────────────────────────────────────
def snap_to_skf_bearing(
    bore_radius_mm: float,
    bearing_type:   str  = "ACBB",
    n_rpm:          float = 4000.0,
    lubrication:    str  = "grease",
) -> Tuple[Bearing, bool, str]:
    """
    Backward-compatible wrapper — behaves identically to the old function
    but now draws from the full multi-manufacturer catalog, returning the
    bearing with highest C_r at the nearest bore size.

    To restrict to a specific manufacturer, use snap_to_bearing() with the
    manufacturer= parameter.
    """
    return snap_to_bearing(bore_radius_mm, bearing_type, n_rpm, lubrication,
                           manufacturer=None, criterion="max_Cr")


def plot_manufacturer_comparison(
    bore_mm:        float,
    n_rpm:          float = 4000.0,
    save_path:      str   = "./08d_manufacturer_comparison.png",
) -> None:
    """
    Fig 08d — Side-by-side bar chart comparing C_r and speed rating
    for all 6 manufacturers at a given bore size and operating speed.
    """
    import matplotlib.pyplot as plt
    from plot_theme import apply_paper_theme, C as PT, savefig_paper
    apply_paper_theme()

    rows_acbb = compare_manufacturers(bore_mm, "ACBB", n_rpm)
    rows_crb  = compare_manufacturers(bore_mm, "CRB",  n_rpm)
    colours   = [PT.BLUE, PT.RED, PT.ORANGE, PT.GREEN, PT.PURPLE, PT.GRAY]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=PT.BG)
    for ax, rows, title in [
        (axes[0], rows_acbb, f"ACBB — Ø{bore_mm:.0f} mm bore"),
        (axes[1], rows_crb,  f"CRB  — Ø{bore_mm:.0f} mm bore"),
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
        ax.set_xlabel("Dynamic load rating C_r [kN]")
        ax.set_title(title)
        for i, r in enumerate(rows):
            clr = PT.GREEN if r["speed_ok"] else PT.RED
            ax.text(0.99, (i + 0.5) / len(rows), "✅" if r["speed_ok"] else "❌",
                    transform=ax.transAxes, ha="right", va="center",
                    fontsize=12, color=clr)

    fig.suptitle(
        f"Bearing Manufacturer Comparison  —  Ø{bore_mm:.0f} mm bore  |  n = {n_rpm:.0f} rpm\n"
        "Sorted by C_r (highest load capacity first)  |  ✅/❌ = grease speed OK",
        fontweight="bold",
    )
    plt.tight_layout()
    savefig_paper(fig, save_path)
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test / demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  BEARING CATALOG — Multi-Manufacturer Database")
    print("=" * 70)

    # T1: Check catalog sizes
    for mfr in SUPPORTED_MANUFACTURERS:
        na = len(ACBB_CATALOG[mfr])
        nc = len(CRB_CATALOG[mfr])
        print(f"  {mfr:<8}: {na} ACBB  {nc} CRB")

    print(f"\n  ALL_ACBB_BORES: {ALL_ACBB_BORES}")
    assert len(ALL_ACBB_BORES) > 0

    # T2: Compare manufacturers for 110mm bore
    print_manufacturer_comparison(110.0, "ACBB", 4000.0)
    print_manufacturer_comparison(110.0, "CRB",  4000.0)

    # T3: snap_to_bearing (new API)
    for mfr in SUPPORTED_MANUFACTURERS:
        brg, ok, warn = snap_to_bearing(55.0, "ACBB", 4000, "grease", manufacturer=mfr)
        print(f"  {mfr:<8} ACBB bore=110mm → {brg}")

    # T4: backward-compat snap_to_skf_bearing (now multi-manufacturer)
    print("\n  snap_to_skf_bearing() (all manufacturers, max C_r):")
    brg, ok, warn = snap_to_skf_bearing(55.0, "ACBB", 4000, "grease")
    print(f"  Best: {brg}")

    # T5: stiffness sanity
    skf = next(b for b in _SKF_ACBB if b.d == 110.0)
    fag = next(b for b in _FAG_ACBB if b.d == 110.0)
    print(f"\n  Stiffness @ d=110mm (single bearing):")
    print(f"    SKF: {skf.radial_stiffness_single_N_mm:,.0f} N/mm")
    print(f"    FAG: {fag.radial_stiffness_single_N_mm:,.0f} N/mm")
    assert skf.radial_stiffness_single_N_mm == fag.radial_stiffness_single_N_mm, \
        "Stiffness should be equal for same bore (Palmgren formula is bore-dependent only)"

    print("\n✅ All smoke tests passed")
