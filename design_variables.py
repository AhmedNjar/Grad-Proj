#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  Design Variables v4 — 21 Variables (adds Positional Tolerances)
================================================================================

  New in v4:
      • pos_tol_front [mm]  — ISO 1101 positional tolerance of front bearing seat
      • pos_tol_rear  [mm]  — ISO 1101 positional tolerance of rear bearing seats
      Total design variables: 19 → 21

  Positional Tolerance (ISO 1101 ⊕):
      Defines the diameter ϕ_pos of the cylindrical tolerance zone within which
      the actual bearing seat axis must lie.  Maximum radial offset of seat axis
      from nominal spindle centreline = ϕ_pos / 2.
      This offset propagates to nose runout exactly like bearing inner-ring
      eccentricity (same geometric amplification factor).

      Front seat (critical — locating):  ϕ_pos = 0.010 mm nominal [0.005–0.030]
      Rear seats (floating, less critical): ϕ_pos = 0.015 mm nominal [0.008–0.040]

  plot_design_space():
      Generates three figures:
        Fig 1 — Variable bounds bar chart (normalised width)
        Fig 2 — Tolerance asymmetry: upper (+) vs lower (−) deviation per variable
        Fig 3 — Manufacturing variation cloud (MC samples, first 4 geometric vars)
================================================================================
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ASYMMETRIC TOLERANCE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AsymmetricTolerance:
    """
    ISO 286 / ASME Y14.5 bilateral asymmetric tolerance.

        D_nominal − lower  ≤  D_actual  ≤  D_nominal + upper

    • Shaft h5 journal : upper=0.000, lower=0.013   → always undersized
    • Bore  H7 inner   : upper=0.021, lower=0.000   → always oversized
    • Bilateral ±0.05  : upper=0.050, lower=0.050
    """
    upper: float
    lower: float
    unit:  str  = "mm"
    note:  str  = ""

    @property
    def range(self) -> float:
        return self.upper + self.lower

    @property
    def midpoint_offset(self) -> float:
        return (self.upper - self.lower) / 2.0

    def contains(self, nominal: float, actual: float) -> bool:
        return (nominal - self.lower) <= actual <= (nominal + self.upper)

    def sample(self, nominal: float, n: int = 1,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Truncated-normal, 3σ = range, centred at mid-band."""
        rng   = rng or np.random.default_rng()
        mid   = nominal + self.midpoint_offset
        sigma = self.range / 6.0
        raw   = rng.normal(mid, sigma, size=n)
        return np.clip(raw, nominal - self.lower, nominal + self.upper)

    def __repr__(self) -> str:
        return f"+{self.upper}/{-self.lower} {self.unit}"


# ─────────────────────────────────────────────────────────────────────────────
# 2a.  SKF ACBB CATALOG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SKFBearing:
    designation:       str
    d:                 float
    D:                 float
    B:                 float
    C_r:               float
    C_0r:              float
    n_grease:          int
    n_oil:             int
    mass_kg:           float
    contact_angle_deg: float = 25.0
    precision_class:   str   = "P5"
    F_preload_MA_N:    float = 500.0
    bearing_type:      str   = "ACBB"

    @property
    def d_m(self) -> float:
        return (self.d + self.D) / 2.0

    @property
    def radius_bore(self) -> float:
        return self.d / 2.0

    @property
    def radial_stiffness_single_N_mm(self) -> float:
        return 5.5 * (self.d ** 0.75) * 1000.0

    @property
    def axial_stiffness_single_N_mm(self) -> float:
        import math
        alpha = math.radians(self.contact_angle_deg)
        return self.radial_stiffness_single_N_mm * math.tan(alpha) ** 2

    def ndm(self, speed_rpm: float) -> float:
        return self.d_m * speed_rpm

    def speed_ok(self, n_rpm: float, lubrication: str = "grease") -> bool:
        limit = self.n_grease if lubrication == "grease" else self.n_oil
        return n_rpm <= limit

    def __str__(self) -> str:
        return (f"SKF {self.designation}  d={self.d}mm  D={self.D}mm  "
                f"C_r={self.C_r/1e3:.1f}kN  n_gr={self.n_grease}RPM")


SKF_ACBB_CATALOG: List[SKFBearing] = [
    SKFBearing("7206 BECBP",  30,  62,  16,  19900,  13700, 18000, 24000, 0.12, F_preload_MA_N=150),
    SKFBearing("7207 BECBP",  35,  72,  17,  25700,  17900, 16000, 21000, 0.17, F_preload_MA_N=190),
    SKFBearing("7208 BECBP",  40,  80,  18,  30500,  21600, 14000, 19000, 0.23, F_preload_MA_N=230),
    SKFBearing("7209 BECBP",  45,  85,  19,  33200,  23600, 12000, 16000, 0.27, F_preload_MA_N=270),
    SKFBearing("7210 BECBP",  50,  90,  20,  35500,  25500, 11000, 15000, 0.31, F_preload_MA_N=320),
    SKFBearing("7211 BECBP",  55, 100,  21,  43000,  32000,  9500, 13000, 0.42, F_preload_MA_N=390),
    SKFBearing("7212 BECBP",  60, 110,  22,  51500,  38500,  8500, 11000, 0.55, F_preload_MA_N=470),
    SKFBearing("7213 BECBP",  65, 120,  23,  61000,  46500,  7500, 10000, 0.70, F_preload_MA_N=560),
    SKFBearing("7214 BECBP",  70, 125,  24,  63000,  49000,  6700,  9000, 0.75, F_preload_MA_N=630),
    SKFBearing("7215 BECBP",  75, 130,  25,  66500,  53000,  6300,  8500, 0.85, F_preload_MA_N=710),
    SKFBearing("7216 BECBP",  80, 140,  26,  73500,  60000,  5600,  7500, 1.05, F_preload_MA_N=800),
    SKFBearing("7217 BECBP",  85, 150,  28,  87000,  73500,  5300,  7000, 1.35, F_preload_MA_N=900),
    SKFBearing("7218 BECBP",  90, 160,  30,  98000,  80000,  4800,  6300, 1.65, F_preload_MA_N=1000),
    SKFBearing("7219 BECBP",  95, 170,  32, 110000,  90000,  4500,  6000, 2.05, F_preload_MA_N=1100),
    SKFBearing("7220 BECBP", 100, 180,  34, 118000,  98000,  4300,  5600, 2.50, F_preload_MA_N=1200),
    SKFBearing("7221 BECBP", 105, 190,  36, 130000, 108000,  4000,  5300, 3.00, F_preload_MA_N=1350),
    SKFBearing("7222 BECBP", 110, 200,  38, 143000, 118000,  3800,  5000, 3.60, F_preload_MA_N=1500),
    SKFBearing("7224 BECBP", 120, 215,  40, 153000, 130000,  3400,  4500, 4.25, F_preload_MA_N=1700),
]
SKF_CATALOG = SKF_ACBB_CATALOG


# ─────────────────────────────────────────────────────────────────────────────
# 2b.  SKF CRB CATALOG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SKFCRBBearing:
    designation:     str
    d:               float
    D:               float
    B:               float
    C_r:             float
    C_0r:            float
    n_grease:        int
    n_oil:           int
    mass_kg:         float
    precision_class: str = "P5"
    bearing_type:    str = "CRB"

    @property
    def d_m(self) -> float:
        return (self.d + self.D) / 2.0

    @property
    def radius_bore(self) -> float:
        return self.d / 2.0

    @property
    def radial_stiffness_single_N_mm(self) -> float:
        return 8.0 * (self.d ** 0.80) * 1000.0

    def ndm(self, speed_rpm: float) -> float:
        return self.d_m * speed_rpm

    def speed_ok(self, n_rpm: float, lubrication: str = "grease") -> bool:
        limit = self.n_grease if lubrication == "grease" else self.n_oil
        return n_rpm <= limit

    def __str__(self) -> str:
        return (f"SKF {self.designation}  d={self.d}mm  D={self.D}mm  "
                f"C_r={self.C_r/1e3:.1f}kN  n_gr={self.n_grease}RPM")


SKF_CRB_CATALOG: List[SKFCRBBearing] = [
    SKFCRBBearing("NU2206 ECP",  30,  62,  20,  28600,  24000, 17000, 22000, 0.19),
    SKFCRBBearing("NU2207 ECP",  35,  72,  23,  37700,  32000, 15000, 19000, 0.28),
    SKFCRBBearing("NU2208 ECP",  40,  80,  23,  47800,  42000, 13000, 17000, 0.37),
    SKFCRBBearing("NU2209 ECP",  45,  85,  23,  51800,  46500, 11000, 15000, 0.41),
    SKFCRBBearing("NU2210 ECP",  50,  90,  23,  55600,  51000, 10000, 14000, 0.46),
    SKFCRBBearing("NU2211 ECP",  55, 100,  25,  68000,  63000,  8500, 11000, 0.65),
    SKFCRBBearing("NU2212 ECP",  60, 110,  28,  86500,  80000,  7500, 10000, 0.88),
    SKFCRBBearing("NU2213 ECP",  65, 120,  31, 108000, 100000,  6700,  9000, 1.20),
    SKFCRBBearing("NU2214 ECP",  70, 125,  31, 110000, 102000,  6000,  8000, 1.25),
    SKFCRBBearing("NU2215 ECP",  75, 130,  31, 114000, 107000,  5600,  7500, 1.30),
    SKFCRBBearing("NU2216 ECP",  80, 140,  33, 137000, 127000,  5000,  6700, 1.70),
    SKFCRBBearing("NU2217 ECP",  85, 150,  36, 163000, 155000,  4800,  6300, 2.25),
    SKFCRBBearing("NU2218 ECP",  90, 160,  40, 190000, 180000,  4300,  5600, 2.90),
    SKFCRBBearing("NU2219 ECP",  95, 170,  43, 216000, 208000,  4000,  5300, 3.60),
    SKFCRBBearing("NU2220 ECP", 100, 180,  46, 228000, 224000,  3800,  5000, 4.50),
    SKFCRBBearing("NU2222 ECP", 110, 200,  53, 270000, 270000,  3400,  4500, 6.40),
    SKFCRBBearing("NU2224 ECP", 120, 215,  58, 305000, 315000,  3000,  4000, 7.90),
]

_ACBB_BY_BORE: Dict[float, SKFBearing]    = {b.d: b for b in SKF_ACBB_CATALOG}
_CRB_BY_BORE:  Dict[float, SKFCRBBearing] = {b.d: b for b in SKF_CRB_CATALOG}
_ACBB_BORES = np.array(sorted(_ACBB_BY_BORE.keys()))
_CRB_BORES  = np.array(sorted(_CRB_BY_BORE.keys()))
BearingRecord = Union[SKFBearing, SKFCRBBearing]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SNAP-TO-CATALOG
# ─────────────────────────────────────────────────────────────────────────────

def snap_to_skf_bearing(
    journal_radius_mm: float,
    bearing_type:      Literal["ACBB", "CRB"],
    n_operating_rpm:   float,
    lubrication:       str  = "grease",
    fallback_to_oil:   bool = True,
) -> Tuple[BearingRecord, bool, str]:
    bore_mm   = journal_radius_mm * 2.0
    catalog   = _ACBB_BY_BORE if bearing_type == "ACBB" else _CRB_BY_BORE
    bores_arr = _ACBB_BORES   if bearing_type == "ACBB" else _CRB_BORES
    nearest_idx = int(np.argmin(np.abs(bores_arr - bore_mm)))
    bearing     = catalog[bores_arr[nearest_idx]]
    ok      = bearing.speed_ok(n_operating_rpm, lubrication)
    warning = ""
    if not ok and lubrication == "grease" and fallback_to_oil:
        if bearing.speed_ok(n_operating_rpm, "oil"):
            warning = (f"⚠️  {bearing.designation}: grease {bearing.n_grease} RPM "
                       f"< {n_operating_rpm:.0f}. Use oil ({bearing.n_oil} RPM).")
            ok = True
        else:
            for step_idx in range(nearest_idx - 1, -1, -1):
                smaller = catalog[bores_arr[step_idx]]
                if smaller.speed_ok(n_operating_rpm, lubrication):
                    warning = (f"⚠️  Stepped to {smaller.designation} "
                               f"(bore {smaller.d}mm, n_gr={smaller.n_grease}).")
                    bearing = smaller; ok = True; break
            else:
                warning = (f"❌  No {bearing_type} for {n_operating_rpm:.0f} RPM "
                           f"under {lubrication}.")
    return bearing, ok, warning


# ─────────────────────────────────────────────────────────────────────────────
# 4.  BEARING STATION & ARRANGEMENT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BearingStation:
    role:              Literal["front", "rear"]
    bearing_type:      Literal["ACBB", "CRB"]
    n_bearings:        int
    sub_arrangement:   Literal["single", "DB", "DF", "DT", "spacer_pair"]
    z_fraction_1:      float
    z_fraction_2:      Optional[float] = None
    spacer_length_mm:  float           = 20.0
    preload_class:     str             = "MA"

    def z_positions_mm(self, L1: float, L2: float) -> List[float]:
        z1 = L1 + self.z_fraction_1 * L2
        if self.n_bearings == 2 and self.z_fraction_2 is not None:
            z2 = L1 + self.z_fraction_2 * L2
            return [z1, z2]
        return [z1]

    @property
    def stiffness_factor(self) -> float:
        if self.n_bearings == 1:              return 1.0
        if self.sub_arrangement in ("DB","DF"): return 1.7
        if self.sub_arrangement == "spacer_pair": return 2.0
        if self.sub_arrangement == "DT":      return 1.5
        return float(self.n_bearings)


@dataclass
class SpindleBearingArrangement:
    stations:        List[BearingStation]
    lubrication:     str = "grease"
    precision_class: str = "P5"
    balance_grade:   str = "G2.5"

    @classmethod
    def default_lathe(cls) -> "SpindleBearingArrangement":
        return cls(stations=[
            BearingStation("front","ACBB",1,"single",0.25,None,0.0,"MA"),
            BearingStation("rear","CRB",2,"spacer_pair",0.70,0.80,20.0,"MA"),
        ])

    @classmethod
    def three_acbb(cls) -> "SpindleBearingArrangement":
        return cls(stations=[
            BearingStation("front","ACBB",2,"DB",0.20,0.30,25.0,"MA"),
            BearingStation("rear","ACBB",1,"single",0.80,None,0.0,"LA"),
        ])

    @classmethod
    def user_defined(cls, stations, lubrication="grease",
                     precision_class="P5", balance_grade="G2.5"):
        return cls(stations=stations, lubrication=lubrication,
                   precision_class=precision_class, balance_grade=balance_grade)

    def resolve(self, R2_mm, n_rpm):
        return [
            snap_to_skf_bearing(R2_mm, st.bearing_type, n_rpm, self.lubrication)
            + (st,)   # returns (bearing, ok, warn, station) — reordered below
            for st in self.stations
        ]

    def description(self) -> str:
        lines = [f"Arrangement ({len(self.stations)} stations)"]
        for i, st in enumerate(self.stations):
            lines.append(
                f"  St{i+1}: {st.role.upper()} | {st.n_bearings}× {st.bearing_type}"
                f" [{st.sub_arrangement}] z={st.z_fraction_1:.2f}"
                + (f"–{st.z_fraction_2:.2f}" if st.z_fraction_2 else "")
            )
        return "\n".join(lines)


def _resolve_arrangement(ds, x, n_rpm, arrangement):
    """Unified resolve: returns List[(station, bearing, ok, warn)]."""
    v   = ds.decode_vector(x)
    R2  = v["R2"]
    arr = arrangement or SpindleBearingArrangement.default_lathe()
    result = []
    for st in arr.stations:
        brg, ok, warn = snap_to_skf_bearing(
            R2, st.bearing_type, n_rpm, arr.lubrication)
        result.append((st, brg, ok, warn))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5.  VARIABLE BOUNDS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VariableBounds:
    nominal:     float
    lower:       float
    upper:       float
    tolerance:   AsymmetricTolerance
    unit:        str
    description: str

    def sample_manufacturing(self, n=1, rng=None):
        return self.tolerance.sample(self.nominal, n=n, rng=rng)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  DESIGN SPACE — 21 VARIABLES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DesignSpace:
    """
    21 continuous design variables for the lathe spindle RDO.

    Additions vs. v3:
        pos_tol_front [mm] — ISO 1101 positional tolerance diameter of front ACBB seat
        pos_tol_rear  [mm] — ISO 1101 positional tolerance diameter of rear CRB seats
    """

    geometry: Dict[str, VariableBounds] = field(default_factory=lambda: {
        "L1": VariableBounds(80.0, 60.0, 90.0,  AsymmetricTolerance(0.500,0.000,note="IT12"), "mm","Nose section length"),
        "L2": VariableBounds(405.0,350.0,450.0, AsymmetricTolerance(0.500,0.000,note="IT12"), "mm","Journal section length"),
        "L3": VariableBounds(24.0, 15.0, 40.0,  AsymmetricTolerance(0.100,0.000,note="spacer IT10"), "mm","Flange section length"),
        "L4": VariableBounds(15.0, 10.0, 25.0,  AsymmetricTolerance(0.500,0.000,note="IT12"), "mm","Tail section length"),
        "R1": VariableBounds(45.0, 35.0, 55.0,  AsymmetricTolerance(0.000,0.011,note="ISO h5 Ø90"), "mm","Nose outer radius"),
        "R2": VariableBounds(35.0, 28.0, 37.5,  AsymmetricTolerance(0.000,0.011,note="ISO h5 Ø50-80"), "mm","Journal radius (bearing seat)"),
        "R3": VariableBounds(82.5, 70.0, 95.0,  AsymmetricTolerance(0.050,0.050,note="bilateral"), "mm","Flange outer radius"),
        "R4": VariableBounds(45.0, 35.0, 55.0,  AsymmetricTolerance(0.000,0.011,note="ISO h5 Ø90"), "mm","Tail outer radius"),
        "ri": VariableBounds(20.0, 15.0, 25.0,  AsymmetricTolerance(0.021,0.000,note="ISO H7 Ø40"), "mm","Inner bore radius"),
    })

    bearings: Dict[str, VariableBounds] = field(default_factory=lambda: {
        "front_z_fraction": VariableBounds(0.15,0.10,0.25, AsymmetricTolerance(0.010,0.010), "—","Front ACBB position (fraction of L2)"),
        "rear_z_fraction":  VariableBounds(0.75,0.65,0.85, AsymmetricTolerance(0.010,0.010), "—","Rear CRB pair centroid (fraction of L2)"),
        # NOTE: K_radial and K_axial are REMOVED as optimizer variables.
        # They are READ-ONLY quantities derived from snap_to_skf_bearing(R2, n_rpm).
        # Keeping them as free variables allowed the optimizer to create impossible
        # designs: a bearing that is simultaneously K_radial→min and K_axial→max,
        # which does not exist in any SKF catalog. Fixed in v4.
        # ── Positional Tolerances (ISO 1101 ⊕) ────────────────────────────
        "pos_tol_front": VariableBounds(
            nominal=0.010, lower=0.005, upper=0.030,
            tolerance=AsymmetricTolerance(0.005,0.003, note="ISO 1101 ⊕, front seat"),
            unit="mm",
            description="Positional tol. diameter of front ACBB seat",
        ),
        "pos_tol_rear": VariableBounds(
            nominal=0.015, lower=0.008, upper=0.040,
            tolerance=AsymmetricTolerance(0.007,0.004, note="ISO 1101 ⊕, rear seats"),
            unit="mm",
            description="Positional tol. diameter of rear CRB seats",
        ),
    })

    material: Dict[str, VariableBounds] = field(default_factory=lambda: {
        "E":       VariableBounds(2.1e5,2.0e5,2.15e5, AsymmetricTolerance(2000.0,2000.0), "MPa","Young's modulus"),
        "rho":     VariableBounds(7.85e-9,7.7e-9,8.0e-9, AsymmetricTolerance(5e-11,5e-11), "ton/mm³","Material density"),
        "sigma_y": VariableBounds(655.0,600.0,750.0, AsymmetricTolerance(10.0,15.0,note="Q&T -15/+10"), "MPa","Yield strength"),
    })

    loads: Dict[str, VariableBounds] = field(default_factory=lambda: {
        "Ft": VariableBounds(1500.0,1000.0,3500.0, AsymmetricTolerance(120.0,120.0,note="±8%"), "N","Tangential cutting force"),
        "Fr": VariableBounds(500.0, 300.0,1500.0, AsymmetricTolerance(40.0,40.0),  "N","Radial cutting force"),
        "Ff": VariableBounds(300.0, 200.0,1000.0, AsymmetricTolerance(24.0,24.0),  "N","Feed (axial) cutting force"),
    })

    # ── helpers ──────────────────────────────────────────────────────────────
    def get_all_variables(self) -> Dict[str, VariableBounds]:
        d: Dict[str, VariableBounds] = {}
        d.update(self.geometry); d.update(self.bearings)
        d.update(self.material); d.update(self.loads)
        return d

    def get_variable_names(self) -> List[str]:
        return sorted(self.get_all_variables().keys())

    def get_bounds(self) -> np.ndarray:
        all_v = self.get_all_variables()
        return np.array([[all_v[n].lower, all_v[n].upper] for n in sorted(all_v)])

    def get_nominal(self) -> np.ndarray:
        all_v = self.get_all_variables()
        return np.array([all_v[n].nominal for n in sorted(all_v)])

    def get_upper_tolerances(self) -> np.ndarray:
        all_v = self.get_all_variables()
        return np.array([all_v[n].tolerance.upper for n in sorted(all_v)])

    def get_lower_tolerances(self) -> np.ndarray:
        all_v = self.get_all_variables()
        return np.array([all_v[n].tolerance.lower for n in sorted(all_v)])

    def get_tolerances(self) -> np.ndarray:
        return (self.get_upper_tolerances() + self.get_lower_tolerances()) / 2.0

    def decode_vector(self, x: np.ndarray) -> Dict[str, float]:
        """
        Convert optimizer design vector to a named dictionary.

        IMPORTANT: This returns the RAW optimizer values.
        R2, K_radial, K_axial will NOT yet be catalog-snapped.
        Always call resolve_to_catalog() before saving or reporting results.
        """
        return {n: float(v) for n, v in zip(self.get_variable_names(), x)}

    def resolve_to_catalog(
        self,
        x:           np.ndarray,
        n_rpm:       float,
        arrangement: Optional["SpindleBearingArrangement"] = None,
    ) -> Dict[str, object]:
        """
        Produce the FULL manufacturable design specification.

        This is the correct function to call when saving or reporting an
        optimal design.  It:
          1. Decodes the raw optimizer vector
          2. Snaps R2 to the nearest real SKF catalog bearing
          3. REPLACES K_radial and K_axial with catalog-derived values
          4. Reports the real catalog designation (part number to order)

        Returns a dict with keys:
            design_variables  : raw optimizer values
            catalog_front     : SKFBearing record (front ACBB)
            catalog_rear      : SKFCRBBearing record (rear CRB)
            K_radial_catalog  : actual pair radial stiffness [N/mm]
            K_axial_catalog   : actual pair axial stiffness [N/mm]
            bore_mm           : actual catalog bore [mm]
            speed_ok          : bool
            speed_warning     : str
            manufacturable_design : flat dict ready for CSV export
        """
        import math as _math
        arr     = arrangement or SpindleBearingArrangement.default_lathe()
        v       = self.decode_vector(x)

        # Snap to nearest catalog bearings
        R2 = v["R2"]
        brg_front, ok_f, warn_f = snap_to_skf_bearing(R2, "ACBB", n_rpm, arr.lubrication)
        brg_rear,  ok_r, warn_r = snap_to_skf_bearing(R2, "CRB",  n_rpm, arr.lubrication)

        # Catalog-derived stiffness (ACBB pair, DB stiffness factor = 1.7)
        alpha     = _math.radians(brg_front.contact_angle_deg)
        K_r_pair  = 1.7 * brg_front.radial_stiffness_single_N_mm
        K_a_pair  = 1.7 * brg_front.radial_stiffness_single_N_mm * _math.tan(alpha)**2

        # Build the flat manufacturable design (what to put on a drawing/BOM)
        mfg = dict(v)
        mfg["R2_nominal_mm"]      = R2                     # optimizer value
        mfg["bore_catalog_mm"]    = float(brg_front.d)     # ← order THIS from SKF
        mfg["front_bearing"]      = brg_front.designation  # part number
        mfg["rear_bearing"]       = brg_rear.designation
        mfg["K_radial_catalog"]   = K_r_pair               # ← use THIS in FEA
        mfg["K_axial_catalog"]    = K_a_pair
        mfg["F_preload_MA_N"]     = float(brg_front.F_preload_MA_N)
        mfg["speed_ok"]           = ok_f and ok_r
        mfg["speed_warning"]      = (warn_f or warn_r or "none")

        return {
            "design_variables":      v,
            "catalog_front":         brg_front,
            "catalog_rear":          brg_rear,
            "K_radial_catalog":      K_r_pair,
            "K_axial_catalog":       K_a_pair,
            "bore_mm":               float(brg_front.d),
            "speed_ok":              ok_f and ok_r,
            "speed_warning":         warn_f or warn_r,
            "manufacturable_design": mfg,
        }

    def print_catalog_sheet(self, x: np.ndarray, n_rpm: float = 4000.0) -> None:
        """
        Print a human-readable design sheet with catalog-snapped values.
        Use this instead of printing raw decode_vector() output.
        """
        res  = self.resolve_to_catalog(x, n_rpm)
        v    = res["design_variables"]
        bf   = res["catalog_front"]
        br   = res["catalog_rear"]
        sep  = "─" * 65

        print(f"\n{'═'*65}")
        print(f"  MANUFACTURABLE DESIGN SPECIFICATION  @ {n_rpm:.0f} RPM")
        print(f"{'═'*65}")
        print(f"\n  Shaft Geometry:")
        for name in ["L1","L2","L3","L4","R1","R2","R3","R4","ri"]:
            print(f"    {name:<6} = {v[name]:>10.3f} mm")

        print(f"\n  Bearing Seats:")
        print(f"    Bore (catalog)  = {bf.d:>5.0f} mm   "
              f"← order this, NOT R2×2={v['R2']*2:.1f}mm")
        print(f"    front_z_frac    = {v['front_z_fraction']:.4f}")
        print(f"    rear_z_frac     = {v['rear_z_fraction']:.4f}")

        print(f"\n  Bearing Selection (SKF Catalog):")
        print(f"    FRONT (locating) : {bf.designation}")
        print(f"      d={bf.d}mm  D={bf.D}mm  B={bf.B}mm  "
              f"C_r={bf.C_r/1e3:.1f}kN  n_gr={bf.n_grease}RPM")
        print(f"      F_preload (MA) = {bf.F_preload_MA_N:.0f} N")
        print(f"    REAR (floating)  : {br.designation} × 2")
        print(f"      d={br.d}mm  D={br.D}mm  B={br.B}mm  "
              f"C_r={br.C_r/1e3:.1f}kN  n_gr={br.n_grease}RPM")

        print(f"\n  Catalog Stiffness (use in ANSYS COMBIN14):")
        print(f"    K_radial pair   = {res['K_radial_catalog']:>10,.0f} N/mm  "
              f"({res['K_radial_catalog']/1000:.0f} N/μm)")
        print(f"    K_axial  pair   = {res['K_axial_catalog']:>10,.0f} N/mm  "
              f"({res['K_axial_catalog']/1000:.0f} N/μm)")

        print(f"\n  Positional Tolerances (ISO 1101 ⊕):")
        print(f"    pos_tol_front   = ϕ{v['pos_tol_front']*1000:.1f} μm")
        print(f"    pos_tol_rear    = ϕ{v['pos_tol_rear']*1000:.1f} μm")

        print(f"\n  Material & Loads:")
        for name in ["E","rho","sigma_y","Ft","Fr","Ff"]:
            print(f"    {name:<8} = {v[name]:>12.4g}")

        warn = res["speed_warning"]
        print(f"\n  Speed check: {'✅ OK' if res['speed_ok'] else '❌'}"
              + (f"  {warn}" if warn else ""))
        print(f"{'═'*65}\n")

    def sample_manufacturing_variation(self, x_nominal, n_samples=1, seed=None) -> np.ndarray:
        rng    = np.random.default_rng(seed)
        names  = self.get_variable_names()
        all_v  = self.get_all_variables()
        bounds = self.get_bounds()
        X = np.zeros((n_samples, len(names)))
        for i, name in enumerate(names):
            X[:, i] = all_v[name].tolerance.sample(x_nominal[i], n=n_samples, rng=rng)
        return np.clip(X, bounds[:, 0], bounds[:, 1])

    def resolve_arrangement(self, x, n_rpm, arrangement=None):
        return _resolve_arrangement(self, x, n_rpm, arrangement)

    def summary(self) -> str:
        all_v = self.get_all_variables()
        hdr = (f"{'Variable':<22} {'Nominal':>10} {'Lower':>10} {'Upper':>10} "
               f"{'Tol +':>8} {'Tol −':>8} {'Unit':<8} Description")
        sep = "=" * 108
        rows = [sep, hdr, "-" * 108]
        for name in sorted(all_v):
            v = all_v[name]; tol = v.tolerance
            rows.append(f"{name:<22} {v.nominal:>10.4g} {v.lower:>10.4g} "
                        f"{v.upper:>10.4g} {tol.upper:>8.4g} {tol.lower:>8.4g} "
                        f"{v.unit:<8} {v.description}")
        rows += [sep, f"Total: {len(all_v)} variables"]
        return "\n".join(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_design_space(ds: DesignSpace, save_dir: str = ".") -> List:
    """
    Generate three design-space diagnostic plots.

    Figure 1 — Normalised variable bounds (search range width)
    Figure 2 — Tolerance asymmetry (upper vs lower deviation)
    Figure 3 — Manufacturing variation cloud (geometric variables, 500 MC samples)

    Returns list of (fig, filepath) tuples.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import os

    os.makedirs(save_dir, exist_ok=True)

    NAVY  = "#0d1b2a"
    TEAL  = "#00b4d8"
    CORAL = "#e63946"
    GOLD  = "#ffd166"
    GRAY  = "#8d99ae"
    plt.rcParams.update({
        "figure.facecolor": NAVY, "axes.facecolor": "#112233",
        "axes.edgecolor": GRAY, "axes.labelcolor": "white",
        "xtick.color": GRAY, "ytick.color": GRAY,
        "text.color": "white", "grid.color": "#2d4060",
        "grid.alpha": 0.5, "font.size": 9,
    })

    all_v  = ds.get_all_variables()
    names  = ds.get_variable_names()
    nom    = ds.get_nominal()
    bounds = ds.get_bounds()
    upper_tols = ds.get_upper_tolerances()
    lower_tols = ds.get_lower_tolerances()

    figs = []

    # ── Fig 1: Normalised search range width ─────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    fig1.patch.set_facecolor(NAVY)
    widths = (bounds[:, 1] - bounds[:, 0]) / np.maximum(np.abs(nom), 1e-12) * 100
    colours = [TEAL if "pos_tol" not in n else GOLD for n in names]
    bars = ax1.barh(names, widths, color=colours, edgecolor=NAVY, linewidth=0.4)
    ax1.set_xlabel("Search range / |nominal|  [%]")
    ax1.set_title("Fig 1 — Design Variable Search Range (Normalised)", pad=10)
    ax1.axvline(20, color=GRAY, linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.text(20.5, -0.5, "20%", color=GRAY, fontsize=8)
    teal_patch = mpatches.Patch(color=TEAL, label="Geometric / material / load")
    gold_patch  = mpatches.Patch(color=GOLD, label="Positional tolerance (new)")
    ax1.legend(handles=[teal_patch, gold_patch], loc="lower right", fontsize=8)
    plt.tight_layout()
    p1 = os.path.join(save_dir, "01a_design_space_bounds.png")
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    figs.append((fig1, p1))
    print(f"  Saved → {p1}")

    # ── Fig 2: Tolerance asymmetry ────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    fig2.patch.set_facecolor(NAVY)
    y_pos = np.arange(len(names))
    ax2.barh(y_pos - 0.2, upper_tols, height=0.35, color=TEAL,  label="+upper deviation", edgecolor=NAVY, linewidth=0.3)
    ax2.barh(y_pos + 0.2, lower_tols, height=0.35, color=CORAL, label="−lower deviation (magnitude)", edgecolor=NAVY, linewidth=0.3)
    ax2.set_yticks(y_pos); ax2.set_yticklabels(names, fontsize=7.5)
    ax2.set_xlabel("Tolerance deviation [same unit as variable]")
    ax2.set_title("Fig 2 — Asymmetric Tolerances (+upper / −lower) per Variable", pad=10)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlim(left=1e-5)
    plt.tight_layout()
    p2 = os.path.join(save_dir, "01b_tolerance_asymmetry.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    figs.append((fig2, p2))
    print(f"  Saved → {p2}")

    # ── Fig 3: Manufacturing variation cloud ──────────────────────────────
    X_mc   = ds.sample_manufacturing_variation(nom, n_samples=500, seed=42)
    geom_names = sorted(ds.geometry.keys())[:4]   # first 4 geometric vars
    geom_idx   = [names.index(n) for n in geom_names]

    fig3, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig3.patch.set_facecolor(NAVY)
    fig3.suptitle("Fig 3 — Manufacturing Variation Cloud (500 MC samples, geometric vars)",
                  color="white", y=1.01)
    pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    for ax, (i, j) in zip(axes.flat, pairs):
        xi = X_mc[:, geom_idx[i]]; xj = X_mc[:, geom_idx[j]]
        nom_i = nom[geom_idx[i]];  nom_j = nom[geom_idx[j]]
        ax.scatter(xi, xj, s=4, c=TEAL, alpha=0.35)
        ax.scatter([nom_i], [nom_j], s=60, c=GOLD, zorder=5, marker="*", label="Nominal")
        ax.set_xlabel(geom_names[i], fontsize=8); ax.set_ylabel(geom_names[j], fontsize=8)
        ax.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=7)
    plt.tight_layout()
    p3 = os.path.join(save_dir, "01c_mfg_variation_cloud.png")
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    figs.append((fig3, p3))
    print(f"  Saved → {p3}")

    plt.close("all")
    return figs


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ds  = DesignSpace()
    nom = ds.get_nominal()
    n   = len(ds.get_variable_names())
    print(ds.summary())
    assert n == 21, f"Expected 21 variables, got {n}"
    print(f"\n✅  {n} design variables confirmed")

    # Asymmetric sampling: h5 R2 must only be ≤ nominal
    idx_R2 = ds.get_variable_names().index("R2")
    X_mc   = ds.sample_manufacturing_variation(nom, 2000, seed=42)
    assert X_mc[:, idx_R2].max() <= nom[idx_R2]
    print("✅  R2 h5 shaft: samples never exceed nominal")

    # pos_tol variables exist
    assert "pos_tol_front" in ds.get_variable_names()
    assert "pos_tol_rear"  in ds.get_variable_names()
    print("✅  pos_tol_front and pos_tol_rear present")

    # Plots
    print("\nGenerating design-space plots...")
    figs = plot_design_space(ds, save_dir="/tmp/spindle_plots")
    print(f"✅  {len(figs)} plots generated\n")
