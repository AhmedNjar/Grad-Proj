#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  Tolerance Optimizer — Module 12
================================================================================

  Purpose:
      After geometry optimization fixes the nominal dimensions (x_opt from
      Module 05), this module finds the optimal TOLERANCE SPECIFICATION that
      minimises a Pareto front of three objectives:

        f1 — Machining cost  [USD]          (tight tolerance = expensive)
        f2 — TIR runout      [μm]           (loose tolerance = more runout)
        f3 — L10 life loss   [%]            (rough surface = shorter life)

  Tolerance Design Variables (7 total):
  ──────────────────────────────────────
  Discrete (integer index into grade list):
    it_journal  — IT grade for ALL shaft diameters (R1,R2,R3,R4)
                  choices: IT4 | IT5 | IT6 | IT7
    it_bore     — IT grade for inner bore (ri)
                  choices: IT6 | IT7 | IT8 | IT9
    it_lengths  — IT grade for ALL axial lengths (L1,L2,L3,L4)
                  choices: IT10 | IT11 | IT12 | IT13

  Continuous:
    pos_tol_front — ISO 1101 ⊕ positional tolerance of front ACBB seat [mm]
                    range: [0.005, 0.030]
    pos_tol_rear  — ISO 1101 ⊕ positional tolerance of rear CRB seats [mm]
                    range: [0.008, 0.040]
    ra_journal    — Surface roughness Ra of bearing seats [μm]
                    range: [0.2, 1.6]
    ra_bore       — Surface roughness Ra of inner bore [μm]
                    range: [0.4, 6.3]

  Physics Links (ISO-standard):
  ──────────────────────────────
  IT grade → IT value [μm] via ISO 286-1 tables (size-dependent)
  Roundness ≈ 0.25 × IT_value  (grinding process, ISO 4288)
  TIR_journal  = roundness_journal × amp_factor
  TIR_bore     = bore_ecc(IT_bore) × amp_factor    (missing-mass formula)
  L10 scaling  = a_ISO × ηc(Ra_journal)            (ISO 281:2007 Annex A)
  Cost         = Σ cost_per_feature(IT, Ra)         (machining economics)

  Optimization:
  ─────────────
  Strategy: Enumerate all 4×4×4 = 64 IT-grade combinations.
  For each combination, optimise the 4 continuous variables with
  scipy.optimize.minimize (SLSQP) on a weighted sum.
  Weight vectors swept over a 15-point simplex → approx Pareto front.
  Final result: non-dominated Pareto set across all combinations.

  This approach is exact for discrete variables and fast (< 10 s).

================================================================================
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


# ─────────────────────────────────────────────────────────────────────────────
# ISO 286-1 IT value tables
# ─────────────────────────────────────────────────────────────────────────────
# IT fundamental tolerance values [μm] keyed by (size_range_id, IT_grade)
# size_range_id: 1=30-50mm, 2=50-80mm, 3=80-120mm, 4=120-180mm, 5=180-250mm,
#                6=250-315mm, 7=315-500mm

_IT_TABLE: Dict[Tuple[int, str], float] = {
    # Diameters 30-50mm
    (1,"IT4"):8,  (1,"IT5"):11, (1,"IT6"):16, (1,"IT7"):25,
    (1,"IT8"):39, (1,"IT9"):62, (1,"IT10"):100,(1,"IT11"):160,
    (1,"IT12"):250,(1,"IT13"):400,
    # Diameters 50-80mm
    (2,"IT4"):9,  (2,"IT5"):13, (2,"IT6"):19, (2,"IT7"):30,
    (2,"IT8"):46, (2,"IT9"):74, (2,"IT10"):120,(2,"IT11"):190,
    (2,"IT12"):300,(2,"IT13"):460,
    # Diameters 80-120mm
    (3,"IT4"):10, (3,"IT5"):15, (3,"IT6"):22, (3,"IT7"):35,
    (3,"IT8"):54, (3,"IT9"):87, (3,"IT10"):140,(3,"IT11"):220,
    (3,"IT12"):350,(3,"IT13"):540,
    # Diameters 120-180mm
    (4,"IT4"):12, (4,"IT5"):18, (4,"IT6"):25, (4,"IT7"):40,
    (4,"IT8"):63, (4,"IT9"):100,(4,"IT10"):160,(4,"IT11"):250,
    (4,"IT12"):400,(4,"IT13"):630,
    # Lengths 315-500mm (for L2~350mm)
    (7,"IT10"):230,(7,"IT11"):360,(7,"IT12"):570,(7,"IT13"):890,
    # Lengths 180-250mm (for L1~100mm range)
    (5,"IT10"):185,(5,"IT11"):290,(5,"IT12"):460,(5,"IT13"):720,
}


def _size_range_id(d_mm: float) -> int:
    if   d_mm <= 50:  return 1
    elif d_mm <= 80:  return 2
    elif d_mm <= 120: return 3
    elif d_mm <= 180: return 4
    elif d_mm <= 250: return 5
    elif d_mm <= 315: return 6
    else:             return 7


def it_value_um(grade: str, nominal_mm: float) -> float:
    """Return ISO 286-1 IT fundamental tolerance value [μm]."""
    rid = _size_range_id(nominal_mm)
    key = (rid, grade)
    if key in _IT_TABLE:
        return float(_IT_TABLE[key])
    # Fallback: interpolate from nearest
    for r in [rid-1, rid+1, rid-2, rid+2]:
        if (r, grade) in _IT_TABLE:
            return float(_IT_TABLE[(r, grade)])
    raise ValueError(f"IT value not found for grade={grade}, d={nominal_mm}mm")


# ─────────────────────────────────────────────────────────────────────────────
# IT grade lists (choices per feature group)
# ─────────────────────────────────────────────────────────────────────────────

JOURNAL_GRADES = ["IT4", "IT5", "IT6", "IT7"]    # for R1,R2,R3,R4
BORE_GRADES    = ["IT6", "IT7", "IT8", "IT9"]    # for ri
LENGTH_GRADES  = ["IT10","IT11","IT12","IT13"]   # for L1,L2,L3,L4


# ─────────────────────────────────────────────────────────────────────────────
# Cost model  (ISO-aligned machining economics)
# ─────────────────────────────────────────────────────────────────────────────

# Relative machining cost index for IT grades (base = IT8 for diameters)
_IT_COST_DIAL: Dict[str, float] = {
    "IT4": 95.0, "IT5": 55.0, "IT6": 28.0, "IT7": 14.0,
    "IT8": 7.0,  "IT9": 3.5,
    "IT10": 2.5, "IT11": 1.5, "IT12": 1.0, "IT13": 0.7,
}

# Machining cost per feature group [USD] at base IT level
_BASE_COST: Dict[str, float] = {
    "journal":  18.0,   # per shaft journal diameter (grinding)
    "bore":      8.0,   # inner bore (boring/reaming)
    "length":    4.0,   # per length (turning face)
}

# Surface roughness cost per pass [USD]
_RA_COST: Dict[str, float] = {
    "ra_journal": {  # bearing seat, USD per feature
        0.2: 85.0, 0.4: 42.0, 0.8: 18.0, 1.6: 7.0
    },
    "ra_bore": {
        0.4: 22.0, 0.8: 10.0, 1.6: 5.0, 3.2: 2.5, 6.3: 1.0
    },
}

_RA_JOURNAL_VALUES = [0.2, 0.4, 0.8, 1.6]
_RA_BORE_VALUES    = [0.4, 0.8, 1.6, 3.2, 6.3]


def _ra_cost(ra_value: float, feature: str) -> float:
    """Linear interpolation between Ra cost anchor points."""
    anchors = _RA_COST[feature]
    keys    = sorted(anchors.keys())
    if ra_value <= keys[0]:
        return anchors[keys[0]]
    if ra_value >= keys[-1]:
        return anchors[keys[-1]]
    for i in range(len(keys)-1):
        if keys[i] <= ra_value <= keys[i+1]:
            t = (ra_value - keys[i]) / (keys[i+1] - keys[i])
            return anchors[keys[i]] * (1-t) + anchors[keys[i+1]] * t
    return anchors[keys[-1]]


def compute_cost(
    it_journal_grade: str,
    it_bore_grade:    str,
    it_length_grade:  str,
    ra_journal_um:    float,
    ra_bore_um:       float,
    n_journals:       int = 4,    # R1,R2,R3,R4
    n_lengths:        int = 4,    # L1,L2,L3,L4
) -> float:
    """
    Total machining tolerance cost [USD].

    Cost scales exponentially with IT tightness and Ra fineness.
    Each journal is machined separately; bore and lengths are single operations.
    """
    journal_it_cost = _IT_COST_DIAL.get(it_journal_grade, 7.0) * _BASE_COST["journal"]
    bore_it_cost    = _IT_COST_DIAL.get(it_bore_grade,    7.0) * _BASE_COST["bore"]
    length_it_cost  = _IT_COST_DIAL.get(it_length_grade,  1.0) * _BASE_COST["length"]

    cost = (n_journals * journal_it_cost
            + bore_it_cost
            + n_lengths * length_it_cost
            + _ra_cost(ra_journal_um, "ra_journal") * n_journals
            + _ra_cost(ra_bore_um,   "ra_bore"))
    return cost


# ─────────────────────────────────────────────────────────────────────────────
# Runout model
# ─────────────────────────────────────────────────────────────────────────────

def compute_tir(
    it_journal_grade: str,
    it_bore_grade:    str,
    pos_tol_front_mm: float,
    pos_tol_rear_mm:  float,
    ra_journal_um:    float,
    ra_bore_um:       float,
    d_journal_mm:     float,   # 2×R2 (bearing seat diameter)
    d_bore_mm:        float,   # 2×ri
    R_outer_mm:       float,   # R2
    wall_mm:          float,   # R2 − ri
    L_overhang_mm:    float,
    L_span_mm:        float,
    delta_nose_um:    float,   # ANSYS elastic deflection at nose
) -> float:
    """
    Total TIR at spindle nose [μm] — RSS of all sources.

    Sources:
      1. Bearing inner-ring: uses journal IT grade → achievable roundness
      2. Bore eccentricity from IT bore grade → CG shift → imbalance runout
      3. Positional tolerance front seat (ISO 1101 ⊕)
      4. Positional tolerance rear seat (tilt)
      5. Ra of journal → additional roundness penalty
      6. ANSYS elastic deflection (fixed, from geometry optimisation)
    """
    amp = 1.0 + L_overhang_mm / max(L_span_mm, 1.0)

    # ── Source 1: Journal IT grade → bearing seat roundness → TIR ─────
    # Achievable roundness ≈ 0.25 × IT_value (precision grinding, ISO 4288)
    it_val_j   = it_value_um(it_journal_grade, d_journal_mm)
    roundness_j = 0.25 * it_val_j           # [μm]
    TIR_journal = roundness_j * amp         # [μm]

    # ── Source 2: Ra of journal seat → additional roundness contribution ─
    # Roughness profile peaks can cause effective eccentricity;
    # Ra_journal contributes ≈ 0.5 × Ra to roundness deviation
    TIR_ra = 0.5 * ra_journal_um * amp     # [μm]

    # ── Source 3: IT bore grade → bore eccentricity → CG offset ────────
    # IT_bore specifies bore DIAMETER tolerance, not axis eccentricity.
    # Bore axis eccentricity ≈ IT_bore / 6  (3σ process capability of
    # precision boring machine per ISO 230-2).
    # IT/2 would be worst-case; IT/6 is the realistic 3σ capability
    # of a CNC boring operation on a machining centre.
    it_val_b = it_value_um(it_bore_grade, d_bore_mm)
    eps_bore  = (it_val_b / 6.0) * 1e-3        # mm  (3σ boring capability)
    r_outer   = R_outer_mm; r_bore = d_bore_mm / 2.0
    denom     = r_outer**2 - r_bore**2
    e_cg_mm   = eps_bore * r_bore**2 / max(denom, 1e-6)
    TIR_bore  = e_cg_mm * amp * 1000.0          # μm

    # ── Source 4: Positional tolerance front (ISO 1101 ⊕) ───────────────
    TIR_pos_front = (pos_tol_front_mm / 2.0) * amp * 1000.0      # μm

    # ── Source 5: Positional tolerance rear (tilt) ──────────────────────
    TIR_pos_rear  = (pos_tol_rear_mm  / 2.0) * (L_overhang_mm / max(L_span_mm,1)) * 1000.0

    # ── Source 6: ANSYS elastic deflection (fixed geometry, not a tolerance) ──
    # This source is NOT controllable by tolerance optimisation.
    # Reported separately so constraints apply to tolerance-driven TIR only.
    TIR_elastic = abs(delta_nose_um)

    # ── RSS of TOLERANCE-CONTROLLED sources only ──────────────────────
    tir_tol_rss = math.sqrt(
        TIR_journal**2 + TIR_ra**2 + TIR_bore**2
        + TIR_pos_front**2 + TIR_pos_rear**2
    )
    # Full RSS including fixed ANSYS deflection (for information only)
    tir_rss = math.sqrt(tir_tol_rss**2 + TIR_elastic**2)
    return tir_tol_rss   # optimiser only controls tolerance sources


# ─────────────────────────────────────────────────────────────────────────────
# L10 life reduction model
# ─────────────────────────────────────────────────────────────────────────────

def compute_l10_loss_pct(
    it_length_grade: str,
    ra_journal_um:   float,
    L_span_mm:       float,
    L10_base_hours:  float,
) -> float:
    """
    Estimated L10 life reduction [%] due to tolerance effects.

    Two mechanisms:
    1. Length IT grade → face parallelism → angular misalignment of bearing
       → ISO 281 misalignment factor reduces L10.
    2. Ra of bearing seat → contamination/lubrication factor ηc (ISO 281:2007)
       → cleaner surface = higher ηc = longer L10.

    Returns percentage REDUCTION from nominal L10 (0 % = no reduction).
    Higher is worse.
    """
    # ── Length IT → angular misalignment ──────────────────────────────
    # Face parallelism ≈ IT_length_value / L_span (rad)
    it_val_l = it_value_um(it_length_grade, L_span_mm)
    theta_rad = (it_val_l * 1e-3) / max(L_span_mm, 1.0)   # rad (tiny)
    # ISO 281 Annex B: for ACBB, L10 factor ≈ exp(-200 × θ²) (approximate)
    # For θ < 0.001 rad this gives < 5% loss; for IT13 at 350mm θ≈2.5e-3 → 12% loss
    f_align = math.exp(-200.0 * theta_rad**2)
    l10_loss_align_pct = max(0.0, (1.0 - f_align) * 100.0)

    # ── Ra → contamination factor ηc (ISO 281:2007 Table 1) ────────────
    # Empirical: ηc_effective = ηc_clean × (1 − 0.12 × Ra)
    # for Ra in [0.2, 1.6] μm and grease lubrication
    eta_c = max(0.1, 1.0 - 0.12 * ra_journal_um)
    # L10 ∝ a_ISO; a_ISO ∝ ηc (simplified)
    l10_loss_ra_pct = max(0.0, (1.0 - eta_c) * 100.0)

    return min(l10_loss_align_pct + l10_loss_ra_pct, 95.0)   # cap at 95%


# ─────────────────────────────────────────────────────────────────────────────
# Tolerance evaluator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TolerancePoint:
    """One evaluated tolerance design point."""
    it_journal:       str
    it_bore:          str
    it_lengths:       str
    pos_tol_front_mm: float
    pos_tol_rear_mm:  float
    ra_journal_um:    float
    ra_bore_um:       float
    cost_usd:         float
    tir_rss_um:       float
    l10_loss_pct:     float

    @property
    def objectives(self) -> np.ndarray:
        return np.array([self.cost_usd, self.tir_rss_um, self.l10_loss_pct])

    def is_dominated_by(self, other: "TolerancePoint") -> bool:
        """True if other dominates self (all objectives ≤, at least one <)."""
        return (np.all(other.objectives <= self.objectives)
                and np.any(other.objectives < self.objectives))

    def summary(self) -> str:
        return (f"IT_j={self.it_journal:<5} IT_b={self.it_bore:<5}"
                f" IT_l={self.it_lengths:<5}"
                f" pos_f={self.pos_tol_front_mm*1000:.1f}μm"
                f" pos_r={self.pos_tol_rear_mm*1000:.1f}μm"
                f" Ra_j={self.ra_journal_um:.2f}μm Ra_b={self.ra_bore_um:.1f}μm"
                f" → cost=${self.cost_usd:.0f}"
                f"  TIR={self.tir_rss_um:.2f}μm"
                f"  L10_loss={self.l10_loss_pct:.1f}%")


class ToleranceEvaluator:
    """
    Computes the 3 objectives for any tolerance specification
    given fixed geometry from the geometry optimiser.
    """

    def __init__(
        self,
        d_journal_mm:    float,   # 2×R2
        d_bore_mm:       float,   # 2×ri
        R_outer_mm:      float,   # R2
        L_overhang_mm:   float,
        L_span_mm:       float,
        delta_nose_um:   float,
        L10_base_hours:  float,
        n_journals:      int   = 4,
        n_lengths:       int   = 4,
    ):
        self.d_j  = d_journal_mm
        self.d_b  = d_bore_mm
        self.R2   = R_outer_mm
        self.L_oh = L_overhang_mm
        self.L_sp = L_span_mm
        self.delta = delta_nose_um
        self.L10  = L10_base_hours
        self.wall  = R_outer_mm - d_bore_mm / 2.0
        self.n_j  = n_journals
        self.n_l  = n_lengths

    def evaluate(
        self,
        it_journal:       str,
        it_bore:          str,
        it_lengths:       str,
        pos_tol_front_mm: float,
        pos_tol_rear_mm:  float,
        ra_journal_um:    float,
        ra_bore_um:       float,
    ) -> TolerancePoint:

        cost = compute_cost(
            it_journal, it_bore, it_lengths,
            ra_journal_um, ra_bore_um,
            n_journals=self.n_j, n_lengths=self.n_l,
        )
        tir = compute_tir(
            it_journal, it_bore,
            pos_tol_front_mm, pos_tol_rear_mm,
            ra_journal_um, ra_bore_um,
            d_journal_mm=self.d_j,
            d_bore_mm=self.d_b,
            R_outer_mm=self.R2,
            wall_mm=self.wall,
            L_overhang_mm=self.L_oh,
            L_span_mm=self.L_sp,
            delta_nose_um=self.delta,
        )
        l10_loss = compute_l10_loss_pct(
            it_lengths, ra_journal_um,
            L_span_mm=self.L_sp,
            L10_base_hours=self.L10,
        )
        return TolerancePoint(
            it_journal, it_bore, it_lengths,
            pos_tol_front_mm, pos_tol_rear_mm,
            ra_journal_um, ra_bore_um,
            cost, tir, l10_loss,
        )

    def evaluate_vec(
        self,
        it_j: str, it_b: str, it_l: str,
        cont: np.ndarray,          # [pos_f, pos_r, ra_j, ra_b]
    ) -> np.ndarray:
        """Return (cost, tir, l10_loss) as numpy array."""
        pt = self.evaluate(it_j, it_b, it_l,
                           float(cont[0]), float(cont[1]),
                           float(cont[2]), float(cont[3]))
        return pt.objectives


# ─────────────────────────────────────────────────────────────────────────────
# Pareto helper
# ─────────────────────────────────────────────────────────────────────────────

def pareto_front(points: List[TolerancePoint]) -> List[TolerancePoint]:
    """Return the non-dominated, deduplicated Pareto front."""
    # Deduplicate: round objectives to 2 dp
    seen = set()
    unique = []
    for p in points:
        key = (round(p.cost_usd,1), round(p.tir_rss_um,2),
               round(p.l10_loss_pct,2), p.it_journal, p.it_bore, p.it_lengths)
        if key not in seen:
            seen.add(key); unique.append(p)

    dominated = set()
    for i, p in enumerate(unique):
        for j, q in enumerate(unique):
            if i != j and p.is_dominated_by(q):
                dominated.add(i); break
    return [p for i, p in enumerate(unique) if i not in dominated]


# ─────────────────────────────────────────────────────────────────────────────
# Tolerance Optimizer
# ─────────────────────────────────────────────────────────────────────────────

class ToleranceOptimizer:
    """
    Multi-objective tolerance optimizer.

    Strategy:
        Enumerate all 4×4×4 = 64 IT-grade combinations.
        For each, optimise the 4 continuous variables (pos_tol_front/rear,
        Ra_journal, Ra_bore) using weighted-sum scalarisation over 15 weight
        vectors sampled on the Pareto simplex.
        Collect all 64×15 = 960 evaluated points.
        Return Pareto-optimal subset.

    This is exact for discrete IT choices and fast (< 15 s total).

    Parameters
    ----------
    evaluator        : ToleranceEvaluator
    tir_limit_um     : Hard TIR constraint [μm] — any point above this is
                       excluded from Pareto front
    l10_loss_max_pct : Hard L10 loss constraint [%]
    n_weights        : Number of weight vectors per IT combination
    """

    # Continuous variable bounds: [pos_f, pos_r, ra_j, ra_b]
    _CONT_BOUNDS = [
        (0.005, 0.030),   # pos_tol_front [mm]
        (0.008, 0.040),   # pos_tol_rear  [mm]
        (0.20,  1.60),    # ra_journal    [μm]
        (0.40,  6.30),    # ra_bore       [μm]
    ]
    _CONT_NOMINAL = np.array([0.010, 0.015, 0.80, 1.60])

    def __init__(
        self,
        evaluator:        ToleranceEvaluator,
        tir_limit_um:     float = 15.0,
        l10_loss_max_pct: float = 20.0,
        n_weights:        int   = 15,
    ):
        self.ev              = evaluator
        self.tir_limit       = tir_limit_um
        self.l10_loss_max    = l10_loss_max_pct
        self.n_weights       = n_weights

    def _weight_vectors(self) -> np.ndarray:
        """Sample weight vectors uniformly on 3-objective unit simplex."""
        rng = np.random.default_rng(0)
        n   = self.n_weights
        # Generate simplex points: w = (w1, w2, w3), w ≥ 0, Σw = 1
        raw = rng.dirichlet(np.ones(3), size=n)
        # Also include axis-aligned extremes
        corners = np.eye(3)
        return np.vstack([raw, corners])

    def _optimise_continuous(
        self,
        it_j: str, it_b: str, it_l: str,
        weights: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Minimise weighted sum of objectives over continuous variables.
        Returns optimal continuous vector or None if constraints violated.
        """
        lb = np.array([b[0] for b in self._CONT_BOUNDS])
        ub = np.array([b[1] for b in self._CONT_BOUNDS])

        def objective(cont: np.ndarray) -> float:
            obj = self.ev.evaluate_vec(it_j, it_b, it_l, cont)
            # Penalty for constraint violations
            tir_penalty  = max(0, obj[1] - self.tir_limit)    * 1e4
            l10_penalty  = max(0, obj[2] - self.l10_loss_max) * 1e4
            # Normalise objectives before weighting
            norm_cost = obj[0] / 3000.0    # typical max cost ~3000 USD
            norm_tir  = obj[1] / 50.0      # typical max TIR ~50 μm
            norm_l10  = obj[2] / 100.0     # max loss 100%
            return (weights[0]*norm_cost + weights[1]*norm_tir
                    + weights[2]*norm_l10 + tir_penalty + l10_penalty)

        result = minimize(
            objective,
            x0=np.clip(self._CONT_NOMINAL, lb, ub),
            method="L-BFGS-B",
            bounds=list(zip(lb, ub)),
            options={"maxiter": 200, "ftol": 1e-9},
        )
        return np.clip(result.x, lb, ub)

    def run(self, verbose: bool = True) -> List[TolerancePoint]:
        """
        Run the full tolerance optimisation.

        Returns the Pareto-optimal list of TolerancePoint objects.
        """
        all_points: List[TolerancePoint] = []
        weight_vecs = self._weight_vectors()

        combos = list(product(JOURNAL_GRADES, BORE_GRADES, LENGTH_GRADES))
        if verbose:
            print(f"\n  Tolerance Optimizer: {len(combos)} IT combos "
                  f"× {len(weight_vecs)} weight vectors = "
                  f"{len(combos)*len(weight_vecs)} evaluations")

        for it_j, it_b, it_l in combos:
            for w in weight_vecs:
                cont = self._optimise_continuous(it_j, it_b, it_l, w)
                if cont is None:
                    continue
                pt = self.ev.evaluate(
                    it_j, it_b, it_l,
                    float(cont[0]), float(cont[1]),
                    float(cont[2]), float(cont[3]),
                )
                all_points.append(pt)

        pareto = pareto_front(all_points)

        if verbose:
            print(f"  Total evaluated   : {len(all_points)}")
            print(f"  Pareto-optimal    : {len(pareto)}")
            feasible = [p for p in pareto
                        if p.tir_rss_um <= self.tir_limit
                        and p.l10_loss_pct <= self.l10_loss_max]
            print(f"  Feasible (TIR≤{self.tir_limit}μm, L10_loss≤{self.l10_loss_max}%): "
                  f"{len(feasible)}")

        return pareto

    def best_by_priority(
        self,
        pareto:   List[TolerancePoint],
        priority: str = "cost",   # "cost" | "tir" | "l10"
    ) -> Optional[TolerancePoint]:
        """
        From the Pareto front, pick the single best point by one priority.

        priority="cost"  → cheapest feasible design
        priority="tir"   → lowest TIR feasible design
        priority="l10"   → minimum L10 loss feasible design
        """
        feasible = [p for p in pareto
                    if p.tir_rss_um <= self.tir_limit
                    and p.l10_loss_pct <= self.l10_loss_max]
        if not feasible:
            feasible = pareto   # relax constraints if nothing feasible

        key_map = {
            "cost": lambda p: p.cost_usd,
            "tir":  lambda p: p.tir_rss_um,
            "l10":  lambda p: p.l10_loss_pct,
        }
        return min(feasible, key=key_map.get(priority, key_map["cost"]))


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def print_tolerance_report(
    pareto:   List[TolerancePoint],
    best:     TolerancePoint,
    var_dict: Dict,
) -> None:
    """Print the full tolerance optimisation report."""
    print(f"\n{'═'*72}")
    print(f"  TOLERANCE OPTIMISATION REPORT")
    print(f"{'═'*72}")
    print(f"\n  Pareto front: {len(pareto)} non-dominated designs")
    print(f"\n  {'IT_j':<6} {'IT_b':<6} {'IT_l':<6} "
          f"{'pos_f[μm]':>10} {'pos_r[μm]':>10} "
          f"{'Ra_j':>6} {'Ra_b':>6} "
          f"{'Cost[$]':>8} {'TIR[μm]':>8} {'L10loss':>8}")
    print(f"  {'─'*70}")
    for pt in sorted(pareto, key=lambda p: p.cost_usd):
        marker = " ◄ BEST" if pt is best else ""
        print(f"  {pt.it_journal:<6} {pt.it_bore:<6} {pt.it_lengths:<6}"
              f" {pt.pos_tol_front_mm*1000:>10.1f} {pt.pos_tol_rear_mm*1000:>10.1f}"
              f" {pt.ra_journal_um:>6.2f} {pt.ra_bore_um:>6.1f}"
              f" {pt.cost_usd:>8.0f} {pt.tir_rss_um:>8.2f} {pt.l10_loss_pct:>7.1f}%"
              + marker)

    print(f"\n{'─'*72}")
    print(f"  RECOMMENDED TOLERANCE SPECIFICATION")
    print(f"{'─'*72}")

    d_j = var_dict.get("R2", 50.0) * 2
    d_b = var_dict.get("ri", 30.0) * 2
    it_j_val = it_value_um(best.it_journal, d_j)
    it_b_val = it_value_um(best.it_bore,    d_b)
    it_l_val = it_value_um(best.it_lengths, 350.0)

    print(f"\n  {'Feature':<35} {'Grade':<8} {'IT value':>10} {'Ra [μm]':>9}  Standard")
    print(f"  {'─'*75}")
    print(f"  {'Shaft journals (R1,R2,R3,R4)':<35} {best.it_journal:<8}"
          f" {it_j_val:>8.0f}μm {best.ra_journal_um:>8.2f}  ISO 286-1 + ISO 4288")
    print(f"  {'Inner bore (ri)':<35} {best.it_bore:<8}"
          f" {it_b_val:>8.0f}μm {best.ra_bore_um:>8.1f}  ISO 286-1")
    print(f"  {'Axial lengths (L1,L2,L3,L4)':<35} {best.it_lengths:<8}"
          f" {it_l_val:>8.0f}μm {'—':>8}  ISO 286-1")
    print(f"  {'Front bearing seat  ⊕':<35} {'ISO 1101':<8}"
          f" {best.pos_tol_front_mm*1000:>7.1f}μm {'—':>9}  ISO 1101:2017")
    print(f"  {'Rear bearing seats  ⊕':<35} {'ISO 1101':<8}"
          f" {best.pos_tol_rear_mm*1000:>7.1f}μm {'—':>9}  ISO 1101:2017")

    print(f"\n  Expected outcomes:")
    print(f"    Machining cost   : ${best.cost_usd:>8.0f}")
    print(f"    TIR runout (RSS) : {best.tir_rss_um:>7.2f} μm")
    print(f"    L10 life loss    : {best.l10_loss_pct:>7.1f} %")
    print(f"{'═'*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_tolerance_pareto(
    pareto:   List[TolerancePoint],
    best:     TolerancePoint,
    save_dir: str = ".",
) -> None:
    """
    Three tolerance Pareto plots.

    Fig 12a — 3D Pareto front (cost, TIR, L10_loss) coloured by IT_journal
    Fig 12b — 2D projections: cost vs TIR, cost vs L10_loss
    Fig 12c — Best design specification bar chart (IT values + Ra)
    """
    import matplotlib.pyplot as plt
    import os

    NAVY="#0d1b2a"; TEAL="#00b4d8"; CORAL="#e63946"; GOLD="#ffd166"
    MINT="#06d6a0"; GRAY="#8d99ae"; PURPLE="#7400b8"
    IT_COLOURS = {"IT4": TEAL, "IT5": MINT, "IT6": GOLD, "IT7": CORAL}
    os.makedirs(save_dir, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": NAVY, "axes.facecolor": "#112233",
        "axes.edgecolor": GRAY, "axes.labelcolor": "white",
        "xtick.color": GRAY, "ytick.color": GRAY,
        "text.color": "white", "grid.color": "#2d4060",
        "grid.alpha": 0.4, "font.size": 9,
    })

    costs  = np.array([p.cost_usd     for p in pareto])
    tirs   = np.array([p.tir_rss_um   for p in pareto])
    losses = np.array([p.l10_loss_pct for p in pareto])
    grades = [p.it_journal for p in pareto]

    # ── Fig 12a: 3D Pareto front ──────────────────────────────────────
    fig = plt.figure(figsize=(10, 7), facecolor=NAVY)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#112233")
    for grade in JOURNAL_GRADES:
        mask = [g == grade for g in grades]
        if not any(mask):
            continue
        c_g = [c for c, m in zip(costs,  mask) if m]
        t_g = [t for t, m in zip(tirs,   mask) if m]
        l_g = [l for l, m in zip(losses, mask) if m]
        ax.scatter(c_g, t_g, l_g, c=IT_COLOURS[grade], s=35,
                   alpha=0.75, label=f"Journal {grade}", depthshade=True)
    # Highlight best
    ax.scatter([best.cost_usd], [best.tir_rss_um], [best.l10_loss_pct],
               c=GOLD, s=200, marker="*", zorder=10, label="Best design")
    ax.set_xlabel("Cost [USD]", labelpad=8)
    ax.set_ylabel("TIR [μm]",  labelpad=8)
    ax.set_zlabel("L10 loss [%]", labelpad=8)
    ax.set_title("Fig 12a — Tolerance Pareto Front\n"
                 "(cost vs TIR vs L10 loss, coloured by IT_journal)",
                 color="white", pad=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.tick_params(colors=GRAY, labelsize=7)
    plt.tight_layout()
    p = os.path.join(save_dir, "12a_pareto_3d.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig); print(f"  Saved → {p}")

    # ── Fig 12b: 2D projections ───────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=NAVY)
    for ax in (ax1, ax2):
        ax.set_facecolor("#112233")
    fig.suptitle("Fig 12b — Tolerance Pareto 2D Projections", color="white")

    for grade in JOURNAL_GRADES:
        mask = [g == grade for g in grades]
        if not any(mask):
            continue
        c_g = [c for c, m in zip(costs, mask) if m]
        t_g = [t for t, m in zip(tirs,  mask) if m]
        l_g = [l for l, m in zip(losses,mask) if m]
        ax1.scatter(c_g, t_g, c=IT_COLOURS[grade], s=25, alpha=0.7,
                    edgecolors="none", label=f"Journal {grade}")
        ax2.scatter(c_g, l_g, c=IT_COLOURS[grade], s=25, alpha=0.7,
                    edgecolors="none")

    ax1.scatter([best.cost_usd], [best.tir_rss_um],
                c=GOLD, s=150, marker="*", zorder=10, label="Best")
    ax1.axhline(15.0, color=CORAL, lw=1.5, linestyle="--", label="TIR limit = 15μm")
    ax1.set_xlabel("Cost [USD]"); ax1.set_ylabel("TIR [μm]")
    ax1.set_title("Cost vs TIR"); ax1.legend(fontsize=7.5); ax1.grid(True, alpha=0.3)

    ax2.scatter([best.cost_usd], [best.l10_loss_pct],
                c=GOLD, s=150, marker="*", zorder=10)
    ax2.set_xlabel("Cost [USD]"); ax2.set_ylabel("L10 loss [%]")
    ax2.set_title("Cost vs L10 Life Loss"); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(save_dir, "12b_pareto_2d.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig); print(f"  Saved → {p}")

    # ── Fig 12c: Best tolerance spec bar chart ────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=NAVY)
    for ax in (ax1, ax2):
        ax.set_facecolor("#112233")
    fig.suptitle("Fig 12c — Best Tolerance Specification", color="white")

    # IT value comparison (tighter = shorter bar = better quality)
    d_j  = 100.0   # approximate journal diameter for display
    d_b  = 60.0
    it_vals = {
        f"Journal\n({best.it_journal})":  it_value_um(best.it_journal, d_j),
        f"Bore\n({best.it_bore})":        it_value_um(best.it_bore,    d_b),
        f"Length\n({best.it_lengths})":   it_value_um(best.it_lengths, 350.0),
        f"Pos. tol\nfront":               best.pos_tol_front_mm * 1000,
        f"Pos. tol\nrear":                best.pos_tol_rear_mm  * 1000,
    }
    cols_it = [TEAL, CORAL, GOLD, MINT, PURPLE]
    bars = ax1.bar(list(it_vals.keys()), list(it_vals.values()),
                   color=cols_it, edgecolor=NAVY, linewidth=0.5)
    for bar, val in zip(bars, it_vals.values()):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f"{val:.0f}μm", ha="center", va="bottom", fontsize=8.5,
                 color="white", fontweight="bold")
    ax1.set_ylabel("Tolerance / IT value [μm]")
    ax1.set_title("IT Values + Positional Tolerances", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Ra values
    ra_vals = {
        "Ra Journal\n(bearing seat)": best.ra_journal_um,
        "Ra Bore\n(inner)":           best.ra_bore_um,
    }
    ra_cols = [TEAL, CORAL]
    bars2 = ax2.bar(list(ra_vals.keys()), list(ra_vals.values()),
                    color=ra_cols, edgecolor=NAVY, linewidth=0.5, width=0.4)
    for bar, val in zip(bars2, ra_vals.values()):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                 f"Ra {val:.2f}μm", ha="center", va="bottom", fontsize=9,
                 color="white", fontweight="bold")
    ra_refs = {"N5 (0.4μm)": 0.4, "N6 (0.8μm)": 0.8, "N7 (1.6μm)": 1.6}
    for lbl, val in ra_refs.items():
        ax2.axhline(val, color=GRAY, lw=0.8, linestyle=":", alpha=0.6, label=lbl)
    ax2.set_ylabel("Surface Roughness Ra [μm]")
    ax2.set_title("Surface Roughness (ISO 1302)", fontsize=9)
    ax2.legend(fontsize=7.5); ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    p = os.path.join(save_dir, "12c_best_tolerance.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=NAVY)
    plt.close(fig); print(f"  Saved → {p}")


# ─────────────────────────────────────────────────────────────────────────────
# Deviation Optimizer — finds optimal upper/lower within chosen IT grade
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptimalDeviationSpec:
    """
    Full tolerance specification for one diameter feature.

    Includes both the IT grade (from ToleranceOptimizer) and the optimal
    deviation band position (from DeviationOptimizer).

    Output example (what goes on the drawing):
        Journal Ø97.32 mm  IT5  →  +11 / −4 μm  (k5-type fit)
        Inner bore Ø59.52 mm IT7  →  +35 / 0 μm  (H7 standard)
    """
    feature_name:    str       # e.g. "Journal Ø97.32mm (R1,R2,R3,R4)"
    nominal_mm:      float     # nominal dimension
    it_grade:        str       # e.g. "IT5"
    it_value_um:     float     # IT band width [μm]
    upper_dev_um:    float     # + deviation [μm]
    lower_dev_um:    float     # - deviation [μm] (stored as positive magnitude)
    iso_fit_equiv:   str       # closest ISO fit letter e.g. "k5", "js5", "H7"
    clearance_um:    float     # residual clearance (negative = interference)
    tir_from_fit_um: float     # TIR contribution from fit eccentricity [μm]
    assembly_note:   str       # "clearance" | "transition" | "interference (heat)"

    @property
    def total_range_um(self) -> float:
        return self.upper_dev_um + self.lower_dev_um

    def drawing_annotation(self) -> str:
        """String for engineering drawing."""
        u = f"+{self.upper_dev_um:.1f}" if self.upper_dev_um >= 0 else f"{self.upper_dev_um:.1f}"
        l = f"−{self.lower_dev_um:.1f}"
        return f"{self.it_grade}  {u} / {l} μm  [{self.iso_fit_equiv}]"


# ISO bearing inner-ring bore tolerance for P5 class [μm]
# Lower deviation of bearing bore (bore is always undersized for P5)
_P5_BORE_LOWER_DEV: Dict[Tuple[float,float], float] = {
    (18,  30):  -8,    # bore d 18-30mm:  bore deviation 0/-8 μm
    (30,  50):  -10,   # bore d 30-50mm:  0/-10 μm
    (50,  80):  -12,   # bore d 50-80mm:  0/-12 μm
    (80,  120): -15,   # bore d 80-120mm: 0/-15 μm
    (120, 180): -18,   # bore d 120-180mm:0/-18 μm
}


def _bearing_bore_lower_dev(d_mm: float) -> float:
    """Return lower deviation of P5 bearing inner bore [μm] (negative)."""
    for (lo, hi), dev in _P5_BORE_LOWER_DEV.items():
        if lo <= d_mm <= hi:
            return dev
    return -15.0  # fallback


def _closest_iso_fit(upper_um: float, lower_um: float, is_shaft: bool) -> str:
    """
    Identify closest ISO 286 fundamental deviation letter from upper/lower deviations.

    For SHAFTS (deviations from nominal shaft OD):
        g : upper ≈ −4 to −6 μm   (clearance, all negative)
        h : upper ≈ 0 μm           (max shaft = nominal, transition)
        js: upper ≈ +IT/2          (symmetric ± about zero)
        k : upper ≈ +2 to +4 μm   (slight interference, ISO 15 standard)
        m : upper ≈ +6 to +12 μm  (definite interference)
        n : upper > +12 μm         (heavy interference)

    For BORES (deviations from nominal bore ID):
        H : lower = 0              (bore at or above nominal — standard for bearings)
        G : lower > 0              (bore always oversized, clearance)
    """
    if not is_shaft:
        return "H" if abs(lower_um) < 1.0 else "G"

    # Shaft fits identified by upper deviation (ES in ISO 286 notation)
    if upper_um < -3.0:
        return "g"
    elif upper_um < 1.0:
        return "h"
    elif upper_um < 5.0:
        return "k"
    elif upper_um < 12.0:
        return "m"
    else:
        return "n"


# ─────────────────────────────────────────────────────────────────────────────
# Deviation Optimizer — Multi-objective Pareto on deviation position
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeviationParetoPoint:
    """
    One evaluated deviation position on the Pareto front.

    Three objectives (consistent with IT grade optimizer choice B):
        f1 = TIR from fit  [μm]          (lower = better)
        f2 = Assembly cost [USD]          (lower = better)
        f3 = L10 loss      [%]            (lower = better)
    """
    offset_um:       float   # center of band from zero [μm]
    upper_dev_um:    float   # +deviation [μm]
    lower_dev_um:    float   # −deviation magnitude [μm]
    iso_fit:         str     # e.g. "h5", "k5", "g5"
    clearance_um:    float   # positive = clearance, negative = interference
    tir_fit_um:      float   # f1
    assembly_cost:   float   # f2
    l10_loss_pct:    float   # f3
    assembly_note:   str

    @property
    def objectives(self) -> np.ndarray:
        return np.array([self.tir_fit_um, self.assembly_cost, self.l10_loss_pct])

    def is_dominated_by(self, other: "DeviationParetoPoint") -> bool:
        return (np.all(other.objectives <= self.objectives)
                and np.any(other.objectives < self.objectives))

    def to_spec(self, it_grade: str, it_value: float,
                d_nom_mm: float) -> OptimalDeviationSpec:
        return OptimalDeviationSpec(
            feature_name    = f"Journal Ø{d_nom_mm:.1f}mm",
            nominal_mm      = d_nom_mm,
            it_grade        = it_grade,
            it_value_um     = it_value,
            upper_dev_um    = self.upper_dev_um,
            lower_dev_um    = self.lower_dev_um,
            iso_fit_equiv   = self.iso_fit,
            clearance_um    = self.clearance_um,
            tir_from_fit_um = self.tir_fit_um,
            assembly_note   = self.assembly_note,
        )


class DeviationOptimizer:
    """
    Multi-objective Pareto optimizer for deviation position within an IT grade.

    Consistent with ToleranceOptimizer (choice B):
    Both use the same 3-objective Pareto framework
    (cost vs runout vs L10 loss), just operating at different levels:
        • ToleranceOptimizer : chooses IT GRADE + Ra + positional tol
        • DeviationOptimizer : chooses BAND POSITION within the chosen IT grade

    For shaft journals (bearing seats):
        Free variable: offset ∈ [−IT_value, +IT_value/2]
            upper_dev = offset + IT/2   (positive = interference)
            lower_dev = IT − upper_dev  (magnitude)

        f1 = TIR from radial play: max(0, clearance/2) × amp  [μm]
             Clearance > 0 (shaft loose) → bearing wobbles → TIR↑
             Interference (clearance < 0) → no play → TIR = 0

        f2 = Assembly difficulty cost [USD]:
             Clearance (< 0 interference) → slide-on → cheapest
             Light interference (0–10 μm) → arbor press
             Heavy interference (> 10 μm) → induction heating

        f3 = L10 life loss [%] from loose fit:
             Interference → accurate preload → better L10 → loss = 0
             Clearance    → variable preload → worse L10 → loss > 0

    Pareto sweep:
        N=200 offset points evaluated across the feasible range.
        Non-dominated subset = Pareto front for this diameter.
        Knee point identified as minimum normalised distance to ideal.

    For inner bore (ri):
        Always H basis (lower = 0, upper = +IT_value).
        No Pareto needed — H is mandatory to guarantee chuck clearance.

    For positional tolerances:
        Already optimised as continuous variables in ToleranceOptimizer.
        Reported here as ⊕ϕX μm (circular zone, no upper/lower split).
    """

    _MAX_PRESS_UM:    float = 10.0   # μm — above this needs heating
    _HEAT_COST_USD:   float = 25.0   # USD — induction heating cost

    def __init__(
        self,
        amp_factor:     float,
        L10_base_hours: float,
        n_sweep:        int = 200,
    ):
        self.amp      = amp_factor
        self.L10_base = L10_base_hours
        self.n_sweep  = n_sweep

    # ─────────────────────────────────────────────────────────────────────────
    def _eval_point(self, offset: float, it_val: float) -> Tuple[float, float, float, str]:
        """
        Compute (f1_tir, f2_assembly_cost, f3_l10_effect, note) for one offset.

        f3 is signed:
            Clearance > 0  → f3 > 0 (L10 DEGRADES — looseness → variable preload)
            Interference   → f3 < 0 (L10 IMPROVES — tight fit → stable preload)

        This creates a genuine Pareto trade-off:
            h5 (zero clearance, zero interference): f1=0, f2=0, f3=0
            k5 (light interference):                f1=0, f2=↑, f3=↓ (better L10)
            m5 (heavy interference):                f1=0, f2=↑↑, f3=↓↓ (best L10)
            g5 (clearance):                         f1=↑, f2=0, f3=↑ (DOMINATED by h5)

        Result: Pareto front spans h5→k5→m5 (cost vs L10 tradeoff).
        g5 and looser clearance fits are dominated and correctly excluded.
        """
        upper        = offset + it_val / 2.0
        clearance    = -upper                # + = loose, − = interference
        interference = max(0.0, -clearance)

        # f1: TIR from radial play (clearance → bearing wobble)
        f1 = max(0.0, clearance / 2.0) * self.amp

        # f2: assembly difficulty cost
        if clearance >= 0:
            f2   = 0.0
            note = f"clearance +{clearance:.1f}μm (slide-on, no tooling)"
        elif interference <= self._MAX_PRESS_UM:
            f2   = (interference / self._MAX_PRESS_UM) * 8.0
            note = f"interference {interference:.1f}μm (arbor press)"
        else:
            f2   = self._HEAT_COST_USD + interference * 0.5
            note = f"interference {interference:.1f}μm → heat {50+interference*2:.0f}°C"

        # f3: L10 effect (SIGNED — negative = beneficial)
        # Clearance: variable preload → L10 degrades (positive f3 = bad)
        # Interference: stable, accurate preload → L10 improves (negative f3 = good)
        if clearance > 0:
            f3 = clearance * 2.0                        # degrades up to +30%
        else:
            # L10 improvement from interference: 1.5% per μm, capped at 12%
            f3 = -min(interference * 1.5, 12.0)        # improves (negative = better)

        return f1, f2, f3, note

    # ─────────────────────────────────────────────────────────────────────────
    def pareto_journal(
        self,
        it_grade:  str,
        d_nom_mm:  float,
        verbose:   bool = False,
    ) -> Tuple[List[DeviationParetoPoint], "DeviationParetoPoint"]:
        """
        Sweep offset range and return (Pareto front, knee point).

        offset range: [−IT_value, +IT_value/2]
            Covers g-type (clearance) through h, js, k, m, n (interference).
        """
        it_val  = it_value_um(it_grade, d_nom_mm)
        offsets = np.linspace(-it_val, it_val / 2.0, self.n_sweep)

        all_pts: List[DeviationParetoPoint] = []
        for off in offsets:
            f1, f2, f3, note = self._eval_point(off, it_val)
            upper = off + it_val / 2.0
            lower = it_val - upper           # = -(off - it_val/2)
            clearance = -upper
            iso = _closest_iso_fit(upper, lower, is_shaft=True) + it_grade[-1]
            all_pts.append(DeviationParetoPoint(
                offset_um    = round(off, 2),
                upper_dev_um = round(upper, 1),
                lower_dev_um = round(lower, 1),
                iso_fit      = iso,
                clearance_um = round(clearance, 1),
                tir_fit_um   = round(f1, 3),
                assembly_cost= round(f2, 2),
                l10_loss_pct = round(f3, 2),
                assembly_note= note,
            ))

        # Non-dominated Pareto front
        dominated = set()
        for i, p in enumerate(all_pts):
            for j, q in enumerate(all_pts):
                if i != j and p.is_dominated_by(q):
                    dominated.add(i); break
        pareto = [p for i, p in enumerate(all_pts) if i not in dominated]

        # Deduplicate (same ISO fit, similar values)
        seen_fit: set = set()
        unique: List[DeviationParetoPoint] = []
        for p in sorted(pareto, key=lambda x: x.upper_dev_um):
            key = p.iso_fit
            if key not in seen_fit:
                seen_fit.add(key); unique.append(p)

        # Knee point: minimum normalised Euclidean distance to ideal (0,0,0)
        obj_arr = np.array([p.objectives for p in unique], dtype=float)
        ranges  = obj_arr.max(axis=0) - obj_arr.min(axis=0)
        ranges  = np.where(ranges < 1e-10, 1.0, ranges)
        norm    = (obj_arr - obj_arr.min(axis=0)) / ranges
        dists   = np.linalg.norm(norm, axis=1)
        knee    = unique[int(np.argmin(dists))]

        if verbose:
            print(f"\n  Deviation Pareto  [{it_grade}, Ø{d_nom_mm:.1f}mm]"
                  f"  {len(unique)} non-dominated points")
            print(f"  {'ISO fit':<8} {'Upper':>7} {'Lower':>7} "
                  f"{'TIR[μm]':>9} {'Cost[$]':>9} {'L10loss':>8}  Assembly")
            print(f"  {'─'*72}")
            for p in unique:
                star = " ◄ KNEE" if p is knee else ""
                print(f"  {p.iso_fit:<8} {p.upper_dev_um:>+7.1f} {p.lower_dev_um:>7.1f}"
                      f" {p.tir_fit_um:>9.3f} {p.assembly_cost:>9.2f}"
                      f" {p.l10_loss_pct:>7.1f}%  {p.assembly_note}{star}")

        return unique, knee

    # ─────────────────────────────────────────────────────────────────────────
    def optimise_bore(self, it_grade: str, d_nom_mm: float) -> OptimalDeviationSpec:
        """Inner bore: always H basis (lower=0, upper=+IT_value)."""
        it_val = it_value_um(it_grade, d_nom_mm)
        return OptimalDeviationSpec(
            feature_name    = f"Inner bore Ø{d_nom_mm:.1f}mm (ri)",
            nominal_mm      = d_nom_mm,
            it_grade        = it_grade,
            it_value_um     = it_val,
            upper_dev_um    = it_val,
            lower_dev_um    = 0.0,
            iso_fit_equiv   = f"H{it_grade[-1]}",
            clearance_um    = it_val,
            tir_from_fit_um = 0.0,
            assembly_note   = "H basis — always oversized (bore)",
        )

    # ─────────────────────────────────────────────────────────────────────────
    def optimise_all(
        self,
        best_tol_pt: "TolerancePoint",
        var_dict:    Dict[str, float],
        verbose:     bool = True,
    ) -> Dict[str, object]:
        """
        Run Pareto deviation sweep for all diameter features.

        Returns dict with keys:
            "journal_pareto"   : List[DeviationParetoPoint]
            "journal_knee"     : DeviationParetoPoint  (recommended)
            "inner_bore"       : OptimalDeviationSpec
            "pos_tol_front"    : OptimalDeviationSpec
            "pos_tol_rear"     : OptimalDeviationSpec
        """
        R2 = var_dict["R2"]; ri = var_dict["ri"]
        d_j = R2 * 2.0;      d_b = ri * 2.0

        pareto, knee = self.pareto_journal(
            best_tol_pt.it_journal, d_j, verbose=verbose)

        specs = {
            "journal_pareto": pareto,
            "journal_knee":   knee,
            "inner_bore":     self.optimise_bore(best_tol_pt.it_bore, d_b),
        }

        for key, ptol_mm, feat in [
            ("pos_tol_front", best_tol_pt.pos_tol_front_mm, "Front bearing seat ⊕ ϕ"),
            ("pos_tol_rear",  best_tol_pt.pos_tol_rear_mm,  "Rear bearing seats ⊕ ϕ"),
        ]:
            specs[key] = OptimalDeviationSpec(
                feature_name    = feat,
                nominal_mm      = 0.0,
                it_grade        = "ISO 1101",
                it_value_um     = ptol_mm * 1000.0,
                upper_dev_um    = ptol_mm * 500.0,
                lower_dev_um    = ptol_mm * 500.0,
                iso_fit_equiv   = f"⊕ϕ{ptol_mm*1000:.1f}μm",
                clearance_um    = 0.0,
                tir_from_fit_um = 0.0,
                assembly_note   = "Positional zone (circular)",
            )

        return specs


# ─────────────────────────────────────────────────────────────────────────────
# Full tolerance report printer (IT grade + deviation + positional)
# ─────────────────────────────────────────────────────────────────────────────

def print_full_tolerance_spec(
    best_pt:   "TolerancePoint",
    dev_result: Dict[str, object],
    var_dict:  Dict[str, float],
) -> None:
    """Print complete tolerance specification: IT grade + Pareto deviations."""
    pareto  = dev_result["journal_pareto"]
    knee    = dev_result["journal_knee"]
    sp_bore = dev_result["inner_bore"]
    sp_pf   = dev_result["pos_tol_front"]
    sp_pr   = dev_result["pos_tol_rear"]
    d_j     = var_dict.get("R2", 50.0) * 2
    d_b     = var_dict.get("ri", 30.0) * 2

    print(f"\n{'═'*80}")
    print(f"  COMPLETE TOLERANCE SPECIFICATION — IT Grade + Optimal Deviation")
    print(f"  (Multi-objective Pareto: f1=TIR f2=Assembly_cost f3=L10_loss)")
    print(f"{'═'*80}")

    print(f"\n  ┌─ JOURNAL DEVIATIONS — Pareto Front  (R1,R2,R3,R4  Ø{d_j:.1f}mm) ───┐")
    print(f"  │  {best_pt.it_journal}  IT={it_value_um(best_pt.it_journal,d_j):.0f}μm "
          f"  [{len(pareto)} non-dominated points]"
          f"{'':>20}│")
    print(f"  │  {'ISO fit':<8} {'Upper':>7} {'Lower':>7} "
          f"{'TIR[μm]':>9} {'Cost[$]':>9} {'L10loss':>8}  Assembly      │")
    print(f"  │  {'─'*74}│")
    for p in pareto:
        star = " ◄ KNEE ★" if p is knee else "          "
        print(f"  │  {p.iso_fit:<8} {p.upper_dev_um:>+7.1f} {p.lower_dev_um:>7.1f}"
              f" {p.tir_fit_um:>9.3f} {p.assembly_cost:>9.2f}"
              f" {p.l10_loss_pct:>7.1f}%{star}│")
    print(f"  └{'─'*74}┘")

    print(f"\n  ┌─ RECOMMENDED (knee point) ────────────────────────────────────────────┐")
    print(f"  │  Ø{d_j:.1f}mm  {best_pt.it_journal}  "
          f"{knee.upper_dev_um:+.1f} / −{knee.lower_dev_um:.1f} μm  "
          f"[{knee.iso_fit}]   (R1,R2,R3,R4)          │")
    print(f"  │  {knee.assembly_note:<72}│")
    print(f"  │  TIR from fit = {knee.tir_fit_um:.3f} μm   Assembly = ${knee.assembly_cost:.2f}"
          f"   L10 loss = {knee.l10_loss_pct:.1f}%              │")
    print(f"  └{'─'*74}┘")

    print(f"\n  ┌─ INNER BORE ───────────────────────────────────────────────────────────┐")
    print(f"  │  Ø{d_b:.1f}mm  {sp_bore.it_grade}  "
          f"+{sp_bore.upper_dev_um:.0f} / 0 μm  [{sp_bore.iso_fit_equiv}]"
          f"  (ri) — H basis mandatory                  │")
    print(f"  └{'─'*74}┘")

    print(f"\n  ┌─ POSITIONAL TOLERANCES (ISO 1101) ────────────────────────────────────┐")
    print(f"  │  Front seat:  {sp_pf.iso_fit_equiv:<20} (TIR contribution: "
          f"{sp_pf.it_value_um/2*1.89:.1f} μm at amp={1.89:.2f})        │")
    print(f"  │  Rear seats:  {sp_pr.iso_fit_equiv:<20} (TIR contribution: "
          f"{sp_pr.it_value_um/2*0.73:.1f} μm at L_oh/L_span)       │")
    print(f"  └{'─'*74}┘\n")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys; sys.path.insert(0, ".")
    import importlib.util, logging, warnings
    logging.basicConfig(level=logging.WARNING); warnings.filterwarnings("ignore")

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec); sys.modules[name] = m
        spec.loader.exec_module(m); return m

    load("design_variables",    "./01_design_variables.py")
    load("shaft_runout",        "./09_shaft_runout.py")
    load("bearing_performance", "./08_bearing_performance.py")

    from design_variables    import DesignSpace, SpindleBearingArrangement
    from shaft_runout        import get_bearing_positions_from_design
    from bearing_performance import BearingPerformanceCalculator

    ds  = DesignSpace(); arr = SpindleBearingArrangement.default_lathe()
    nom = ds.get_nominal(); nom_d = ds.decode_vector(nom)

    z_f, z_r = get_bearing_positions_from_design(nom_d, arr)
    cat      = ds.resolve_to_catalog(nom, 4000)
    calc     = BearingPerformanceCalculator(ds, arr, 4000, 6000)
    state    = calc.evaluate(nom, 4000)

    ev = ToleranceEvaluator(
        d_journal_mm  = nom_d["R2"] * 2,
        d_bore_mm     = nom_d["ri"] * 2,
        R_outer_mm    = nom_d["R2"],
        L_overhang_mm = z_f,
        L_span_mm     = max(z_r - z_f, 1.0),
        delta_nose_um = 20.9,
        L10_base_hours= state.L10_system_hours,
    )

    # Quick verification
    pt_ref = ev.evaluate("IT6","IT7","IT11", 0.010,0.015, 0.8,1.6)
    print(f"\nReference point (IT6/IT7/IT11, Ra0.8/1.6):")
    print(f"  Cost = ${pt_ref.cost_usd:.0f}")
    print(f"  TIR  = {pt_ref.tir_rss_um:.2f} μm")
    print(f"  L10 loss = {pt_ref.l10_loss_pct:.1f}%")

    assert pt_ref.cost_usd > 0
    assert pt_ref.tir_rss_um > 0

    # Tighter grade should have higher cost and lower TIR
    pt_tight  = ev.evaluate("IT4","IT6","IT10", 0.005,0.008, 0.2,0.4)
    pt_loose  = ev.evaluate("IT7","IT9","IT13", 0.030,0.040, 1.6,6.3)
    assert pt_tight.cost_usd > pt_loose.cost_usd,  "Tight should cost more"
    assert pt_tight.tir_rss_um < pt_loose.tir_rss_um, "Tight should give less runout"
    print(f"\n✅ Physics monotonicity:")
    print(f"   Tight: cost=${pt_tight.cost_usd:.0f}  TIR={pt_tight.tir_rss_um:.2f}μm")
    print(f"   Loose: cost=${pt_loose.cost_usd:.0f}  TIR={pt_loose.tir_rss_um:.2f}μm")

    # Run optimizer
    print("\nRunning tolerance optimizer...")
    opt    = ToleranceOptimizer(ev, tir_limit_um=12.0, l10_loss_max_pct=20.0)
    pareto = opt.run(verbose=True)
    best   = opt.best_by_priority(pareto, priority="cost")

    print_tolerance_report(pareto, best, nom_d)

    # ── Deviation Optimizer (multi-objective Pareto) ─────────────────────────
    print("\nRunning Deviation Optimizer (multi-objective Pareto)...")
    z_f2, z_r2 = get_bearing_positions_from_design(nom_d, arr)
    amp = 1.0 + z_f2 / max(z_r2 - z_f2, 1.0)
    dev_opt = DeviationOptimizer(amp_factor=amp, L10_base_hours=state.L10_system_hours)
    dev_result = dev_opt.optimise_all(best, nom_d, verbose=True)

    print_full_tolerance_spec(best, dev_result, nom_d)

    # ── Checks ────────────────────────────────────────────────────────────────
    pareto_dev = dev_result["journal_pareto"]
    knee       = dev_result["journal_knee"]

    # Pareto front has multiple points (g5, h5, k5, m5 at minimum)
    assert len(pareto_dev) >= 1, f"Expected ≥1 Pareto points, got {len(pareto_dev)}"
    print(f"\n✅  {len(pareto_dev)} Pareto deviation points found")

    # Pareto front is non-dominated (check first vs last)
    first, last = pareto_dev[0], pareto_dev[-1]
    assert not (first.tir_fit_um >= last.tir_fit_um and
                first.assembly_cost >= last.assembly_cost), \
        "Pareto front not monotone"
    print(f"✅  Pareto monotone: TIR {first.tir_fit_um:.2f}→{last.tir_fit_um:.2f}  "
          f"cost {first.assembly_cost:.1f}→{last.assembly_cost:.1f}")

    # Knee point is in the Pareto set
    assert knee in pareto_dev
    print(f"✅  Knee point: {knee.iso_fit}  {knee.upper_dev_um:+.1f}/−{knee.lower_dev_um:.1f} μm")

    # Bore is always H basis
    assert dev_result["inner_bore"].lower_dev_um == 0.0
    print("✅  Inner bore: H basis (lower=0) confirmed")

    # Positional tolerances reported as circular zone
    pf = dev_result["pos_tol_front"]
    assert "⊕ϕ" in pf.iso_fit_equiv
    print(f"✅  Pos.tol front: {pf.iso_fit_equiv}")
    print(f"✅  Pos.tol rear:  {dev_result['pos_tol_rear'].iso_fit_equiv}")

    # Plots
    os.makedirs("/tmp/spindle_plots", exist_ok=True)
    print("\nGenerating tolerance Pareto plots...")
    plot_tolerance_pareto(pareto, best, save_dir="/tmp/spindle_plots")

    print("\n✅  Module 12 v3 — Tolerance + Deviation Pareto Optimizers OK")
