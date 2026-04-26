#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  Selective Assembly Module — Precision Spindle Mating Analysis
================================================================================

  Purpose:
      Model the selective assembly process for precision spindle components.
      Selective assembly is the manufacturing strategy of:
        1. Machining parts to wider individual tolerances (cheaper)
        2. Measuring each part after machining
        3. Sorting parts into bins by measured deviation
        4. Pairing parts from compatible bins to achieve a tight assembly fit

  Physics / Engineering Context:
      For a CNC spindle, selective assembly is critical for three interfaces:

      Interface 1 — Shaft journal ↔ Bearing inner race (radial fit):
        • Controls bearing preload and radial runout
        • Target: transition fit (0–8 μm clearance) or slight interference
        • Without SA: ±50 μm shaft tolerance → unacceptable runout scatter
        • With SA (5 bins): effective clearance scatter ≤ ±5 μm

      Interface 2 — Inner spacer ↔ Outer spacer (axial preload):
        • In DB (back-to-back) arrangement, the difference in spacer lengths
          controls axial preload on both bearings.
        • Δ = L_inner_spacer − L_outer_spacer = target_preload / K_axial
        • With SA: Δ held to ±1 μm → preload variation ≤ ±K_axial μm

      Interface 3 — Housing bore ↔ Bearing outer ring (housing fit):
        • Controls outer-ring micro-creep under load
        • Target: light interference (5–15 μm) for rotating outer ring
        • SA applied similarly to Interface 1

  Cost Model:
      C_sa = n_stages × (C_measurement + C_sorting × n_bins + C_rework × p_unmatched)

      where p_unmatched = P(shaft bin has no matching bearing bin)
                       = f(part count ratio, distribution shape, n_bins)

  Integration with RDO:
      • Wider shaft tolerances → cheaper machining, but more SA stages
      • Tighter shaft tolerances → expensive grinding, but trivial SA
      • GA explores this trade-off: SA cost is one GA objective
================================================================================
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
import logging

log = logging.getLogger("SelectiveAssembly")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BinConfig:
    """
    Selective-assembly bin layout for one mating interface.

    Args:
        n_bins:         Number of sorted bins (typically 3-10)
        nominal_gap:    Target clearance (+ = clearance, − = interference) [mm]
        allowed_gap_tol:  Maximum allowable gap scatter within a matched pair [mm]
        interface_name: Human-readable label
    """
    n_bins:           int
    nominal_gap:      float          # mm (positive = clearance)
    allowed_gap_tol:  float          # mm (±)
    interface_name:   str = "unnamed"


@dataclass
class MatingInterface:
    """
    Pair of mating features (shaft + hole / spacer pair).

    shaft_dim   : nominal shaft diameter or length [mm]
    shaft_tol   : bilateral machining tolerance ± [mm]
    housing_dim : nominal housing/bore diameter [mm]
    housing_tol : bilateral machining tolerance ± [mm]
    """
    shaft_dim:    float
    shaft_tol:    float
    housing_dim:  float
    housing_tol:  float
    bin_config:   BinConfig


@dataclass
class SelectiveAssemblyResult:
    """Results of a selective-assembly analysis for one design."""
    interface_name:    str
    n_bins:            int
    mean_gap_um:       float         # μm
    std_gap_um:        float         # μm, after SA
    std_gap_no_sa_um:  float         # μm, without SA (reference)
    improvement_ratio: float         # std_no_sa / std_with_sa
    match_yield:       float         # fraction 0–1 of parts that find a match
    sa_cost_usd:       float
    machining_cost_usd: float
    total_cost_usd:    float


# ─────────────────────────────────────────────────────────────────────────────
# Cost parameters (tune to your shop-rate)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ShopCostParams:
    """Shop-floor cost parameters for spindle manufacturing."""
    # Machining costs (per mm of tolerance width, per mm² of surface area)
    cost_per_tol_mm_turning:  float = 5.0     # $/mm tolerance (turning)
    cost_per_tol_mm_grinding: float = 80.0    # $/mm tolerance (grinding finish)
    grinding_threshold_mm:    float = 0.02    # below this tolerance → grinding

    # Material costs
    mat_price_4140_per_kg:    float = 3.5     # USD/kg
    mat_price_4340_per_kg:    float = 5.5
    mat_price_EN24_per_kg:    float = 4.5

    # Bearing costs (pair)
    bearing_base_cost_usd:    float = 220.0   # USD per pair (basic ACBB)
    bearing_stiffness_ref:    float = 5.0e5   # N/mm (reference stiffness)
    bearing_stiffness_exp:    float = 0.6     # cost ∝ K^exp

    # Selective assembly costs (per interface, per batch)
    cmm_measurement_usd:      float = 8.0    # USD per part for CMM measurement
    sorting_cost_per_bin:     float = 2.0    # USD per bin × per part
    rework_cost_per_part:     float = 45.0   # USD per unmatched part (rework)
    storage_cost_per_day:     float = 0.5    # USD per bin-set held in storage


class SpindleCostModel:
    """
    Comprehensive cost model for a spindle design.

    Computes:
      C_material   — raw billet cost
      C_machining  — turning + grinding (tolerance-dependent)
      C_bearings   — angular contact bearing pair cost
      C_sa         — selective assembly overhead

    All costs in USD per spindle unit.
    """

    def __init__(self, params: Optional[ShopCostParams] = None):
        self.p = params or ShopCostParams()

    # ─────────────────────────────────────────────────────────────────────────
    # Material cost
    # ─────────────────────────────────────────────────────────────────────────
    def material_cost(self, var_dict: Dict[str, float]) -> float:
        """
        Cost of raw billet (excess stock ×1.3 for turning stock).

        Uses mm-ton unit system; converts to kg for pricing.
        """
        L_total  = var_dict["L1"] + var_dict["L2"] + var_dict["L3"] + var_dict["L4"]
        R_max    = max(var_dict["R1"], var_dict["R2"], var_dict["R3"], var_dict["R4"])
        ri       = var_dict["ri"]

        # Billet is a cylinder of outer radius R_max (worst case)
        V_billet_mm3 = np.pi * R_max**2 * L_total * 1.3   # 30 % machining stock
        mass_kg      = V_billet_mm3 * var_dict["rho"] * 1e9   # ton→kg: ×1e3; mm³→m³: ×1e-9 → ×1e-6 kg? 
        # ton/mm³ → kg/m³: multiply by 1e9 (1 ton = 1000 kg, 1 mm³ = 1e-9 m³, so ton/mm³ × 1e9 = kg/m³)
        # Then mass_kg = (V_billet [m³]) × density [kg/m³] = V_billet_mm3 × 1e-9 × rho × 1e9
        mass_kg      = V_billet_mm3 * var_dict["rho"]   # ton (mm-ton system) × 1000 = kg
        mass_kg      = mass_kg * 1000.0                 # ton → kg

        # Pick material price based on E (proxy for alloy)
        E = var_dict.get("E", 2.1e5)
        if E >= 2.08e5:
            price = self.p.mat_price_4340_per_kg
        elif E >= 2.05e5:
            price = self.p.mat_price_EN24_per_kg
        else:
            price = self.p.mat_price_4140_per_kg

        return mass_kg * price

    # ─────────────────────────────────────────────────────────────────────────
    # Machining cost
    # ─────────────────────────────────────────────────────────────────────────
    def machining_cost(self, var_dict: Dict[str, float]) -> float:
        """
        Turning + grinding cost.

        Tolerances below the grinding_threshold_mm require a grinding pass,
        which costs significantly more than turning.  The cost scales inversely
        with tolerance width (tighter → more expensive).
        """
        # Critical tolerances for each feature (bilateral ±)
        tolerances = {
            "journal_dia":   0.025,    # ±25 μm turning, ±5 μm ground
            "bore_dia":      0.010,    # inner bore
            "flange_face":   0.005,    # flatness (grinding)
            "outer_profile": 0.050,    # rough OD (turning)
        }

        R2 = var_dict["R2"]           # journal radius → diameter = 2×R2
        L2 = var_dict["L2"]           # journal length

        cost = 0.0

        for feature, tol in tolerances.items():
            # Representative surface area for cost scaling
            if "journal" in feature:
                area = 2 * np.pi * R2 * L2     # mm²
            else:
                area = 2 * np.pi * R2 * 20.0   # nominal 20 mm for other features

            # Tolerance premium: exponential as tol → 0
            tol_premium = np.exp(-(tol / 0.05) * 2.0)   # normalised

            if tol < self.p.grinding_threshold_mm:
                rate = self.p.cost_per_tol_mm_grinding
            else:
                rate = self.p.cost_per_tol_mm_turning

            cost += rate * (area / 1e4) * (1.0 + tol_premium)   # area in cm²

        return cost

    # ─────────────────────────────────────────────────────────────────────────
    # Bearing cost
    # ─────────────────────────────────────────────────────────────────────────
    def bearing_cost(self, var_dict: Dict[str, float]) -> float:
        """
        Angular contact ball bearing pair cost as a power-law of stiffness.

        Empirical fit from NSK/SKF catalogue data:
          cost ≈ C_base × (K / K_ref)^0.6
        """
        K = var_dict["K_radial"]
        cost_per_pair = (
            self.p.bearing_base_cost_usd
            * (K / self.p.bearing_stiffness_ref) ** self.p.bearing_stiffness_exp
        )
        return cost_per_pair * 2   # front + rear bearings

    # ─────────────────────────────────────────────────────────────────────────
    # Selective assembly cost (new)
    # ─────────────────────────────────────────────────────────────────────────
    def sa_cost(self, sa_results: List[SelectiveAssemblyResult]) -> float:
        """
        Total selective-assembly cost across all interfaces.

        Includes:
          • CMM measurement of shaft + housing parts
          • Bin sorting labour
          • Rework of unmatched parts
        """
        total = 0.0
        for r in sa_results:
            n_bins         = r.n_bins
            unmatched_rate = 1.0 - r.match_yield
            # Cost per spindle unit:
            meas_cost   = self.p.cmm_measurement_usd * 2        # shaft + housing
            sort_cost   = self.p.sorting_cost_per_bin * n_bins
            rework_cost = self.p.rework_cost_per_part * unmatched_rate
            total      += meas_cost + sort_cost + rework_cost

        return total

    # ─────────────────────────────────────────────────────────────────────────
    # Total cost
    # ─────────────────────────────────────────────────────────────────────────
    def total_cost(
        self,
        var_dict: Dict[str, float],
        sa_results: Optional[List[SelectiveAssemblyResult]] = None,
    ) -> Dict[str, float]:
        """
        Compute and return all cost components plus the total.

        Args:
            var_dict:   Decoded design vector
            sa_results: Output from SelectiveAssemblyAnalyser.analyse_all()

        Returns:
            Dict with keys: material, machining, bearings, sa, total
        """
        c_mat  = self.material_cost(var_dict)
        c_mach = self.machining_cost(var_dict)
        c_bear = self.bearing_cost(var_dict)
        c_sa   = self.sa_cost(sa_results) if sa_results else 0.0
        total  = c_mat + c_mach + c_bear + c_sa

        return {
            "material_usd":   c_mat,
            "machining_usd":  c_mach,
            "bearings_usd":   c_bear,
            "sa_usd":         c_sa,
            "total_usd":      total,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Selective assembly analyser
# ─────────────────────────────────────────────────────────────────────────────

class SelectiveAssemblyAnalyser:
    """
    Simulate selective-assembly binning and compute assembly quality metrics.

    For each mating interface:
      1. Sample shaft and housing dimensions from their respective tolerance bands
      2. Assign each part to a bin (uniform bins spanning ± tolerance)
      3. Compute the actual assembly gap for each shaft-housing pair
      4. Calculate the yield (fraction matched) and gap statistics
    """

    def __init__(
        self,
        n_parts: int = 500,
        random_seed: int = 42,
    ):
        self.n_parts = n_parts
        self.rng     = np.random.default_rng(random_seed)

    # ─────────────────────────────────────────────────────────────────────────
    # Core binning algorithm
    # ─────────────────────────────────────────────────────────────────────────
    def analyse_interface(
        self,
        interface: MatingInterface,
    ) -> SelectiveAssemblyResult:
        """
        Simulate selective assembly for a single mating pair.

        Gap = housing_bore − shaft_journal  (positive = clearance)

        Without SA: gap varies over full tolerance range → large σ_gap
        With SA:    shaft and housing bins are matched so gap is confined
                    within one bin-width → σ_gap dramatically smaller

        Args:
            interface: Geometry + tolerance + bin configuration

        Returns:
            SelectiveAssemblyResult with all quality + cost metrics
        """
        cfg = interface.bin_config
        n   = self.n_parts

        # Sample actual part dimensions (uniform manufacturing distribution)
        shaft_dims   = self.rng.uniform(
            interface.shaft_dim   - interface.shaft_tol,
            interface.shaft_dim   + interface.shaft_tol,
            size=n,
        )
        housing_dims = self.rng.uniform(
            interface.housing_dim - interface.housing_tol,
            interface.housing_dim + interface.housing_tol,
            size=n,
        )

        # ── Without SA: random pairing ─────────────────────────────────────
        gaps_random = housing_dims - shaft_dims        # mm
        std_no_sa   = float(np.std(gaps_random)) * 1e3   # μm

        # ── With SA: bin assignment ────────────────────────────────────────
        n_bins = cfg.n_bins

        shaft_bins   = self._assign_bins(shaft_dims,   interface.shaft_tol,   n_bins)
        housing_bins = self._assign_bins(housing_dims, interface.housing_tol, n_bins)

        # Bin compatibility matrix:
        #   shaft bin i is compatible with housing bin j if the resulting gap
        #   is within the allowed tolerance around nominal_gap.
        matched_gaps = []
        matched_pairs = 0

        for i in range(n_bins):
            shaft_in_bin   = shaft_dims[shaft_bins   == i]
            housing_in_bin = housing_dims[housing_bins == i]

            if len(shaft_in_bin) == 0 or len(housing_in_bin) == 0:
                continue

            # Match bin-i shaft with bin-i housing (same-bin pairing)
            n_match = min(len(shaft_in_bin), len(housing_in_bin))
            gaps    = housing_in_bin[:n_match] - shaft_in_bin[:n_match]
            matched_gaps.append(gaps)
            matched_pairs += n_match

        if not matched_gaps:
            matched_gaps_arr = np.array([cfg.nominal_gap])
        else:
            matched_gaps_arr = np.concatenate(matched_gaps)

        mean_gap_mm = float(np.mean(matched_gaps_arr))
        std_gap_mm  = float(np.std(matched_gaps_arr))

        yield_rate  = matched_pairs / n

        # Cost stub (actual dollar assignment in SpindleCostModel.sa_cost)
        sa_cost_usd = 0.0   # filled in by SpindleCostModel

        return SelectiveAssemblyResult(
            interface_name    = cfg.interface_name,
            n_bins            = n_bins,
            mean_gap_um       = mean_gap_mm * 1e3,
            std_gap_um        = std_gap_mm  * 1e3,
            std_gap_no_sa_um  = std_no_sa,
            improvement_ratio = std_no_sa / max(std_gap_mm * 1e3, 0.001),
            match_yield       = yield_rate,
            sa_cost_usd       = sa_cost_usd,
            machining_cost_usd= 0.0,
            total_cost_usd    = 0.0,
        )

    def _assign_bins(
        self,
        dims: np.ndarray,
        tolerance: float,
        n_bins: int,
    ) -> np.ndarray:
        """Assign parts to bins 0…n_bins-1 based on measured dimension."""
        lo   = dims.min() - 1e-9
        hi   = dims.max() + 1e-9
        bins = np.floor((dims - lo) / (hi - lo) * n_bins).astype(int)
        bins = np.clip(bins, 0, n_bins - 1)
        return bins

    # ─────────────────────────────────────────────────────────────────────────
    # Analyse all three spindle interfaces for a given design vector
    # ─────────────────────────────────────────────────────────────────────────
    def analyse_all(
        self,
        var_dict: Dict[str, float],
        n_bins: int = 5,
    ) -> List[SelectiveAssemblyResult]:
        """
        Run selective-assembly analysis for:
          1. Journal ↔ Bearing inner race (radial fit)
          2. Inner spacer ↔ Outer spacer (axial preload control)
          3. Housing bore ↔ Bearing outer ring

        Args:
            var_dict: Decoded design vector (from DesignSpace.decode_vector)
            n_bins:   Bins per interface (3=coarse, 5=standard, 10=precision)

        Returns:
            List of three SelectiveAssemblyResult objects.
        """
        results = []

        # ── Interface 1: Shaft journal ↔ Bearing inner race ───────────────
        journal_diam   = 2.0 * var_dict["R2"]   # mm (journal outer diameter)
        bearing_bore   = journal_diam            # nominal clearance = 0 → snug fit
        # Typical h5 / H6 tolerances for 100mm diameter journal
        shaft_tol_rad  = 0.006     # ±6 μm on radius = ±12 μm on diameter
        bearing_tol_d  = 0.008     # ±8 μm on diameter

        iface1 = MatingInterface(
            shaft_dim   = journal_diam,
            shaft_tol   = shaft_tol_rad * 2,     # diameter
            housing_dim = bearing_bore + 0.004,  # 4 μm nominal clearance
            housing_tol = bearing_tol_d,
            bin_config  = BinConfig(
                n_bins          = n_bins,
                nominal_gap     = 0.004,          # mm (4 μm clearance)
                allowed_gap_tol = 0.002,
                interface_name  = "Journal_InnerRace",
            ),
        )
        results.append(self.analyse_interface(iface1))

        # ── Interface 2: Inner spacer ↔ Outer spacer (axial preload) ──────
        # In a DB pair, preload Δ = L_inner − L_outer.  The flange section
        # length L3 proxies the spacer length.
        spacer_len  = var_dict["L3"]
        target_delta_mm = var_dict["K_axial"] * 1e-5   # very small preload gap

        iface2 = MatingInterface(
            shaft_dim   = spacer_len,
            shaft_tol   = 0.005,     # ±5 μm inner spacer
            housing_dim = spacer_len + target_delta_mm,
            housing_tol = 0.005,     # ±5 μm outer spacer
            bin_config  = BinConfig(
                n_bins          = n_bins,
                nominal_gap     = target_delta_mm,
                allowed_gap_tol = 0.001,          # ±1 μm preload tolerance
                interface_name  = "InnerSpacer_OuterSpacer",
            ),
        )
        results.append(self.analyse_interface(iface2))

        # ── Interface 3: Housing bore ↔ Bearing outer ring ────────────────
        bearing_OD    = 2.0 * var_dict["R2"] + 30.0    # approx (bearing OD > shaft OD)
        housing_bore  = bearing_OD - 0.008             # 8 μm interference fit

        iface3 = MatingInterface(
            shaft_dim   = bearing_OD,
            shaft_tol   = 0.006,
            housing_dim = housing_bore,
            housing_tol = 0.010,
            bin_config  = BinConfig(
                n_bins          = n_bins,
                nominal_gap     = -0.008,         # mm (negative = interference)
                allowed_gap_tol = 0.004,
                interface_name  = "Housing_OuterRing",
            ),
        )
        results.append(self.analyse_interface(iface3))

        return results

    def summary_table(self, results: List[SelectiveAssemblyResult]) -> pd.DataFrame:
        """Return a formatted DataFrame summarising all SA interfaces."""
        rows = []
        for r in results:
            rows.append({
                "Interface":        r.interface_name,
                "Bins":             r.n_bins,
                "Mean gap (μm)":    f"{r.mean_gap_um:+.2f}",
                "σ with SA (μm)":   f"{r.std_gap_um:.2f}",
                "σ without SA (μm)":f"{r.std_gap_no_sa_um:.2f}",
                "Improvement ×":    f"{r.improvement_ratio:.1f}",
                "Yield %":          f"{r.match_yield*100:.1f}",
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from design_variables import DesignSpace

    ds       = DesignSpace()
    nom      = ds.decode_vector(ds.get_nominal())

    analyser  = SelectiveAssemblyAnalyser(n_parts=1000)
    cost_mdl  = SpindleCostModel()

    print("\n🔩 SELECTIVE ASSEMBLY ANALYSIS — Nominal Design\n")

    for n_bins in [3, 5, 10]:
        sa_results = analyser.analyse_all(nom, n_bins=n_bins)
        print(f"\n  ── {n_bins} bins ──────────────────────────────")
        print(analyser.summary_table(sa_results).to_string(index=False))

        costs = cost_mdl.total_cost(nom, sa_results)
        print(f"\n  Cost breakdown:")
        for k, v in costs.items():
            print(f"    {k:<22} ${v:>8.2f}")

    print("\n✅ Selective Assembly module OK")
