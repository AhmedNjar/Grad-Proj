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
from plot_theme import apply_paper_theme, C, patch_ax, savefig_paper

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
        self._machining_cost_cache: Dict[Tuple[object, ...], float] = {}

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
        Tolerance-driven machining cost.

        This replaces the old rough turning/grinding estimate with a direct
        tolerance-cost model based on IT grades and surface roughness, so the
        machining line in the selective-assembly cost breakdown reflects the
        tolerance specification expense.
        """
        try:
            from tolerance_optimizer import compute_cost
        except Exception:
            # Fallback if the tolerance module is unavailable.
            return 0.0

        key = (
            round(float(var_dict.get("R2", 0.0)), 5),
            round(float(var_dict.get("L2", 0.0)), 5),
            str(var_dict.get("it_journal", "IT5")),
            str(var_dict.get("it_bore", "IT7")),
            str(var_dict.get("it_lengths", "IT12")),
            str(var_dict.get("housing_it_grade", 6.0)),
            round(float(var_dict.get("ra_journal_um", 0.8)), 5),
            round(float(var_dict.get("ra_bore_um", 1.6)), 5),
        )
        if key in self._machining_cost_cache:
            return self._machining_cost_cache[key]

        journal_it = str(var_dict.get("it_journal", "IT5"))
        if not isinstance(var_dict.get("it_journal", "IT5"), str):
            journal_it = f"IT{int(round(float(var_dict.get('it_journal', 5))))}"

        bore_it = str(var_dict.get("it_bore", "IT7"))
        if not isinstance(var_dict.get("it_bore", "IT7"), str):
            bore_it = f"IT{int(round(float(var_dict.get('it_bore', 7))))}"

        length_it = str(var_dict.get("it_lengths", "IT12"))
        if not isinstance(var_dict.get("it_lengths", "IT12"), str):
            length_it = f"IT{int(round(float(var_dict.get('it_lengths', 12))))}"

        housing_it = var_dict.get("housing_it_grade", 6.0)
        if isinstance(housing_it, str):
            housing_it_grade = housing_it
        else:
            housing_it_grade = f"IT{int(round(float(housing_it)))}"

        ra_journal_um = float(var_dict.get("ra_journal_um", 0.8))
        ra_bore_um    = float(var_dict.get("ra_bore_um", 1.6))

        cost = float(compute_cost(
            it_journal_grade=journal_it,
            it_bore_grade=bore_it,
            it_length_grade=length_it,
            it_housing_grade=housing_it_grade,
            ra_journal_um=ra_journal_um,
            ra_bore_um=ra_bore_um,
            n_journals=4,
            n_lengths=4,
        ))
        self._machining_cost_cache[key] = cost
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
        # K_radial no longer in design vector (removed in v4) — derive from catalog
        from design_variables import snap_to_skf_bearing
        brg_front, _, _ = snap_to_skf_bearing(var_dict["R2"], "ACBB", 4000.0, "grease")
        K = brg_front.radial_stiffness_single_N_mm * 1.7   # DB pair
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
        tolerance_cost_usd: Optional[float] = None,
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
        c_mach = (
            float(tolerance_cost_usd)
            if tolerance_cost_usd is not None
            else self.machining_cost(var_dict)
        )
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
        # K_axial derived from catalog (removed from design vector in v4)
        from design_variables import snap_to_skf_bearing
        import math as _math
        _brg, _, _ = snap_to_skf_bearing(var_dict["R2"], "ACBB", 4000.0, "grease")
        _alpha = _math.radians(_brg.contact_angle_deg)
        K_axial_catalog = 1.7 * _brg.radial_stiffness_single_N_mm * _math.tan(_alpha)**2
        target_delta_mm = K_axial_catalog * 1e-5   # very small preload gap

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
        # Housing bore tolerance derived from housing_it_grade design variable.
        # IT grade number → IT value → tolerance half-width for SA model.
        # IT values [μm] for housing bore diameter 120-180mm (ISO 286-1):
        _HOUSING_IT_UM = {5: 18.0, 6: 25.0, 7: 40.0, 8: 63.0}
        grade_num    = int(round(float(var_dict.get("housing_it_grade", 6.0))))
        grade_num    = max(5, min(8, grade_num))
        housing_tol_um = _HOUSING_IT_UM.get(grade_num, 25.0)   # μm full IT band
        housing_tol_mm = housing_tol_um * 1e-3                  # mm

        bearing_OD   = 2.0 * var_dict["R2"] + 30.0    # approx bearing OD
        housing_bore = bearing_OD - 0.008              # 8 μm nominal interference

        iface3 = MatingInterface(
            shaft_dim   = bearing_OD,
            shaft_tol   = 0.006,
            housing_dim = housing_bore,
            housing_tol = housing_tol_mm,       # IT-grade-dependent
            bin_config  = BinConfig(
                n_bins          = n_bins,
                nominal_gap     = -0.008,
                allowed_gap_tol = housing_tol_mm / 2.0,   # ±IT/2
                interface_name  = f"Housing_OuterRing_H{grade_num}",
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
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_selective_assembly(
    sa_results:  list,
    costs:       Dict[str, float],
    var_dict:    Dict[str, float],
    n_bins_list: Optional[List[int]] = None,
    analyser:    Optional["SelectiveAssemblyAnalyser"] = None,
    save_dir:    str = ".",
) -> None:
    """
    Three SA analysis plots.

    Fig 07a — Gap distribution before/after SA for each interface
    Fig 07b — Cost breakdown pie + bar comparison across n_bins
    Fig 07c — Improvement ratio and yield % vs. n_bins
    """
    import matplotlib.pyplot as plt
    import os

    NAVY=C.NAVY; TEAL=C.TEAL; CORAL=C.RED; GOLD=C.ORANGE
    MINT=C.GREEN; GRAY=C.GRAY; PURPLE=C.PURPLE
    os.makedirs(save_dir, exist_ok=True)
    apply_paper_theme()
    if n_bins_list is None:
        n_bins_list = [3, 5, 7, 10]

    SEG_COLS = [TEAL, CORAL, GOLD]

    # ── Fig 07a: Gap distributions before/after SA ────────────────────
    n_intf = len(sa_results)
    fig, axes = plt.subplots(n_intf, 2, figsize=(12, 4 * n_intf), facecolor=C.BG)
    if n_intf == 1:
        axes = [axes]
    fig.suptitle("Fig 07a — Gap Distributions Before/After Selective Assembly", color=C.TEXT, y=1.01)
    for row_axes, res, col in zip(axes, sa_results, SEG_COLS):
        ax_before, ax_after = row_axes[0], row_axes[1]
        for ax in (ax_before, ax_after):
            ax.set_facecolor(C.BG)
            patch_ax(ax)
        σ_before = res.std_gap_no_sa_um
        σ_after  = res.std_gap_um
        μ        = res.mean_gap_um
        x_no_sa  = np.random.normal(μ, σ_before, 500)
        x_sa     = np.random.normal(μ, σ_after,  500)
        ax_before.hist(x_no_sa, bins=30, color=CORAL, alpha=0.8, edgecolor=NAVY, lw=0.3)
        ax_before.axvline(μ - 3*σ_before, color=GOLD, lw=1, linestyle="--")
        ax_before.axvline(μ + 3*σ_before, color=GOLD, lw=1, linestyle="--",
                          label=f"±3σ = {σ_before:.1f} μm")
        ax_before.set_title(f"{res.interface_name}\nBefore SA  σ={σ_before:.1f} μm", fontsize=8)
        ax_before.legend(fontsize=7.5)

        ax_after.hist(x_sa, bins=30, color=col, alpha=0.8, edgecolor=NAVY, lw=0.3)
        ax_after.axvline(μ - 3*σ_after, color=GOLD, lw=1, linestyle="--")
        ax_after.axvline(μ + 3*σ_after, color=GOLD, lw=1, linestyle="--",
                         label=f"±3σ = {σ_after:.1f} μm")
        ax_after.set_title(f"After SA ({res.n_bins} bins)\nImproved ×{res.improvement_ratio:.1f}",
                           fontsize=8)
        ax_after.legend(fontsize=7.5)
        for ax in (ax_before, ax_after):
            ax.set_xlabel("Gap [μm]"); ax.set_ylabel("Count")
    plt.tight_layout()
    p = os.path.join(save_dir, "07a_sa_gap_distribution.png")
    savefig_paper(fig, p, dpi=150)
    plt.close(fig)

    # ── Fig 07b: Cost breakdown ───────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=C.BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(C.BG)
        patch_ax(ax)
    fig.suptitle("Fig 07b — Cost Breakdown", color=C.TEXT)

    cost_keys = ["material_usd", "machining_usd", "bearings_usd", "sa_usd"]
    cost_labels = ["Material", "Machining", "Bearings", "SA"]
    cost_vals = [costs.get(k, 0) for k in cost_keys]
    cost_colours = [TEAL, CORAL, GOLD, MINT]
    wedges, texts, autos = ax1.pie(
        cost_vals, labels=cost_labels, colors=cost_colours,
        autopct=lambda p: f"{p:.0f}%" if p > 3 else "",
        textprops={"color": "white", "fontsize": 9},
        wedgeprops={"edgecolor": NAVY, "linewidth": 0.6},
    )
    for at in autos:
        at.set_color("white")
    ax1.set_facecolor(C.BG)
    ax1.set_title(f"Total = ${costs.get('total_usd', sum(cost_vals)):.2f}", fontsize=9)

    # Bar chart of cost components
    ax2.bar(cost_labels, cost_vals, color=cost_colours, edgecolor=NAVY, linewidth=0.5)
    for i, v in enumerate(cost_vals):
        ax2.text(i, v + costs.get("total_usd", 1) * 0.01,
                 f"${v:.1f}", ha="center", fontsize=8.5, color="white")
    ax2.set_ylabel("Cost [USD]")
    ax2.set_title("Cost Components", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(save_dir, "07b_cost_breakdown.png")
    savefig_paper(fig, p, dpi=150)
    plt.close(fig)

    # ── Fig 07c: Improvement ratio and yield vs. n_bins ───────────────
    if analyser is not None:
        improve_per_bins = {n: [] for n in n_bins_list}
        yield_per_bins   = {n: [] for n in n_bins_list}
        for nb in n_bins_list:
            res_nb = analyser.analyse_all(var_dict, n_bins=nb)
            for r in res_nb:
                improve_per_bins[nb].append(r.improvement_ratio)
                yield_per_bins[nb].append(r.match_yield * 100)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), facecolor=C.BG)
        for ax in (ax1, ax2):
            ax.set_facecolor(C.BG)
            patch_ax(ax)
        fig.suptitle("Fig 07c — SA Performance vs. Number of Bins", color=C.TEXT)

        for j, res in enumerate(sa_results):
            iname = res.interface_name[:20]
            col   = SEG_COLS[j % len(SEG_COLS)]
            imps  = [np.mean(improve_per_bins[nb]) for nb in n_bins_list]
            ylds  = [np.mean(yield_per_bins[nb])   for nb in n_bins_list]
            ax1.plot(n_bins_list, [improve_per_bins[nb][j] for nb in n_bins_list],
                     marker="o", color=col, lw=1.8, label=iname)
            ax2.plot(n_bins_list, [yield_per_bins[nb][j]  for nb in n_bins_list],
                     marker="o", color=col, lw=1.8, label=iname)

        ax1.set_xlabel("Number of bins"); ax1.set_ylabel("Scatter improvement ×")
        ax1.set_title("Gap scatter improvement vs. bins"); ax1.legend(fontsize=7.5)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Number of bins"); ax2.set_ylabel("Match yield [%]")
        ax2.set_title("Assembly yield vs. bins"); ax2.legend(fontsize=7.5)
        ax2.axhline(90, color=GOLD, lw=1.2, linestyle="--", label="90% target")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        p = os.path.join(save_dir, "07c_sa_vs_bins.png")
        savefig_paper(fig, p, dpi=150)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os; sys.path.insert(0, ".")
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

    os.makedirs("/tmp/spindle_plots", exist_ok=True)
    sa5    = analyser.analyse_all(nom, n_bins=5)
    costs5 = cost_mdl.total_cost(nom, sa5)
    print("\nGenerating SA plots...")
    plot_selective_assembly(sa5, costs5, nom,
                            n_bins_list=[3, 5, 7, 10],
                            analyser=analyser,
                            save_dir="/tmp/spindle_plots")
    print("\n✅ Selective Assembly module OK")
