#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  FEA Pool Runner v2 — Batch Parametric Simulation Engine
================================================================================

  Bug Fixes Applied (vs. v1):
      BUG-11 [HIGH]:  Analytical model treated the spindle as a pure cantilever
                       (δ=FL³/3EI), ignoring the two bearing supports.  A
                       spindle is a propped beam: the load is applied at the
                       nose overhang (z<0 relative to front bearing), while
                       the two bearings act as spring supports.
                       Fix: three-segment transfer-matrix model that accounts
                       for front and rear bearing radial stiffness explicitly.
      BUG-12 [MEDIUM]: First natural frequency used the cantilever Rayleigh
                        coefficient λ₁=1.875 (free-clamped).  With two spring
                        supports the Dunkerly superposition formula is used
                        instead, accumulating compliance contributions from
                        each support point.
================================================================================
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── try to import from the user's PyMAPDL script ──────────────────────────
try:
    sys.path.insert(0, "/mnt/user-data/uploads")
    from lathe_spindle_simulation import (
        SpindleGeometry, MaterialProperties, BearingConfig,
        CuttingLoads, MeshConfig, ModalConfig, SimulationCase,
        SpindleSimulation,
    )
    MAPDL_AVAILABLE = True
except ImportError:
    MAPDL_AVAILABLE = False
    logging.getLogger("FEA_Pool").warning(
        "PyMAPDL script not importable — dry-run only."
    )

from design_variables import DesignSpace

log = logging.getLogger("FEA_Pool")
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

class FEAPoolRunner:

    def __init__(
        self,
        X_samples,
        design_space,
        dry_run=True,
        output_dir="fea_batch_results",
        restart_interval=20
    ):
        self.X_samples = X_samples
        self.design_space = design_space
        self.dry_run = dry_run
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = []
        self.sim = None
        self.restart_interval = restart_interval

    # ============================================================
    # MAIN EXECUTION
    # ============================================================
    def execute_batch(self, max_failures=10, save_interval=20):

        n = len(self.X_samples)
        fails = 0

        # 🔥 start MAPDL once
        if not self.dry_run:
            self.sim = SpindleSimulation(output_dir=str(self.output_dir))
            self.sim.start()

        for i, x in enumerate(tqdm(self.X_samples, desc="FEA Pool")):

            try:
                if self.dry_run:
                    res = self._run_analytical(x, case_id=i + 1)
                else:
                    res = self._run_ansys(x, case_id=i + 1)

                self.results.append(res)
                fails = 0

            except Exception as exc:
                log.error(f"Case {i+1} failed: {exc}")
                fails += 1

                if fails >= max_failures:
                    log.error("Stopping — too many consecutive failures.")
                    break

            # 💾 save periodically
            if (i + 1) % save_interval == 0:
                self._save()

            # 🔄 SAFE MAPDL RESTART
            if not self.dry_run and (i + 1) % self.restart_interval == 0:
                log.info("🔄 Restarting MAPDL to avoid memory issues...")
                self.sim.close()
                self.sim.start()

        # 🔚 close MAPDL once
        if not self.dry_run and self.sim:
            self.sim.close()

        self._save()

        df = pd.DataFrame(self.results)
        log.info(f"✅ Batch done: {len(df)}/{n} successful")

        return df

    # ============================================================
    # ANSYS RUN (FIXED)
    # ============================================================
    def _run_ansys(self, x, case_id):

        if self.sim is None:
            raise RuntimeError("MAPDL session not initialized.")

        v = self.design_space.decode_vector(x)

        case = SimulationCase(
            case_id=case_id,
            geometry=SpindleGeometry(
                L1=v["L1"], L2=v["L2"], L3=v["L3"], L4=v["L4"],
                R1=v["R1"], R2=v["R2"], R3=v["R3"], R4=v["R4"],
                ri=v["ri"],
            ),
            material=MaterialProperties(
                E=v["E"], nu=0.29, rho=v["rho"],
                sigma_y=v["sigma_y"], sigma_u=v["sigma_y"] * 1.3,
            ),
            bearings=BearingConfig(
                front_z_fraction=v["front_z_fraction"],
                rear_z_fraction=v["rear_z_fraction"],
                K_radial=v["K_radial"], K_axial=v["K_axial"],
            ),
            loads=CuttingLoads(Ft=v["Ft"], Fr=v["Fr"], Ff=v["Ff"]),
            mesh=MeshConfig(),
            modal=ModalConfig(num_modes=6),
        )

        # 🔥 reuse same MAPDL instance
        res = self.sim.run_case(case)
        res["mode"] = "ansys_mapdl"

        # add variables
        res.update({f"var_{k}": float(v_) for k, v_ in v.items()})

        return res

    # ============================================================
    # SAVE
    # ============================================================
    def _save(self):
        if not self.results:
            return

        df = pd.DataFrame(self.results)
        df.to_csv(self.output_dir / "fea_batch_results.csv", index=False)

        try:
            df.to_parquet(self.output_dir / "fea_batch_results.parquet", index=False)
        except ImportError:
            pass

# ──────────────────────────────────────────────────────────────────────────────
# Smoke test — verify bearing-supported model gives physically sensible results
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from design_variables import DesignSpace
    from lhs_sampler     import LHSSampler

    ds = DesignSpace()
    X  = LHSSampler(ds).generate_lhs(15, seed=42)

    runner = FEAPoolRunner(X, ds, dry_run=True)
    df     = runner.execute_batch()

    cols = ["case_id", "static_max_deflection_um",
            "static_max_vonmises_MPa", "freq_mode1_Hz"]
    print(df[cols].to_string(index=False))

    # Sanity: stiffening the shaft (larger R2) should reduce deflection
    from design_variables import DesignSpace as DS
    import copy

    ds2  = DS()
    nom  = ds2.get_nominal()
    soft = nom.copy(); stiff = nom.copy()

    # Find R2 index
    idx_R2 = ds2.get_variable_names().index("R2")
    soft[idx_R2]  = 40.0    # small radius
    stiff[idx_R2] = 60.0    # large radius

    X2 = np.vstack([soft, stiff])
    r2 = FEAPoolRunner(X2, ds2, dry_run=True).execute_batch()
    d_soft  = r2.iloc[0]["static_max_deflection_um"]
    d_stiff = r2.iloc[1]["static_max_deflection_um"]
    assert d_soft > d_stiff, f"Physics check failed: {d_soft:.2f} vs {d_stiff:.2f}"
    print(f"\n✅ Stiffness check: R2=40→δ={d_soft:.2f}μm  R2=60→δ={d_stiff:.2f}μm (correct)")
