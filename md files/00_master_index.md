# TechPulse Spindle Suite — Complete Technical Documentation
**Framework:** Robust Design Optimization for CNC Lathe Spindles
**Version:** v3 (post-review, all bugs fixed)

---

## Framework Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TECHPULSE SPINDLE RDO FRAMEWORK                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  01_design_variables.py          ← Design Space (19 vars)               │
│      AsymmetricTolerance          ← ISO 286 asymmetric fits             │
│      SKF_ACBB_CATALOG (18 brgs)   ← 7200 BE series, 25°                 │
│      SKF_CRB_CATALOG  (17 brgs)   ← NU 2xx ECP                          │
│      SpindleBearingArrangement    ← User-configurable stations          │
│                │                                                        │
│                ▼                                                        │
│  02_lhs_sampler.py               ← Space-filling DoE                    │
│      LHSSampler (maximin/lloyd)   ← scipy.stats.qmc                     │
│      Sobol, Monte Carlo, Hybrid   ← For robustness inner loop           │
│                │                                                        │
│                ▼                                                        │
│  03_fea_pool_runner.py           ← Batch FEA Execution                  │
│      Analytical (dry-run)         ← Propped-beam + Dunkerly             │
│      ANSYS MAPDL (production)     ← BEAM188, COMBIN14, MASS21           │
│                │                                                        │
│                ▼                                                        │
│  04_ml_surrogate.py              ← Digital Twin (1000× speedup)         │
│      GP (Matérn 2.5)              ← Per-output scalers (bug-1 fix)      │
│      XGBoost, MLP (fallbacks)     ← Auto cross-validation               │
│                │                                                        │
│         ┌──────┴──────────────────────────┐                             │
│         ▼                                 ▼                             │
│  05_robust_optimizer.py         06_inverse_design.py                    │
│      NSGA-II (4 objectives)         DNN: Y* → X*                        │
│      Taguchi S/N (inner MC)         TF / sklearn MLP                    │
│      Cost + Weight objectives       Forward validation                  │
│         │                                                               │
│         ▼                                                               │
│  07_selective_assembly.py        ← SA Quality + Cost                    │
│      3 interfaces analysed          Feeds f3 (cost) in GA               │
│      SpindleCostModel               4 cost components                   │
│                                                                         │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─                │
│                                                                         │
│  08_bearing_performance.py       ← Bearing Mechanics                    │
│      Multi-station force dist.      ISO 281 L10 life                    │
│      COMBIN14 spring table          ndm / speed constraints             │
│                                                                         │
│  09_shaft_runout.py              ← Analytical Runout                    │
│      4 sources (ISO 492/1101)       + ANSYS δ_nose overlay              │
│      Pre-FEA + Post-FEA phases      RSS + linear combination            │
│                                                                         │
│  10_rotor_eccentricity.py        ← CG + Imbalance                       │
│      Bore-offset formula            ISO 1940-1 G2.5                     │
│      ANSYS force block export       F = m·e·ω² at z_CG                  │
│                                                                         │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─                │
│                                                                         │
│  ANSYS MAPDL                     ← Full FEA / Harmonic                  │
│      BEAM188 shaft elements         Static deflection / stress          │
│      COMBIN14 bearing springs       Modal frequencies (Campbell)        │
│      MASS21 end masses              Harmonic: vibration from F_imbal    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Through the Framework

```
Design Vector x[19]  (continuous, from optimizer)
         │
         ▼
01  snap_to_skf_bearing(R2, n_rpm)
         │ → SKFBearing or SKFCRBBearing (catalog record)
         │ → K_radial, K_axial, F_preload, n_grease, C_r
         │
         ▼
02  LHSSampler.generate_lhs(n=500)
         │ → X_samples (500 × 19)
         │
         ▼
03  FEAPoolRunner.execute_batch()
         │ → [dry-run]  analytical propped-beam + Dunkerly
         │ → [ANSYS]    BEAM188 + COMBIN14 + MASS21
         │ → df_results: {deflection_um, stress_MPa, freq_Hz, ...}
         │
         ▼
04  SurrogateModel.train(X, y)
         │ → GP models (one per output, one scaler per output)
         │ → predict(x) in 0.001 s
         │
         ▼
05  RobustOptimizer.evaluate(x, n_rpm)
         │
         ├── taguchi_sn_ratio(x)
         │       └── inner MC (n=20): surrogate × 20 calls
         │           → S/N_deflection, S/N_stress
         │
         ├── SpindleCostModel.total_cost(x)
         │       └── material + machining + bearings + SA
         │
         └── weight = Σ ρ π (R²-r²) L × 1000
                 │
                 ▼
         f = [-SN_defl, -SN_stress, cost_USD, mass_kg]
                 │
                 ▼
         NSGA-II → Pareto front (100 pop × 50 gen)
                 │
                 ▼
         Optimal design x*
         │
         ├── 08  BearingPerformanceCalculator.evaluate(x*)
         │       → L10_system, K_pair, F_preload, COMBIN14 table
         │
         ├── 09  ShaftRunoutAnalyser.analyse(x*, δ_ANSYS)
         │       → TIR_geometric, TIR_loaded
         │
         └── 10  RotorEccentricityAnalyser.analyse(x*)
                 → z_CG, F_imbalance → ANSYS MAPDL input block
```

---

## Module Summary Table

| Module | File | Lines | Purpose | Key Class |
|--------|------|-------|---------|-----------|
| 01 | `01_design_variables.py` | 719 | Design space, SKF catalogs, arrangements | `DesignSpace`, `SpindleBearingArrangement` |
| 02 | `02_lhs_sampler.py` | 320 | Space-filling DoE | `LHSSampler` |
| 03 | `03_fea_pool_runner.py` | 380 | Batch FEA execution | `FEAPoolRunner` |
| 04 | `04_ml_surrogate.py` | 350 | ML digital twin | `SurrogateModel` |
| 05 | `05_robust_optimizer.py` | 480 | GA + Taguchi RDO | `RobustOptimizer` |
| 06 | `06_inverse_design.py` | 410 | Target → geometry | `InverseDesignEngine` |
| 07 | `07_selective_assembly.py` | 520 | SA quality + cost | `SelectiveAssemblyAnalyser`, `SpindleCostModel` |
| 08 | `08_bearing_performance.py` | 420 | Bearing L10, stiffness | `BearingPerformanceCalculator` |
| 09 | `09_shaft_runout.py` | 350 | TIR analytical | `ShaftRunoutAnalyser` |
| 10 | `10_rotor_eccentricity.py` | 370 | CG + imbalance | `RotorEccentricityAnalyser` |

---

## All Bugs Fixed (12 total)

| # | Module | Severity | Description | Fix |
|---|--------|----------|-------------|-----|
| 1 | 04 | 🔴 Critical | Single shared `scaler_y` refitted per output → wrong inverse-transform for all outputs except last | `scalers_y: Dict[str, StandardScaler]` — one per output |
| 2 | 04 | 🔴 Critical | `List`, `Dict` not imported from `typing` → `NameError` | Added to imports |
| 3 | 04 | 🟠 High | GP std unscaling used wrong (stale) scaler | Uses `scalers_y[out_name].scale_[0]` |
| 4 | 02 | 🔴 Critical | pyDOE2 criterion strings passed to scipy → `ValueError` | `_CRITERION_MAP` dict + lloyd fallback |
| 5 | 02 | 🟡 Medium | `tuple[...]` return type (Python 3.9+ only) | `Tuple` from `typing` |
| 6 | 05 | 🔴 Critical | Hardcoded key lookup silently returned 0 → GA on flat landscape | Index resolved at `__init__`, `ValueError` on missing key |
| 7 | 05 | 🟠 High | `n_obj=3` hardcoded in pymoo Problem | Parameterised from `self.n_objectives` |
| 8 | 05 | 🟠 High | Weight vector length-3 crashed with 4 objectives | Built dynamically from `n_objectives` |
| 9 | 05 | 🟡 Medium | `log10(1/y²)` → −∞ when y≈0 | `y_c = max(\|y\|, eps=1e-10)` |
| 10 | 06 | 🔴 Critical | `List` used in signature but not imported → `NameError` | Added to `typing` imports |
| 11 | 03 | 🟠 High | Cantilever formula ignores bearing supports (wrong physics) | Propped-beam with `R_front`, `R_rear` reactions |
| 12 | 03 | 🟡 Medium | Free-free Rayleigh coefficient for bearing-supported shaft | Dunkerly superposition |

---

## Complete ISO Standard References

| Standard | Year | Module | Topic |
|----------|------|--------|-------|
| **ISO 281** | 2007 | 08 | Dynamic load ratings and rating life for rolling bearings. Defines $L_{10}$, equivalent dynamic load $P$, exponents $p=3$ (ball), $p=10/3$ (roller), and system life combination formula. |
| **ISO 281 Annex C** | 2007 | 08 | System life of bearing arrangements: $1/L_{10,\text{sys}}^e = \sum 1/L_{10,i}^e$ |
| **ISO 286-1** | 2010 | 01 | ISO system of limits and fits. Defines fundamental deviations (h5 shaft, H6/H7 housing), tolerance grades IT5–IT12. Used for all shaft-bearing interface fits. |
| **ISO 492** | 2014 | 09 | Rolling bearings — Dimensional and geometrical tolerances. Table 4: inner ring radial runout per precision class: P5 = 5 μm. |
| **ISO 1101** | 2017 | 09 | Geometrical Product Specifications (GPS) — Geometrical tolerancing. Form tolerances: straightness, roundness. Used for shaft bow model. |
| **ISO 1940-1** | 2003 | 10 | Mechanical vibration — Balance quality requirements for rotors in a constant (rigid) state. Defines G grades: $U_{\text{allow}} = m \times G / \omega$. G2.5 for machine tool spindles. |
| **ISO 1940-2** | 1997 | 10 | Balance errors — procedures and tolerances for determining and verifying residual imbalance. |
| **ISO 2768-1** | 1989 | 09 | General tolerances for linear and angular dimensions. Fine class (f): straightness 0.012 mm per 300 mm. |
| **ISO 15** | 2017 | 01 | Rolling bearings — Radial bearings — Boundary dimensions, general plan. Defines shaft fits h5/h6 and housing fits H6/H7 for ACBB. |
| **ISO 230-1** | 2012 | 09 | Test code for machine tools — Part 1: Geometric accuracy of machines. §4.3.1.1: Radial deviation under working conditions (loaded runout). |
| **ISO 21940-11** | 2016 | 08 (prev), ANSYS | Mechanical vibration — Rotor balancing — Part 11: Procedures and tolerances for rotors with flexible behaviour. |
| **ISO 10816-3** | 2009 | 08 (prev) | Mechanical vibration — Evaluation of machine vibration by measurements on non-rotating parts. Velocity limit 2.8 mm/s RMS for machine tools. |
| **SKF GC 6000/EN** | 2018 | 01, 08 | SKF General Catalogue. Chapter 1: ACBB 7200 BE series. Chapter 4: CRB NU 2xx ECP series. Speed limits, load capacities, stiffness data. |

---

## Design Variables Quick Reference

| # | Name | Nominal | Bounds | Tol + | Tol − | Unit |
|---|------|---------|--------|-------|-------|------|
| 1 | E | 2.10×10⁵ | [2.00, 2.15]×10⁵ | 2000 | 2000 | MPa |
| 2 | Ff | 300 | [200, 1000] | 24 | 24 | N |
| 3 | Fr | 500 | [300, 1500] | 40 | 40 | N |
| 4 | Ft | 1500 | [1000, 3500] | 120 | 120 | N |
| 5 | K_axial | 8.0×10⁵ | [4.0, 14.0]×10⁵ | 8×10⁴ | 8×10⁴ | N/mm |
| 6 | K_radial | 5.0×10⁵ | [2.5, 9.0]×10⁵ | 5×10⁴ | 5×10⁴ | N/mm |
| 7 | L1 | 122 | [100, 150] | 0.500 | 0.000 | mm |
| 8 | L2 | 405 | [350, 450] | 0.500 | 0.000 | mm |
| 9 | L3 | 24 | [15, 40] | 0.100 | 0.000 | mm |
| 10 | L4 | 15 | [10, 25] | 0.500 | 0.000 | mm |
| 11 | R1 | 45 | [35, 55] | 0.000 | 0.011 | mm |
| 12 | **R2** | **50** | **[35, 60]** | **0.000** | **0.013** | **mm** |
| 13 | R3 | 82.5 | [70, 95] | 0.050 | 0.050 | mm |
| 14 | R4 | 45 | [35, 55] | 0.000 | 0.011 | mm |
| 15 | front_z_fraction | 0.25 | [0.15, 0.35] | 0.010 | 0.010 | — |
| 16 | rear_z_fraction | 0.75 | [0.65, 0.85] | 0.010 | 0.010 | — |
| 17 | ri | 30 | [25, 35] | 0.021 | 0.000 | mm |
| 18 | rho | 7.85×10⁻⁹ | [7.7, 8.0]×10⁻⁹ | 5×10⁻¹¹ | 5×10⁻¹¹ | ton/mm³ |
| 19 | sigma_y | 655 | [600, 750] | 10 | 15 | MPa |

**R2** (bearing seat radius) is the master variable — it drives `snap_to_skf_bearing()`.

---

## GA Objectives and Constraints Summary

### Objectives (all minimised)

| $f_i$ | Formula | Physical meaning |
|--------|---------|-----------------|
| $f_1$ | $-\eta_{\text{SB,deflection}}$ [dB] | Robustness of nose deflection vs. tolerance scatter |
| $f_2$ | $-\eta_{\text{SB,stress}}$ [dB] | Robustness of max von Mises stress |
| $f_3$ | $C_{\text{total}}$ [USD] | Total manufacturing + SA cost per unit |
| $f_4$ | $m_{\text{rotor}}$ [kg] | Rotor mass |

### Constraints (all $g \leq 0$ = satisfied)

**Bearing performance (Module 08):**

| Constraint | Formula |
|-----------|---------|
| Speed (each station) | $(n_{\text{op}} - n_{\text{limit}}) / n_{\text{limit}}$ |
| ndm (each station) | $(ndm - ndm_{\text{limit}}) / ndm_{\text{limit}}$ |
| System L10 life | $(L_{10,\text{target}} - L_{10,\text{sys}}) / L_{10,\text{target}}$ |
| Preload (low) | $(F_{\text{pre,min}} - F_{\text{pre}}) / F_{\text{pre,min}}$ |
| Preload (high) | $(F_{\text{pre}} - F_{\text{pre,max}}) / F_{\text{pre,max}}$ |

**Runout (Module 09):**

| Constraint | Formula |
|-----------|---------|
| Geometric TIR | $(\text{TIR}_{\text{geom}} - 10\;\mu\text{m}) / 10$ |
| Loaded TIR | $(\text{TIR}_{\text{loaded}} - 15\;\mu\text{m}) / 15$ |

**Eccentricity (Module 10):**

| Constraint | Formula |
|-----------|---------|
| Residual imbalance | $(U_{\text{static}} - U_{\text{allow}}) / U_{\text{allow}}$ |
| Imbalance force | $(F_{\text{imbal}} - 50\;\text{N}) / 50$ |

---

## Quick-Start Commands

```bash
# 1. Install dependencies
pip install numpy scipy pandas scikit-learn tqdm

# 2. Test all modules (no ANSYS required)
python3 run_tests.py

# 3. Run analytical dry-run workflow (50 samples)
python3 master_workflow.py --dry_run --n_samples 50

# 4. Full workflow with ANSYS (200 samples, GP surrogate)
export ANSYS_MAPDL_EXEC=/path/to/ansys/mapdl
python3 master_workflow.py --n_samples 200 --surrogate_type gp

# 5. Multi-objective optimization (NSGA-II)
python3 master_workflow.py --opt_method nsga2 --n_samples 500
```

---

## Pending Open Questions

These require answers before going to production:

| # | Question | Impact | Default applied |
|---|---------|--------|----------------|
| 1 | **Lubrication:** Grease or oil-air mist? | Oil removes ndm violation at 6,000 RPM | Grease (triggers warning) |
| 2 | **Runout limit:** Is 10 μm at nose the spec, or is 15 μm acceptable? | Changes which designs are feasible | 10 μm geometric, 15 μm loaded |
| 3 | **Contact angle:** 25° confirmed, or 15° for higher speed? | 15° gives higher ndm limit at same bore | 25° (7200 BE series) |
| 4 | **Shop bore_offset:** Actual CMM measurement of bore eccentricity per segment | Better than 5 μm assumed | 5 μm (conservative) |
| 5 | **Balancing equipment:** Dynamic balancing available in shop? | G1 vs G2.5 → 2.5× tighter eccentricity | G2.5 |
