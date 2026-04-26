# Module 01 — Design Variables
**File:** `01_design_variables.py`

---

## Purpose

Defines the complete **19-variable design space** for the lathe spindle RDO framework. Every quantity that the optimizer is allowed to change is declared here with:

- Its nominal (baseline) value
- Its search bounds for the optimizer `[lower, upper]`
- Its **asymmetric manufacturing tolerance** (separate `+upper` and `−lower` deviations)
- Its physical unit and description

In addition, this module embeds the **SKF bearing catalog** (ACBB + CRB series) and provides snap-to-catalog resolution, so every optimizer evaluation uses a real, purchasable bearing.

---

## 1. `AsymmetricTolerance` — ISO 286 / ASME Y14.5

### Why asymmetric?

Engineering drawings specify tolerances as:

$$D_{\text{nominal}} \; {}^{+\text{upper}}_{-\text{lower}}$$

For example, an h5 shaft journal at Ø100 mm is specified as `100 −0.013/+0.000 mm`. This means the shaft **can only be undersized**, never oversized.  Storing a single bilateral `±tolerance` float loses this information and incorrectly allows the shaft to be both over- and under-nominal.

### Implementation

```python
@dataclass
class AsymmetricTolerance:
    upper: float   # + deviation (positive number)
    lower: float   # magnitude of − deviation (stored positive; means subtract)
```

### Sampling method — Truncated Normal

For robust design evaluation (Monte Carlo inner loop), each dimension is sampled from a **truncated normal distribution** centred at the mid-band of the tolerance:

$$\mu_{\text{sample}} = D_{\text{nominal}} + \frac{\text{upper} - \text{lower}}{2}$$

$$\sigma = \frac{\text{upper} + \text{lower}}{6}$$

The 6σ choice means ±3σ = total tolerance band, consistent with a Cpk = 1.0 process.  Samples are hard-clipped to `[nominal − lower, nominal + upper]`.

**Standard reference:** ISO 286-1:2010 §4 — Fundamental deviation and tolerance grade.

---

## 2. SKF ACBB Catalog — `SKFBearing` / `SKF_ACBB_CATALOG`

### Bearing series

| Property | Value |
|----------|-------|
| Series | 7200 BE-2RZP |
| Contact angle α | 25° |
| Precision class | P5 |
| Bore range | 30 – 120 mm |
| Data source | SKF General Catalogue 6000/EN, Chapter 1 |

### Key catalog fields

| Field | Symbol | Unit | Meaning |
|-------|--------|------|---------|
| `C_r` | $C_r$ | N | Dynamic radial load capacity |
| `C_0r` | $C_{0r}$ | N | Static radial load capacity |
| `n_grease` | $n_{\text{gr}}$ | RPM | Limiting speed under grease |
| `n_oil` | $n_{\text{oil}}$ | RPM | Limiting speed under oil |
| `F_preload_MA_N` | $F_{\text{pre}}$ | N | Medium (MA) class preload force |

### Pitch diameter

$$d_m = \frac{d + D}{2} \quad [\text{mm}]$$

Used in the ndm speed factor check.

### Radial stiffness — empirical fit

$$K_{r,\text{single}} \;[\text{N/μm}] \approx 5.5 \times d^{0.75}$$

Fitted to published SKF spindle bearing stiffness data (P5 precision, MA preload). Error ±15 % vs catalogue values.

**Verification:**

| Bearing | d [mm] | Formula [N/μm] | Catalogue [N/μm] | Error |
|---------|--------|----------------|-----------------|-------|
| 7212 | 60 | 121 | 130 | −7 % |
| 7216 | 80 | 153 | 160 | −4 % |
| 7220 | 100 | 174 | 200 | −13 % |

### Axial stiffness

$$K_{a,\text{single}} = K_{r,\text{single}} \times \tan^2(\alpha) \quad \alpha = 25°$$

$$\tan^2(25°) = 0.2169$$

**Standard reference:** SKF Engineering Handbook, §3.4 — Spring stiffness of ball bearings under preload.

---

## 3. SKF CRB Catalog — `SKFCRBBearing` / `SKF_CRB_CATALOG`

### Bearing series

| Property | Value |
|----------|-------|
| Series | NU 2xx ECP |
| Type | Single row, NU (inner ring free axially) |
| Precision class | P5 |
| Data source | SKF General Catalogue 6000/EN, Chapter 4 |

### Why CRB at the rear position?

The rear bearing is the **floating** (non-locating) position. Its requirements are:

1. **High radial stiffness** — CRB has line contact vs. ACBB point contact → 1.9× stiffer at same bore
2. **No axial constraint** — NU type inner ring is free to slide axially, allowing thermal expansion of the shaft without inducing axial force
3. **Higher radial capacity** — NU2220 carries 228 kN vs. 7220 ACBB 118 kN at same bore

### CRB radial stiffness

$$K_{r,\text{CRB}} \;[\text{N/μm}] \approx 8.0 \times d^{0.80}$$

Line contact (roller) gives a steeper bore-exponent (0.80 vs. 0.75 for ACBB) and a higher coefficient.

**Verification at d = 80 mm:**
- ACBB formula: 5.5 × 80^0.75 = 153 N/μm
- CRB formula: 8.0 × 80^0.80 = 291 N/μm → ratio 1.90× ✓ (expected ~1.8–2.0×)

---

## 4. `snap_to_skf_bearing()` — Snap-to-Catalog Function

### Why snap-to-nearest?

The optimizer (NSGA-II, Differential Evolution) works in **continuous** space. If we made the bore a discrete variable, gradient-free optimizers would still work but the landscape would be discontinuous. Instead:

1. Optimizer proposes a continuous journal radius R2 ∈ [35, 60] mm
2. At **evaluation time only**, `snap_to_skf_bearing(R2, n_rpm)` finds the nearest SKF bore
3. The catalog bearing's stiffness, speed limit, and preload are used
4. The optimizer sees a smooth objective landscape (the surrogate interpolates across snap boundaries)

### Speed check cascade

```
Request: R2 = 50 mm, n = 6,000 RPM, grease
Step 1: nearest bore = 100 mm → 7220 BECBP → n_grease = 4,300 RPM < 6,000 ❌
Step 2: try oil → n_oil = 5,600 RPM < 6,000 ❌
Step 3: step down bores until n_grease ≥ 6,000 → 7215 (75 mm) → n_grease = 6,300 ✅
Return: 7215 BECBP with warning
```

---

## 5. `BearingStation` and `SpindleBearingArrangement`

### BearingStation fields

| Field | Type | Meaning |
|-------|------|---------|
| `role` | "front" / "rear" | Locating vs. floating |
| `bearing_type` | "ACBB" / "CRB" | Bearing family |
| `n_bearings` | int | Count at this station |
| `sub_arrangement` | "single" / "DB" / "spacer_pair" | How bearings relate |
| `z_fraction_1` | float | Position as fraction of L2 |
| `z_fraction_2` | float | Second bearing position (pairs) |
| `preload_class` | "LA" / "MA" / "HA" | Preload level (ACBB only) |

### Pair stiffness factors

| `sub_arrangement` | `stiffness_factor` | Rationale |
|-------------------|--------------------|-----------|
| `single` | 1.0 | One bearing |
| `DB` | 1.7 | SKF data: preloaded DB pair ≈ 1.7× single |
| `spacer_pair` (CRB) | 2.0 | Two independent CRBs in parallel |
| `DT` | 1.5 | Tandem shares axial, adds radial |

### Default arrangement (user-confirmed)

```
Station 1 (FRONT):  1 × ACBB single, z_frac = 0.25
Station 2 (REAR):   2 × CRB spacer_pair, z_frac = 0.70 – 0.80
```

**Engineering rationale:**
- Front ACBB handles combined axial + radial loads (locating bearing)
- Two rear CRBs with spacer give 2× radial stiffness and symmetric thermal expansion

---

## 6. `DesignSpace` — 19 Design Variables

### Geometry (9 variables)

| Variable | Nominal | Bounds | Tolerance | Notes |
|----------|---------|--------|-----------|-------|
| L1 | 122 mm | 100–150 | +0.5/0 | Nose length, IT12 |
| L2 | 405 mm | 350–450 | +0.5/0 | Journal length, IT12 |
| L3 | 24 mm | 15–40 | +0.1/0 | Flange (spacer), IT10 |
| L4 | 15 mm | 10–25 | +0.5/0 | Tail, IT12 |
| R1 | 45 mm | 35–55 | +0/−0.011 | Nose journal, ISO h5 |
| **R2** | **50 mm** | **35–60** | **+0/−0.013** | **Bearing seat, ISO h5** |
| R3 | 82.5 mm | 70–95 | ±0.05 | Flange OD |
| R4 | 45 mm | 35–55 | +0/−0.011 | Tail journal |
| ri | 30 mm | 25–35 | +0.021/0 | Inner bore, ISO H7 |

**R2 is the master bearing variable.** Its value drives `snap_to_skf_bearing()` which sets the actual bore, OD, stiffness, speed, and preload for the entire simulation.

**Why h5 for shaft, H7 for bore?**
ISO 286 recommends h5 / H6 for angular contact ball bearings under rotating inner ring (ISO fits for rolling bearings, ISO 15:2017 §6.3). The h5 shaft ensures a transition fit: 0 to −13 μm → always slight interference or exact fit, preventing inner ring creep.

### Bearings (4 variables)

| Variable | Nominal | Bounds | Notes |
|----------|---------|--------|-------|
| front_z_fraction | 0.25 | 0.15–0.35 | Front ACBB position |
| rear_z_fraction | 0.75 | 0.65–0.85 | CRB pair centroid |
| K_radial | 5×10⁵ N/mm | 2.5×10⁵–9×10⁵ | Fallback / surrogate |
| K_axial | 8×10⁵ N/mm | 4×10⁵–1.4×10⁶ | Fallback / surrogate |

K_radial and K_axial are kept as variables for surrogate model compatibility. At real evaluation, they are overwritten by catalog values.

### Material (3 variables)

| Variable | Nominal | Notes |
|----------|---------|-------|
| E | 2.1×10⁵ MPa | Batch variation ±2,000 MPa |
| ρ | 7.85×10⁻⁹ ton/mm³ | Alloy variation |
| σ_y | 655 MPa | Q&T scatter: **asymmetric** −15/+10 MPa |

The σ_y tolerance is intentionally asymmetric: quenching scatter is more likely to produce under-hardened parts (lower than target) than over-hardened.

### Loads (3 variables)

| Variable | Nominal | Notes |
|----------|---------|-------|
| Ft | 1,500 N | Tangential cutting force ±8% oscillation |
| Fr | 500 N | Radial cutting force |
| Ff | 300 N | Feed (axial) force |

Load tolerances represent **cutting force oscillation** due to chip variation and chatter. Used in the Monte Carlo robustness loop.

---

## 7. Key Helper Methods

### `sample_manufacturing_variation(x_nominal, n_samples)`

Draws `n_samples` Monte Carlo samples from the asymmetric tolerance distributions of all 19 variables simultaneously. Used in:
- Taguchi S/N ratio inner loop (robust optimizer)
- Selective assembly yield simulation

### `resolve_arrangement(x, n_rpm, arrangement)`

Given a design vector `x` and a `SpindleBearingArrangement`, calls `snap_to_skf_bearing()` for each station and returns a list of `(station, bearing, speed_ok, warning)` tuples. This is the single function that connects the optimizer's design vector to real SKF catalog data.

---

## Summary

```
DesignSpace (19 variables)
    │
    ├── 9 Geometry  → shaft dimensions (snap-to-catalog via R2)
    ├── 4 Bearings  → position fractions + stiffness fallbacks
    ├── 3 Material  → E, ρ, σ_y
    └── 3 Loads     → Ft, Fr, Ff

AsymmetricTolerance (+upper / −lower)
    └── Used in: Monte Carlo sampling, Selective Assembly, Taguchi S/N

SKF_ACBB_CATALOG (18 bearings, 7206–7224 BECBP)
SKF_CRB_CATALOG  (17 bearings, NU2206–NU2224 ECP)
    └── Both accessed via snap_to_skf_bearing()

SpindleBearingArrangement
    └── default_lathe(): 1 ACBB front + 2 CRB spacer-pair rear
```
