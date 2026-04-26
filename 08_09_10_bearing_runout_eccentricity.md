# Modules 08–10 — Bearing Mechanics, Shaft Runout, Rotor Eccentricity

---

# Module 08 — Bearing Performance Calculator
**File:** `08_bearing_performance.py`

---

## Purpose

Computes all **bearing-mechanics quantities** that are independent of ANSYS:

1. SKF catalog resolution per station
2. Static force distribution (propped-cantilever)
3. Pair stiffness (K_radial, K_axial)
4. Preload force and spacer delta
5. Speed and ndm checks
6. ISO 281 L10 life (per bearing + system combined)
7. ANSYS COMBIN14 spring table export

**Not in this module:** runout, eccentricity, vibration (moved to modules 09/10/ANSYS).

---

## 1. Force Distribution — Static Equilibrium

### Physical model

The spindle is treated as an **Euler-Bernoulli beam** with:
- Point load $F_r$ at the nose ($z = 0$)
- Two spring supports (front bearing at $z_f$, rear bearing centroid at $z_r$)

### Two-station case (default)

Taking moments about the rear bearing centroid:

$$R_{\text{front}} = F_r \cdot \frac{z_r}{z_r - z_f}$$

$$R_{\text{rear}} = F_r - R_{\text{front}}$$

**Important:** Because $z_r > z_f$ and $z_f > 0$ (nose to front bearing is the overhang), we get $R_{\text{front}} > F_r$. The front bearing is **overloaded** relative to the applied force. This is the classic cantilever overreach effect.

**Numerical example (nominal design):**
```
z_front = 223.2 mm,  z_rear = 425.8 mm
R_front = 1,581 × 425.8 / 202.5 = 3,324 N  (> 1,581 N applied)
R_rear  = 1,581 − 3,324 = −1,743 N  (upward reaction)
```

### Three-station case

If a third station is added (e.g., additional intermediate bearing), the rear group centroid is used for the front vs. group split, then the group reaction is distributed **proportionally to K_radial** of each station:

$$R_k = R_{\text{rear,total}} \times \frac{K_k}{\sum K_j}$$

### Axial load

The entire feed force $F_f$ is taken by the **front ACBB station only** (locating bearing). The rear CRBs (NU type) carry zero axial load.

---

## 2. Pair Stiffness

### ACBB — Angular Contact Ball Bearing

Single-bearing radial stiffness (empirical fit to SKF P5, MA preload data):

$$K_{r,\text{single}} \;[\text{N/mm}] = 5.5 \times d^{0.75} \times 1000$$

Single-bearing axial stiffness:

$$K_{a,\text{single}} = K_{r,\text{single}} \times \tan^2(\alpha) \quad \alpha = 25°, \; \tan^2(25°) = 0.2169$$

**DB pair (back-to-back):**

$$K_{r,\text{DB}} = 1.7 \times K_{r,\text{single}}$$

$$K_{a,\text{DB}} = 1.7 \times K_{a,\text{single}}$$

The 1.7 factor comes from SKF Spindle Bearings catalogue data: a preloaded DB pair has approximately 70% more stiffness than a single bearing due to the preload interaction between the two bearings.

### CRB — Cylindrical Roller Bearing

$$K_{r,\text{single}} \;[\text{N/mm}] = 8.0 \times d^{0.80} \times 1000$$

Axial stiffness: **0** (NU type, inner ring free axially).

Spacer-pair (two CRBs in parallel):

$$K_{r,\text{pair}} = 2.0 \times K_{r,\text{single}}$$

---

## 3. Preload Force & Spacer Delta

For ACBB with Medium (MA) preload, the preload force $F_{\text{pre}}$ is taken directly from the SKF catalog field `F_preload_MA_N`.

The required spacer length difference (between inner and outer spacers) to achieve this preload:

$$\Delta_{\text{spacer}} = \frac{F_{\text{pre}}}{K_{a,\text{pair}}} \quad [\text{mm}]$$

This is the value that Selective Assembly Module 07 needs to control with tight tolerance.

---

## 4. Speed and ndm Checks

### ndm factor

$$ndm = d_m \times n \quad [\text{mm} \cdot \text{RPM}]$$

$$d_m = \frac{d + D}{2} \quad [\text{mm}]$$

Limits per SKF Engineering Handbook, §2.4:

| Lubrication | ndm limit [mm·RPM] |
|-------------|-------------------|
| Grease | 500,000 |
| Oil-air mist | 1,000,000 |

**Constraint:** $ndm \leq ndm_{\text{limit}}$

### Speed check

Bearing speed limit depends on lubrication:
- Grease: $n \leq n_{\text{grease}}$
- Oil: $n \leq n_{\text{oil}}$

---

## 5. ISO 281 Bearing Life

### Equivalent Dynamic Load

**ACBB (25° contact angle):**

Ratio factor $e = 0.68$ (from SKF catalogue, 7200 BE series, 25°).

$$\text{If } \frac{F_a}{F_r} \leq e: \quad P = F_r$$

$$\text{If } \frac{F_a}{F_r} > e: \quad P = X_2 F_r + Y_2 F_a = 0.41 F_r + 0.87 F_a$$

Minimum load floor: $P \geq 0.01 \times C_r$ (prevent roller slippage at light load).

**CRB (NU type):**

$$P = F_r \quad \text{(no axial capacity)}$$

**Standard reference:** ISO 281:2007 §5.2 — Equivalent dynamic bearing load.

### Basic Rating Life

$$L_{10} \;[\text{rev}] = \left(\frac{C_r}{P}\right)^p \times 10^6$$

$$L_{10h} \;[\text{hours}] = \frac{L_{10}}{60 \, n}$$

Exponents per ISO 281:2007 §4:

| Bearing type | $p$ |
|-------------|-----|
| Ball (ACBB) | 3 |
| Roller (CRB) | 10/3 |

### System L10 — Combined Life

For a set of $n$ independent bearings, ISO 281 Annex C:

$$\frac{1}{L_{10,\text{sys}}^{e}} = \sum_{i=1}^{n} \frac{1}{L_{10,i}^{e}}$$

where $e = 10/9$ for ball bearings, $9/8$ for roller bearings.

When a system contains both types (ACBB front + CRB rear), the conservative choice $e = 10/9$ is used.

**Physical meaning:** The system fails when the weakest bearing fails. The system L10 is always less than the minimum individual L10.

**Standard reference:** ISO 281:2007 Annex C — System life of bearing arrangements.

---

## 6. ANSYS COMBIN14 Spring Table

The `stiffness_for_ansys()` method outputs one row per bearing z-position:

```
z_mm     K_radial [N/mm]   K_axial [N/mm]   bearing
223.2        173,925          37,819         7220 BECBP  (front ACBB)
405.5        318,486               0         NU2220 ECP  (rear CRB-1)
446.0        318,486               0         NU2220 ECP  (rear CRB-2)
```

In ANSYS MAPDL, these values feed the `COMBIN14` spring elements:

```apdl
ET, 10, COMBIN14          ! Spring element type
KEYOPT, 10, 1, 1          ! UX (radial)
R, 10, K_radial           ! Spring constant = K_radial

ET, 11, COMBIN14
KEYOPT, 11, 1, 3          ! UZ (axial)
R, 11, K_axial

! Create spring at z=223.2 mm (front bearing)
N, 9001, 0, 0, 223.2      ! Ground node
E, node_shaft_front, 9001  ! Spring element
```

The rear CRBs get K_axial = 0 (free axial DOF, representing the floating NU bearing).

---

## 7. Constraints Summary

| Constraint | Formula | Satisfied when |
|-----------|---------|----------------|
| Speed | $(n - n_{\text{lim}}) / n_{\text{lim}}$ | $\leq 0$ |
| ndm | $(ndm - ndm_{\text{lim}}) / ndm_{\text{lim}}$ | $\leq 0$ |
| L10 system | $(L_{10,\text{target}} - L_{10,\text{sys}}) / L_{10,\text{target}}$ | $\leq 0$ |
| Preload low | $(F_{\text{pre,min}} - F_{\text{pre}}) / F_{\text{pre,min}}$ | $\leq 0$ |
| Preload high | $(F_{\text{pre}} - F_{\text{pre,max}}) / F_{\text{pre,max}}$ | $\leq 0$ |

All constraints normalised → dimensionless → directly comparable in the GA penalty function.

---

---

# Module 09 — Shaft Runout Analysis
**File:** `09_shaft_runout.py`

---

## Purpose

Compute the **Total Indicated Runout (TIR)** at the spindle nose from all analytical sources, and combine with ANSYS elastic deflection to produce the **loaded runout** — the actual machining accuracy error.

---

## Physical Model: 4 Sources of Runout

### Source 1 — Bearing Inner Ring Runout (ISO 492)

The precision class of the bearing limits how eccentric the inner ring raceway can be relative to the bore.

**Standard reference:** ISO 492:2014 Table 4 — Radial runout of assembled bearing, inner ring.

| Class | Max inner ring runout |
|-------|-----------------------|
| P0 | 15 μm |
| P6 | 10 μm |
| **P5** | **5 μm** |
| P4 | 2.5 μm |
| P2 | 1.5 μm |

This eccentricity $\delta_{ir}$ at the front bearing is **amplified** as it propagates to the nose. For a rigid spindle pivoting about the rear bearing:

$$\text{TIR}_{\text{bearing}} = \delta_{ir} \times \left(1 + \frac{L_{\text{oh}}}{L_{\text{span}}}\right)$$

**Derivation:** If the front bearing inner ring has eccentricity $\delta_{ir}$, the shaft tilts. The nose is further from the pivot (rear bearing) than the front bearing by the lever ratio:

$$\frac{z_{\text{rear}}}{z_{\text{rear}} - z_{\text{front}}} = 1 + \frac{z_{\text{front}}}{z_{\text{rear}} - z_{\text{front}}} = 1 + \frac{L_{\text{oh}}}{L_{\text{span}}}$$

**Numerical example (nominal):**
```
δ_ir = 5 μm,  L_oh = 223 mm,  L_span = 202 mm
TIR_bearing = 5 × (1 + 223/202) = 5 × 2.10 = 10.51 μm
```

### Source 2 — Shaft Straightness Error (ISO 1101)

A real shaft has a **bow** bounded by the straightness tolerance $t_s$ [mm per mm of length].

**Standard reference:** ISO 1101:2017 — Geometrical tolerancing, Form tolerances.

For a half-sine bow mode on a shaft of length $L_{\text{total}}$:

$$f_{\text{bow}} = t_s \times \frac{L_{\text{total}}}{2} \quad [\text{mm}]$$

The contribution at the nose (linear interpolation):

$$\text{TIR}_{\text{straightness}} = f_{\text{bow}} \times \frac{L_{\text{oh}}}{L_{\text{total}} / 2} \times 1000 \quad [\mu\text{m}]$$

**Straightness grades used:**

| Grade | $t_s$ [mm/mm] | Surface finish |
|-------|--------------|---------------|
| standard | 4×10⁻⁵ | ISO 2768-1 fine |
| precision | 1×10⁻⁵ | P5 ground shaft |
| ultra | 5×10⁻⁶ | P4/P2 lapped |

### Source 3 — ANSYS Elastic Deflection

Under steady cutting forces, the spindle nose displaces laterally by $\delta_{\text{nose}}$. On every revolution, the cutting tool (fixed) traces this displacement as a sinusoidal error on the workpiece.

$$\text{TIR}_{\text{elastic}} = |\delta_{\text{nose,ANSYS}}| \quad [\mu\text{m}]$$

This is typically the **dominant source** in production machining. It is returned from ANSYS static analysis and overlaid here.

**Standard reference:** ISO 230-1:2012 §4.3.1.1 — Radial deviation under working conditions.

### Source 4 — Bending Slope Amplification

The elastic bending creates a **slope** $\theta$ at the front bearing. Over the overhang length, this slope generates an additional nose displacement:

$$\theta_{\text{front}} = \frac{F_r \times L_{\text{oh}}^2}{2 E I} \quad [\text{rad}]$$

$$\text{TIR}_{\text{slope}} = |\theta_{\text{front}} \times L_{\text{oh}}| \times 1000 \quad [\mu\text{m}]$$

where:
- $I = \frac{\pi}{4}(R_1^4 - r_i^4)$ — second moment of area at the nose cross-section
- $E$ — Young's modulus

This correction is most significant when $L_{\text{oh}}$ is large (long nose overhang).

---

## Combination Methods

### Linear (worst-case — for acceptance testing)

$$\text{TIR}_{\text{linear}} = \text{TIR}_{\text{bearing}} + \text{TIR}_{\text{straight}} + \text{TIR}_{\text{elastic}} + \text{TIR}_{\text{slope}}$$

All errors assumed co-directional. Use for manufacturing acceptance limits.

### RSS (probabilistic — for design optimisation)

$$\text{TIR}_{\text{RSS}} = \sqrt{\text{TIR}_{\text{bearing}}^2 + \text{TIR}_{\text{straight}}^2 + \text{TIR}_{\text{elastic}}^2 + \text{TIR}_{\text{slope}}^2}$$

Errors assumed statistically independent in direction. Use for the GA objective.

---

## Two-Phase Usage (Pre-FEA and Post-FEA)

```python
# Phase 1: Before ANSYS (geometric feasibility check)
bd = analyser.analyse(var_dict, z_front, z_rear, delta_nose_ansys_um=0.0)
# bd.TIR_without_ansys_um = sqrt(bearing² + straight²)

# Phase 2: After ANSYS returns deflection
bd = analyser.analyse(var_dict, z_front, z_rear, delta_nose_ansys_um=35.0)
# bd.TIR_loaded_um = sqrt(bearing² + straight² + elastic² + slope²)
```

This two-phase approach allows the GA to use the fast analytical pre-FEA runout as an initial filter, then refine with ANSYS results.

---

## Constraints

| Constraint | Formula | Limit |
|-----------|---------|-------|
| Geometric TIR | $(\text{TIR}_{\text{geom}} - \text{limit}) / \text{limit}$ | 10 μm |
| Loaded TIR | $(\text{TIR}_{\text{loaded}} - \text{limit}) / \text{limit}$ | 15 μm |

---

---

# Module 10 — Rotor Eccentricity Analysis
**File:** `10_rotor_eccentricity.py`

---

## Purpose

Compute the **Centre of Gravity (CG)** of the stepped hollow spindle shaft, derive the **static eccentricity** (radial offset of CG from the rotation axis), and compute the **imbalance force** that ANSYS uses as excitation input in harmonic analysis.

**This module does NOT compute vibration amplitude.** That is ANSYS's job.

---

## Physical Model

### Segment CG positions

For 4 hollow-cylinder segments, indexed $i = 1 \ldots 4$:

**Segment mass:**
$$m_i = \rho \times \pi \times (R_i^2 - r_i^2) \times L_i \quad [\text{ton}]$$

$$m_{i,\text{kg}} = m_i \times 1000 \quad [\text{kg}]$$

**Axial CG of segment** (measured from spindle nose $z = 0$):
$$z_{CG,i} = z_{i,\text{start}} + \frac{L_i}{2} \quad [\text{mm}]$$

### System axial CG

$$z_{CG} = \frac{\sum_i m_{i,\text{kg}} \times z_{CG,i}}{\sum_i m_{i,\text{kg}}} \quad [\text{mm}]$$

---

## Bore Eccentricity → Radial CG Shift

### Physical mechanism

The hollow bore is machined in a separate operation. If the bore centre is offset from the outer surface axis by $\varepsilon_{\text{bore}}$ [mm], the hollow annulus CG shifts radially.

### Derivation

Consider a hollow segment (outer radius $R$, inner radius $r$, bore offset $\varepsilon$). By superposition:

$$m_{\text{hollow}} \times e_{i} = -m_{\text{bore}} \times \varepsilon$$

where:
- $m_{\text{bore}} = \rho \pi r^2 L$ — mass of the "removed" bore material
- $m_{\text{hollow}} = \rho \pi (R^2 - r^2) L$ — hollow segment mass

Solving:

$$\boxed{e_i = \varepsilon_{\text{bore}} \times \frac{r_i^2}{R_i^2 - r_i^2}} \quad [\text{mm}]$$

**Physical interpretation:** The larger the bore relative to the wall, the more sensitive the CG is to bore eccentricity. A thin-walled section (large $r/R$) shifts CG more per unit of bore offset.

**Numerical example (nose segment: R=45, r=30):**
$$e_{\text{nose}} = 0.005 \times \frac{900}{2025 - 900} = 0.005 \times \frac{900}{1125} = 0.004 \; \text{mm} = 4.0 \; \mu\text{m}$$

---

## System Eccentricity

### Worst-case (all bore offsets in same direction)

$$e_{\text{static}} = \frac{\sum_i m_{i,\text{kg}} \times e_i}{\sum_i m_{i,\text{kg}}} \quad [\text{mm}]$$

Use for conservative ANSYS excitation input.

### RSS (bore offsets random in direction)

$$e_{\text{RSS}} = \frac{\sqrt{\sum_i (m_{i,\text{kg}} \times e_i)^2}}{\sum_i m_{i,\text{kg}}} \quad [\text{mm}]$$

Use for probabilistic design analysis.

---

## Static Imbalance

### Imbalance vector U

$$U = m_{\text{total}} \times e_{CG} \times 1000 \quad [\text{g·mm}]$$

(converting kg → g and keeping mm → g·mm is the standard imbalance unit)

### Allowable residual imbalance — ISO 1940-1

For balance quality grade G:

$$U_{\text{allow}} = m_{\text{total}} \;[\text{kg}] \times \frac{G \;[\text{mm/s}]}{\omega \;[\text{rad/s}]} \times 1000 \quad [\text{g·mm}]$$

**Standard reference:** ISO 1940-1:2003 §5 — Balance quality grades.

Grades for reference:

| Grade | G [mm/s] | Typical application |
|-------|----------|---------------------|
| G1 | 1.0 | Turbines, gyroscopes |
| **G2.5** | **2.5** | **Machine tool spindles** |
| G6.3 | 6.3 | General machining |

**At 4,000 RPM (ω = 418.9 rad/s):**
$$e_{\text{allow}} = \frac{2.5}{418.9} = 0.00597 \; \text{mm} = 5.97 \; \mu\text{m}$$

For nominal spindle mass 23.3 kg:
$$U_{\text{allow}} = 23.3 \times 5.97 \times 1000 = 139{,}000 \; \text{g·mm}$$

The nominal design has $e_{\text{static}} = 2.70 \; \mu\text{m}$, giving $U = 62,800 \; \text{g·mm}$ → excess factor 0.45× → no balancing required.

---

## Imbalance Force — ANSYS Input

### Rotating imbalance force

$$F_{\text{imbal}} = m_{\text{total}} \;[\text{kg}] \times e_{CG} \;[\text{m}] \times \omega^2 \;[\text{rad}^2/\text{s}^2] \quad [\text{N}]$$

Note unit conversion: $e_{CG}$ in mm × 10⁻³ → m.

**At 4,000 RPM:**
$$F = 23.3 \times (2.70 \times 10^{-6}) \times 418.9^2 = 11.0 \; \text{N}$$

### ANSYS application

The imbalance force is a **rotating force** applied at the node nearest to the axial CG position:

```apdl
/PREP7
NSEL, S, LOC, Z, 322.58   ! select node at z_CG
*GET, N_CG, NODE, 0, NUM, MIN
F, N_CG, FX, 11.0211      ! amplitude in X (phase = 0)
NSEL, ALL
ALLSEL
```

In ANSYS Harmonic Analysis (HARFRQ), this single-amplitude force becomes the rotating excitation. ANSYS returns the complex response (amplitude + phase) at all nodes, including the nose deflection which feeds back to Module 09 as `delta_nose_ansys_um`.

---

## Integration Diagram

```
Module 10 Output               ANSYS Input
─────────────────────────────────────────────────────
F_imbalance_N = 11.0 N  ──→   F command at z_CG node
z_CG = 322.6 mm          ──→   Node selection
ω = 418.9 rad/s           ──→   HARFRQ frequency sweep

ANSYS Output               Module 09 Input
─────────────────────────────────────────────────────
δ_nose_ANSYS = 35 μm  ──→   delta_nose_ansys_um argument
```

---

## Constraints

| Constraint | Formula | Physical meaning |
|-----------|---------|-----------------|
| Imbalance | $(U - U_{\text{allow}}) / U_{\text{allow}}$ | Must not exceed G2.5 residual limit |
| Force | $(F_{\text{imbal}} - F_{\text{limit}}) / F_{\text{limit}}$ | Imbalance force must not over-excite bearings |

---

## Sensitivity Analysis

The following relationships govern design decisions:

| Parameter increases | Effect on $e_{\text{static}}$ | Effect on $F_{\text{imbal}}$ |
|--------------------|------------------------------|------------------------------|
| Bore offset $\varepsilon$ | Proportionally ↑ | ↑ |
| Inner radius $r_i$ | ↑ (via $r^2/(R^2-r^2)$) | ↑ |
| Operating speed $n$ | No effect | $\propto \omega^2$ ↑ |
| Rotor mass $m$ | ↓ (heavier outer wall dilutes) | ↑ |

The most effective way to reduce $F_{\text{imbal}}$ is to **reduce bore eccentricity** $\varepsilon$ (tighter boring tolerance or honing) or to **reduce bore-to-wall ratio** $r/R$ (thicker walls at high-speed sections).
