# Modules 05–07 — Robust Optimization, Inverse Design, Selective Assembly

---

# Module 05 — Robust Optimizer
**File:** `05_robust_optimizer.py`

---

## Purpose

Find the **optimal AND robust** spindle design using a Genetic Algorithm
with Taguchi's Signal-to-Noise (S/N) ratio as the robustness objective.

"Robust" means the design performs well not just at the nominal dimensions
but also when those dimensions vary within their manufacturing tolerances.
A non-robust design might be optimal on paper but degrade badly in production.

---

## 1. The Four Objectives

All four are **minimised** simultaneously:

| # | Objective | Symbol | Formula | Physical meaning |
|---|-----------|--------|---------|-----------------|
| f1 | Deflection robustness | $-\eta_\delta$ | $-\text{S/N}_{\text{deflection}}$ | Larger S/N = more robust deflection |
| f2 | Stress robustness | $-\eta_\sigma$ | $-\text{S/N}_{\text{stress}}$ | Larger S/N = more robust stress |
| f3 | Total cost | $C_{\text{total}}$ | Material + machining + bearings + SA | Lower cost = better |
| f4 | Weight | $m$ | $\rho \sum \pi(R_i^2 - r_i^2) L_i$ | Lower mass = better |

Minimising $-\eta$ is equivalent to maximising $\eta$ (S/N ratio).

---

## 2. Taguchi Signal-to-Noise Ratio — `taguchi_sn_ratio()`

### Background

Genichi Taguchi's key insight: optimising only the **mean** response is
insufficient.  A design that achieves the target on average but varies widely
under manufacturing scatter is unacceptable.  The S/N ratio combines mean
and variance into a single robustness metric:

$$\eta = f(\mu_y, \sigma_y) \quad [\text{dB}]$$

A higher S/N ratio means the desired signal (performance) is strong relative
to the noise (variability from tolerances).

### Smaller-is-Better (used for deflection and stress)

$$\eta = -10 \log_{10}\left(\frac{1}{n} \sum_{i=1}^{n} y_i^2\right) \quad [\text{dB}]$$

This form penalises both high mean AND high variance simultaneously.
If all $y_i$ are small and consistent, $\eta$ is large (close to 0 or positive).

**Standard reference:** Taguchi, G. "Introduction to Quality Engineering"
(1986), Asian Productivity Organisation, Chapter 3.

### Inner-loop Monte Carlo

For each candidate design $\mathbf{x}_0$ evaluated by the GA:

1. Generate $n_{\text{MC}}$ perturbed designs by sampling from the
   asymmetric manufacturing tolerance distributions:
   $$\mathbf{x}_k = \mathbf{x}_0 + \boldsymbol{\varepsilon}_k, \quad
     \boldsymbol{\varepsilon}_k \sim \mathcal{N}(0, \boldsymbol{\sigma}_{\text{tol}})$$
   where $\sigma_j = (\text{upper}_j + \text{lower}_j) / 6$

2. Clip to hard bounds: $\mathbf{x}_k \in [\mathbf{l}, \mathbf{u}]$

3. Evaluate all $n_{\text{MC}}$ designs via the ML surrogate
   (not ANSYS — this loop runs thousands of times)

4. Compute S/N ratio from the $n_{\text{MC}}$ responses

**Bug-9 fix — epsilon guard:**
The original code could produce $-\infty$ for "larger-is-better" S/N when
$y \approx 0$:
```python
# BUG: log10(1/y²) → -inf when y → 0
sn = -10 * np.log10(np.mean(1.0 / y**2))

# FIX:
y_c = np.maximum(np.abs(y), 1e-10)   # guard before log
sn  = -10 * np.log10(np.mean(1.0 / y_c**2))
```

---

## 3. Bug-6 Fix — Output Index Resolution

### The bug

v1 accessed surrogate outputs by hardcoded key name:
```python
sn_results.get("static_max_deflection_um", {}).get("sn_ratio", 0)
```

If the surrogate was trained with columns named differently
(e.g., `"deflection_um"` instead of `"static_max_deflection_um"`),
the `.get()` silently returned `{}` → S/N ratio = 0 for every candidate.
The GA was optimising a perfectly flat landscape — producing random
"optimal" designs.

### The fix

Output indices are resolved **once at construction time**:

```python
omap = output_name_map or _DEFAULT_OUTPUT_MAP
for role, col in omap.items():
    if col in surrogate.output_names:
        self._idx[role] = surrogate.output_names.index(col)
    else:
        raise ValueError(f"Surrogate does not contain: {col}")
```

Now the optimizer fails immediately with a clear error message if the
surrogate and optimizer are misconfigured, rather than silently producing
wrong results.

---

## 4. Differential Evolution — `optimize_de()`

Scipy's `differential_evolution` is a single-objective GA used when a
weighted sum of the four objectives is acceptable:

$$f_{\text{scalar}} = \mathbf{w}^\top \frac{\mathbf{f}}{\bar{\mathbf{f}}}$$

Default weights: $\mathbf{w} = [0.30, 0.30, 0.25, 0.15]$

Normalisation $\bar{\mathbf{f}}$ is computed from a 2-point pilot evaluation
at the nominal design and the bounds midpoint.  This prevents one objective
from dominating due to scale differences (e.g., cost in USD vs. S/N in dB).

**Bug-8 fix:** Weight vector length was hardcoded to 3.  Now built
dynamically from `self.n_objectives`:
```python
w = np.array([0.30, 0.30, 0.25, 0.15][:n_obj])
w = w / w.sum()
```

### Algorithm parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Strategy | `best1bin` (scipy default) | Robust for smooth landscapes |
| Population size | 15 × n_vars = 285 (scipy default) | Adequate coverage |
| Mutation | [0.5, 1.0] (scipy default) | Exploration-exploitation balance |
| Recombination | 0.7 (scipy default) | Standard |

---

## 5. NSGA-II — `optimize_nsga2()`

NSGA-II (Non-dominated Sorting Genetic Algorithm II) is the standard
multi-objective GA that produces a **Pareto front** — the full set of
trade-off solutions where no objective can be improved without worsening
another.

### Algorithm

1. Initialize random population of size `pop_size`
2. Evaluate fitness: $\mathbf{f}(\mathbf{x}) \in \mathbb{R}^4$
3. Non-dominated sorting: rank solutions by Pareto dominance
4. Crowding distance: preserve diversity within each front
5. Selection, crossover, mutation → offspring
6. Combine parents + offspring, truncate to `pop_size`
7. Repeat for `n_gen` generations

**Standard reference:** Deb, K. et al. "A Fast and Elitist Multiobjective
Genetic Algorithm: NSGA-II." IEEE Transactions on Evolutionary Computation,
6(2), 182–197, 2002.

### Bug-7 fix — dynamic n_obj

v1 hardcoded `n_obj=3` in the pymoo `Problem` subclass.  Adding the 4th
cost objective broke the shape of the fitness matrix.  Fixed:

```python
class _Problem(Problem):
    def __init__(self):
        super().__init__(
            n_var = outer.n_vars,
            n_obj = outer.n_objectives,   # BUG-7 FIX: not hardcoded 3
            xl    = outer.bounds[:, 0],
            xu    = outer.bounds[:, 1],
        )
```

### Pareto front selection

After NSGA-II, the single "best" design is selected by weighted-sum on the
normalised Pareto front:

```python
scale    = np.abs(pareto_F).mean(axis=0)
best_idx = np.argmin((pareto_F / scale) @ w)
```

This is a **post-hoc preference articulation** — the engineer can change
the weights after seeing the Pareto front to select a different trade-off.

---

## 6. Complete Report — `report_best()`

Returns a nested dictionary containing:

```python
{
    "design_variables": {L1: 122.3, R2: 47.8, ...},
    "objectives": {
        "−S/N_deflection": -34.2,
        "−S/N_stress":     -31.7,
        "Total cost (USD)": 842.5,
        "Weight (kg)":       23.1,
    },
    "costs": {
        "material_usd":  145.2,
        "machining_usd": 312.8,
        "bearings_usd":  220.0,
        "sa_usd":         164.5,
        "total_usd":      842.5,
    },
    "selective_assembly": {
        "Journal_InnerRace": {
            "improvement_x": 4.2,
            "std_with_sa_um": 2.1,
            "std_no_sa_um": 8.8,
            "yield_pct": 87.4,
        },
        ...
    }
}
```

---

---

# Module 06 — Inverse Design Engine
**File:** `06_inverse_design.py`

---

## Purpose

Perform **inverse design**: given desired performance targets, predict the
required spindle dimensions.

$$\text{Forward:} \quad \mathbf{x} \xrightarrow{\text{FEA}} \mathbf{y}$$

$$\text{Inverse:} \quad \mathbf{y}^* \xrightarrow{\text{DNN}} \hat{\mathbf{x}}$$

**Use case:** The engineer says "I need deflection < 8 μm, stress < 350 MPa,
and first frequency > 800 Hz.  What geometry achieves this?"

The inverse model answers in milliseconds rather than requiring the engineer
to manually iterate through the optimizer.

---

## 1. Why Inverse Design Is Hard

The forward map $\mathbf{x} \rightarrow \mathbf{y}$ is a many-to-one
function: many different geometries can achieve the same performance.  This
**non-uniqueness** means the inverse problem is ill-posed.

The DNN handles this by learning the **mean** inverse mapping — the most
likely geometry that produces the target performance given the training
distribution.  It will not return all possible solutions, but it will return
one geometrically sensible solution close to the training data distribution.

---

## 2. Training Data — Role Swap

The inverse DNN is trained with the roles of $\mathbf{x}$ and $\mathbf{y}$
swapped:

```python
# Forward surrogate:   X_design  → y_performance
# Inverse DNN:         y_performance → X_design  (inputs and outputs swapped)

X_train_dnn = y_performance   # shape (n_samples, n_perf_vars)
y_train_dnn = X_design         # shape (n_samples, n_design_vars)
```

The same FEA pool data feeds both the forward surrogate and the inverse DNN.

---

## 3. DNN Architecture

```
Input  (n_perf = 3)
   ↓
Dense(128, ReLU)  →  Dropout(0.2)
   ↓
Dense(256, ReLU)  →  Dropout(0.2)
   ↓
Dense(128, ReLU)  →  Dropout(0.2)
   ↓
Dense(64, ReLU)
   ↓
Output (n_design = 19, Linear)
```

**Dropout (0.2):** 20% of neurons randomly zeroed during training.  Prevents
overfitting by forcing the network to learn redundant representations.

**ReLU activation:** $\text{ReLU}(z) = \max(0, z)$.  Avoids the vanishing
gradient problem of sigmoid/tanh for deep networks.

**Linear output:** No activation on the output layer — we need unbounded
real-valued predictions for the design variables.

### Optimizer and callbacks

- **Adam** with initial learning rate 0.001
- **ReduceLROnPlateau**: halves learning rate when validation loss plateaus
  for 10 epochs (minimum lr = 10⁻⁶)
- **EarlyStopping**: restores best weights when validation loss does not
  improve for 20 epochs

---

## 4. Bug-10 Fix — Missing `List` Import

v1 used `List[str]` in the `train()` signature but never imported `List`
from `typing`:

```python
# BUG: NameError at runtime
def train(self, X_design, y_performance, performance_names: List[str] = None):

# FIX:
from typing import Dict, List, Optional, Tuple   # List now present
def train(self, X_design, y_performance, performance_names: Optional[List[str]] = None):
```

---

## 5. Prediction and Validation

```python
target = {"deflection_um": 8.0, "stress_MPa": 350.0, "frequency_Hz": 800.0}
x_pred = engine.predict_design(target)
```

After prediction, the result is validated using the **forward surrogate**:

```python
y_achieved = forward_surrogate.predict(x_pred.reshape(1, -1))
# y_achieved ≈ [8.3 μm, 342 MPa, 812 Hz]
```

The validation table shows target vs. achieved vs. relative error for each
performance variable.

**Bounds clipping:** Predicted design variables are hard-clipped to the
design space bounds, ensuring the returned geometry is physically realizable
even if the DNN extrapolates slightly.

---

## 6. sklearn Fallback

When TensorFlow is not installed, the module falls back to
`sklearn.neural_network.MLPRegressor` with the same layer structure.
The fallback is less accurate (no dropout, less flexible training control)
but fully functional.

---

---

# Module 07 — Selective Assembly
**File:** `07_selective_assembly.py`

---

## Purpose

Model the **selective assembly** manufacturing strategy for the three critical
mating interfaces of the spindle, compute the assembly quality improvement,
and feed the SA cost into the GA as objective f3.

Selective assembly: instead of machining all parts to tight tolerances
(expensive), parts are machined to wider tolerances (cheaper), measured,
sorted into bins, and **matched** by bin number so the assembly gap stays
tight.

---

## 1. The Three Interfaces

| # | Interface | Gap nominal | Effect on performance |
|---|-----------|------------|----------------------|
| 1 | Shaft journal ↔ Bearing inner race | +4 μm (clearance) | Radial runout |
| 2 | Inner spacer ↔ Outer spacer | $F_{\text{pre}} / K_a$ | Axial preload scatter |
| 3 | Housing bore ↔ Bearing outer ring | −8 μm (interference) | Outer ring micro-creep |

---

## 2. Binning Algorithm — `_assign_bins()`

Each part is assigned to bin $b \in \{0, 1, \ldots, n_{\text{bins}}-1\}$:

$$b_i = \left\lfloor \frac{d_i - d_{\min}}{d_{\max} - d_{\min}} \times n_{\text{bins}} \right\rfloor$$

clipped to $[0, n_{\text{bins}}-1]$.

A shaft in bin $k$ is paired with a housing in bin $k$.  Within the matched
pair, the assembly gap is:

$$\Delta_{\text{pair}} = d_{\text{housing,k}} - d_{\text{shaft,k}}$$

### Gap statistics

Without SA (random pairing):
$$\sigma_{\text{gap,no-SA}} = \sqrt{\sigma_{\text{shaft}}^2 + \sigma_{\text{housing}}^2}$$

With SA ($n$ bins):
$$\sigma_{\text{gap,SA}} \approx \frac{\sigma_{\text{gap,no-SA}}}{n}$$

In practice the simulation directly measures this from the sampled
`matched_gaps` array.

**Improvement ratio:**
$$\text{Improvement} = \frac{\sigma_{\text{no-SA}}}{\sigma_{\text{SA}}}$$

Ideally equals $n_{\text{bins}}$ (perfect linear scaling).  Actual value
is slightly less due to unequal bin populations.

---

## 3. Match Yield

Not every shaft finds a compatible housing in its bin — if the batch sizes
are unequal, some parts are left unmatched.

$$\text{Yield} = \frac{\text{matched pairs}}{n_{\text{parts total}}}$$

Unmatched parts require rework (remachining to fit a different bin) which
adds SA cost.

---

## 4. Cost Model — `SpindleCostModel`

### Material cost

$$C_{\text{material}} = V_{\text{billet}} \times \rho \times 1000 \times P_{\text{alloy}}$$

where:
- $V_{\text{billet}} = \pi R_{\max}^2 \times L_{\text{total}} \times 1.3$ (30% machining stock)
- $\rho$ in ton/mm³ × $10^9$ → kg/m³, then convert to kg
- $P_{\text{alloy}}$ = alloy price in USD/kg (4140: \$3.5, 4340: \$5.5, EN24: \$4.5)

### Machining cost

Models the tolerance-dependent premium: tighter tolerance = exponentially
higher machining cost.

For each feature (journal OD, bore ID, flange face, outer profile):

$$C_{\text{feature}} = \text{rate} \times \frac{A}{10000} \times (1 + e^{-2 \text{tol}/0.05})$$

where:
- `rate` = turning rate (\$5/mm-tol) or grinding rate (\$80/mm-tol)
  depending on whether `tol < 0.02 mm` (grinding threshold)
- $A$ = feature surface area in mm² (normalised to cm²)
- The exponential factor adds a tolerance premium as `tol → 0`

### Bearing cost

Empirical power-law fit to SKF/FAG price lists for ACBB series:

$$C_{\text{bearings}} = 2 \times C_{\text{base}} \times \left(\frac{K_{\text{radial}}}{K_{\text{ref}}}\right)^{0.6}$$

where:
- $C_{\text{base}} = \$220$ per pair (basic ACBB reference)
- $K_{\text{ref}} = 5 \times 10^5$ N/mm (reference stiffness)
- Exponent 0.6 fitted to catalogue prices across the 7200 series

A higher stiffness bearing (achieved through larger bore or higher preload
class) costs more — this is a key trade-off the GA explores.

### Selective assembly cost

Per interface, per spindle unit:

$$C_{\text{SA}} = \underbrace{C_{\text{CMM}} \times 2}_{\text{measure shaft + housing}} + \underbrace{C_{\text{sort}} \times n_{\text{bins}}}_{\text{bin sorting}} + \underbrace{C_{\text{rework}} \times (1 - \text{yield})}_{\text{unmatched parts}}$$

Parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| $C_{\text{CMM}}$ | \$8 | CMM measurement per part |
| $C_{\text{sort}}$ | \$2/bin | Sorting labour per bin per part |
| $C_{\text{rework}}$ | \$45 | Rework cost per unmatched part |

**Total SA cost = sum over all 3 interfaces.**

### Total cost

$$C_{\text{total}} = C_{\text{material}} + C_{\text{machining}} + C_{\text{bearings}} + C_{\text{SA}}$$

This feeds directly into GA objective f3.

---

## 5. Design Trade-off Modelled

The SA module captures the key trade-off the optimizer must navigate:

```
Wider machining tolerance
    → Lower machining cost  ✅
    → More SA stages needed → Higher SA cost  ❌
    → Larger gap scatter without SA  ❌

Tighter machining tolerance
    → Higher machining cost  ❌
    → Less SA needed → Lower SA cost  ✅
    → Better fit quality directly  ✅
```

The GA explores this Pareto trade-off automatically by simultaneously
minimising f3 (total cost) and f1/f2 (robustness S/N ratios).

---

## 6. Interface Details

### Interface 1: Journal ↔ Bearing inner race

- Target gap: +4 μm (clearance for easy assembly)
- Shaft tolerance: h5 → upper=0, lower=13 μm (always undersized)
- Bearing bore: H6 → upper=22 μm, lower=0 (always oversized)
- Without SA at 5 bins: $\sigma_{\text{gap}} \approx 8$ μm
- With SA at 5 bins: $\sigma_{\text{gap}} \approx 2$ μm → 4× improvement

### Interface 2: Inner spacer ↔ Outer spacer

Controls axial preload through the DB bearing pair.  The target delta:

$$\Delta_{\text{target}} = \frac{F_{\text{preload}}}{K_{\text{axial,pair}}}$$

For $F_{\text{pre}} = 1200$ N and $K_a = 37,800$ N/mm:
$$\Delta_{\text{target}} = 1200 / 37800 = 0.032 \text{ mm} = 32 \; \mu\text{m}$$

SA ensures this delta is held to ±1–2 μm, giving predictable preload and
thus predictable bearing stiffness.

### Interface 3: Housing bore ↔ Bearing outer ring

- Target: −8 μm interference (slight press fit)
- SA prevents over-interference (damage to outer ring) and under-interference
  (outer ring creep under cyclic load)
