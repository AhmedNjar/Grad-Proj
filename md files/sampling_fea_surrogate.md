# Modules 02–04 — Sampling, FEA Pool, ML Surrogate

---

# Module 02 — LHS Sampler
**File:** `02_lhs_sampler.py`

---

## Purpose

Generate **space-filling design samples** for the DoE (Design of Experiments)
phase of the RDO workflow.  A good sample set covers the 19-dimensional design
space uniformly so the subsequent ML surrogate can interpolate accurately
between sample points.

Three sampling strategies are implemented:

| Method | When to use |
|--------|------------|
| Latin Hypercube (LHS) | Primary DoE for ML training (best coverage) |
| Sobol sequence | High-dimensional spaces, power-of-2 sample counts |
| Monte Carlo (normal) | Manufacturing variation simulation |
| Hybrid | LHS nominal + MC perturbations for robustness |

---

## 1. Latin Hypercube Sampling — `generate_lhs()`

### Concept

LHS partitions each dimension into $n$ equal intervals and places exactly one
sample per interval per dimension.  This guarantees no two samples share the
same "row" or "column" in any 2D projection — unlike pure random sampling
which can cluster.

For $n = 6$ samples and $d = 2$ dimensions:

```
  Dim-2
  │  ·
  │        ·
  │  · 
  │              ·
  │        ·
  │  ·
  └─────────────── Dim-1
    each column & row has exactly 1 sample
```

### scipy implementation

`scipy.stats.qmc.LatinHypercube` generates samples in $[0,1]^d$, then
`_scale()` maps them to actual variable bounds:

$$x_{ij} = \text{lower}_j + u_{ij} \times (\text{upper}_j - \text{lower}_j)$$

where $u_{ij} \in [0,1]$ is the normalised LHS sample.

### Criterion mapping — Bug-4 fix

The original code passed pyDOE2 criterion strings directly to scipy, causing
`ValueError` at runtime.  The fix is an explicit mapping table:

| User-facing name | scipy `optimization=` | Behaviour |
|-----------------|----------------------|-----------|
| `"maximin"` | `"random-cd"` | Maximise minimum inter-point distance (best for ML) |
| `"center"` | `None` | Plain LHS, cell-centred |
| `"centermaximin"` | `"lloyd"` | Lloyd's Voronoi tessellation |
| `"correlation"` | `"random-cd"` | Minimise linear correlation |

**Lloyd / centermaximin caveat:** Voronoi tessellation requires
$n_{\text{samples}} > n_{\text{vars}}$.  When this condition is violated
(e.g., quick 10-sample test on a 19-variable space), the code automatically
falls back to `"random-cd"` with a warning rather than crashing with a
`QhullError`.

### Recommendation for ML training

Use `criterion="maximin"` with at least $10 \times n_{\text{vars}}$ samples:

$$n_{\text{min}} = 10 \times 19 = 190 \text{ samples}$$

For high-fidelity surrogate (target R² > 0.97): 500 samples.

---

## 2. Sobol Quasi-Random Sequence — `generate_sobol()`

Sobol sequences are **quasi-random** (low-discrepancy) sequences that fill
space more uniformly than pseudo-random numbers, especially in high dimensions.

**Key property:** The star discrepancy $D^*$ of a Sobol sequence of $n$
points in $d$ dimensions satisfies:

$$D^* \sim \frac{(\log n)^d}{n}$$

vs. Monte Carlo: $D^* \sim \sqrt{d/n}$ — Sobol wins for $n \gg 2^d$.

**Best practice:** Use $n = 2^k$ samples (64, 128, 256, …) for best
uniformity.  Owen scrambling (`scramble=True`) removes the fixed-pattern
artefacts of raw Sobol.

**When to prefer Sobol over LHS:**
- $n_{\text{vars}} > 15$ and $n_{\text{samples}} > 256$
- When the DoE will be extended later (Sobol sequences extend cleanly)

---

## 3. Monte Carlo Sampling — `generate_montecarlo()`

### Uniform distribution

Pure random sampling in $[\text{lower}, \text{upper}]^d$.  Used only for
exploratory analysis; LHS or Sobol is always preferable for DoE.

### Normal distribution (manufacturing variation)

Samples centred at the **nominal design** with spread derived from the
asymmetric manufacturing tolerances:

$$\sigma_j = \frac{\text{upper}_j + \text{lower}_j}{6}$$

(3σ = total tolerance band → Cpk = 1.0 process assumption)

The mean is shifted to the mid-band of the asymmetric tolerance:

$$\mu_j = x_{j,\text{nominal}} + \frac{\text{upper}_j - \text{lower}_j}{2}$$

Samples are hard-clipped to `[lower_j, upper_j]`.

**Use case:** Feed this into the Taguchi inner-loop Monte Carlo to estimate
$\sigma_y$ (response standard deviation under manufacturing variation) for
the S/N ratio calculation.

---

## 4. Hybrid Sampling — `generate_hybrid()`

Combines LHS (for exploring the design space) with Monte Carlo perturbations
(for evaluating robustness of each LHS point):

```
Step 1: Generate n_lhs LHS points  →  X_nominal  (n_lhs × n_vars)
Step 2: For each LHS point, generate n_mc_per_lhs normal perturbations
        →  X_perturbed  (n_lhs × n_mc_per_lhs × n_vars)
```

**Returns:** `(X_nominal, X_perturbed)` as `Tuple[np.ndarray, np.ndarray]`

**Bug-5 fix:** Return type annotation was `tuple[...]` (Python 3.9+ only).
Fixed to `Tuple` from `typing` for Python 3.8 compatibility.

**Use case in RDO workflow:**
- Train ML surrogate on `X_nominal` (space-filling)
- Evaluate Taguchi S/N robustness on `X_perturbed` (manufacturing scatter)

---

## 5. Scale and Save

### `_scale()` — Normalised to actual bounds

$$x_{ij}^{\text{actual}} = \text{lower}_j + u_{ij} \times (\text{upper}_j - \text{lower}_j)$$

All sampling methods generate $u \in [0,1]$ first, then call `_scale()`.

### `save_samples()` — CSV and JSON output

```python
sampler.save_samples(X, "doe_samples/lhs_200.csv", fmt="csv")
sampler.save_samples(X, "doe_samples/lhs_200.json", fmt="json")
```

CSV uses variable names as column headers.  JSON produces a list of dicts.

---

## 6. Recommended Sample Counts

| Goal | Method | n_samples |
|------|--------|-----------|
| Quick test / dry run | LHS maximin | 20–50 |
| ML training (good) | LHS maximin | 200 |
| ML training (excellent, R²>0.97) | LHS maximin | 500 |
| Robustness verification | Hybrid (20 LHS × 20 MC) | 400 total |
| Full production DoE | Sobol 2^9 | 512 |

---

---

# Module 03 — FEA Pool Runner
**File:** `03_fea_pool_runner.py`

---

## Purpose

Execute parametric FEA simulations in batch, converting design vectors from
the LHS sampler into `SimulationCase` objects and feeding them to ANSYS
MAPDL.  Also provides an **analytical dry-run mode** for rapid prototyping
without an ANSYS licence.

---

## 1. Design Vector → SimulationCase Conversion

The `_vector_to_simulation_case()` method maps the continuous optimizer
vector to the PyMAPDL dataclass hierarchy:

```
x[19]  →  SpindleGeometry(L1, L2, L3, L4, R1, R2, R3, R4, ri)
       →  MaterialProperties(E, rho, sigma_y, …)
       →  BearingConfig(front_z_fraction, rear_z_fraction, K_radial, K_axial)
       →  CuttingLoads(Ft, Fr, Ff)
       →  SimulationCase(all of the above)
```

---

## 2. Analytical Beam Theory (Dry-Run Mode) — Bug-11/12 Fixes

### Original cantilever model (Bug-11 — wrong physics)

The v1 model used a pure cantilever:

$$\delta_{\text{nose}} = \frac{F_r L_{\text{total}}^3}{3EI}$$

This is **physically wrong** for a spindle.  A spindle has two bearing
supports — it is a **propped beam**, not a free cantilever.  The cantilever
formula:
- Overestimates deflection (no bearing supports)
- Has wrong sensitivity to bearing stiffness (ignores $K_{\text{radial}}$)
- Cannot predict the sign-change of the rear reaction

### Corrected propped-cantilever model (Bug-11 fix)

**Geometry:**
```
F_r (transverse, at nose z=0)
     ↓
─────|──────────|──────────────────|───
     z=0      z=a                z=b
              front bearing      rear bearing
              K_front             K_rear
```

**Static equilibrium (moment about rear bearing $z = b$):**

$$R_{\text{front}} = F_r \times \frac{b}{b - a}$$

$$R_{\text{rear}} = F_r - R_{\text{front}}$$

**Spring compliance deflections:**

$$\delta_{\text{front}} = \frac{R_{\text{front}}}{K_{\text{front}}} \quad [\text{mm}]$$

$$\delta_{\text{rear}} = \frac{R_{\text{rear}}}{K_{\text{rear}}} \quad [\text{mm}]$$

**Nose deflection** (superposition of cantilever bending + rigid-body tilt):

$$\delta_{\text{nose}} = \underbrace{\frac{F_r \, a^3}{3EI}}_{\text{cantilever bend}} + \underbrace{\delta_{\text{front}} + (\delta_{\text{front}} - \delta_{\text{rear}}) \times \frac{a}{b-a}}_{\text{rigid-body tilt}}$$

where $a = L_1 + f_{\text{front}} \times L_2$ (front bearing z-position).

**Physics check verified in test suite:**
```
R2 = 42 mm (soft) → δ_nose = 26.4 μm
R2 = 58 mm (stiff) → δ_nose = 21.7 μm  ✅ (stiffer = less deflection)
```

### Frequency model (Bug-12 fix)

**Original: free-clamped Rayleigh (wrong)**

Used $\lambda_1 = 1.875$ (cantilever eigenvalue), ignoring bearing supports.

**Corrected: Dunkerly superposition**

Dunkerly's method gives a lower-bound approximation to the first critical
speed by treating the shaft and each bearing spring as independent
single-DOF systems and combining their compliances:

$$\frac{1}{f_1^2} = \frac{1}{f_{\text{beam}}^2} + \frac{1}{f_{\text{front}}^2} + \frac{1}{f_{\text{rear}}^2}$$

**Shaft beam frequency** (Euler-Bernoulli, pinned-pinned approximation):

$$f_{\text{beam}} = \frac{\pi^2}{2\pi L^2} \sqrt{\frac{EI}{\rho A}} \quad [\text{Hz}]$$

**Bearing spring frequencies** (mass on spring):

$$f_{\text{front}} = \frac{1}{2\pi} \sqrt{\frac{K_{\text{front}}}{m/2}} \quad [\text{Hz}]$$

$$f_{\text{rear}} = \frac{1}{2\pi} \sqrt{\frac{K_{\text{rear}}}{m/2}}$$

where $m/2$ approximates the effective modal mass assigned to each support.

**Physics check verified:**
```
L2 = 440 mm (long) → f1 = 496 Hz
L2 = 360 mm (short) → f1 = 587 Hz  ✅ (shorter span = higher frequency)
```

**Standard reference for Dunkerly:**
Rao, S.S. "Mechanical Vibrations" §7.6 — Dunkerly's method for multi-DOF
rotor systems.

---

## 3. Results Format

Every case (analytical or ANSYS) returns a dictionary with standardised keys:

| Key | Unit | Description |
|-----|------|-------------|
| `static_max_deflection_um` | μm | Spindle nose lateral deflection |
| `static_max_vonmises_MPa` | MPa | Maximum von Mises stress |
| `static_factor_of_safety` | — | σ_y / σ_max |
| `freq_mode1_Hz` | Hz | First natural frequency |
| `freq_mode2_Hz` | Hz | Second natural frequency |
| `freq_mode3_Hz` | Hz | Third natural frequency |
| `var_L1`, `var_R2`, … | mm | Design variables (prefixed `var_`) |

This consistent schema means the ML surrogate can be trained identically on
analytical or ANSYS results.

---

## 4. Batch Execution

```python
runner = FEAPoolRunner(X_lhs, design_space, dry_run=True)
df = runner.execute_batch(max_failures=10, save_interval=20)
```

- `max_failures`: stops if this many consecutive cases fail (ANSYS crash protection)
- `save_interval`: saves intermediate CSV every N cases (crash recovery)
- Progress shown with `tqdm` progress bar

---

## 5. Parquet Fallback — Bug Fix

v1 always wrote `.parquet` which requires `pyarrow` or `fastparquet`.
v2 wraps the parquet write in try/except:

```python
try:
    df.to_parquet(path)
except ImportError:
    pass   # pyarrow not installed; CSV is sufficient
```

---

---

# Module 04 — ML Surrogate Model
**File:** `04_ml_surrogate.py`

---

## Purpose

Train machine learning models that act as **high-speed emulators** of ANSYS
FEA.  Once trained, the surrogate predicts deflection, stress, and natural
frequencies in ~0.001 seconds vs. 5–10 minutes for a full ANSYS run —
a **1,000× speedup** that makes robust optimisation feasible.

---

## 1. Bug-1 Fix — Per-Output Scalers

### The bug

v1 stored a single `self.scaler_y = StandardScaler()` and called
`scaler_y.fit_transform()` inside the output loop:

```python
# BUG: scaler_y was REFITTED on each output
for i, out_name in enumerate(output_names):
    y_i_scaled = self.scaler_y.fit_transform(y_train[:, i])   # ← refit each time
    model.fit(X_scaled, y_i_scaled)
    # predict later would always use scaler_y fitted to LAST output
```

When `predict()` called `self.scaler_y.inverse_transform()`, it used the
scale factors from the **final output only** (e.g., `frequency_Hz` with
range 400–900 Hz).  Applying those scale factors to deflection (range 5–200 μm)
gave completely wrong predictions for all outputs except the last.

### The fix

```python
# FIX: one independent scaler per output
self.scalers_y: Dict[str, StandardScaler] = {}   # dict, not single scaler

for i, out_name in enumerate(output_names):
    scaler_i = StandardScaler()                   # fresh scaler
    y_i_scaled = scaler_i.fit_transform(y_i)
    self.scalers_y[out_name] = scaler_i           # store it
    model.fit(X_scaled, y_i_scaled)
    self.models[out_name] = model
```

In `predict()`:

```python
scaler_i = self.scalers_y[out_name]              # BUG-3 FIX: correct scaler
y_pred[:, i] = scaler_i.inverse_transform(y_s)
y_std[:, i]  = std_s * scaler_i.scale_[0]       # BUG-3 FIX: correct scale
```

**Proof the bug is fixed:** The smoke test deliberately uses three outputs
with very different scales (≈15, ≈220, ≈9000) and verifies that each is
predicted accurately.

---

## 2. Model Types

### Gaussian Process (Kriging) — `model_type="gp"` — Recommended

The GP models the response as a realisation of a stochastic process with
Matérn covariance kernel:

$$k(\mathbf{x}, \mathbf{x}') = \sigma^2 \cdot \text{Matérn}(\mathbf{x}, \mathbf{x}' \mid \nu=2.5) + \sigma_n^2 \delta(\mathbf{x}, \mathbf{x}')$$

**Kernel choice — Matérn ν=2.5:**

The Matérn kernel with $\nu = 2.5$ corresponds to twice-differentiable
sample paths.  This is the appropriate choice for smooth engineering
responses (deflection, stress, frequency) that are $C^2$ functions of
geometry.

$$\text{Matérn}_{2.5}(r) = \left(1 + \frac{\sqrt{5}\,r}{\ell} + \frac{5r^2}{3\ell^2}\right) \exp\left(-\frac{\sqrt{5}\,r}{\ell}\right)$$

where $r = \|\mathbf{x} - \mathbf{x}'\|$ and $\ell$ is the length-scale
(fitted by maximum likelihood during training).

**WhiteKernel** models observation noise (FEA solver numerical error,
analytical approximation error in dry-run mode).

**Prediction with uncertainty:**
$$\hat{y}(\mathbf{x}) = \mu_{\text{posterior}}(\mathbf{x})$$
$$\hat{\sigma}(\mathbf{x}) = \sqrt{\sigma^2_{\text{posterior}}(\mathbf{x})}$$

The posterior standard deviation $\hat{\sigma}$ is the **uncertainty estimate**:
- Low near training points (surrogate has seen similar designs)
- High in unexplored regions (surrogate is extrapolating)

This uncertainty feeds the **Active Learning** extension where new FEA
samples are placed in high-uncertainty regions.

**Standard reference:** Rasmussen, C.E. & Williams, C.K.I. "Gaussian
Processes for Machine Learning" (2006), Chapter 2.

### XGBoost / Gradient Boosting — `model_type="xgb"`

Ensemble of 500 decision trees trained with gradient boosting.  Better for:
- High-dimensional input spaces (>15 variables)
- Discontinuous or non-smooth responses
- Large training datasets (>1,000 samples)

No uncertainty estimates available.

### MLP Neural Network — `model_type="mlp"`

Three hidden layers: `[2n, n, n//2]` where `n = n_features`.  Early stopping
with 20% validation split.  Best for highly non-linear mappings but requires
the most training data.

---

## 3. Feature Scaling

All inputs are standardised with `StandardScaler`:

$$\tilde{x}_j = \frac{x_j - \mu_j}{\sigma_j}$$

This is essential for GP (kernel length-scales become comparable) and MLP
(gradient descent stability).

XGBoost is scale-invariant but scaling does not hurt it.

---

## 4. Cross-Validation

5-fold cross-validation is run during training:

```python
cv_r2 = cross_val_score(model, X_scaled, y_scaled, cv=5, scoring="r2")
```

**Target R² > 0.95** for deployment.  If R² < 0.90:

1. Increase sample count (200 → 500)
2. Try XGBoost (handles nonlinearity better)
3. Check for outliers in FEA data (failed ANSYS runs)

---

## 5. Serialisation — Joblib

```python
# Save (includes all per-output scalers)
joblib.dump({
    "model_type":   self.model_type,
    "models":       self.models,           # Dict[name → estimator]
    "scaler_X":     self.scaler_X,
    "scalers_y":    self.scalers_y,        # Dict[name → StandardScaler]
    "output_names": self.output_names,
}, filepath)

# Load
data = joblib.load(filepath)
surrogate.scalers_y = data["scalers_y"]   # Bug-1 fix: dict restored
```

---

## 6. Output Variables Predicted

Default outputs from the FEA pool:

| Output | Symbol | Unit | Physics |
|--------|--------|------|---------|
| `static_max_deflection_um` | $\delta_{\text{nose}}$ | μm | Nose lateral displacement under Ft, Fr, Ff |
| `static_max_vonmises_MPa` | $\sigma_{\text{VM}}$ | MPa | Peak von Mises stress (typically at bearing seat) |
| `static_factor_of_safety` | FoS | — | $\sigma_y / \sigma_{\text{VM}}$ |
| `freq_mode1_Hz` | $f_1$ | Hz | First bending natural frequency |
| `freq_mode2_Hz` | $f_2$ | Hz | Second natural frequency |
| `freq_mode3_Hz` | $f_3$ | Hz | Third natural frequency |

The surrogate predicts all outputs simultaneously but with **independent
models** (one per output), avoiding the single-scaler bug.

---

## 7. Performance Benchmarks

| Configuration | Training time | Prediction time |
|---------------|--------------|----------------|
| GP, 100 samples, 19 vars | ~30 s | 0.001 s |
| GP, 500 samples, 19 vars | ~300 s | 0.005 s |
| XGB, 500 samples, 19 vars | ~10 s | 0.0001 s |
| ANSYS full FEA | — | 5–10 min |

**Speedup at evaluation time:** GP: ×300,000 vs. ANSYS.
This enables 10,000 GA fitness evaluations in ~10 seconds.
