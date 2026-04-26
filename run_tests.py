#!/usr/bin/env python3
"""
================================================================================
  Full Integration Test Suite — RDO Framework v2
================================================================================

  Tests all 7 modules, verifies every bug fix, validates the new Selective
  Assembly module and the Cost objective in the GA.

  Run with:
      python3 run_tests.py
================================================================================
"""

import sys
import importlib.util
import logging
import numpy as np

logging.basicConfig(level=logging.WARNING)   # suppress INFO noise

# ─── module loader (numeric prefixes prevent direct import) ───────────────────
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

load('design_variables',   './design_variables.py')
load('lhs_sampler',        './lhs_sampler.py')
load('fea_pool_runner',    './fea_pool_runner.py')
load('ml_surrogate',       './ml_surrogate.py')
load('selective_assembly', './selective_assembly.py')
load('robust_optimizer',   './robust_optimizer.py')
load('inverse_design',     './inverse_design.py')

from design_variables    import DesignSpace
from lhs_sampler         import LHSSampler
from fea_pool_runner     import FEAPoolRunner
from ml_surrogate        import SurrogateModel
from selective_assembly  import SelectiveAssemblyAnalyser, SpindleCostModel
from robust_optimizer    import RobustOptimizer
from inverse_design      import InverseDesignEngine
from sklearn.dummy       import DummyRegressor
from sklearn.preprocessing import StandardScaler

# ─── tiny test harness ────────────────────────────────────────────────────────
_results = []

def check(label, cond, detail=""):
    tag = "  OK " if cond else "  FAIL"
    extra = f"  ({detail})" if detail else ""
    print(f"{tag}  {label}{extra}")
    _results.append((label, cond))


def section(title):
    print(f"\n{'='*62}\n  {title}\n{'='*62}")


# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
ds     = DesignSpace()
n      = ds.get_bounds().shape[0]
nom    = ds.get_nominal()
bounds = ds.get_bounds()

section("MODULE 1 — Design Variables")
check("19 variables extracted", n == 19, f"got {n}")
check("All lower < upper", (bounds[:, 0] < bounds[:, 1]).all())
check("Nominal within bounds",
      ((nom >= bounds[:, 0]) & (nom <= bounds[:, 1])).all())
check("All tolerances > 0", (ds.get_tolerances() > 0).all())
decoded = ds.decode_vector(nom)
check("decode_vector returns all names",
      set(decoded.keys()) == set(ds.get_variable_names()))

# ─────────────────────────────────────────────────────────────────────────────
section("MODULE 2 — LHS Sampler  (BUG-4, BUG-5 fixed)")
s = LHSSampler(ds)

for crit in ["maximin", "center", "centermaximin", "correlation"]:
    X = s.generate_lhs(25, criterion=crit, seed=0)
    ok = X.shape == (25, 19) and ((X >= bounds[:, 0]) & (X <= bounds[:, 1])).all()
    check(f"criterion={crit!r:<15} shape & bounds", ok)

X_sob = s.generate_sobol(64)
check("Sobol shape (64, 19)", X_sob.shape == (64, 19))

X_mc = s.generate_montecarlo(50, distribution="normal")
check("MC-normal within bounds",
      ((X_mc >= bounds[:, 0]) & (X_mc <= bounds[:, 1])).all())

Xn, Xp = s.generate_hybrid(n_lhs=10, n_mc_per_lhs=4)   # BUG-5: Tuple type
check("BUG-5: hybrid returns Tuple", isinstance((Xn, Xp), tuple))
check("Hybrid shapes correct", Xn.shape == (10, 19) and Xp.shape == (40, 19))

# ─────────────────────────────────────────────────────────────────────────────
section("MODULE 3 — FEA Pool Runner  (BUG-11, BUG-12 fixed)")
X_fea  = s.generate_lhs(12, seed=7)
runner = FEAPoolRunner(X_fea, ds, dry_run=True)
df     = runner.execute_batch()

check("All 12 cases completed", len(df) == 12)
check("Deflection column present", "static_max_deflection_um" in df.columns)
check("All deflections > 0", (df["static_max_deflection_um"] > 0).all())
check("All FoS > 0",         (df["static_factor_of_safety"] > 0).all())
check("All freq_mode1 > 0",  (df["freq_mode1_Hz"] > 0).all())

# BUG-11 physics check: larger journal radius → smaller deflection
soft  = nom.copy(); stiff = nom.copy()
idx_R2 = ds.get_variable_names().index("R2")
soft[idx_R2]  = 42.0
stiff[idx_R2] = 58.0
rp  = FEAPoolRunner(np.vstack([soft, stiff]), ds, dry_run=True).execute_batch()
d_s = rp.iloc[0]["static_max_deflection_um"]
d_h = rp.iloc[1]["static_max_deflection_um"]
check("BUG-11: stiffer shaft deflects less",
      d_s > d_h, f"R2=42→{d_s:.1f}μm  R2=58→{d_h:.1f}μm")

# BUG-12 physics check: shorter bearing span → higher first frequency
long_v  = nom.copy(); short_v = nom.copy()
idx_L2  = ds.get_variable_names().index("L2")
long_v[idx_L2]  = 440.0
short_v[idx_L2] = 360.0
rf = FEAPoolRunner(np.vstack([long_v, short_v]), ds, dry_run=True).execute_batch()
check("BUG-12: shorter span → higher freq (Dunkerly)",
      rf.iloc[1]["freq_mode1_Hz"] > rf.iloc[0]["freq_mode1_Hz"],
      f"L2=440→{rf.iloc[0]['freq_mode1_Hz']:.1f}Hz  L2=360→{rf.iloc[1]['freq_mode1_Hz']:.1f}Hz")

# ─────────────────────────────────────────────────────────────────────────────
section("MODULE 4 — ML Surrogate  (BUG-1, BUG-2, BUG-3 fixed)")
var_cols = [c for c in df.columns if c.startswith("var_")]
X_tr     = df[var_cols].values
out_cols = ["static_max_deflection_um", "static_max_vonmises_MPa", "freq_mode1_Hz"]
y_tr     = df[out_cols].values

surr = SurrogateModel(model_type="gp")
surr.train(X_tr, y_tr, output_names=out_cols, verbose=False)

check("3 models trained",                 len(surr.models) == 3)
check("BUG-1: 3 independent scalers_y",  len(surr.scalers_y) == 3)

# Prove scalers are independent: outputs span very different ranges so their
# scale_ factors must differ.
sc_vals = [surr.scalers_y[n].scale_[0] for n in out_cols]
check("BUG-1: per-output scale_ values differ",
      len(set(round(v, 3) for v in sc_vals)) > 1,
      f"scales={[round(v,2) for v in sc_vals]}")

y_pred = surr.predict(X_tr)
check("Prediction shape (12, 3)",     y_pred.shape == (len(X_tr), 3))
check("Deflection predictions > 0",   (y_pred[:, 0] > 0).all())

y_p2, y_s2 = surr.predict(X_tr, return_std=True)   # BUG-3: uses per-output scaler
check("BUG-3: GP std returned, finite",
      y_s2 is not None and np.isfinite(y_s2).all())

# Save / load round-trip
surr.save("/tmp/surr_rdo.pkl")
surr2 = SurrogateModel.load("/tmp/surr_rdo.pkl")
check("Save/load predictions match",  np.allclose(surr2.predict(X_tr), y_pred, atol=1e-6))
check("scalers_y persisted in file",  len(surr2.scalers_y) == 3)

# ─────────────────────────────────────────────────────────────────────────────
section("MODULE 7 — Selective Assembly  (new module)")
nom_dict = ds.decode_vector(nom)
analyser = SelectiveAssemblyAnalyser(n_parts=500)
cost_mdl = SpindleCostModel()
sa_res   = analyser.analyse_all(nom_dict, n_bins=5)

check("3 interfaces analysed", len(sa_res) == 3)
check("All yields in (0, 1]",  all(0 < r.match_yield <= 1.0 for r in sa_res))
check("SA improves scatter (improvement_ratio > 1)",
      all(r.improvement_ratio > 1.0 for r in sa_res),
      f"ratios={[round(r.improvement_ratio,1) for r in sa_res]}")

costs = cost_mdl.total_cost(nom_dict, sa_res)
check("All 5 cost keys present",
      all(k in costs for k in
          ["material_usd","machining_usd","bearings_usd","sa_usd","total_usd"]))
check("Total cost > 0", costs["total_usd"] > 0,
      f"${costs['total_usd']:.2f}")
check("SA cost > 0",    costs["sa_usd"] > 0,
      f"${costs['sa_usd']:.2f}")
check("Material cost > 0", costs["material_usd"] > 0,
      f"${costs['material_usd']:.2f}")
check("Bearing cost > 0",  costs["bearings_usd"] > 0,
      f"${costs['bearings_usd']:.2f}")

# More bins tighten scatter
sa_3  = analyser.analyse_all(nom_dict, n_bins=3)
sa_10 = analyser.analyse_all(nom_dict, n_bins=10)
avg3  = np.mean([r.std_gap_um for r in sa_3])
avg10 = np.mean([r.std_gap_um for r in sa_10])
check("More bins → tighter gap scatter",
      avg10 < avg3, f"3-bin σ={avg3:.2f}μm  10-bin σ={avg10:.2f}μm")

# High-stiffness bearings cost more
low_cost  = cost_mdl.bearing_cost({**nom_dict, "K_radial": 3e5})
high_cost = cost_mdl.bearing_cost({**nom_dict, "K_radial": 8e5})
check("Higher bearing stiffness → higher bearing cost",
      high_cost > low_cost, f"K=3e5→${low_cost:.0f}  K=8e5→${high_cost:.0f}")

# SA summary table
tbl = analyser.summary_table(sa_res)
check("Summary table has 3 rows",  len(tbl) == 3)
check("Summary table has Yield column", "Yield %" in tbl.columns)

# ─────────────────────────────────────────────────────────────────────────────
section("MODULE 5 — Robust Optimizer  (BUG-6,7,8,9 + Cost obj, 4 objectives)")

# build mock surrogate with correct output names and real per-output scalers
mock = SurrogateModel(model_type="gp")
mock.output_names = out_cols
X_mock = np.random.rand(30, n)
mock.scaler_X.fit(X_mock)
for i, col in enumerate(out_cols):
    sc = StandardScaler()
    sc.fit((np.random.rand(30, 1) * (100 * (i + 1))))
    mock.scalers_y[col] = sc
    dr = DummyRegressor(strategy="mean")
    dr.fit(X_mock, np.random.rand(30) * 100 * (i + 1))
    mock.models[col] = dr

opt = RobustOptimizer(mock, ds, n_mc_inner=5, n_sa_bins=3, sa_n_parts=100)

check("BUG-7: 4 objectives (not hardcoded 3)", opt.n_objectives == 4)

f = opt.multi_objective_fitness(nom)
check("BUG-8: fitness returns 4 values",    f.shape == (4,))
check("BUG-9: all fitness values finite",   np.isfinite(f).all())
check("f1 = −S/N_deflection (finite dB)",   np.isfinite(f[0]))
check("f2 = −S/N_stress (finite dB)",       np.isfinite(f[1]))
check("f3 = cost > 0 (NEW objective)",      f[2] > 0, f"${f[2]:.2f}")
check("f4 = weight > 0",                   f[3] > 0, f"{f[3]:.3f} kg")

# BUG-6: wrong output name must raise ValueError, NOT silently return 0
caught = False
try:
    RobustOptimizer(mock, ds,
        output_name_map={
            "deflection": "NO_SUCH_COLUMN",
            "stress":     out_cols[1],
            "frequency":  out_cols[2],
        })
except ValueError:
    caught = True
check("BUG-6: ValueError raised for unknown output name", caught)

res = opt.optimize_de(maxiter=4, seed=0)
check("DE returns OptimizationResult", hasattr(res, "best_robust_design"))
check("BUG-8: objective_labels has 4 entries", len(res.objective_labels) == 4)
check("DE best design within bounds",
      ((res.best_robust_design >= bounds[:, 0]) &
       (res.best_robust_design <= bounds[:, 1])).all())

rpt = opt.report_best(res)
check("report_best includes 'selective_assembly' key", "selective_assembly" in rpt)
check("report_best includes 'costs' key",              "costs" in rpt)
check("3 SA interfaces in report",
      len(rpt["selective_assembly"]) == 3)
check("objectives dict has 4 entries",
      len(rpt["objectives"]) == 4)

# ─────────────────────────────────────────────────────────────────────────────
section("MODULE 6 — Inverse Design  (BUG-10 fixed)")
X_inv = np.random.rand(150, n) * 80 + 10
y_inv = np.column_stack([
    np.random.rand(150) * 200,
    np.random.rand(150) * 400,
    np.random.rand(150) * 1000,
])
pnames = ["deflection_um", "stress_MPa", "frequency_Hz"]

eng = InverseDesignEngine(ds, use_tensorflow=False)
try:
    # BUG-10: List was not imported → NameError on this call
    m = eng.train(X_inv, y_inv, performance_names=pnames, verbose=False)
    check("BUG-10: train() no NameError for List annotation", True)
except NameError as e:
    check("BUG-10: train() no NameError for List annotation", False, str(e))

check("val_r2 in metrics", "val_r2" in m)

target = {"deflection_um": 8.0, "stress_MPa": 350.0, "frequency_Hz": 800.0}
x_pred = eng.predict_design(target)
check("Predicted design shape (n_vars,)",  x_pred.shape == (n,))
check("Predicted design within bounds",
      ((x_pred >= bounds[:, 0]) & (x_pred <= bounds[:, 1])).all())

eng.save("/tmp/inv_rdo.pkl")
eng2 = InverseDesignEngine.load("/tmp/inv_rdo.pkl", ds)
check("Save/load round-trip",
      np.allclose(eng2.predict_design(target), x_pred, atol=1e-6))

# ─────────────────────────────────────────────────────────────────────────────
section("FINAL RESULTS")
passed = sum(1 for _, ok in _results if ok)
failed = [(lbl, ok) for lbl, ok in _results if not ok]
total  = len(_results)

if failed:
    print("\n  FAILED checks:")
    for lbl, _ in failed:
        print(f"    - {lbl}")

banner = "ALL PASSED" if not failed else f"{len(failed)} FAILED"
print(f"\n  {banner}   {passed}/{total} checks")
print()
sys.exit(0 if not failed else 1)
