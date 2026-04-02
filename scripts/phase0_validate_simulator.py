#!/usr/bin/env python3
"""
Phase 0 — Simulator Validation (Tests 1–5)
==========================================

Runs all five validation tests and reports pass/fail for each. Results are saved to
outputs/phase0_validation.txt and outputs/phase0_timing.csv.

Tests
-----
1. Ideal-case check   : zero parasitics, known analytical solution
2. Parasitic check    : parasitics reduce V_out and efficiency
3. Monotonicity checks: sweep each input, confirm expected monotonic trends
4. Convergence check  : compare rtol = 1e-4 vs 1e-6 vs 1e-8
5. Timing             : 100 simulations, mean ± std wall-clock time
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.simulation.boost_converter_ode import simulate, DESIGN_SPACE, FIXED_PARAMS

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

PASS = "PASS"
FAIL = "FAIL"

results_log: list[str] = []


def log(msg: str = "") -> None:
    print(msg)
    results_log.append(msg)


# ---------------------------------------------------------------------------
# Test 1 — Ideal-case check (no parasitics)
# ---------------------------------------------------------------------------
def test1_ideal_case() -> bool:
    log("=" * 60)
    log("TEST 1: Ideal-case check (R_L=0, R_sw=0, V_d=0)")
    log("=" * 60)

    D, V_in, R = 0.5, 12.0, 10.0
    ideal_params = {"R_L": 0.0, "R_sw": 0.0, "V_d": 0.0}

    out = simulate(D=D, V_in=V_in, R=R, f_sw=50e3, fixed_params=ideal_params)

    V_out_expected = V_in / (1 - D)          # 24.0 V
    I_out_expected = V_out_expected / R       # 2.4 A
    I_in_expected  = I_out_expected / (1 - D) # 4.8 A (= I_L in averaged model)
    eta_expected   = 1.0

    v_err   = abs(out["v_out_mean"] - V_out_expected) / V_out_expected
    i_err   = abs(out["i_l_mean"]   - I_in_expected)  / I_in_expected
    eta_err = abs(out["efficiency"] - eta_expected)

    log(f"  V_out:      {out['v_out_mean']:.4f} V  (expected {V_out_expected:.1f} V,  error {v_err*100:.3f}%)")
    log(f"  I_L:        {out['i_l_mean']:.4f} A  (expected {I_in_expected:.1f} A,   error {i_err*100:.3f}%)")
    log(f"  efficiency: {out['efficiency']:.6f}   (expected {eta_expected:.1f},     error {eta_err*100:.4f}%)")
    log(f"  CCM OK:     {out['ccm_ok']}")

    passed = v_err < 0.01 and i_err < 0.01 and eta_err < 0.01
    log(f"  Result: {PASS if passed else FAIL}")
    return passed


# ---------------------------------------------------------------------------
# Test 2 — Parasitic check
# ---------------------------------------------------------------------------
def test2_parasitic_check() -> bool:
    log("")
    log("=" * 60)
    log("TEST 2: Parasitic check (default parasitics)")
    log("=" * 60)

    D, V_in, R = 0.5, 12.0, 10.0
    out_ideal = simulate(D=D, V_in=V_in, R=R, f_sw=50e3,
                         fixed_params={"R_L": 0.0, "R_sw": 0.0, "V_d": 0.0})
    out_para  = simulate(D=D, V_in=V_in, R=R, f_sw=50e3)

    v_lower  = out_para["v_out_mean"] < out_ideal["v_out_mean"]
    eta_lt1  = out_para["efficiency"] < 1.0
    v_range  = 20.0 <= out_para["v_out_mean"] <= 24.0
    eta_range = 0.75 <= out_para["efficiency"] <= 0.98

    log(f"  V_out (ideal):     {out_ideal['v_out_mean']:.4f} V")
    log(f"  V_out (parasitic): {out_para['v_out_mean']:.4f} V  (lower? {v_lower})")
    log(f"  efficiency:        {out_para['efficiency']:.4f}   (< 1? {eta_lt1})  in [0.75, 0.98]? {eta_range}")
    log(f"  V_out in [20,24]?  {v_range}")

    passed = v_lower and eta_lt1 and v_range and eta_range
    log(f"  Result: {PASS if passed else FAIL}")
    return passed


# ---------------------------------------------------------------------------
# Test 3 — Monotonicity checks
# ---------------------------------------------------------------------------
def test3_monotonicity() -> bool:
    log("")
    log("=" * 60)
    log("TEST 3: Monotonicity checks")
    log("=" * 60)

    subtests: list[tuple[str, bool]] = []

    # 3a: Sweep D → V_out increases
    D_vals = np.linspace(0.2, 0.75, 12)
    v_outs = [simulate(D=d, V_in=12.0, R=20.0, f_sw=50e3)["v_out_mean"] for d in D_vals]
    mono_D = all(v_outs[i] < v_outs[i + 1] for i in range(len(v_outs) - 1))
    log(f"  3a: D sweep  → V_out monotone increasing? {mono_D}")
    subtests.append(("3a D sweep", mono_D))

    # 3b: Sweep V_in → V_out increases
    Vin_vals = np.linspace(8, 24, 10)
    v_outs_vin = [simulate(D=0.5, V_in=v, R=20.0, f_sw=50e3)["v_out_mean"] for v in Vin_vals]
    mono_Vin = all(v_outs_vin[i] < v_outs_vin[i + 1] for i in range(len(v_outs_vin) - 1))
    log(f"  3b: V_in sweep → V_out monotone increasing? {mono_Vin}")
    subtests.append(("3b V_in sweep", mono_Vin))

    # 3c: Sweep R → efficiency generally increases (lighter load = less I²R loss)
    # Efficiency vs R is monotone but nonlinear (plateaus at high R), so Spearman
    # rank correlation is the correct measure (Pearson underestimates nonlinear trends).
    R_vals = np.linspace(5, 100, 10)
    etas = [simulate(D=0.5, V_in=12.0, R=r, f_sw=50e3)["efficiency"] for r in R_vals]
    from scipy.stats import spearmanr
    r_spearman, _ = spearmanr(R_vals, etas)
    mono_R = r_spearman > 0.8
    log(f"  3c: R sweep → efficiency Spearman r={r_spearman:.3f} (want > 0.8)? {mono_R}")
    subtests.append(("3c R sweep", mono_R))

    # 3d: Sweep f_sw → i_l_ripple decreases  (ΔI = V_in·D / (L·f_sw))
    fsw_vals = np.linspace(20e3, 200e3, 10)
    ripples = [simulate(D=0.5, V_in=12.0, R=20.0, f_sw=f)["i_l_ripple"] for f in fsw_vals]
    mono_fsw = all(ripples[i] > ripples[i + 1] for i in range(len(ripples) - 1))
    log(f"  3d: f_sw sweep → i_l_ripple monotone decreasing? {mono_fsw}")
    subtests.append(("3d f_sw sweep", mono_fsw))

    passed = all(ok for _, ok in subtests)
    log(f"  Result: {PASS if passed else FAIL}")
    return passed


# ---------------------------------------------------------------------------
# Test 4 — Convergence check
# ---------------------------------------------------------------------------
def test4_convergence() -> bool:
    log("")
    log("=" * 60)
    log("TEST 4: Convergence check (rtol = 1e-4 vs 1e-6 vs 1e-8)")
    log("=" * 60)

    D, V_in, R, f_sw = 0.5, 12.0, 10.0, 50e3
    results_by_tol = {}
    for rtol in [1e-4, 1e-6, 1e-8]:
        out = simulate(D=D, V_in=V_in, R=R, f_sw=f_sw, rtol=rtol, atol=rtol * 1e-3)
        results_by_tol[rtol] = out

    v4 = results_by_tol[1e-4]["v_out_mean"]
    v6 = results_by_tol[1e-6]["v_out_mean"]
    v8 = results_by_tol[1e-8]["v_out_mean"]

    diff_4_6 = abs(v4 - v6) / abs(v6) if v6 != 0 else float("inf")
    diff_6_8 = abs(v6 - v8) / abs(v8) if v8 != 0 else float("inf")

    log(f"  V_out (rtol=1e-4): {v4:.6f} V")
    log(f"  V_out (rtol=1e-6): {v6:.6f} V   change from 1e-4: {diff_4_6*100:.4f}%")
    log(f"  V_out (rtol=1e-8): {v8:.6f} V   change from 1e-6: {diff_6_8*100:.4f}%")

    # Pass: tightening from 1e-6 to 1e-8 changes V_out by < 0.1 %
    passed = diff_6_8 < 0.001
    log(f"  Convergence at rtol=1e-6 (< 0.1% change to 1e-8)? {PASS if passed else FAIL}")
    log(f"  Result: {PASS if passed else FAIL}")
    return passed


# ---------------------------------------------------------------------------
# Test 5 — Timing
# ---------------------------------------------------------------------------
def test5_timing() -> tuple[bool, pd.DataFrame]:
    log("")
    log("=" * 60)
    log("TEST 5: Timing (100 simulations)")
    log("=" * 60)

    rng = np.random.default_rng(42)
    n = 100
    D_vals    = rng.uniform(*DESIGN_SPACE["D"],    n)
    Vin_vals  = rng.uniform(*DESIGN_SPACE["V_in"], n)
    R_vals    = rng.uniform(*DESIGN_SPACE["R"],    n)
    fsw_vals  = rng.uniform(*DESIGN_SPACE["f_sw"], n)

    times = []
    for i in range(n):
        out = simulate(D=D_vals[i], V_in=Vin_vals[i],
                       R=R_vals[i], f_sw=fsw_vals[i])
        times.append(out["wall_time_s"])

    times_arr = np.array(times)
    mean_t = times_arr.mean()
    std_t  = times_arr.std()
    min_t  = times_arr.min()
    max_t  = times_arr.max()
    total  = times_arr.sum()

    log(f"  n simulations : {n}")
    log(f"  mean time     : {mean_t*1000:.2f} ms")
    log(f"  std           : {std_t*1000:.2f} ms")
    log(f"  min / max     : {min_t*1000:.2f} / {max_t*1000:.2f} ms")
    log(f"  total wall    : {total:.2f} s")

    timing_df = pd.DataFrame({"wall_time_s": times})
    timing_df.to_csv(OUTPUT_DIR / "phase0_timing.csv", index=False)
    log(f"  Timing saved  : outputs/phase0_timing.csv")

    passed = True   # Timing test always passes; it's informational
    log(f"  Result: {PASS} (informational)")
    return passed, timing_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log("PHASE 0 — SIMULATOR VALIDATION")
    log(f"Fixed params: {FIXED_PARAMS}")
    log(f"Design space: {DESIGN_SPACE}")
    log("")

    t1 = test1_ideal_case()
    t2 = test2_parasitic_check()
    t3 = test3_monotonicity()
    t4 = test4_convergence()
    t5, _ = test5_timing()

    log("")
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    all_tests = [
        ("Test 1 — Ideal-case",    t1),
        ("Test 2 — Parasitic",     t2),
        ("Test 3 — Monotonicity",  t3),
        ("Test 4 — Convergence",   t4),
        ("Test 5 — Timing",        t5),
    ]
    for name, ok in all_tests:
        log(f"  {name:30s}  {PASS if ok else FAIL}")

    overall = all(ok for _, ok in all_tests)
    log("")
    log(f"  OVERALL: {'ALL TESTS PASSED' if overall else 'SOME TESTS FAILED'}")

    report_path = OUTPUT_DIR / "phase0_validation.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(results_log))
    print(f"\nReport saved: {report_path}")

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
