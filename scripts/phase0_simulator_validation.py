"""
Phase 0 validation checks for the averaged simulator.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time
from src.simulation.boost_converter import boost_converter_sim, DESIGN_BOUNDS

np.set_printoptions(precision=6)

PASS = "PASS"
FAIL = "FAIL"


def test1_ideal_case():
    """Test 1: Ideal-case check — no parasitics."""
    print("=" * 70)
    print("TEST 1: Ideal-Case Check (no parasitics)")
    print("=" * 70)
    print("Setup: R_L=0, R_sw=0, V_d=0, D=0.5, V_in=12V, R=10Ω")
    print()

    result = boost_converter_sim(
        D=0.5, V_in=12.0, R=10.0, f_sw=50_000.0,
        R_L=0.0, R_sw=0.0, V_d=0.0,
    )

    expected_v_out = 24.0
    expected_i_out = 2.4
    expected_i_in = 4.8

    measured_i_out = result.v_out_mean / 10.0

    checks = {
        "V_out (V)": (expected_v_out, result.v_out_mean),
        "I_out (A)": (expected_i_out, measured_i_out),
        "I_in (A)": (expected_i_in, result.i_l_mean),
    }

    all_pass = True
    print(f"{'Output':<15} {'Expected':>10} {'Measured':>12} {'Error (%)':>10} {'Pass?':>6}")
    print("-" * 55)
    for name, (exp, meas) in checks.items():
        err_pct = abs(meas - exp) / exp * 100
        passed = err_pct < 1.0
        all_pass = all_pass and passed
        print(f"{name:<15} {exp:>10.4f} {meas:>12.6f} {err_pct:>10.4f} {PASS if passed else FAIL:>6}")

    eff_err = abs(result.efficiency - 1.0) * 100 if not np.isnan(result.efficiency) else 100
    passed = eff_err < 1.0
    all_pass = all_pass and passed
    print(f"{'Efficiency':<15} {'1.0000':>10} {result.efficiency:>12.6f} {eff_err:>10.4f} {PASS if passed else FAIL:>6}")

    print(f"\nTest 1 overall: {PASS if all_pass else FAIL}")
    return all_pass, {
        "v_out": result.v_out_mean,
        "i_out": measured_i_out,
        "i_in": result.i_l_mean,
        "efficiency": result.efficiency,
    }


def test2_parasitic_check():
    """Test 2: Parasitic check — default parasitics."""
    print("\n" + "=" * 70)
    print("TEST 2: Parasitic Check (default parasitics)")
    print("=" * 70)
    print("Setup: Default parasitics, D=0.5, V_in=12V, R=10Ω")
    print()

    result = boost_converter_sim(D=0.5, V_in=12.0, R=10.0, f_sw=50_000.0)

    v_pass = 20.0 <= result.v_out_mean <= 24.0
    e_pass = 0.75 <= result.efficiency <= 0.98

    print(f"{'Output':<15} {'Expected Range':>20} {'Measured':>12} {'Pass?':>6}")
    print("-" * 55)
    print(f"{'V_out (V)':<15} {'[20, 24]':>20} {result.v_out_mean:>12.4f} {PASS if v_pass else FAIL:>6}")
    print(f"{'Efficiency':<15} {'[0.75, 0.98]':>20} {result.efficiency:>12.4f} {PASS if e_pass else FAIL:>6}")

    all_pass = v_pass and e_pass
    print(f"\nTest 2 overall: {PASS if all_pass else FAIL}")
    return all_pass, {
        "v_out": result.v_out_mean,
        "efficiency": result.efficiency,
    }


def test3_monotonicity():
    """Test 3: Monotonicity checks — sweep each input."""
    print("\n" + "=" * 70)
    print("TEST 3: Monotonicity Checks")
    print("=" * 70)
    print()

    N = 50
    D_nom, V_in_nom, R_nom, f_sw_nom = 0.5, 12.0, 20.0, 50_000.0

    all_pass = True

    D_vals = np.linspace(0.2, 0.8, N)
    v_out_D = [boost_converter_sim(D=d, V_in=V_in_nom, R=R_nom, f_sw=f_sw_nom).v_out_mean for d in D_vals]
    mono_D = all(v_out_D[i] <= v_out_D[i+1] for i in range(len(v_out_D)-1))
    all_pass = all_pass and mono_D
    print(f"Sweep D (0.2→0.8): V_out {v_out_D[0]:.2f} → {v_out_D[-1]:.2f}  "
          f"Monotonically increasing: {PASS if mono_D else FAIL}")

    V_in_vals = np.linspace(8, 24, N)
    v_out_Vin = [boost_converter_sim(D=D_nom, V_in=v, R=R_nom, f_sw=f_sw_nom).v_out_mean for v in V_in_vals]
    mono_Vin = all(v_out_Vin[i] <= v_out_Vin[i+1] for i in range(len(v_out_Vin)-1))
    all_pass = all_pass and mono_Vin
    print(f"Sweep V_in (8→24): V_out {v_out_Vin[0]:.2f} → {v_out_Vin[-1]:.2f}  "
          f"Monotonically increasing: {PASS if mono_Vin else FAIL}")

    R_vals = np.linspace(5, 100, N)
    eff_R = [boost_converter_sim(D=D_nom, V_in=V_in_nom, R=r, f_sw=f_sw_nom).efficiency for r in R_vals]
    trend_R = eff_R[-1] > eff_R[0]
    n_increases = sum(1 for i in range(len(eff_R)-1) if eff_R[i] <= eff_R[i+1])
    mostly_increasing = n_increases / (len(eff_R) - 1) > 0.8
    pass_R = trend_R and mostly_increasing
    all_pass = all_pass and pass_R
    print(f"Sweep R (5→100): Efficiency {eff_R[0]:.4f} → {eff_R[-1]:.4f}  "
          f"Generally increasing ({n_increases}/{N-1} steps up): {PASS if pass_R else FAIL}")

    f_sw_vals = np.linspace(20_000, 200_000, N)
    ripple_fsw = [boost_converter_sim(D=D_nom, V_in=V_in_nom, R=R_nom, f_sw=f).i_l_ripple for f in f_sw_vals]
    mono_fsw = all(ripple_fsw[i] >= ripple_fsw[i+1] for i in range(len(ripple_fsw)-1))
    all_pass = all_pass and mono_fsw
    print(f"Sweep f_sw (20k→200k): I_L ripple {ripple_fsw[0]:.4f} → {ripple_fsw[-1]:.4f}  "
          f"Monotonically decreasing: {PASS if mono_fsw else FAIL}")

    print(f"\nTest 3 overall: {PASS if all_pass else FAIL}")
    return all_pass, {
        "D_sweep": (v_out_D[0], v_out_D[-1], mono_D),
        "Vin_sweep": (v_out_Vin[0], v_out_Vin[-1], mono_Vin),
        "R_sweep": (eff_R[0], eff_R[-1], pass_R),
        "fsw_sweep": (ripple_fsw[0], ripple_fsw[-1], mono_fsw),
    }


def test4_convergence():
    """Test 4: ODE convergence check — compare tolerance levels."""
    print("\n" + "=" * 70)
    print("TEST 4: ODE Convergence Check")
    print("=" * 70)
    print("Setup: D=0.5, V_in=12V, R=10Ω, f_sw=50kHz")
    print()

    tolerances = [1e-4, 1e-6, 1e-8]
    results = {}
    for rtol in tolerances:
        r = boost_converter_sim(D=0.5, V_in=12.0, R=10.0, f_sw=50_000.0, rtol=rtol, atol=rtol * 1e-3)
        results[rtol] = r

    ref = results[1e-8]

    print(f"{'RelTol':<10} {'V_out (V)':>12} {'I_L (A)':>12} {'Max Δ from 1e-8 (%)':>22} {'Pass?':>6}")
    print("-" * 65)

    all_pass = True
    convergence_data = {}
    for rtol in tolerances:
        r = results[rtol]
        delta_v = abs(r.v_out_mean - ref.v_out_mean) / abs(ref.v_out_mean) * 100
        delta_i = abs(r.i_l_mean - ref.i_l_mean) / abs(ref.i_l_mean) * 100
        max_delta = max(delta_v, delta_i)
        passed = True if rtol == 1e-8 else max_delta < 0.1
        if rtol != 1e-8:
            all_pass = all_pass and passed
        print(f"{rtol:<10.0e} {r.v_out_mean:>12.6f} {r.i_l_mean:>12.6f} {max_delta:>22.6f} {PASS if passed else FAIL:>6}")
        convergence_data[rtol] = {
            "v_out": r.v_out_mean,
            "i_l": r.i_l_mean,
            "max_delta_pct": max_delta,
        }

    delta_6_v = abs(results[1e-6].v_out_mean - ref.v_out_mean) / abs(ref.v_out_mean) * 100
    delta_6_i = abs(results[1e-6].i_l_mean - ref.i_l_mean) / abs(ref.i_l_mean) * 100
    converged = max(delta_6_v, delta_6_i) < 0.1
    print(f"\n1e-6 vs 1e-8 max delta: {max(delta_6_v, delta_6_i):.6f}% — {'Converged' if converged else 'NOT converged'}")
    print(f"\nTest 4 overall: {PASS if all_pass else FAIL}")
    return all_pass, convergence_data


def test5_timing():
    """Test 5: Timing — 100 random simulations."""
    print("\n" + "=" * 70)
    print("TEST 5: Timing (100 simulations)")
    print("=" * 70)
    print()

    rng = np.random.default_rng(42)
    N = 100

    D_vals = rng.uniform(*DESIGN_BOUNDS["D"], N)
    Vin_vals = rng.uniform(*DESIGN_BOUNDS["V_in"], N)
    R_vals = rng.uniform(*DESIGN_BOUNDS["R"], N)
    fsw_vals = rng.uniform(*DESIGN_BOUNDS["f_sw"], N)

    times = []
    n_ccm_valid = 0
    t_total_start = time.perf_counter()

    for i in range(N):
        r = boost_converter_sim(D=D_vals[i], V_in=Vin_vals[i], R=R_vals[i], f_sw=fsw_vals[i])
        times.append(r.sim_time)
        if r.ccm_valid:
            n_ccm_valid += 1

    t_total = time.perf_counter() - t_total_start
    times = np.array(times)

    print(f"{'Metric':<30} {'Value':>15}")
    print("-" * 47)
    print(f"{'Total time (s)':<30} {t_total:>15.3f}")
    print(f"{'Mean per sim (s)':<30} {np.mean(times):>15.6f}")
    print(f"{'Std per sim (s)':<30} {np.std(times):>15.6f}")
    print(f"{'Min per sim (s)':<30} {np.min(times):>15.6f}")
    print(f"{'Max per sim (s)':<30} {np.max(times):>15.6f}")
    print(f"{'CCM valid':<30} {n_ccm_valid:>15d}/{N}")

    timing_data = {
        "total_s": t_total,
        "mean_s": float(np.mean(times)),
        "std_s": float(np.std(times)),
        "min_s": float(np.min(times)),
        "max_s": float(np.max(times)),
        "n_ccm_valid": n_ccm_valid,
        "n_total": N,
    }

    print(f"\nTest 5: COMPLETE (timing recorded)")
    return True, timing_data


if __name__ == "__main__":
    print("PHASE 0: SIMULATOR VALIDATION")
    print("Five validation checks")
    print("=" * 70)
    print()

    results = {}
    all_tests_pass = True

    for test_fn in [test1_ideal_case, test2_parasitic_check, test3_monotonicity,
                    test4_convergence, test5_timing]:
        passed, data = test_fn()
        results[test_fn.__name__] = {"passed": passed, "data": data}
        all_tests_pass = all_tests_pass and passed
        print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, r in results.items():
        print(f"  {name}: {PASS if r['passed'] else FAIL}")
    print(f"\n  Overall: {PASS if all_tests_pass else FAIL}")
    print("=" * 70)
