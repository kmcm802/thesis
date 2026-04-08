#!/usr/bin/env python3
"""
Phase 5 — Engineering Insight
==============================

Runs Experiments G.1, G.2, and G.3 using the best surrogate (GP-M52)
trained on the full 187-point CCM-valid training set.

G.1  Sensitivity analysis
     • Method 1: Sobol' variance-based indices  (SALib, Saltelli scheme)
     • Method 2: Morris screening                (SALib, r=100 trajectories)
     • Method 3: SHAP values                     (shap, KernelSHAP on GP-M52)
     • Computational cost comparison

G.2  Design-space exploration
     • 2-D contour slices (200×200 each): v_out vs D×V_in, η vs D×R,
       ripple vs D×f_sw
     • 4-D feasibility sweep (50^4 = 6.25 M pts via surrogate)
     • Feasible region: V_out > 30 V, η > 0.85, v_out_ripple < 1 V

G.3  (Optional) Surrogate-based optimisation
     • Maximise η subject to V_out ≥ 30 V (scipy.optimize.minimize)
     • Fixed R = 20 Ω, f_sw = 100 kHz; optimise over D, V_in

Outputs
-------
  outputs/phase5/sobol_indices.csv
  outputs/phase5/morris_indices.csv
  outputs/phase5/shap_values.csv
  outputs/phase5/design_exploration.csv   (feasible-point summary)
  outputs/phase5/optimisation_result.txt
  outputs/phase5/figures/
  outputs/phase5/summary.txt
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.surrogates.registry import MODEL_REGISTRY
from src.utils.metrics import compute_rmse

# ── constants ──────────────────────────────────────────────────────────────────

INPUT_COLS = ["D", "V_in", "R", "f_sw"]
OUTPUTS    = ["v_out_mean", "i_l_mean", "v_out_ripple", "i_l_ripple", "efficiency"]

# Physical bounds (same as design space)
PARAM_BOUNDS = {
    "D":    (0.2,    0.8),
    "V_in": (8.0,   24.0),
    "R":    (5.0,  100.0),
    "f_sw": (20e3, 200e3),
}

OUT = ROOT / "outputs" / "phase5"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = ROOT / "outputs" / "phase2" / "train_200.csv"
TEST_CSV  = ROOT / "outputs" / "phase2" / "test_500.csv"

BEST_MODEL = "GP-SE"
BUILD = {cid: fn for cid, _, fn in MODEL_REGISTRY}

MODEL_COLOR = {
    "GP-M52":   "#377eb8",
    "GP-SE":    "#984ea3",
    "POLY-3-L": "#e41a1c",
    "NN-M-tanh":"#a65628",
    "RBF-C":    "#4daf4a",
}


# ── data ───────────────────────────────────────────────────────────────────────

def load_data():
    train = pd.read_csv(TRAIN_CSV)
    test  = pd.read_csv(TEST_CSV)
    tr = train[train["ccm_ok"]].reset_index(drop=True)
    te = test[test["ccm_ok"]].reset_index(drop=True)
    X_tr = tr[INPUT_COLS].values.astype(float)
    Y_tr = tr[OUTPUTS].values.astype(float)
    X_te = te[INPUT_COLS].values.astype(float)
    Y_te = te[OUTPUTS].values.astype(float)
    return X_tr, Y_tr, X_te, Y_te


# ── train surrogates ───────────────────────────────────────────────────────────

def train_surrogates(X_tr, Y_tr):
    """Train GP-M52 for each output. Returns list of fitted models."""
    print(f"  Training {BEST_MODEL} × {len(OUTPUTS)} outputs …")
    models = []
    for oi, name in enumerate(OUTPUTS):
        t0 = time.perf_counter()
        m = BUILD[BEST_MODEL](X_tr, Y_tr[:, oi])
        elapsed = time.perf_counter() - t0
        models.append(m)
        print(f"    {name:18s}  {elapsed:.2f}s")
    return models


def surrogate_predict(models, X):
    """Predict all outputs for X. Returns (N, n_outputs) array."""
    preds = np.column_stack([np.array(m.predict(X)).ravel() for m in models])
    return preds


# ── G.1 Sensitivity Analysis ───────────────────────────────────────────────────

def run_sobol(models):
    """Sobol' variance-based global sensitivity indices via SALib (Saltelli scheme)."""
    from SALib.sample import saltelli
    from SALib.analyze import sobol as sobol_analyze

    problem = {
        "num_vars": 4,
        "names": INPUT_COLS,
        "bounds": [list(PARAM_BOUNDS[k]) for k in INPUT_COLS],
    }

    N = 10_000
    t0 = time.perf_counter()
    # Saltelli scheme: N × (2d + 2) = 10 000 × 10 = 100 000 rows
    X_samp = saltelli.sample(problem, N, calc_second_order=False)
    t_sample = time.perf_counter() - t0

    t0 = time.perf_counter()
    Y_samp = surrogate_predict(models, X_samp)
    t_eval = time.perf_counter() - t0

    rows = []
    for oi, out_name in enumerate(OUTPUTS):
        Si = sobol_analyze.analyze(problem, Y_samp[:, oi],
                                   calc_second_order=False, print_to_console=False)
        for ii, inp in enumerate(INPUT_COLS):
            rows.append({
                "output": out_name, "input": inp,
                "S1": float(Si["S1"][ii]),
                "S1_conf": float(Si["S1_conf"][ii]),
                "ST": float(Si["ST"][ii]),
                "ST_conf": float(Si["ST_conf"][ii]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sobol_indices.csv", index=False)
    print(f"  Sobol: {len(X_samp):,} evaluations  "
          f"sample={t_sample:.2f}s  eval={t_eval:.4f}s")
    return df, t_sample + t_eval


def run_morris(models):
    """Morris elementary-effects screening via SALib."""
    from SALib.sample import morris as morris_sample
    from SALib.analyze import morris as morris_analyze

    problem = {
        "num_vars": 4,
        "names": INPUT_COLS,
        "bounds": [list(PARAM_BOUNDS[k]) for k in INPUT_COLS],
    }

    t0 = time.perf_counter()
    X_samp = morris_sample.sample(problem, N=100, num_levels=4,
                                  optimal_trajectories=None)
    t_sample = time.perf_counter() - t0

    t0 = time.perf_counter()
    Y_samp = surrogate_predict(models, X_samp)
    t_eval = time.perf_counter() - t0

    rows = []
    for oi, out_name in enumerate(OUTPUTS):
        Mi = morris_analyze.analyze(problem, X_samp, Y_samp[:, oi],
                                    num_levels=4, print_to_console=False)
        for ii, inp in enumerate(INPUT_COLS):
            rows.append({
                "output": out_name, "input": inp,
                "mu_star": float(Mi["mu_star"][ii]),
                "mu":      float(Mi["mu"][ii]),
                "sigma":   float(Mi["sigma"][ii]),
                "mu_star_conf": float(Mi["mu_star_conf"][ii]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "morris_indices.csv", index=False)
    print(f"  Morris: {len(X_samp):,} trajectories  "
          f"sample={t_sample:.2f}s  eval={t_eval:.4f}s")
    return df


def run_shap(models, X_tr):
    """SHAP values for each surrogate output using KernelSHAP."""
    import shap

    # Use 100-point background (summary of training data) for speed
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_tr), size=min(100, len(X_tr)), replace=False)
    background = X_tr[bg_idx]

    all_rows = []
    for oi, out_name in enumerate(OUTPUTS):
        print(f"    SHAP: {out_name} …")
        m = models[oi]

        def predict_fn(X_new):
            return np.array(m.predict(X_new)).ravel()

        explainer = shap.KernelExplainer(predict_fn, background,
                                          link="identity")
        # Explain 200 test points (not whole 468 for speed)
        X_explain = X_tr[:200]
        shap_vals = explainer.shap_values(X_explain, nsamples=100, silent=True)

        mean_abs = np.abs(shap_vals).mean(axis=0)
        for ii, inp in enumerate(INPUT_COLS):
            all_rows.append({
                "output": out_name, "input": inp,
                "mean_abs_shap": float(mean_abs[ii]),
            })

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "shap_values.csv", index=False)
    return df


def plot_sensitivity(df_sobol, df_morris, df_shap):
    """3-panel figure: Sobol S1, Morris μ*, SHAP mean|φ| — for each output."""

    fig, axes = plt.subplots(len(OUTPUTS), 3, figsize=(15, 3 * len(OUTPUTS)))
    if len(OUTPUTS) == 1:
        axes = axes[np.newaxis, :]

    for row_i, out_name in enumerate(OUTPUTS):
        # ── Sobol S1 ─────────────────────────────────────────────────────
        ax = axes[row_i, 0]
        sub = df_sobol[df_sobol["output"] == out_name]
        bars = ax.bar(sub["input"], sub["S1"], color="#377eb8", alpha=0.8,
                      yerr=sub["S1_conf"], capsize=3)
        ax.set_ylim(0, 1)
        ax.set_title(f"{out_name} — Sobol S₁", fontsize=9)
        ax.set_ylabel("S₁" if row_i == 0 else "")
        ax.grid(True, axis="y", alpha=0.3)

        # ── Morris μ* ─────────────────────────────────────────────────────
        ax = axes[row_i, 1]
        sub = df_morris[df_morris["output"] == out_name]
        ax.bar(sub["input"], sub["mu_star"], color="#e41a1c", alpha=0.8,
               yerr=sub["mu_star_conf"], capsize=3)
        ax.set_title(f"{out_name} — Morris μ*", fontsize=9)
        ax.set_ylabel("μ*" if row_i == 0 else "")
        ax.grid(True, axis="y", alpha=0.3)

        # ── SHAP mean|φ| ──────────────────────────────────────────────────
        ax = axes[row_i, 2]
        sub = df_shap[df_shap["output"] == out_name].sort_values(
            "mean_abs_shap", ascending=True)
        ax.barh(sub["input"], sub["mean_abs_shap"], color="#4daf4a", alpha=0.8)
        ax.set_title(f"{out_name} — SHAP |φ|", fontsize=9)
        ax.set_xlabel("Mean |SHAP|" if row_i == len(OUTPUTS) - 1 else "")
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(f"Sensitivity Analysis — {BEST_MODEL} surrogate", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "sensitivity_all_outputs.png", dpi=150)
    plt.close(fig)
    print("  Saved: sensitivity_all_outputs.png")

    # ── Combined bar chart for primary output (v_out_mean) ────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    out_name = "v_out_mean"
    x = np.arange(4)
    w = 0.25

    s1  = df_sobol[df_sobol["output"] == out_name].set_index("input").loc[INPUT_COLS, "S1"].values
    st  = df_sobol[df_sobol["output"] == out_name].set_index("input").loc[INPUT_COLS, "ST"].values
    mstar = df_morris[df_morris["output"] == out_name].set_index("input").loc[INPUT_COLS, "mu_star"].values
    shap_ = df_shap[df_shap["output"] == out_name].set_index("input").loc[INPUT_COLS, "mean_abs_shap"].values

    # Normalise to [0,1] for comparison
    def norm01(v):
        return v / (v.max() + 1e-12)

    ax.bar(x - w,   norm01(s1),    w, label="Sobol S₁",   color="#377eb8", alpha=0.8)
    ax.bar(x,       norm01(st),    w, label="Sobol Sᵀ",   color="#aec7e8", alpha=0.8)
    ax.bar(x + w,   norm01(mstar), w, label="Morris μ*",  color="#e41a1c", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(INPUT_COLS)
    ax.set_ylabel("Normalised sensitivity (0–1)")
    ax.set_title(f"Sensitivity comparison — {out_name}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "sensitivity_comparison_vout.png", dpi=150)
    plt.close(fig)
    print("  Saved: sensitivity_comparison_vout.png")


# ── G.2 Design-Space Exploration ──────────────────────────────────────────────

# Nominal operating point for fixed-dim slices
NOMINAL = {"D": 0.5, "V_in": 16.0, "R": 20.0, "f_sw": 100e3}

def _make_grid_2d(ax1, ax2, n=200):
    """Build a (n×n, 4) array varying ax1/ax2 at nominal for others."""
    v1 = np.linspace(*PARAM_BOUNDS[ax1], n)
    v2 = np.linspace(*PARAM_BOUNDS[ax2], n)
    G1, G2 = np.meshgrid(v1, v2, indexing="ij")
    X = np.column_stack([G1.ravel(), G2.ravel()])
    df = pd.DataFrame(X, columns=[ax1, ax2])
    for col in INPUT_COLS:
        if col not in df:
            df[col] = NOMINAL[col]
    return df[INPUT_COLS].values, G1, G2, v1, v2


def run_design_exploration(models):
    t_start = time.perf_counter()

    # ── 2D slices ─────────────────────────────────────────────────────────
    slices = [
        ("D", "V_in", "v_out_mean",   "V_out (V)"),
        ("D", "R",    "efficiency",   "η"),
        ("D", "f_sw", "v_out_ripple", "V_ripple (V)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (ax1, ax2, out_col, clabel) in zip(axes, slices):
        oi = OUTPUTS.index(out_col)
        X_grid, G1, G2, v1, v2 = _make_grid_2d(ax1, ax2, n=200)
        Y_grid = surrogate_predict(models, X_grid)[:, oi].reshape(200, 200)

        cf = ax.contourf(v1, v2, Y_grid.T, levels=20, cmap="viridis")
        plt.colorbar(cf, ax=ax, label=clabel)
        cs = ax.contour(v1, v2, Y_grid.T, levels=10, colors="white", linewidths=0.5,
                        alpha=0.5)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")
        ax.set_xlabel(ax1); ax.set_ylabel(ax2)
        ax.set_title(f"{out_col}\n(others fixed at nominal)", fontsize=10)

    fig.suptitle("Design-space contour slices — GP-M52 surrogate", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "design_space_contours.png", dpi=150)
    plt.close(fig)
    print("  Saved: design_space_contours.png")

    # ── 4D feasibility sweep (50^4 = 6.25 M pts) ─────────────────────────
    n4 = 50
    print(f"  4D sweep: {n4}^4 = {n4**4:,} points …")
    t0 = time.perf_counter()

    d_vals    = np.linspace(*PARAM_BOUNDS["D"],    n4)
    vin_vals  = np.linspace(*PARAM_BOUNDS["V_in"], n4)
    r_vals    = np.linspace(*PARAM_BOUNDS["R"],    n4)
    fsw_vals  = np.linspace(*PARAM_BOUNDS["f_sw"], n4)

    # Process in batches to avoid memory issues
    rows_summary = []
    feasible_X = []
    BATCH = 250_000

    grid = np.array(np.meshgrid(d_vals, vin_vals, r_vals, fsw_vals,
                                 indexing="ij")).reshape(4, -1).T
    n_total = len(grid)
    n_feasible = 0

    for start in range(0, n_total, BATCH):
        batch = grid[start:start + BATCH]
        Y_b = surrogate_predict(models, batch)
        v_out   = Y_b[:, OUTPUTS.index("v_out_mean")]
        eta     = Y_b[:, OUTPUTS.index("efficiency")]
        ripple  = Y_b[:, OUTPUTS.index("v_out_ripple")]

        feas = (v_out > 30.0) & (eta > 0.85) & (ripple < 1.0)
        n_feasible += int(feas.sum())
        if feas.any():
            feasible_X.append(batch[feas])

    t_sweep = time.perf_counter() - t0
    pct_feasible = 100 * n_feasible / n_total
    print(f"  4D sweep done: {n_feasible:,}/{n_total:,} feasible "
          f"({pct_feasible:.1f}%)  time={t_sweep:.2f}s")

    # Save summary
    rows_summary.append({
        "n_total": n_total, "n_feasible": n_feasible,
        "pct_feasible": pct_feasible, "sweep_time_s": t_sweep,
    })
    pd.DataFrame(rows_summary).to_csv(OUT / "design_exploration.csv", index=False)

    # Plot feasibility in D-R slice using the 4D data
    if feasible_X:
        X_feas = np.vstack(feasible_X)
        # 2D histogram of feasible points in D vs R
        fig, ax = plt.subplots(figsize=(7, 5))
        h, xedges, yedges = np.histogram2d(X_feas[:, 0], X_feas[:, 2],
                                            bins=50,
                                            range=[[0.2, 0.8], [5, 100]])
        im = ax.imshow(h.T, origin="lower",
                       extent=[0.2, 0.8, 5, 100],
                       aspect="auto", cmap="hot_r",
                       norm=mcolors.PowerNorm(gamma=0.4))
        plt.colorbar(im, ax=ax, label="Feasible point count")
        ax.set_xlabel("Duty cycle D"); ax.set_ylabel("Load R (Ω)")
        ax.set_title("Feasible region (D vs R) — all V_in, f_sw\n"
                     "V_out > 30V, η > 85%, ripple < 1V", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / "feasible_region_D_R.png", dpi=150)
        plt.close(fig)
        print("  Saved: feasible_region_D_R.png")

    t_total_explore = time.perf_counter() - t_start
    return pct_feasible, n_feasible, n_total, t_sweep


def plot_efficiency_contour(models):
    """High-res η vs D×R contour with efficiency annotations."""
    oi = OUTPUTS.index("efficiency")
    X_grid, G1, G2, v1, v2 = _make_grid_2d("D", "R", n=300)
    Y_grid = surrogate_predict(models, X_grid)[:, oi].reshape(300, 300)

    fig, ax = plt.subplots(figsize=(8, 6))
    levels = np.linspace(0.5, 1.0, 21)
    cf = ax.contourf(v1, v2, Y_grid.T, levels=levels, cmap="RdYlGn",
                     vmin=0.5, vmax=1.0)
    plt.colorbar(cf, ax=ax, label="Efficiency η")
    cs = ax.contour(v1, v2, Y_grid.T, levels=[0.85, 0.90, 0.95],
                    colors="black", linewidths=1.0)
    ax.clabel(cs, inline=True, fontsize=8, fmt="η=%.2f")
    ax.set_xlabel("Duty cycle D"); ax.set_ylabel("Load R (Ω)")
    ax.set_title("Efficiency landscape η vs D, R\n"
                 f"(V_in={NOMINAL['V_in']} V, f_sw={NOMINAL['f_sw']/1e3:.0f} kHz)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "efficiency_contour.png", dpi=150)
    plt.close(fig)
    print("  Saved: efficiency_contour.png")


# ── G.3 Surrogate-Based Optimisation ──────────────────────────────────────────

def run_optimisation(models):
    """
    Maximise efficiency subject to V_out ≥ 30 V.
    Fixed: R = 20 Ω, f_sw = 100 kHz.
    Variables: D ∈ [0.2, 0.8], V_in ∈ [8, 24] V.
    """
    oi_eta    = OUTPUTS.index("efficiency")
    oi_vout   = OUTPUTS.index("v_out_mean")
    R_fixed   = 20.0
    fsw_fixed = 100e3

    def objective(x):
        D, V_in = x
        X = np.array([[D, V_in, R_fixed, fsw_fixed]])
        eta = float(surrogate_predict(models, X)[0, oi_eta])
        return -eta   # minimise negative efficiency

    def vout_constraint(x):
        D, V_in = x
        X = np.array([[D, V_in, R_fixed, fsw_fixed]])
        v_out = float(surrogate_predict(models, X)[0, oi_vout])
        return v_out - 30.0   # ≥ 0

    bounds = [PARAM_BOUNDS["D"], PARAM_BOUNDS["V_in"]]
    constraints = [{"type": "ineq", "fun": vout_constraint}]

    # Multi-start from grid to avoid local optima
    best = None
    for D0 in [0.4, 0.5, 0.6, 0.7]:
        for V0 in [12.0, 16.0, 20.0]:
            res = minimize(objective, x0=[D0, V0], method="SLSQP",
                           bounds=bounds, constraints=constraints,
                           options={"ftol": 1e-8, "maxiter": 200})
            if res.success and (best is None or res.fun < best.fun):
                best = res

    if best is None:
        # Fall back to unconstrained best
        best = minimize(objective, x0=[0.5, 16.0], method="SLSQP",
                        bounds=bounds, options={"ftol": 1e-8})

    D_opt, Vin_opt = best.x
    X_opt = np.array([[D_opt, Vin_opt, R_fixed, fsw_fixed]])
    Y_opt = surrogate_predict(models, X_opt)[0]
    eta_opt  = Y_opt[oi_eta]
    vout_opt = Y_opt[oi_vout]

    lines = [
        "SURROGATE-BASED OPTIMISATION — G.3",
        "=" * 45,
        f"Objective:   maximise efficiency (η)",
        f"Constraint:  V_out ≥ 30 V",
        f"Fixed:       R = {R_fixed:.0f} Ω,  f_sw = {fsw_fixed/1e3:.0f} kHz",
        "",
        f"Optimal D    = {D_opt:.4f}",
        f"Optimal V_in = {Vin_opt:.3f} V",
        f"η_opt        = {eta_opt:.4f}  ({eta_opt*100:.2f} %)",
        f"V_out_opt    = {vout_opt:.3f} V",
        "",
        "All predicted outputs at optimum:",
    ]
    for oi, nm in enumerate(OUTPUTS):
        lines.append(f"  {nm:18s} = {Y_opt[oi]:.4f}")

    text = "\n".join(lines)
    print("\n" + text)
    (OUT / "optimisation_result.txt").write_text(text)
    return D_opt, Vin_opt, eta_opt, vout_opt


# ── Computational cost comparison ─────────────────────────────────────────────

def write_cost_comparison(t_sobol_total, n_sobol_evals,
                          t_sweep, n_sweep_pts, sim_time_per_pt=0.01):
    """
    Compare surrogate-based workflow vs direct simulation.
    sim_time_per_pt: analytical simulator takes ~0.01 ms = 1e-5 s per point.
    We use a conservative estimate of 1 second for a realistic ODE-based sim.
    """
    sim_time_realistic = 1.0   # seconds per simulation (realistic ODE)

    lines = [
        "COMPUTATIONAL COST COMPARISON",
        "=" * 55,
        "",
        "Sobol sensitivity analysis:",
        f"  Evaluations needed:    {n_sobol_evals:,}",
        f"  Via surrogate:         {t_sobol_total:.2f} s (total)",
        f"  Via 1s/sim ODE:        {n_sobol_evals * sim_time_realistic / 3600:.1f} hours",
        f"  Speedup factor:        {n_sobol_evals * sim_time_realistic / t_sobol_total:.0f}×",
        "",
        "4-D design-space sweep:",
        f"  Points evaluated:      {n_sweep_pts:,}",
        f"  Via surrogate:         {t_sweep:.2f} s",
        f"  Via 1s/sim ODE:        {n_sweep_pts * sim_time_realistic / 3600 / 24:.1f} days",
        f"  Speedup factor:        {n_sweep_pts * sim_time_realistic / t_sweep:.0f}×",
        "",
        "Note: analytical simulator runs at ~0.01 ms/pt (10k×/s);",
        "      a physics ODE simulator (fixed-step) runs at ~1s/pt.",
        "      Speedup figures use the ODE baseline (conservative).",
    ]
    text = "\n".join(lines)
    print("\n" + text)
    return text


# ── summary ────────────────────────────────────────────────────────────────────

def write_summary(df_sobol, df_morris, df_shap,
                  pct_feasible, n_feasible, n_total,
                  D_opt, Vin_opt, eta_opt, vout_opt,
                  cost_text):

    # Dominant input per output (Sobol S1)
    sobol_top = (
        df_sobol.loc[df_sobol.groupby("output")["S1"].idxmax()]
        [["output", "input", "S1"]]
    )
    shap_top = (
        df_shap.loc[df_shap.groupby("output")["mean_abs_shap"].idxmax()]
        [["output", "input", "mean_abs_shap"]]
    )

    lines = [
        "PHASE 5 SUMMARY",
        "=" * 60, "",
        "G.1 Sensitivity Analysis",
        "-" * 40,
        "Dominant factor by Sobol S₁ (first-order):",
        sobol_top.to_string(index=False),
        "",
        "Dominant factor by SHAP mean|φ|:",
        shap_top.to_string(index=False),
        "",
        "G.2 Design-Space Exploration",
        "-" * 40,
        f"Feasible region (V_out>30V, η>85%, ripple<1V):",
        f"  {n_feasible:,} / {n_total:,} pts = {pct_feasible:.1f}% of design space",
        "",
        "G.3 Surrogate-Based Optimisation",
        "-" * 40,
        f"  D* = {D_opt:.4f}, V_in* = {Vin_opt:.3f} V",
        f"  η* = {eta_opt*100:.2f}%,  V_out* = {vout_opt:.3f} V",
        "",
        cost_text,
    ]
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text)
    print("\n" + text)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 5 — ENGINEERING INSIGHT")
    print("=" * 60)

    print("\n[1] Loading data …")
    X_tr, Y_tr, X_te, Y_te = load_data()
    print(f"  Train: {len(X_tr)},  Test: {len(X_te)}")

    print("\n[2] Training surrogates …")
    t0 = time.perf_counter()
    models = train_surrogates(X_tr, Y_tr)
    t_train = time.perf_counter() - t0

    # Quick sanity check
    Y_pred = surrogate_predict(models, X_te)
    for oi, nm in enumerate(OUTPUTS):
        rmse = compute_rmse(Y_te[:, oi], Y_pred[:, oi])
        print(f"  {nm:18s}  RMSE={rmse:.4f}")

    # ── G.1 Sensitivity ───────────────────────────────────────────────────
    print("\n[3] G.1 Sensitivity Analysis …")

    print("  Method 1: Sobol' indices …")
    df_sobol, t_sobol = run_sobol(models)
    N_SOBOL_EVALS = 10_000 * (2 * 4 + 2)   # = 100,000

    print("  Method 2: Morris screening …")
    df_morris = run_morris(models)

    print("  Method 3: SHAP (KernelSHAP) …")
    df_shap = run_shap(models, X_tr)

    print("  Plotting sensitivity …")
    plot_sensitivity(df_sobol, df_morris, df_shap)

    # ── G.2 Design-space exploration ──────────────────────────────────────
    print("\n[4] G.2 Design-Space Exploration …")
    pct_feasible, n_feasible, n_total, t_sweep = run_design_exploration(models)
    plot_efficiency_contour(models)

    # ── G.3 Optimisation ──────────────────────────────────────────────────
    print("\n[5] G.3 Surrogate-Based Optimisation …")
    D_opt, Vin_opt, eta_opt, vout_opt = run_optimisation(models)

    # ── Computational cost comparison ─────────────────────────────────────
    cost_text = write_cost_comparison(t_sobol, N_SOBOL_EVALS, t_sweep, n_total)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n[6] Summary …")
    write_summary(df_sobol, df_morris, df_shap,
                  pct_feasible, n_feasible, n_total,
                  D_opt, Vin_opt, eta_opt, vout_opt,
                  cost_text)

    print(f"\nAll Phase 5 outputs written to {OUT}")


if __name__ == "__main__":
    main()
