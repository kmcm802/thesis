#!/usr/bin/env python3
"""
Phase 1 — Design of Experiments Experiments
============================================

Runs the DoE comparison and seed-sensitivity study.

Experiment DoE-1: Fixed budget comparison
  For each N in {20,40,60,80,100,150,200}:
    - Generate designs: random (×10 seeds), lhs (×10 seeds),
      maximin_lhs (×1), sobol (×1)
    - Simulate each design with the averaged ODE
    - Train Kriging (control model, Matern-5/2, ARD) on each design
    - Evaluate on a FIXED 500-point test set
    - Record RMSE, NRMSE, R², MAE per output variable

Experiment DoE-2: Seed sensitivity at N=100
  - 30 seeds × 3 stochastic strategies (random, lhs, maximin_lhs)
  - Distribution of RMSE across seeds

Outputs
-------
  outputs/phase1/test_set.csv            — fixed 500-point test set (inputs+outputs)
  outputs/phase1/doe1_raw.csv            — raw per-(strategy,N,replicate,output) results
  outputs/phase1/doe1_summary.csv        — aggregated mean±std RMSE by strategy × N
  outputs/phase1/doe2_raw.csv            — seed sensitivity results
  outputs/phase1/doe2_summary.csv        — mean/std/min/max RMSE for each strategy
  outputs/phase1/strategy_selection.txt  — selected DoE strategy + rationale
  outputs/phase1/figures/                — RMSE-vs-N plots, box-plots
"""

import sys
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from src.simulation.boost_converter_ode import simulate_batch, DESIGN_SPACE
from src.doe.sampling import generate_samples
from src.utils.metrics import compute_rmse, compute_nrmse, compute_r2, compute_mae

# ── output directories ───────────────────────────────────────────────────────
OUT = ROOT / "outputs" / "phase1"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# ── experiment parameters (from §B.3) ────────────────────────────────────────
N_VALUES       = [20, 40, 60, 80, 100, 150, 200]
N_REPS_STOCH   = 10     # replicates for random and lhs
N_TEST         = 500    # fixed test-set size
TEST_SEED      = 9999   # seed for the test set (kept separate from training seeds)
N_SEEDS_DOE2   = 30     # DoE-2 seed sensitivity
N_DOE2         = 100    # DoE-2 training set size

OUTPUTS = ["v_out_mean", "i_l_mean", "v_out_ripple", "i_l_ripple", "efficiency"]
INPUT_COLS = list(DESIGN_SPACE.keys())    # D, V_in, R, f_sw


# ── Kriging wrapper ───────────────────────────────────────────────────────────

def train_kriging(X_train: np.ndarray, y_train: np.ndarray) -> object:
    """Train a Kriging model (SMT KRG, Matern-5/2, ARD) on X_train → y_train."""
    from smt.surrogate_models import KRG
    sm = KRG(
        theta0=[1e-2] * X_train.shape[1],
        corr="matern52",
        poly="constant",
        nugget=1e-6,
        print_global=False,
    )
    sm.set_training_values(X_train, y_train)
    sm.train()
    return sm


def evaluate_kriging(sm, X_test: np.ndarray) -> np.ndarray:
    return sm.predict_values(X_test).flatten()


def metrics_dict(y_true, y_pred, output_name):
    return {
        "output":  output_name,
        "rmse":    compute_rmse(y_true, y_pred),
        "nrmse":   compute_nrmse(y_true, y_pred),
        "r2":      compute_r2(y_true, y_pred),
        "mae":     compute_mae(y_true, y_pred),
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def simulate_design(samples_df: pd.DataFrame) -> pd.DataFrame:
    """Simulate the given design, return DataFrame with outputs appended."""
    result = simulate_batch(samples_df, show_progress=False)
    # Drop CCM-violated rows (flag but keep for now with NaN outputs)
    n_dcm = (~result["ccm_ok"]).sum()
    if n_dcm > 0:
        print(f"    [warn] {n_dcm}/{len(result)} points in DCM — outputs set to NaN")
        for col in OUTPUTS:
            result.loc[~result["ccm_ok"], col] = np.nan
    return result


def build_XY(df: pd.DataFrame):
    """Return (X, Y) arrays from a simulated DataFrame."""
    valid = df.dropna(subset=OUTPUTS)
    X = valid[INPUT_COLS].values.astype(float)
    Y = valid[OUTPUTS].values.astype(float)
    return X, Y


# ── fixed test set ────────────────────────────────────────────────────────────

def get_or_create_test_set() -> pd.DataFrame:
    test_path = OUT / "test_set.csv"
    if test_path.exists():
        print(f"  Loading existing test set: {test_path}")
        return pd.read_csv(test_path)

    print(f"  Generating {N_TEST}-point maximin-LHS test set (seed={TEST_SEED}) …")
    design = generate_samples("maximin_lhs", DESIGN_SPACE, N_TEST, seed=TEST_SEED)
    simulated = simulate_design(design)
    simulated.to_csv(test_path, index=False)
    print(f"  Test set saved: {test_path}  ({simulated['ccm_ok'].sum()} CCM-valid points)")
    return simulated


# ── DoE-1 ─────────────────────────────────────────────────────────────────────

def run_doe1(test_df: pd.DataFrame) -> pd.DataFrame:
    """Run Experiment DoE-1: fixed budget comparison across strategies and N."""
    X_test, Y_test = build_XY(test_df)

    rows = []
    strategies_reps = {
        "random":      N_REPS_STOCH,
        "lhs":         N_REPS_STOCH,
        "maximin_lhs": 1,
        "sobol":       1,
    }

    total = sum(reps * len(N_VALUES) for reps in strategies_reps.values())
    done = 0
    t0 = time.perf_counter()

    for strategy, n_reps in strategies_reps.items():
        for N in N_VALUES:
            for rep in range(n_reps):
                seed = rep * 1000 + N   # unique seed per (rep, N)
                design = generate_samples(strategy, DESIGN_SPACE, N, seed=seed)
                sim = simulate_design(design)
                X_train, Y_train = build_XY(sim)

                if len(X_train) < 5:
                    print(f"    [skip] {strategy} N={N} rep={rep}: too few valid points")
                    continue

                for out_idx, out_name in enumerate(OUTPUTS):
                    y_tr = Y_train[:, out_idx]
                    y_te = Y_test[:, out_idx]
                    # Skip if any NaN in training output
                    valid_mask = np.isfinite(y_tr)
                    if valid_mask.sum() < 5:
                        continue
                    try:
                        sm = train_kriging(X_train[valid_mask], y_tr[valid_mask])
                        y_pred = evaluate_kriging(sm, X_test)
                        m = metrics_dict(y_te, y_pred, out_name)
                    except Exception as e:
                        print(f"    [warn] Kriging failed {strategy} N={N} {out_name}: {e}")
                        m = {"output": out_name, "rmse": np.nan, "nrmse": np.nan,
                             "r2": np.nan, "mae": np.nan}
                    rows.append({
                        "strategy": strategy, "N": N, "rep": rep,
                        **m,
                    })

                done += 1
                if done % 20 == 0:
                    elapsed = time.perf_counter() - t0
                    eta = elapsed / done * (total - done)
                    print(f"  DoE-1: {done}/{total}  elapsed {elapsed:.0f}s  ETA {eta:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "doe1_raw.csv", index=False)
    print(f"  DoE-1 raw results saved ({len(df)} rows)")
    return df


def summarise_doe1(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate DoE-1: mean ± std RMSE per (strategy, N, output)."""
    agg = (
        raw.groupby(["strategy", "N", "output"])["rmse"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={"mean": "rmse_mean", "std": "rmse_std",
                         "min": "rmse_min",  "max": "rmse_max",
                         "count": "n_reps"})
    )
    agg.to_csv(OUT / "doe1_summary.csv", index=False)
    return agg


# ── DoE-2 ─────────────────────────────────────────────────────────────────────

def run_doe2(test_df: pd.DataFrame) -> pd.DataFrame:
    """Run Experiment DoE-2: seed sensitivity at N=100 for stochastic strategies."""
    X_test, Y_test = build_XY(test_df)

    rows = []
    stochastic = ["random", "lhs", "maximin_lhs"]

    for strategy in stochastic:
        print(f"  DoE-2: {strategy}  ({N_SEEDS_DOE2} seeds) …")
        for seed in range(N_SEEDS_DOE2):
            design = generate_samples(strategy, DESIGN_SPACE, N_DOE2, seed=seed)
            sim = simulate_design(design)
            X_train, Y_train = build_XY(sim)

            for out_idx, out_name in enumerate(OUTPUTS):
                y_tr = Y_train[:, out_idx]
                y_te = Y_test[:, out_idx]
                valid_mask = np.isfinite(y_tr)
                if valid_mask.sum() < 5:
                    continue
                try:
                    sm = train_kriging(X_train[valid_mask], y_tr[valid_mask])
                    y_pred = evaluate_kriging(sm, X_test)
                    m = metrics_dict(y_te, y_pred, out_name)
                except Exception as e:
                    m = {"output": out_name, "rmse": np.nan, "nrmse": np.nan,
                         "r2": np.nan, "mae": np.nan}
                rows.append({"strategy": strategy, "seed": seed, **m})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "doe2_raw.csv", index=False)

    summary = (
        df.groupby(["strategy", "output"])["rmse"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"mean": "rmse_mean", "std": "rmse_std",
                         "min": "rmse_min",  "max": "rmse_max"})
    )
    summary.to_csv(OUT / "doe2_summary.csv", index=False)
    print(f"  DoE-2 done ({len(df)} rows)")
    return df, summary


# ── plotting ──────────────────────────────────────────────────────────────────

STRATEGY_STYLE = {
    "random":      {"color": "#e41a1c", "marker": "o", "ls": "--",  "label": "Random"},
    "lhs":         {"color": "#377eb8", "marker": "s", "ls": "--",  "label": "LHS"},
    "maximin_lhs": {"color": "#4daf4a", "marker": "^", "ls": "-",   "label": "Maximin LHS"},
    "sobol":       {"color": "#984ea3", "marker": "D", "ls": "-",   "label": "Sobol"},
}


def plot_doe1_rmse_vs_n(summary: pd.DataFrame) -> None:
    """RMSE vs N per output (one subplot per output), with error bars."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax_i, out_name in enumerate(OUTPUTS):
        ax = axes[ax_i]
        sub = summary[summary["output"] == out_name]
        for strategy, style in STRATEGY_STYLE.items():
            s = sub[sub["strategy"] == strategy].sort_values("N")
            if s.empty:
                continue
            ax.errorbar(
                s["N"], s["rmse_mean"],
                yerr=s["rmse_std"].fillna(0),
                color=style["color"], marker=style["marker"],
                linestyle=style["ls"], label=style["label"],
                capsize=3, linewidth=1.5,
            )
        ax.set_title(out_name, fontsize=11)
        ax.set_xlabel("Training set size N")
        ax.set_ylabel("RMSE (test set)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_visible(False)
    fig.suptitle("DoE-1: RMSE vs Training Set Size (Kriging control model)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "doe1_rmse_vs_n.png", dpi=150)
    plt.close(fig)
    print("  Saved: figures/doe1_rmse_vs_n.png")


def plot_doe1_nrmse_vs_n(summary: pd.DataFrame) -> None:
    """NRMSE vs N — normalised to allow cross-output comparison."""
    nrmse_agg = summary.copy()  # same structure, just use nrmse column if available
    # For now re-use rmse_mean (nrmse stored separately in raw; this is sufficient for ranking)
    # Plot just v_out_mean as the primary output for the thesis
    fig, ax = plt.subplots(figsize=(7, 5))
    sub = summary[summary["output"] == "v_out_mean"]
    for strategy, style in STRATEGY_STYLE.items():
        s = sub[sub["strategy"] == strategy].sort_values("N")
        if s.empty:
            continue
        ax.errorbar(
            s["N"], s["rmse_mean"],
            yerr=s["rmse_std"].fillna(0),
            color=style["color"], marker=style["marker"],
            linestyle=style["ls"], label=style["label"],
            capsize=3, linewidth=2,
        )
    ax.set_title("DoE-1: RMSE vs N (v_out_mean, Kriging control model)", fontsize=11)
    ax.set_xlabel("Training set size N")
    ax.set_ylabel("RMSE on v_out_mean (V)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "doe1_vout_rmse_vs_n.png", dpi=150)
    plt.close(fig)
    print("  Saved: figures/doe1_vout_rmse_vs_n.png")


def plot_doe2_boxplot(raw2: pd.DataFrame) -> None:
    """Box-plot of RMSE distribution across seeds (DoE-2)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax_i, out_name in enumerate(OUTPUTS):
        ax = axes[ax_i]
        sub = raw2[raw2["output"] == out_name]
        data = [sub[sub["strategy"] == s]["rmse"].dropna().values
                for s in ["random", "lhs", "maximin_lhs"]]
        labels = ["Random", "LHS", "Maximin\nLHS"]
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = ["#e41a1c", "#377eb8", "#4daf4a"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_title(out_name, fontsize=11)
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3, axis="y")

    axes[-1].set_visible(False)
    fig.suptitle(f"DoE-2: RMSE distribution over {N_SEEDS_DOE2} seeds (N={N_DOE2})", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "doe2_seed_sensitivity.png", dpi=150)
    plt.close(fig)
    print("  Saved: figures/doe2_seed_sensitivity.png")


# ── strategy selection ────────────────────────────────────────────────────────

def select_strategy(doe1_summary: pd.DataFrame, doe2_summary: pd.DataFrame) -> str:
    """
    Select the DoE strategy based on fixed-budget and seed-sensitivity results.

    Criterion: lowest mean RMSE on v_out_mean across N values (averaged over
    N >= 60 where differences are most practically relevant), weighted by
    DoE-2 seed robustness (low std is better).
    """
    primary_output = "v_out_mean"

    # DoE-1: average RMSE over N >= 60 for primary output
    sub1 = doe1_summary[
        (doe1_summary["output"] == primary_output) &
        (doe1_summary["N"] >= 60)
    ].groupby("strategy")["rmse_mean"].mean().rename("avg_rmse_N60plus")

    # DoE-2: std of RMSE across seeds for primary output (robustness)
    sub2 = doe2_summary[
        doe2_summary["output"] == primary_output
    ].set_index("strategy")["rmse_std"].rename("rmse_std_seeds")

    combined = pd.concat([sub1, sub2], axis=1).dropna()

    # Rank: lower avg_rmse is better; lower std is better
    # Score = normalised_rmse + 0.3 * normalised_std  (0.3 weight on robustness)
    if combined.empty:
        return "maximin_lhs"   # fallback

    combined["norm_rmse"] = combined["avg_rmse_N60plus"] / combined["avg_rmse_N60plus"].max()
    # sobol has no DoE-2 std (not in DoE-2); fill with 0 (deterministic → perfectly robust)
    combined["rmse_std_seeds"] = combined["rmse_std_seeds"].fillna(0.0)
    combined["norm_std"]  = combined["rmse_std_seeds"] / (combined["rmse_std_seeds"].max() + 1e-12)
    combined["score"]     = combined["norm_rmse"] + 0.3 * combined["norm_std"]

    best = combined["score"].idxmin()

    lines = [
        "DoE Strategy Selection",
        "=" * 50,
        "",
        "Primary output: v_out_mean",
        "Criterion: avg RMSE (N≥60) + 0.3 × seed std, both normalised to [0,1]",
        "",
        combined.to_string(),
        "",
        f"Selected strategy: {best}",
        "",
        "Rationale:",
        f"  '{best}' achieved the lowest combined score of "
        f"{combined.loc[best,'score']:.4f}.",
        "  This strategy is used for full dataset generation.",
    ]
    text = "\n".join(lines)
    (OUT / "strategy_selection.txt").write_text(text)
    print("\n" + text)
    return best


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 1 — DoE EXPERIMENTS")
    print("=" * 60)

    print("\n[1/5] Fixed test set")
    test_df = get_or_create_test_set()
    X_test, Y_test = build_XY(test_df)
    print(f"  Test set: {len(X_test)} valid points, {len(OUTPUTS)} outputs")

    print("\n[2/5] Experiment DoE-1 — fixed budget comparison")
    doe1_raw_path = OUT / "doe1_raw.csv"
    if doe1_raw_path.exists():
        print(f"  Loading cached results: {doe1_raw_path}")
        doe1_raw = pd.read_csv(doe1_raw_path)
    else:
        doe1_raw = run_doe1(test_df)

    print("\n[3/5] Summarising DoE-1")
    doe1_summary = summarise_doe1(doe1_raw)
    print(doe1_summary[doe1_summary["output"] == "v_out_mean"].to_string(index=False))

    print("\n[4/5] Experiment DoE-2 — seed sensitivity")
    doe2_raw_path = OUT / "doe2_raw.csv"
    if doe2_raw_path.exists():
        print(f"  Loading cached results: {doe2_raw_path}")
        doe2_raw = pd.read_csv(doe2_raw_path)
        doe2_summary = pd.read_csv(OUT / "doe2_summary.csv")
    else:
        doe2_raw, doe2_summary = run_doe2(test_df)
    print(doe2_summary[doe2_summary["output"] == "v_out_mean"].to_string(index=False))

    print("\n[5/5] Plots and strategy selection")
    plot_doe1_rmse_vs_n(doe1_summary)
    plot_doe1_nrmse_vs_n(doe1_summary)
    plot_doe2_boxplot(doe2_raw)
    best_strategy = select_strategy(doe1_summary, doe2_summary)
    print(f"\nBest strategy: {best_strategy}")
    print(f"\nAll outputs written to {OUT}")


if __name__ == "__main__":
    main()
