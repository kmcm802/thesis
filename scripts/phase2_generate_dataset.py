#!/usr/bin/env python3
"""
Phase 2 — Full Dataset Generation
===================================

Generates the production training dataset and validates the fixed test set,
using the selected maximin LHS design:

  - Strategy  : maximin_lhs
  - N_train   : 200 (supports learning curves up to N=200)
  - N_test    : 500 (fixed test set reused from Phase 1)
  - Seed      : 42 (canonical seed for all production runs)

Outputs
-------
  outputs/phase2/train_200.csv          — 200-point training dataset
  outputs/phase2/test_500.csv           — copy of Phase 1 test set (canonical path)
  outputs/phase2/ccm_report.txt         — CCM boundary analysis
  outputs/phase2/figures/               — scatter plots, correlation heatmap
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

from src.simulation.boost_converter_ode import (
    simulate_batch, DESIGN_SPACE,
)
from src.doe.sampling import generate_samples

# ── config ────────────────────────────────────────────────────────────────────
STRATEGY   = "maximin_lhs"
N_TRAIN    = 200
TRAIN_SEED = 42
N_TEST     = 500
TEST_SEED  = 9999

INPUT_COLS  = list(DESIGN_SPACE.keys())        # D, V_in, R, f_sw
OUTPUT_COLS = ["v_out_mean", "i_l_mean", "v_out_ripple", "i_l_ripple", "efficiency"]

OUT = ROOT / "outputs" / "phase2"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

PHASE1_TEST = ROOT / "outputs" / "phase1" / "test_set.csv"


# ── CCM analysis ──────────────────────────────────────────────────────────────

def ccm_report(df: pd.DataFrame, label: str) -> str:
    n_total = len(df)
    n_ccm   = int(df["ccm_ok"].sum())
    n_dcm   = n_total - n_ccm

    lines = [
        f"CCM Analysis — {label}",
        "-" * 40,
        f"  Total points  : {n_total}",
        f"  CCM valid     : {n_ccm}  ({n_ccm/n_total*100:.1f}%)",
        f"  DCM violations: {n_dcm}  ({n_dcm/n_total*100:.1f}%)",
    ]

    if n_dcm > 0:
        dcm = df[~df["ccm_ok"]]
        lines += [
            "",
            "  DCM parameter ranges:",
            f"    D    : [{dcm['D'].min():.3f}, {dcm['D'].max():.3f}]",
            f"    V_in : [{dcm['V_in'].min():.2f}, {dcm['V_in'].max():.2f}] V",
            f"    R    : [{dcm['R'].min():.1f}, {dcm['R'].max():.1f}] Ω",
            f"    f_sw : [{dcm['f_sw'].min()/1e3:.1f}, {dcm['f_sw'].max()/1e3:.1f}] kHz",
            "",
            "  Note: DCM points excluded from surrogate training.",
            "  The averaged ODE model is only valid in CCM.",
            "  Condition: i_L_min = i_l_mean - ΔI_L/2 > 0",
            "             where ΔI_L = V_in·D / (L·f_sw)",
        ]

    return "\n".join(lines)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_input_distributions(df: pd.DataFrame, label: str, fname: str) -> None:
    """Histograms + pair scatter (D vs V_in, R vs f_sw) for input space coverage."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 4, figure=fig)

    # Top row: histograms of each input
    for i, col in enumerate(INPUT_COLS):
        ax = fig.add_subplot(gs[0, i])
        ax.hist(df[col], bins=20, color="#4daf4a", edgecolor="white", linewidth=0.5)
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    # Middle row: 2D scatter pairs
    pairs = [("D", "V_in"), ("D", "R"), ("V_in", "R"), ("R", "f_sw")]
    for i, (x, y) in enumerate(pairs):
        ax = fig.add_subplot(gs[1, i])
        ccm_mask = df["ccm_ok"]
        ax.scatter(df.loc[ccm_mask, x], df.loc[ccm_mask, y],
                   s=15, alpha=0.7, color="#4daf4a", label="CCM")
        if (~ccm_mask).any():
            ax.scatter(df.loc[~ccm_mask, x], df.loc[~ccm_mask, y],
                       s=15, alpha=0.7, color="#e41a1c", marker="x", label="DCM")
        ax.set_xlabel(x); ax.set_ylabel(y)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    # Bottom row: output distributions (CCM-valid only)
    valid = df[df["ccm_ok"]]
    out_cols = [c for c in OUTPUT_COLS if c in valid.columns]
    for i, col in enumerate(out_cols[:4]):
        ax = fig.add_subplot(gs[2, i])
        ax.hist(valid[col], bins=20, color="#377eb8", edgecolor="white", linewidth=0.5)
        ax.set_title(col, fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{label} — Input Space Coverage & Output Distributions", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def plot_output_correlations(df: pd.DataFrame, label: str, fname: str) -> None:
    """Correlation heatmap of inputs vs outputs (CCM-valid only)."""
    valid = df[df["ccm_ok"]]
    cols = INPUT_COLS + OUTPUT_COLS
    cols = [c for c in cols if c in valid.columns]
    corr = valid[cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=9)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(corr.values[i,j]) < 0.7 else "white")
    ax.set_title(f"{label} — Pearson Correlation (inputs & outputs)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


def plot_output_scatter(df: pd.DataFrame, label: str, fname: str) -> None:
    """Scatter of V_out_mean and efficiency vs key inputs."""
    valid = df[df["ccm_ok"]]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    targets = ["v_out_mean", "efficiency"]
    for row_i, target in enumerate(targets):
        if target not in valid.columns:
            continue
        for col_i, inp in enumerate(INPUT_COLS):
            ax = axes[row_i, col_i]
            ax.scatter(valid[inp], valid[target], s=10, alpha=0.6, color="#377eb8")
            ax.set_xlabel(inp, fontsize=9)
            ax.set_ylabel(target, fontsize=9)
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"{label} — Outputs vs Inputs (scatter)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=150)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 2 — FULL DATASET GENERATION")
    print("=" * 60)

    # ── 1. Training dataset ───────────────────────────────────────────────────
    train_path = OUT / "train_200.csv"
    if train_path.exists():
        print(f"\n[1/3] Loading existing training dataset: {train_path}")
        train_df = pd.read_csv(train_path)
    else:
        print(f"\n[1/3] Generating {N_TRAIN}-point {STRATEGY} training dataset (seed={TRAIN_SEED}) …")
        t0 = time.perf_counter()
        design = generate_samples(STRATEGY, DESIGN_SPACE, N_TRAIN, seed=TRAIN_SEED)
        train_df = simulate_batch(design, show_progress=True)
        elapsed = time.perf_counter() - t0
        train_df.to_csv(train_path, index=False)
        print(f"  Done in {elapsed:.1f}s — saved: {train_path}")

    n_ccm = int(train_df["ccm_ok"].sum())
    n_dcm = len(train_df) - n_ccm
    print(f"  CCM valid: {n_ccm}/{len(train_df)}  ({n_ccm/len(train_df)*100:.1f}%)")

    # ── 2. Test dataset (canonical copy from Phase 1) ─────────────────────────
    test_path = OUT / "test_500.csv"
    if test_path.exists():
        print(f"\n[2/3] Loading existing test dataset: {test_path}")
        test_df = pd.read_csv(test_path)
    elif PHASE1_TEST.exists():
        print(f"\n[2/3] Copying Phase 1 test set → {test_path}")
        test_df = pd.read_csv(PHASE1_TEST)
        test_df.to_csv(test_path, index=False)
    else:
        print(f"\n[2/3] Phase 1 test set not found — regenerating ({N_TEST} points, seed={TEST_SEED}) …")
        design = generate_samples("maximin_lhs", DESIGN_SPACE, N_TEST, seed=TEST_SEED)
        test_df = simulate_batch(design, show_progress=True)
        test_df.to_csv(test_path, index=False)

    n_test_ccm = int(test_df["ccm_ok"].sum())
    print(f"  Test set: {len(test_df)} points, {n_test_ccm} CCM-valid")

    # ── 3. CCM report ─────────────────────────────────────────────────────────
    print("\n[3/3] CCM boundary analysis …")
    report_lines = [
        "PHASE 2 — CCM BOUNDARY REPORT",
        "=" * 50,
        "",
        ccm_report(train_df, f"Training set (N={len(train_df)})"),
        "",
        ccm_report(test_df,  f"Test set     (N={len(test_df)})"),
        "",
        "Action taken: DCM rows kept in CSV (ccm_ok=False) but excluded",
        "from all surrogate training and evaluation via NaN masking.",
    ]
    report_text = "\n".join(report_lines)
    print(report_text)
    (OUT / "ccm_report.txt").write_text(report_text)

    # ── 4. Plots ───────────────────────────────────────────────────────────────
    print("\nGenerating figures …")
    plot_input_distributions(train_df, f"Training set (N={N_TRAIN}, {STRATEGY})",
                             "train_input_coverage.png")
    print("  Saved: train_input_coverage.png")

    plot_output_correlations(train_df, f"Training set (N={N_TRAIN})",
                             "train_correlation_heatmap.png")
    print("  Saved: train_correlation_heatmap.png")

    plot_output_scatter(train_df, f"Training set (N={N_TRAIN})",
                        "train_output_scatter.png")
    print("  Saved: train_output_scatter.png")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY")
    print("=" * 60)
    valid_train = train_df[train_df["ccm_ok"]]
    print(f"\nTraining dataset (CCM-valid, N={len(valid_train)}):")
    print(valid_train[OUTPUT_COLS].describe().round(4).to_string())
    print(f"\nAll outputs written to {OUT}")


if __name__ == "__main__":
    main()
