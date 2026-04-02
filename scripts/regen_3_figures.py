#!/usr/bin/env python3
"""
regen_3_figures.py
==================
Regenerate 3 specific figures using GP-SE trained on all 100 points
from data/raw/simulation_results_hifi.csv.

Figures produced:
  design_space_contours.png  — V_out_mean & efficiency over D×R grid
  sobol_bar_chart.png        — Sobol first-order indices, all inputs × all outputs
  reliability_diagram.png    — Reliability diagram from calibration.csv (GP-SE)
"""

from __future__ import annotations
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).resolve().parent.parent
DEST    = ROOT / "thesis_figures"
DEST.mkdir(parents=True, exist_ok=True)

INPUT_COLS  = ["D", "V_in", "R", "f_sw"]
OUTPUT_COLS = ["v_out_mean", "i_l_mean", "v_out_ripple", "i_l_ripple", "efficiency"]

STYLE = {
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelsize":    10,
    "axes.titlesize":    10,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "font.family":       "sans-serif",
}

def save(fig, name):
    p = DEST / name
    fig.savefig(p, dpi=300, bbox_inches="tight")
    print(f"  Saved → {p}")


# ─── Build GP-SE model ───────────────────────────────────────────────────────

def train_gp_se(X: np.ndarray, y: np.ndarray):
    """Train GP-SE (squar_exp, ARD, n_start=10) with StandardScaler on X and y."""
    from smt.surrogate_models import KRG
    sc_x = StandardScaler()
    Xs   = sc_x.fit_transform(X)
    ym   = float(y.mean())
    ys   = float(y.std()) + 1e-8
    yn   = (y.ravel() - ym) / ys
    d    = Xs.shape[1]
    sm   = KRG(
        theta0      = [1e-2] * d,
        corr        = "squar_exp",
        poly        = "constant",
        nugget      = 1e-6,
        n_start     = 10,
        print_global= False,
    )
    sm.set_training_values(Xs, yn.reshape(-1, 1))
    sm.train()

    class _Model:
        def predict(self_, Xnew):
            return sm.predict_values(sc_x.transform(Xnew)).ravel() * ys + ym
        def predict_std(self_, Xnew):
            var = sm.predict_variances(sc_x.transform(Xnew)).ravel()
            return np.sqrt(np.maximum(var, 0.0)) * ys

    return _Model()


# ─── Load data & train one GP per output ─────────────────────────────────────

print("Loading data …")
sim = pd.read_csv(ROOT / "data/raw/simulation_results_hifi.csv")
X   = sim[INPUT_COLS].values.astype(float)
Y   = sim[OUTPUT_COLS].values.astype(float)
print(f"  {X.shape[0]} samples, {X.shape[1]} inputs, {Y.shape[1]} outputs")

print("Training GP-SE on all 100 points …")
gp_models = {}
for oi, out in enumerate(OUTPUT_COLS):
    print(f"  [{out}] … ", end="", flush=True)
    gp_models[out] = train_gp_se(X, Y[:, oi])
    print("done")


# ─── Fig 1: design_space_contours.png ────────────────────────────────────────
print("\nFig 1: design_space_contours.png")

V_IN_FIXED = 15.0    # V
F_SW_FIXED = 55_000  # Hz  (55 kHz)
N_GRID     = 200

# Input ranges from data
D_range = (float(X[:, 0].min()), float(X[:, 0].max()))
R_range = (float(X[:, 2].min()), float(X[:, 2].max()))

D_vals = np.linspace(*D_range, N_GRID)
R_vals = np.linspace(*R_range, N_GRID)
DD, RR = np.meshgrid(D_vals, R_vals)

# Build query array: [D, V_in, R, f_sw]
X_grid = np.column_stack([
    DD.ravel(),
    np.full(N_GRID * N_GRID, V_IN_FIXED),
    RR.ravel(),
    np.full(N_GRID * N_GRID, F_SW_FIXED),
])

print(f"  Predicting on {N_GRID}×{N_GRID} grid …")
vout_grid = gp_models["v_out_mean"].predict(X_grid).reshape(N_GRID, N_GRID)
eta_grid  = gp_models["efficiency"].predict(X_grid).reshape(N_GRID, N_GRID)

with plt.rc_context(STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        r"Kriging (GP-SE) Design Space — $V_{in}=15\,\mathrm{V}$, $f_{sw}=55\,\mathrm{kHz}$",
        fontsize=12)

    # Left: V_out_mean
    ax = axes[0]
    cf = ax.contourf(D_vals, R_vals, vout_grid, levels=20, cmap="plasma")
    cs = ax.contour(D_vals, R_vals, vout_grid, levels=8,
                    colors="white", linewidths=0.5, alpha=0.6)
    ax.clabel(cs, fmt="%.0f V", fontsize=7, inline=True)
    cb = fig.colorbar(cf, ax=ax, shrink=0.92)
    cb.set_label(r"$\bar{V}_{out}$ (V)", fontsize=9)
    ax.set_xlabel("Duty cycle $D$", fontsize=10)
    ax.set_ylabel(r"Load resistance $R\;(\Omega)$", fontsize=10)
    ax.set_title(r"Mean Output Voltage $\bar{V}_{out}$", fontsize=10)
    # Mark training points
    ax.scatter(X[:, 0], X[:, 2], s=8, c="white", alpha=0.5,
               linewidths=0, label="Training pts")
    ax.legend(loc="upper left", fontsize=7, frameon=False)

    # Right: efficiency
    ax = axes[1]
    cf2 = ax.contourf(D_vals, R_vals, eta_grid, levels=20, cmap="RdYlGn",
                      vmin=0.80, vmax=0.98)
    cs2 = ax.contour(D_vals, R_vals, eta_grid, levels=8,
                     colors="black", linewidths=0.5, alpha=0.5)
    ax.clabel(cs2, fmt="%.2f", fontsize=7, inline=True)
    cb2 = fig.colorbar(cf2, ax=ax, shrink=0.92)
    cb2.set_label(r"$\eta$", fontsize=9)
    ax.set_xlabel("Duty cycle $D$", fontsize=10)
    ax.set_ylabel(r"Load resistance $R\;(\Omega)$", fontsize=10)
    ax.set_title(r"Conversion Efficiency $\eta$", fontsize=10)
    ax.scatter(X[:, 0], X[:, 2], s=8, c="black", alpha=0.4,
               linewidths=0, label="Training pts")
    ax.legend(loc="upper left", fontsize=7, frameon=False)

    fig.tight_layout()

save(fig, "design_space_contours.png")
plt.close(fig)


# ─── Fig 2: sobol_bar_chart.png ──────────────────────────────────────────────
print("\nFig 2: sobol_bar_chart.png  (Saltelli sampling, N=2048 base)")

from SALib.sample import saltelli
from SALib.analyze import sobol as sobol_analyze

# Define problem bounds from data ranges
problem = {
    "num_vars": 4,
    "names":    ["D", "V_in", "R", "f_sw"],
    "bounds":   [
        [float(X[:, 0].min()), float(X[:, 0].max())],
        [float(X[:, 1].min()), float(X[:, 1].max())],
        [float(X[:, 2].min()), float(X[:, 2].max())],
        [float(X[:, 3].min()), float(X[:, 3].max())],
    ],
}

N_BASE = 2048
print(f"  Generating Saltelli samples (N_base={N_BASE}) …")
np.random.seed(42)
param_values = saltelli.sample(problem, N_BASE, calc_second_order=False)
print(f"  {len(param_values)} surrogate evaluations …")

# Predict all outputs at once
Y_sobol = np.zeros((len(param_values), len(OUTPUT_COLS)))
for oi, out in enumerate(OUTPUT_COLS):
    Y_sobol[:, oi] = gp_models[out].predict(param_values)

print("  Analysing Sobol indices …")
S1 = np.zeros((len(OUTPUT_COLS), 4))
for oi, out in enumerate(OUTPUT_COLS):
    si = sobol_analyze.analyze(problem, Y_sobol[:, oi],
                               calc_second_order=False, seed=42)
    S1[oi] = np.clip(si["S1"], 0, 1)

OUT_LABELS = [r"$\bar{V}_{out}$", r"$\bar{I}_L$",
              r"$\Delta V_{out}$", r"$\Delta I_L$", r"$\eta$"]
INPUT_LABELS = ["$D$", r"$V_{in}$", "$R$", r"$f_{sw}$"]
COLORS = ["#d6604d", "#4393c3", "#2ca25f", "#7b2d8b"]

n_out  = len(OUTPUT_COLS)
n_inp  = 4
x      = np.arange(n_out)
width  = 0.18
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(11, 5))
    for ii in range(n_inp):
        bars = ax.bar(x + offsets[ii], S1[:, ii], width=width * 0.92,
                      label=INPUT_LABELS[ii], color=COLORS[ii],
                      alpha=0.88, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, S1[:, ii]):
            if val > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.008,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=6.5, color=COLORS[ii])
    ax.set_xticks(x)
    ax.set_xticklabels(OUT_LABELS, fontsize=11)
    ax.set_ylabel(r"First-order Sobol index $S_i$", fontsize=10)
    ax.set_title(
        "Sobol Sensitivity Analysis — First-Order Indices (Kriging Surrogate, GP-SE)",
        fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=9)
    ax.axhline(0, color="k", lw=0.6, alpha=0.4)
    fig.tight_layout()

save(fig, "sobol_bar_chart.png")
plt.close(fig)


# ─── Fig 3: reliability_diagram.png ──────────────────────────────────────────
print("\nFig 3: reliability_diagram.png")

cal_path = ROOT / "outputs/phase4/calibration.csv"
cal      = pd.read_csv(cal_path)
gp_cal   = cal[cal["model"] == "GP-SE"].sort_values(["output", "nominal"])

OUT_COLORS = ["#2166ac", "#d6604d", "#4dac26", "#7b2d8b", "#ff7f00"]
OUT_LABELS_PLAIN = [r"$\bar{V}_{out}$", r"$\bar{I}_L$",
                    r"$\Delta V_{out}$", r"$\Delta I_L$", r"$\eta$"]

# Known 95% empirical coverage values from the run
cov_95 = {
    "v_out_mean":   0.91,
    "i_l_mean":     0.85,
    "v_out_ripple": 0.79,
    "i_l_ripple":   0.92,
    "efficiency":   0.86,
}

with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6, label="Perfect calibration",
            zorder=1)

    # Shaded region: over-confident (above diagonal)
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.04, color="red")
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.04, color="green")
    ax.text(0.68, 0.28, "over-confident", fontsize=7.5, color="#cc4444",
            ha="center", rotation=38, alpha=0.7)
    ax.text(0.28, 0.72, "conservative", fontsize=7.5, color="#3a7a3a",
            ha="center", rotation=38, alpha=0.7)

    for out, lbl, c in zip(OUTPUT_COLS, OUT_LABELS_PLAIN, OUT_COLORS):
        sub = gp_cal[gp_cal["output"] == out].sort_values("nominal")
        if len(sub) == 0:
            continue
        ax.plot(sub["nominal"], sub["empirical"],
                marker="o", ms=5, lw=1.8, color=c, label=lbl, zorder=3)
        # Annotate the 95% point
        p95 = cov_95.get(out, None)
        if p95 is not None:
            ax.annotate(f"{p95:.0%}",
                        xy=(0.95, p95),
                        xytext=(0.95 + 0.015, p95 + (-0.035 if p95 > 0.88 else 0.01)),
                        fontsize=7, color=c,
                        arrowprops=dict(arrowstyle="-", color=c,
                                        lw=0.8, alpha=0.6))

    # Mark 95% nominal level
    ax.axvline(0.95, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.text(0.955, 0.12, "95% nominal", fontsize=7.5, color="gray",
            rotation=90, va="bottom")

    ax.set_xlabel("Nominal coverage probability", fontsize=10)
    ax.set_ylabel("Empirical coverage probability", fontsize=10)
    ax.set_title("Reliability Diagram — Kriging (GP-SE), 5-Fold CV", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(True, alpha=0.2, lw=0.5)
    fig.tight_layout()

save(fig, "reliability_diagram.png")
plt.close(fig)

print("\nDone. All 3 figures saved to thesis_figures/.")
