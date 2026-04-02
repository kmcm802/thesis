#!/usr/bin/env python3
"""
Generate three missing thesis figures:
  R2_residual_plots.png   — CV residuals for Linear, Poly, Kriging, NN × 5 outputs
  F3_lhs_pair_plot.png    — Pairwise scatter of LHS 100-pt design (doe_inputs.csv)
  F4_output_boxplots.png  — Scatter matrix of 5 outputs (simulation_results_hifi.csv)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── destinations ──────────────────────────────────────────────────────────────
LOCAL_FIG  = ROOT / "thesis_figures"
LOCAL_FIG.mkdir(parents=True, exist_ok=True)

# ── shared style ──────────────────────────────────────────────────────────────
THESIS_STYLE = {
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.labelsize":     10,
    "axes.titlesize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "figure.dpi":         150,
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica", "Arial", "DejaVu Sans"],
}

OUTPUTS    = ["v_out_mean", "i_l_mean", "v_out_ripple", "i_l_ripple", "efficiency"]
OUT_LABELS = [r"$\bar{V}_{out}$", r"$\bar{I}_L$", r"$\Delta V_{out}$",
              r"$\Delta I_L$", r"$\eta$"]
INPUT_COLS = ["D", "V_in", "R", "f_sw"]


# ── load training data ────────────────────────────────────────────────────────
def load_train_100():
    df = pd.read_csv(ROOT / "outputs" / "phase2" / "train_200.csv")
    df = df[df["ccm_ok"]].reset_index(drop=True).iloc[:100]
    X = df[INPUT_COLS].values.astype(float)
    Y = df[OUTPUTS].values.astype(float)
    return X, Y


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — R2_residual_plots.png
# ─────────────────────────────────────────────────────────────────────────────

def cv_residuals(config_id: str, X: np.ndarray, Y: np.ndarray, n_folds: int = 5):
    """Return (y_true, y_pred) arrays for every sample via K-fold CV."""
    from src.surrogates.registry import MODEL_REGISTRY
    build_fn = next(fn for cid, _, fn in MODEL_REGISTRY if cid == config_id)

    n = len(X)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    n_out = Y.shape[1]
    all_true = np.full((n, n_out), np.nan)
    all_pred = np.full((n, n_out), np.nan)

    for fold_i, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        for oi in range(n_out):
            y_tr = Y[tr_idx, oi]
            try:
                model = build_fn(X_tr, y_tr)
                y_hat = np.array(model.predict(X_val)).ravel()
                all_true[val_idx, oi] = Y[val_idx, oi]
                all_pred[val_idx, oi] = y_hat
            except Exception as e:
                warnings.warn(f"  [{config_id}] fold {fold_i} out {oi}: {e}")

    return all_true, all_pred


def make_residual_figure():
    print("── Figure R2: CV residuals ──────────────────────────────────────")
    X, Y = load_train_100()

    MODELS = [
        ("POLY-1", "Linear"),
        ("POLY-3", "Polynomial (deg 3)"),
        ("GP-SE",  "Kriging (GP-SE)"),
        ("NN-M",   "Neural Network"),
    ]

    n_models = len(MODELS)
    n_outs   = len(OUTPUTS)

    with plt.rc_context(THESIS_STYLE):
        fig, axes = plt.subplots(
            n_models, n_outs,
            figsize=(3.5 * n_outs, 3.0 * n_models),
            squeeze=False,
        )
        fig.suptitle("5-Fold CV Residuals  (predicted − actual)", fontsize=12, y=1.01)

        palette = ["#2166ac", "#d6604d", "#4dac26", "#7b2d8b"]

        for mi, (cid, label) in enumerate(MODELS):
            print(f"  Running CV for {cid} ({label}) …")
            y_true, y_pred = cv_residuals(cid, X, Y)
            residuals = y_pred - y_true          # shape (100, 5)
            color = palette[mi]

            for oi, (out, out_lbl) in enumerate(zip(OUTPUTS, OUT_LABELS)):
                ax = axes[mi][oi]
                res = residuals[:, oi]
                y_t = y_true[:, oi]

                valid = ~np.isnan(res)
                ax.scatter(y_t[valid], res[valid], s=18, alpha=0.65,
                           color=color, linewidths=0, rasterized=True)
                ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.7)

                # annotate RMSE
                rmse = np.sqrt(np.mean(res[valid] ** 2))
                r2   = 1 - np.sum(res[valid]**2) / np.sum((y_t[valid] - y_t[valid].mean())**2)
                ax.text(0.03, 0.97,
                        f"RMSE={rmse:.3g}\n$R^2$={r2:.3f}",
                        transform=ax.transAxes, va="top", ha="left",
                        fontsize=7, color=color,
                        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))

                if mi == 0:
                    ax.set_title(out_lbl, fontsize=10)
                if oi == 0:
                    ax.set_ylabel(f"{label}\nResidual", fontsize=8)
                if mi == n_models - 1:
                    ax.set_xlabel("Actual", fontsize=8)

        fig.tight_layout()
        path = LOCAL_FIG / "R2_residual_plots.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — F3_lhs_pair_plot.png
# ─────────────────────────────────────────────────────────────────────────────

def make_lhs_pair_plot():
    print("── Figure F3: LHS pair plot ─────────────────────────────────────")
    doe = pd.read_csv(ROOT / "data" / "raw" / "doe_inputs.csv")
    # rename to match thesis variable names
    doe.columns = ["D", "V_in", "R", "f_sw"]
    doe["f_sw"] = doe["f_sw"] / 1e3   # convert to kHz for readability

    labels = {
        "D":    r"Duty cycle $D$",
        "V_in": r"Input voltage $V_{in}$ (V)",
        "R":    r"Load resistance $R$ ($\Omega$)",
        "f_sw": r"Switching freq $f_{sw}$ (kHz)",
    }
    cols  = list(doe.columns)
    n     = len(cols)

    with plt.rc_context(THESIS_STYLE):
        fig, axes = plt.subplots(n, n, figsize=(8, 8))
        fig.suptitle("100-Point Maximin LHS Design — Input Space Coverage",
                     fontsize=11, y=1.01)

        color = "#2166ac"
        hist_color = "#a8cce0"

        for row in range(n):
            for col in range(n):
                ax = axes[row][col]
                if row == col:
                    # diagonal: marginal histogram
                    ax.hist(doe[cols[row]], bins=14, color=hist_color,
                            edgecolor="white", linewidth=0.4)
                    ax.set_yticks([])
                    if row == n - 1:
                        ax.set_xlabel(labels[cols[col]], fontsize=8)
                    else:
                        ax.set_xlabel("")
                        ax.set_xticklabels([])
                else:
                    ax.scatter(doe[cols[col]], doe[cols[row]],
                               s=12, alpha=0.7, color=color, linewidths=0)
                    if row == n - 1:
                        ax.set_xlabel(labels[cols[col]], fontsize=8)
                    else:
                        ax.set_xticklabels([])
                    if col == 0:
                        ax.set_ylabel(labels[cols[row]], fontsize=8)
                    else:
                        ax.set_yticklabels([])

                ax.tick_params(axis="both", labelsize=7)

        fig.tight_layout()
        path = LOCAL_FIG / "F3_lhs_pair_plot.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — F4_output_boxplots.png
# ─────────────────────────────────────────────────────────────────────────────

def make_output_scatter_matrix():
    print("── Figure F4: Output scatter matrix ────────────────────────────")
    sim = pd.read_csv(ROOT / "data" / "raw" / "simulation_results_hifi.csv")

    # keep only the five output columns
    out_cols = ["v_out_mean", "i_l_mean", "v_out_ripple", "i_l_ripple", "efficiency"]
    # handle possible column name variants
    col_map = {}
    for c in sim.columns:
        c_l = c.lower().replace(" ", "_")
        if "v_out" in c_l and "rip" in c_l:
            col_map[c] = "v_out_ripple"
        elif "v_out" in c_l:
            col_map[c] = "v_out_mean"
        elif "i_l" in c_l and "rip" in c_l:
            col_map[c] = "i_l_ripple"
        elif "i_l" in c_l:
            col_map[c] = "i_l_mean"
        elif "effic" in c_l:
            col_map[c] = "efficiency"
    if col_map:
        sim = sim.rename(columns=col_map)

    # filter to the five outputs that exist
    present = [c for c in out_cols if c in sim.columns]
    df = sim[present].dropna().reset_index(drop=True)
    print(f"  Using {len(df)} rows, columns: {present}")

    nice_labels = {
        "v_out_mean":    r"$\bar{V}_{out}$ (V)",
        "i_l_mean":      r"$\bar{I}_L$ (A)",
        "v_out_ripple":  r"$\Delta V_{out}$ (V)",
        "i_l_ripple":    r"$\Delta I_L$ (A)",
        "efficiency":    r"$\eta$ (—)",
    }
    n = len(present)

    with plt.rc_context(THESIS_STYLE):
        fig, axes = plt.subplots(n, n, figsize=(10, 9))
        fig.suptitle("Output Variable Scatter Matrix — 100-Sample Training Dataset",
                     fontsize=11, y=1.01)

        color = "#d6604d"
        hist_color = "#f4b89a"

        for row in range(n):
            for col in range(n):
                ax  = axes[row][col]
                cx  = present[col]
                cy  = present[row]
                lx  = nice_labels.get(cx, cx)
                ly  = nice_labels.get(cy, cy)

                if row == col:
                    ax.hist(df[cx], bins=14, color=hist_color,
                            edgecolor="white", linewidth=0.4)
                    ax.set_yticks([])
                    if row == n - 1:
                        ax.set_xlabel(lx, fontsize=8)
                    else:
                        ax.set_xticklabels([])
                else:
                    ax.scatter(df[cx], df[cy], s=10, alpha=0.6,
                               color=color, linewidths=0)
                    corr = df[cx].corr(df[cy])
                    ax.text(0.96, 0.04, f"r={corr:.2f}",
                            transform=ax.transAxes, ha="right", va="bottom",
                            fontsize=6.5, color="#555555")
                    if row == n - 1:
                        ax.set_xlabel(lx, fontsize=8)
                    else:
                        ax.set_xticklabels([])
                    if col == 0:
                        ax.set_ylabel(ly, fontsize=8)
                    else:
                        ax.set_yticklabels([])

                ax.tick_params(axis="both", labelsize=7)

        fig.tight_layout()
        path = LOCAL_FIG / "F4_output_boxplots.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    make_lhs_pair_plot()
    make_output_scatter_matrix()
    make_residual_figure()
    print("\nAll three figures generated successfully.")
