#!/usr/bin/env python3
"""
make_all_thesis_figures.py
==========================
Generate the thesis figure set and save local copies.
"""

from __future__ import annotations
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ─── Output destinations ──────────────────────────────────────────────────────
LOCAL    = ROOT / "thesis_figures"
LOCAL.mkdir(parents=True, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
INPUT_COLS = ["D", "V_in", "R", "f_sw"]
OUTPUTS    = ["v_out_mean", "i_l_mean", "v_out_ripple", "i_l_ripple", "efficiency"]
OUT_LABELS = [r"$\bar{V}_{out}$", r"$\bar{I}_L$",
              r"$\Delta V_{out}$", r"$\Delta I_L$", r"$\eta$"]

# Four families: (config_id_in_registry, thesis_label, colour)
FAMILIES = [
    ("POLY-1", "Linear Regression", "#d6604d"),
    ("POLY-2", "Polynomial",        "#4393c3"),
    ("GP-SE",  "Kriging",           "#2ca25f"),
    ("NN-M",   "Neural Network",    "#7b2d8b"),
]

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
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
}

# ─── Data helpers ─────────────────────────────────────────────────────────────
def load_train_100():
    df = pd.read_csv(ROOT / "outputs/phase2/train_200.csv")
    df = df[df["ccm_ok"]].reset_index(drop=True).iloc[:100]
    return df[INPUT_COLS].values.astype(float), df[OUTPUTS].values.astype(float)


def load_test(n: int = 200):
    df = pd.read_csv(ROOT / "outputs/phase2/test_500.csv")
    df = df[df["ccm_ok"]].reset_index(drop=True).iloc[:n]
    return df[INPUT_COLS].values.astype(float), df[OUTPUTS].values.astype(float)


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rng = float(y_true.max() - y_true.min())
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / rng) if rng > 1e-12 else np.nan


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def save_fig(fig: plt.Figure, name: str, dpi: int = 300):
    p = LOCAL / name
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    print(f"  Saved → {p}")


# ─── Fast model builders (avoid importing registry globally) ──────────────────
def _build_poly(degree: int):
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    def build(X, y):
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("poly",  PolynomialFeatures(degree=degree, include_bias=False)),
            ("ols",   LinearRegression()),
        ])
        pipe.fit(X, y.ravel())
        return pipe
    return build


def _build_gp_se(n_start: int = 1):
    """GP with squared-exponential kernel, ARD, small nugget."""
    def build(X, y):
        from smt.surrogate_models import KRG
        d = X.shape[1]
        sm = KRG(theta0=[1e-2] * d, corr="squar_exp", poly="constant",
                 nugget=1e-6, n_start=n_start, print_global=False)
        sm.set_training_values(X, y.reshape(-1, 1))
        sm.train()
        class _W:
            def __init__(self, m): self._m = m
            def predict(self, Xn): return self._m.predict_values(Xn).ravel()
            def predict_std(self, Xn):
                return np.sqrt(np.maximum(self._m.predict_variances(Xn).ravel(), 0.0))
        return _W(sm)
    return build


def _build_nn_m():
    """2-hidden-layer MLP [64, 32] ReLU, Adam, early-stop patience=50."""
    def build(X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        torch.manual_seed(42)
        np.random.seed(42)

        Xm, Xs = X.mean(0), X.std(0) + 1e-8
        ym, ys = y.mean(), y.std() + 1e-8
        Xn = (X - Xm) / Xs
        yn = (y.ravel() - ym) / ys

        n = len(Xn)
        n_val = max(1, int(0.1 * n))
        idx = np.random.permutation(n)
        Xt = torch.tensor(Xn[idx[n_val:]], dtype=torch.float32)
        yt_ = torch.tensor(yn[idx[n_val:]], dtype=torch.float32).unsqueeze(1)
        Xv = torch.tensor(Xn[idx[:n_val]], dtype=torch.float32)
        yv_ = torch.tensor(yn[idx[:n_val]], dtype=torch.float32).unsqueeze(1)

        loader = DataLoader(TensorDataset(Xt, yt_), batch_size=16, shuffle=True)

        net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(),
                            nn.Linear(64, 32), nn.ReLU(),
                            nn.Linear(32, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        best_val, patience, best_sd = np.inf, 0, None
        for epoch in range(1000):
            net.train()
            for xb, yb in loader:
                opt.zero_grad(); loss_fn(net(xb), yb).backward(); opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(loss_fn(net(Xv), yv_))
            if vl < best_val - 1e-6:
                best_val, patience = vl, 0
                best_sd = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                patience += 1
                if patience >= 50: break
        if best_sd: net.load_state_dict(best_sd)

        class _W:
            def __init__(self, m, Xm, Xs, ym, ys):
                self._m = m.eval(); self._Xm=Xm; self._Xs=Xs; self._ym=ym; self._ys=ys
            def predict(self, Xnew):
                import torch
                Xn2 = (Xnew - self._Xm) / self._Xs
                t = torch.tensor(Xn2, dtype=torch.float32)
                with torch.no_grad():
                    return self._m(t).numpy().ravel() * self._ys + self._ym
        return _W(net, Xm, Xs, ym, ys)
    return build


# Map config_id → build function
BUILDERS = {
    "POLY-1": _build_poly(1),
    "POLY-2": _build_poly(2),
    "GP-SE":  _build_gp_se(n_start=1),   # n_start=1 for speed in CV/LC
    "NN-M":   _build_nn_m(),
}


# ─── CV runner ────────────────────────────────────────────────────────────────
def run_cv(cid: str, X: np.ndarray, Y: np.ndarray, n_folds: int = 5) -> tuple:
    """K-fold CV. Returns (y_true, y_pred) arrays of shape (N, n_out)."""
    build = BUILDERS[cid]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    n, n_out = len(X), Y.shape[1]
    yt_all = np.full((n, n_out), np.nan)
    yp_all = np.full((n, n_out), np.nan)
    for fi, (tr, val) in enumerate(kf.split(X)):
        print(f"    fold {fi+1}/{n_folds}", end="\r", flush=True)
        for oi in range(n_out):
            try:
                m = build(X[tr], Y[tr, oi])
                yp_all[val, oi] = np.array(m.predict(X[val])).ravel()
                yt_all[val, oi] = Y[val, oi]
            except Exception as e:
                warnings.warn(f"[{cid}] fold {fi} out {oi}: {e}")
    print()
    return yt_all, yp_all


# ─── Learning-curve runner ────────────────────────────────────────────────────
def run_lc(cid: str, X_tr: np.ndarray, Y_tr: np.ndarray,
           X_te: np.ndarray, Y_te: np.ndarray,
           N_vals: list[int]) -> pd.DataFrame:
    """Train on first N samples; evaluate on fixed test set."""
    build = BUILDERS[cid]
    rows = []
    n_out = Y_tr.shape[1]
    for N in N_vals:
        print(f"    N={N:<4}", end="\r", flush=True)
        for oi, out in enumerate(OUTPUTS):
            try:
                m = build(X_tr[:N], Y_tr[:N, oi])
                yp = np.array(m.predict(X_te)).ravel()
                rows.append({"N": N, "output": out,
                             "nrmse": nrmse(Y_te[:, oi], yp),
                             "r2":    r2(Y_te[:, oi], yp)})
            except Exception as e:
                warnings.warn(f"[{cid}] N={N} oi={oi}: {e}")
                rows.append({"N": N, "output": out, "nrmse": np.nan, "r2": np.nan})
    print()
    return pd.DataFrame(rows)


# =============================================================================
# FIGURE 1 — mc1_parity_top3.png
# Parity plots: predicted vs actual for v_out_mean (4 panels)
# =============================================================================
def fig_parity(cv_data: dict):
    print("── Figure 1: mc1_parity_top3.png")
    oi = OUTPUTS.index("v_out_mean")

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(14, 4.0))
        fig.suptitle(r"5-Fold CV Parity Plots — Output: $\bar{V}_{out}$",
                     fontsize=12, y=1.02)

        for ax, (cid, label, color) in zip(axes, FAMILIES):
            yt = cv_data[cid][0][:, oi]
            yp = cv_data[cid][1][:, oi]
            ok = ~(np.isnan(yt) | np.isnan(yp))
            yt, yp = yt[ok], yp[ok]

            lo = min(yt.min(), yp.min()) * 0.97
            hi = max(yt.max(), yp.max()) * 1.03
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.55)
            ax.scatter(yt, yp, s=22, alpha=0.75, color=color, linewidths=0)

            rv = rmse(yt, yp)
            r2v = r2(yt, yp)
            ax.set_title(f"{label}\nRMSE = {rv:.3g} V,  $R^2$ = {r2v:.4f}", fontsize=9)
            ax.set_xlabel(r"Actual $\bar{V}_{out}$ (V)", fontsize=9)
            if ax is axes[0]:
                ax.set_ylabel(r"Predicted $\bar{V}_{out}$ (V)", fontsize=9)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect("equal", adjustable="box")

        fig.tight_layout()

    save_fig(fig, "mc1_parity_top3.png")
    plt.close(fig)


# =============================================================================
# FIGURE 2 — mc1_rmse_bar_vout.png
# Grouped NRMSE (%) bar chart for all 5 outputs × 4 models
# =============================================================================
def fig_nrmse_bar():
    print("── Figure 2: mc1_rmse_bar_vout.png")
    df = pd.read_csv(ROOT / "outputs/phase3/mc1_summary.csv")

    CIDS    = [f[0] for f in FAMILIES]
    LABELS  = [f[1] for f in FAMILIES]
    COLORS  = [f[2] for f in FAMILIES]

    n_out = len(OUTPUTS)
    n_mod = len(CIDS)
    x     = np.arange(n_out)
    width = 0.19
    offsets = np.linspace(-(n_mod-1)/2, (n_mod-1)/2, n_mod) * width

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))

        for cid, label, color, offset in zip(CIDS, LABELS, COLORS, offsets):
            vals = []
            for out in OUTPUTS:
                row = df[(df["config_id"] == cid) & (df["output"] == out)]
                vals.append(float(row["nrmse"].iloc[0]) * 100 if len(row) else np.nan)
            bars = ax.bar(x + offset, vals, width=width * 0.9, label=label,
                          color=color, alpha=0.88, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                if not np.isnan(val) and val < 30:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.08,
                            f"{val:.2f}", ha="center", va="bottom",
                            fontsize=6, color="#333")

        ax.set_xticks(x)
        ax.set_xticklabels(OUT_LABELS, fontsize=11)
        ax.set_ylabel("NRMSE (%)", fontsize=10)
        ax.set_title("5-Fold CV NRMSE by Output Variable and Model Family", fontsize=11)
        ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=9)
        ax.set_ylim(0, None)
        fig.tight_layout()

    save_fig(fig, "mc1_rmse_bar_vout.png")
    plt.close(fig)


# =============================================================================
# FIGURE 3 — ss1_learning_curves_all.png
# NRMSE vs training set size for all 5 outputs, 4 lines per subplot
# =============================================================================
def fig_learning_curves(lc_data: dict):
    print("── Figure 3: ss1_learning_curves_all.png")
    N_VALS = [20, 40, 60, 80, 100]

    CIDS   = [f[0] for f in FAMILIES]
    LABELS = [f[1] for f in FAMILIES]
    COLORS = [f[2] for f in FAMILIES]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 5, figsize=(17, 4.2))
        fig.suptitle("Learning Curves — NRMSE (%) vs Training Set Size",
                     fontsize=12, y=1.02)

        for oi, (out, lbl, ax) in enumerate(zip(OUTPUTS, OUT_LABELS, axes)):
            for cid, label, color in zip(CIDS, LABELS, COLORS):
                sub = lc_data[cid][lc_data[cid]["output"] == out].sort_values("N")
                ax.plot(sub["N"], sub["nrmse"] * 100, marker="o", ms=5,
                        lw=1.8, color=color, label=label)
            ax.set_title(lbl, fontsize=10)
            ax.set_xlabel("Training size $N$", fontsize=9)
            if oi == 0:
                ax.set_ylabel("NRMSE (%)", fontsize=9)
            ax.set_xticks(N_VALS)
            ax.set_xticklabels([str(n) for n in N_VALS], fontsize=7)
            ax.set_ylim(bottom=0)

        handles, lbls = axes[0].get_legend_handles_labels()
        fig.legend(handles, lbls, loc="lower center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, -0.07), fontsize=9)
        fig.tight_layout()

    save_fig(fig, "ss1_learning_curves_all.png")
    plt.close(fig)


# =============================================================================
# FIGURE 4 — calibration_and_spearman.png
# Left: reliability diagram for GP-SE, all 5 outputs
# Right: Spearman ρ heatmap for GP model variants
# =============================================================================
def fig_calibration_spearman():
    print("── Figure 4: calibration_and_spearman.png")
    cal = pd.read_csv(ROOT / "outputs/phase4/calibration.csv")
    spe = pd.read_csv(ROOT / "outputs/phase4/spearman.csv")

    gp_cal = cal[cal["model"] == "GP-SE"].sort_values(["output", "nominal"])
    gp_spe = spe[spe["model"].isin(["GP-SE", "GP-M52", "GP-M32", "GP-SE-noARD"])]

    OUT_COLORS = ["#2166ac", "#d6604d", "#4dac26", "#7b2d8b", "#ff7f00"]

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # ── Left: Reliability diagram ──────────────────────────────────────
        ax1.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.55, label="Perfect calibration")
        for out, lbl, c in zip(OUTPUTS, OUT_LABELS, OUT_COLORS):
            sub = gp_cal[gp_cal["output"] == out].sort_values("nominal")
            ax1.plot(sub["nominal"], sub["empirical"], marker="o", ms=5,
                     lw=1.6, color=c, label=lbl)

        ax1.set_xlabel("Nominal coverage", fontsize=10)
        ax1.set_ylabel("Empirical coverage", fontsize=10)
        ax1.set_title("Reliability Diagram — Kriging (GP-SE)", fontsize=10)
        ax1.legend(frameon=False, fontsize=8, loc="upper left")
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
        ax1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

        # ── Right: Spearman heatmap ────────────────────────────────────────
        MODEL_ORDER  = ["GP-SE", "GP-M52", "GP-M32", "GP-SE-noARD"]
        MODEL_LABELS = {
            "GP-SE":        "Kriging (SE, ARD)",
            "GP-M52":       "Kriging (Matérn 5/2)",
            "GP-M32":       "Kriging (Matérn 3/2)",
            "GP-SE-noARD":  "Kriging (SE, isotropic)",
        }
        pivot = (gp_spe
                 .pivot_table(index="model", columns="output", values="spearman_rho")
                 .reindex(index=MODEL_ORDER, columns=OUTPUTS))

        rho = pivot.values.astype(float)
        im = ax2.imshow(rho, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
        plt.colorbar(im, ax=ax2, shrink=0.85, label=r"Spearman $\rho$")

        ax2.set_xticks(range(len(OUTPUTS)))
        ax2.set_xticklabels(OUT_LABELS, fontsize=9)
        ax2.set_yticks(range(len(MODEL_ORDER)))
        ax2.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=8)
        ax2.set_title(r"Spearman $\rho\,(\hat{\sigma},\,|error|)$ — Kriging models",
                      fontsize=10)

        for i in range(len(MODEL_ORDER)):
            for j in range(len(OUTPUTS)):
                v = rho[i, j]
                if not np.isnan(v):
                    ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                             fontsize=8, color="white" if v > 0.65 else "black")

        fig.tight_layout()

    save_fig(fig, "calibration_and_spearman.png")
    plt.close(fig)


# =============================================================================
# FIGURE 5 — R2_residual_plots.png
# Residual plots (predicted − actual) for 4 models × 5 outputs
# =============================================================================
def fig_residuals(cv_data: dict):
    print("── Figure 5: R2_residual_plots.png")

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(4, 5, figsize=(18, 14), squeeze=False)
        fig.suptitle("5-Fold CV Residuals  (Predicted − Actual)",
                     fontsize=13, y=1.005)

        for mi, (cid, label, color) in enumerate(FAMILIES):
            yt_all, yp_all = cv_data[cid]
            for oi, (out, lbl) in enumerate(zip(OUTPUTS, OUT_LABELS)):
                ax  = axes[mi][oi]
                yt  = yt_all[:, oi]
                yp  = yp_all[:, oi]
                ok  = ~(np.isnan(yt) | np.isnan(yp))
                res = yp[ok] - yt[ok]

                ax.scatter(yt[ok], res, s=16, alpha=0.65,
                           color=color, linewidths=0, rasterized=True)
                ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.6)

                rv  = rmse(yt[ok], yp[ok])
                r2v = r2(yt[ok], yp[ok])
                ax.text(0.03, 0.97,
                        f"RMSE={rv:.3g}\n$R^2$={r2v:.3f}",
                        transform=ax.transAxes, va="top", ha="left",
                        fontsize=6.5, color=color,
                        bbox=dict(facecolor="white", alpha=0.75,
                                  edgecolor="none", pad=1.5))

                if mi == 0: ax.set_title(lbl, fontsize=10)
                if oi == 0: ax.set_ylabel(f"{label}\nResidual", fontsize=8)
                if mi == 3: ax.set_xlabel("Actual", fontsize=8)

        fig.tight_layout(h_pad=1.2, w_pad=0.6)

    save_fig(fig, "R2_residual_plots.png")
    plt.close(fig)


# =============================================================================
# FIGURE 6 — F3_lhs_pair_plot.png
# Pairwise scatter of the 100-point LHS design
# =============================================================================
def fig_lhs_pair():
    print("── Figure 6: F3_lhs_pair_plot.png")
    doe = pd.read_csv(ROOT / "data/raw/doe_inputs.csv")
    doe.columns = ["D", "V_in", "R", "f_sw"]
    doe["f_sw"] = doe["f_sw"] / 1e3   # → kHz

    var_labels = {
        "D":    r"Duty cycle $D$",
        "V_in": r"Input voltage $V_{in}$ (V)",
        "R":    r"Load resistance $R$ ($\Omega$)",
        "f_sw": r"Switching freq $f_{sw}$ (kHz)",
    }
    cols   = list(doe.columns)
    n_vars = len(cols)
    blue   = "#2166ac"
    lblue  = "#a8cce0"

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(n_vars, n_vars, figsize=(8.5, 8.5))
        fig.suptitle("100-Point Maximin LHS Design — Input Space Coverage",
                     fontsize=11, y=1.01)

        for row in range(n_vars):
            for col in range(n_vars):
                ax = axes[row][col]
                if row == col:
                    ax.hist(doe[cols[row]], bins=14, color=lblue,
                            edgecolor="white", linewidth=0.4)
                    ax.set_yticks([])
                else:
                    ax.scatter(doe[cols[col]], doe[cols[row]],
                               s=13, alpha=0.75, color=blue, linewidths=0)
                # axis labels only on edges
                if row == n_vars - 1:
                    ax.set_xlabel(var_labels[cols[col]], fontsize=8)
                else:
                    ax.set_xticklabels([])
                if col == 0 and row != col:
                    ax.set_ylabel(var_labels[cols[row]], fontsize=8)
                elif col != 0:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=7)

        fig.tight_layout()

    save_fig(fig, "F3_lhs_pair_plot.png")
    plt.close(fig)


# =============================================================================
# FIGURE 7 — F4_output_boxplots.png
# Scatter matrix of the 5 output variables
# =============================================================================
def fig_output_scatter():
    print("── Figure 7: F4_output_boxplots.png")
    sim = pd.read_csv(ROOT / "data/raw/simulation_results_hifi.csv")

    # Map column names defensively
    col_map = {}
    for c in sim.columns:
        cl = c.lower().replace(" ", "_")
        if   "v_out" in cl and "rip" in cl: col_map[c] = "v_out_ripple"
        elif "v_out" in cl:                 col_map[c] = "v_out_mean"
        elif "i_l"   in cl and "rip" in cl: col_map[c] = "i_l_ripple"
        elif "i_l"   in cl:                 col_map[c] = "i_l_mean"
        elif "effic" in cl:                 col_map[c] = "efficiency"
    if col_map:
        sim = sim.rename(columns=col_map)

    present = [c for c in OUTPUTS if c in sim.columns]
    df      = sim[present].dropna().reset_index(drop=True)
    print(f"  {len(df)} rows, columns: {present}")

    nice = {
        "v_out_mean":   r"$\bar{V}_{out}$ (V)",
        "i_l_mean":     r"$\bar{I}_L$ (A)",
        "v_out_ripple": r"$\Delta V_{out}$ (V)",
        "i_l_ripple":   r"$\Delta I_L$ (A)",
        "efficiency":   r"$\eta$",
    }
    n     = len(present)
    red   = "#d6604d"
    lred  = "#f4b89a"

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(n, n, figsize=(11, 10))
        fig.suptitle("Output Variable Scatter Matrix — 100-Sample Training Dataset",
                     fontsize=11, y=1.01)

        for row in range(n):
            for col in range(n):
                ax = axes[row][col]
                cx, cy = present[col], present[row]
                if row == col:
                    ax.hist(df[cx], bins=14, color=lred,
                            edgecolor="white", linewidth=0.4)
                    ax.set_yticks([])
                else:
                    ax.scatter(df[cx], df[cy], s=10, alpha=0.6,
                               color=red, linewidths=0)
                    corr = df[cx].corr(df[cy])
                    ax.text(0.96, 0.04, f"r = {corr:.2f}",
                            transform=ax.transAxes, ha="right", va="bottom",
                            fontsize=6.5, color="#555")
                if row == n - 1:
                    ax.set_xlabel(nice.get(cx, cx), fontsize=8)
                else:
                    ax.set_xticklabels([])
                if col == 0 and row != col:
                    ax.set_ylabel(nice.get(cy, cy), fontsize=8)
                elif col != 0:
                    ax.set_yticklabels([])
                ax.tick_params(labelsize=7)

        fig.tight_layout()

    save_fig(fig, "F4_output_boxplots.png")
    plt.close(fig)


# =============================================================================
# FIGURE 8 — all_variant_comparison.png
# Horizontal bar chart of NRMSE on v_out_mean for all 14 configs
# =============================================================================
def fig_all_variants():
    print("── Figure 8: all_variant_comparison.png")
    df = pd.read_csv(ROOT / "outputs/phase3/mc1_summary.csv")

    CONFIG_LIST = [
        "POLY-1", "POLY-2", "POLY-3", "POLY-4",
        "GP-SE", "GP-M52", "GP-M32", "GP-SE-noARD", "GP-SE-nonug",
        "NN-S", "NN-M", "NN-L", "NN-M-tanh", "NN-M-wd",
    ]
    SELECTED = {"POLY-2", "GP-SE", "NN-M"}   # best-in-family (highlighted)

    DISPLAY = {
        "POLY-1":       "Linear (OLS)",
        "POLY-2":       "Polynomial deg-2  ★",
        "POLY-3":       "Polynomial deg-3",
        "POLY-4":       "Polynomial deg-4",
        "GP-SE":        "Kriging GP-SE  ★",
        "GP-M52":       "Kriging GP-Matérn 5/2",
        "GP-M32":       "Kriging GP-Matérn 3/2",
        "GP-SE-noARD":  "Kriging GP-SE (isotropic)",
        "GP-SE-nonug":  "Kriging GP-SE (no nugget)",
        "NN-S":         "Neural Net (small)",
        "NN-M":         "Neural Net (medium)  ★",
        "NN-L":         "Neural Net (large)",
        "NN-M-tanh":    "Neural Net (tanh)",
        "NN-M-wd":      "Neural Net (weight decay)",
    }

    vout = (df[(df["output"] == "v_out_mean") & (df["config_id"].isin(CONFIG_LIST))]
            .copy()
            .assign(nrmse_pct=lambda d: d["nrmse"] * 100)
            .sort_values("nrmse_pct"))

    bar_colors    = ["#2ca25f" if r in SELECTED else "#9ecae1"
                     for r in vout["config_id"]]
    bar_edges     = ["#006d2c" if r in SELECTED else "#4393c3"
                     for r in vout["config_id"]]
    bar_lws       = [1.8 if r in SELECTED else 0.5
                     for r in vout["config_id"]]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 6.5))

        bars = ax.barh(range(len(vout)), vout["nrmse_pct"],
                       color=bar_colors, edgecolor=bar_edges,
                       linewidth=bar_lws, height=0.72)

        for bar, (_, row) in zip(bars, vout.iterrows()):
            ax.text(row["nrmse_pct"] + 0.04,
                    bar.get_y() + bar.get_height() / 2,
                    f"{row['nrmse_pct']:.3f}%",
                    va="center", fontsize=7.5)

        ax.set_yticks(range(len(vout)))
        ax.set_yticklabels(
            [DISPLAY.get(c, c) for c in vout["config_id"]], fontsize=8.5)
        ax.set_xlabel(r"NRMSE (%) on $\bar{V}_{out}$  [5-fold CV]", fontsize=10)
        ax.set_title(r"All-Configuration Comparison — $\bar{V}_{out}$ NRMSE",
                     fontsize=11)

        legend_elems = [
            mpatches.Patch(facecolor="#2ca25f", edgecolor="#006d2c",
                           lw=1.5, label="Selected best-in-family  (★)"),
            mpatches.Patch(facecolor="#9ecae1", edgecolor="#4393c3",
                           lw=0.5, label="Other configurations"),
        ]
        ax.legend(handles=legend_elems, frameon=False, loc="lower right", fontsize=9)
        fig.tight_layout()

    save_fig(fig, "all_variant_comparison.png")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    t_start = time.perf_counter()

    print("=" * 60)
    print("Loading data …")
    X_train, Y_train = load_train_100()
    X_test,  Y_test  = load_test(200)
    print(f"  Train: {X_train.shape}   Test: {X_test.shape}")

    # ── Run 5-fold CV for parity + residual figures ──────────────────────────
    cv_data: dict[str, tuple] = {}
    N_LC = [20, 40, 60, 80, 100]
    lc_data: dict[str, pd.DataFrame] = {}

    for cid, label, _ in FAMILIES:
        print(f"\n[{label}]  CV …")
        cv_data[cid] = run_cv(cid, X_train, Y_train)
        v_idx = OUTPUTS.index("v_out_mean")
        yt0, yp0 = cv_data[cid]
        ok = ~(np.isnan(yt0[:, v_idx]) | np.isnan(yp0[:, v_idx]))
        print(f"  v_out_mean NRMSE = "
              f"{nrmse(yt0[ok, v_idx], yp0[ok, v_idx])*100:.3f}%  "
              f"R² = {r2(yt0[ok, v_idx], yp0[ok, v_idx]):.4f}")

        print(f"[{label}]  Learning curves N={N_LC} …")
        lc_data[cid] = run_lc(cid, X_train, Y_train, X_test, Y_test, N_LC)

    print()

    # ── Generate all 8 figures ───────────────────────────────────────────────
    fig_parity(cv_data)
    fig_nrmse_bar()
    fig_learning_curves(lc_data)
    fig_calibration_spearman()
    fig_residuals(cv_data)
    fig_lhs_pair()
    fig_output_scatter()
    fig_all_variants()

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"All 8 figures generated in {elapsed:.1f} s")

    # ── NRMSE verification ───────────────────────────────────────────────────
    print("\n── NRMSE verification (mc1_summary.csv, v_out_mean) ──")
    df_sum = pd.read_csv(ROOT / "outputs/phase3/mc1_summary.csv")
    targets = {
        "GP-SE":  ("Kriging",          0.250),
        "POLY-2": ("Polynomial",        2.26),
        "NN-M":   ("Neural Network",    1.24),
        "POLY-1": ("Linear Regression", 12.4),
    }
    for cid, (name, expected) in targets.items():
        row = df_sum[(df_sum["config_id"] == cid) & (df_sum["output"] == "v_out_mean")]
        actual = float(row["nrmse"].iloc[0]) * 100 if len(row) else np.nan
        diff   = abs(actual - expected) / expected * 100
        flag   = "OK" if diff < 15 else f"MISMATCH (diff {diff:.0f}%)"
        print(f"  {name:<22s}: stored={actual:.3f}%   thesis={expected:.3f}%   [{flag}]")
