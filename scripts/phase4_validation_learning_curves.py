#!/usr/bin/env python3
"""
Phase 4 — Validation Strategies & Learning Curves
===================================================

Runs Experiments VAL-1, SS-1, and the calibration analysis (Part F).

Experiment VAL-1: Validation strategy comparison (top 3 models, N=100)
  - Holdout 80/20  × 30 random splits
  - 5-fold CV      × 5 repeats
  - 10-fold CV     × 5 repeats
  - LOOCV          × 1
  Do strategies agree on model ranking and RMSE magnitude?

Experiment SS-1: Learning curves (N_train = 10→200)
  - Top 3 Kriging configs + POLY-3-L + NN-M-tanh + RBF-C (top across families)
  - Fixed test set for evaluation
  - Reveals where performance plateaus

Part F: Uncertainty calibration for Kriging models
  - Reliability diagram: nominal coverage vs empirical coverage
  - Spearman ρ(σ_pred, |error|) — is uncertainty informative?

Outputs
-------
  outputs/phase4/val1_raw.csv           — per-(model,strategy,rep,output) RMSE
  outputs/phase4/val1_summary.csv       — aggregated mean±std RMSE
  outputs/phase4/ss1_raw.csv            — per-(model,N,output) RMSE
  outputs/phase4/calibration.csv        — coverage at each nominal level
  outputs/phase4/spearman.csv           — Spearman ρ per model × output
  outputs/phase4/figures/               — plots
  outputs/phase4/summary.txt            — key findings + recommendations
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
from scipy.stats import spearmanr, norm as sp_norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.utils.metrics import compute_rmse, compute_nrmse, compute_r2, compute_mae
from src.surrogates.registry import MODEL_REGISTRY

INPUT_COLS = ["D", "V_in", "R", "f_sw"]
OUTPUTS    = ["v_out_mean", "i_l_mean", "v_out_ripple", "i_l_ripple", "efficiency"]

OUT = ROOT / "outputs" / "phase4"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = ROOT / "outputs" / "phase2" / "train_200.csv"
TEST_CSV  = ROOT / "outputs" / "phase2" / "test_500.csv"

# Top models from Phase 3
TOP3 = ["GP-M52", "GP-SE", "GP-SE-noARD"]
# Models for learning curves (one per family)
LC_MODELS = ["GP-M52", "GP-SE", "POLY-3-L", "NN-M-tanh", "RBF-C"]
N_TRAIN_LC = [10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 200]

BUILD = {cid: fn for cid, _, fn in MODEL_REGISTRY}

# Fast GP build functions (n_start=1) for repeated VAL-1 cross-validation
def _fast_krg(corr: str, ARD: bool, nugget: float | None):
    """n_start=1 Kriging factory for speed in repeated CV experiments."""
    import time, numpy as np
    def build(X, y):
        from smt.surrogate_models import KRG
        d = X.shape[1]
        nug = 1e-6 if nugget is None else nugget
        theta0 = [1e-2] * (d if ARD else 1)
        sm = KRG(theta0=theta0, corr=corr, poly="constant",
                 nugget=nug, n_start=1, print_global=False)
        t0 = time.perf_counter()
        sm.set_training_values(X, y.reshape(-1, 1))
        sm.train()
        train_time = time.perf_counter() - t0
        class _W:
            def __init__(self, m, tt): self._sm = m; self._train_time = tt
            def predict(self, X_new): return self._sm.predict_values(X_new).ravel()
            def predict_std(self, X_new):
                return np.sqrt(np.maximum(self._sm.predict_variances(X_new).ravel(), 0.0))
        return _W(sm, train_time)
    return build

FAST_BUILD = {
    "GP-M52":       _fast_krg("matern52",  ARD=True,  nugget=None),
    "GP-SE":        _fast_krg("squar_exp", ARD=True,  nugget=None),
    "GP-SE-noARD":  _fast_krg("squar_exp", ARD=False, nugget=None),
}


# ── data ──────────────────────────────────────────────────────────────────────

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


# ── helpers ────────────────────────────────────────────────────────────────────

def fit_predict(cid, X_tr, y_tr, X_te, build_dict=None):
    d = (build_dict or BUILD)[cid]
    m = d(X_tr, y_tr)
    return np.array(m.predict(X_te)).ravel(), m


def cv_rmse(cid, X, Y, oi, n_folds, rng, build_dict=None):
    """Return list of per-fold RMSE values."""
    n = len(X)
    fold_idx = np.array_split(rng.permutation(n), n_folds)
    rmses = []
    for val_idx in fold_idx:
        tr_idx = np.setdiff1d(np.arange(n), val_idx)
        try:
            y_pred, _ = fit_predict(cid, X[tr_idx], Y[tr_idx, oi], X[val_idx], build_dict)
            rmses.append(compute_rmse(Y[val_idx, oi], y_pred))
        except Exception:
            rmses.append(np.nan)
    return rmses


# ── VAL-1 ──────────────────────────────────────────────────────────────────────

def run_val1(X_tr_full, Y_tr_full, X_te, Y_te):
    """
    Compare validation strategies on the top 3 models.
    Training data = first 100 CCM-valid rows (same as MC-1).
    """
    N = 100
    X_tr = X_tr_full[:N]
    Y_tr = Y_tr_full[:N]
    rows = []
    rng  = np.random.default_rng(42)

    # Primary outputs for LOOCV (computationally expensive — run all 5 for holdout/CV,
    # but limit LOOCV to v_out_mean + efficiency to stay within time budget).
    LOOCV_OUTPUTS = {"v_out_mean", "efficiency"}

    for cid in TOP3:
        print(f"  VAL-1: {cid}")
        # Use fast (n_start=1) Kriging for repeated CV to stay in time budget
        bld = FAST_BUILD if cid in FAST_BUILD else BUILD

        for oi, out_name in enumerate(OUTPUTS):
            y_te = Y_te[:, oi]

            # ── Holdout 80/20 × 30 splits ─────────────────────────────────
            for rep in range(30):
                n_val = int(0.2 * N)
                idx   = rng.permutation(N)
                tr_idx, val_idx = idx[n_val:], idx[:n_val]
                try:
                    y_pred, _ = fit_predict(cid, X_tr[tr_idx], Y_tr[tr_idx, oi],
                                            X_tr[val_idx], bld)
                    rmse_cv = compute_rmse(Y_tr[val_idx, oi], y_pred)
                    y_pred_te, _ = fit_predict(cid, X_tr[tr_idx], Y_tr[tr_idx, oi], X_te, bld)
                    rmse_te = compute_rmse(y_te, y_pred_te)
                except Exception:
                    rmse_cv = rmse_te = np.nan
                rows.append({"model": cid, "output": out_name, "strategy": "holdout_80_20",
                             "rep": rep, "rmse_cv": rmse_cv, "rmse_test": rmse_te})

            # ── 5-fold CV × 5 repeats ─────────────────────────────────────
            for rep in range(5):
                fold_rng = np.random.default_rng(rep * 100)
                rmses = cv_rmse(cid, X_tr, Y_tr, oi, 5, fold_rng, bld)
                rows.append({"model": cid, "output": out_name, "strategy": "5fold_cv",
                             "rep": rep, "rmse_cv": float(np.nanmean(rmses)),
                             "rmse_test": np.nan})

            # ── 10-fold CV × 5 repeats ────────────────────────────────────
            for rep in range(5):
                fold_rng = np.random.default_rng(rep * 200)
                rmses = cv_rmse(cid, X_tr, Y_tr, oi, 10, fold_rng, bld)
                rows.append({"model": cid, "output": out_name, "strategy": "10fold_cv",
                             "rep": rep, "rmse_cv": float(np.nanmean(rmses)),
                             "rmse_test": np.nan})

            # ── LOOCV — restricted to primary outputs for time budget ──────
            if out_name in LOOCV_OUTPUTS:
                try:
                    loocv_rng = np.random.default_rng(0)
                    rmses = cv_rmse(cid, X_tr, Y_tr, oi, N, loocv_rng, bld)
                    rows.append({"model": cid, "output": out_name, "strategy": "loocv",
                                 "rep": 0, "rmse_cv": float(np.nanmean(rmses)),
                                 "rmse_test": np.nan})
                except Exception:
                    rows.append({"model": cid, "output": out_name, "strategy": "loocv",
                                 "rep": 0, "rmse_cv": np.nan, "rmse_test": np.nan})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "val1_raw.csv", index=False)

    # Also record actual test RMSE as reference
    test_ref = []
    for cid in TOP3:
        for oi, out_name in enumerate(OUTPUTS):
            try:
                y_pred, _ = fit_predict(cid, X_tr, Y_tr[:, oi], X_te)
                test_ref.append({"model": cid, "output": out_name,
                                 "strategy": "test_set",
                                 "rep": 0,
                                 "rmse_cv": compute_rmse(Y_te[:, oi], y_pred),
                                 "rmse_test": compute_rmse(Y_te[:, oi], y_pred)})
            except Exception:
                pass
    pd.concat([df, pd.DataFrame(test_ref)]).to_csv(OUT / "val1_raw.csv", index=False)

    summary = (
        df.groupby(["model", "output", "strategy"])["rmse_cv"]
        .agg(["mean", "std"]).reset_index()
        .rename(columns={"mean": "rmse_mean", "std": "rmse_std"})
    )
    summary.to_csv(OUT / "val1_summary.csv", index=False)
    return df, summary


# ── SS-1: Learning curves ─────────────────────────────────────────────────────

def run_ss1(X_tr_full, Y_tr_full, X_te, Y_te):
    rows = []
    for cid in LC_MODELS:
        print(f"  SS-1: {cid}")
        for N in N_TRAIN_LC:
            if N > len(X_tr_full):
                break
            X_tr = X_tr_full[:N]
            Y_tr = Y_tr_full[:N]
            for oi, out_name in enumerate(OUTPUTS):
                try:
                    t0 = time.perf_counter()
                    y_pred, _ = fit_predict(cid, X_tr, Y_tr[:, oi], X_te)
                    t_train = time.perf_counter() - t0
                    rows.append({
                        "model": cid, "N": N, "output": out_name,
                        "rmse":  compute_rmse(Y_te[:, oi], y_pred),
                        "nrmse": compute_nrmse(Y_te[:, oi], y_pred),
                        "r2":    compute_r2(Y_te[:, oi], y_pred),
                        "train_s": t_train,
                    })
                except Exception as e:
                    rows.append({"model": cid, "N": N, "output": out_name,
                                 "rmse": np.nan, "nrmse": np.nan,
                                 "r2": np.nan, "train_s": np.nan})

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ss1_raw.csv", index=False)
    return df


# ── Part F: Calibration ───────────────────────────────────────────────────────

def run_calibration(X_tr_full, Y_tr_full, X_te, Y_te):
    """
    Kriging uncertainty calibration analysis.
    Uses the GP-M52 and GP-SE models trained on N=100.
    """
    from src.surrogates.registry import _smt_krg

    N = 100
    X_tr = X_tr_full[:N]
    Y_tr = Y_tr_full[:N]
    nominal_levels = [0.50, 0.80, 0.90, 0.95]

    cov_rows   = []
    spear_rows = []

    krg_models = [cid for cid in TOP3 if cid.startswith("GP")]

    for cid in krg_models:
        print(f"  Calibration: {cid}")
        build_fn = BUILD[cid]

        for oi, out_name in enumerate(OUTPUTS):
            try:
                # Get both mean and std predictions
                m = build_fn(X_tr, Y_tr[:, oi])
                mu    = np.array(m.predict(X_te)).ravel()
                sigma = np.array(m.predict_std(X_te)).ravel()

                y_true = Y_te[:, oi]
                errors = np.abs(y_true - mu)

                # ── reliability diagram ────────────────────────────────
                for alpha in nominal_levels:
                    z = sp_norm.ppf((1 + alpha) / 2)
                    lo, hi = mu - z * sigma, mu + z * sigma
                    emp_coverage = float(np.mean((y_true >= lo) & (y_true <= hi)))
                    cov_rows.append({
                        "model": cid, "output": out_name,
                        "nominal": alpha, "empirical": emp_coverage,
                    })

                # ── Spearman ρ(σ, |error|) ────────────────────────────
                rho, pval = spearmanr(sigma, errors)
                spear_rows.append({
                    "model": cid, "output": out_name,
                    "spearman_rho": float(rho),
                    "p_value": float(pval),
                })

            except Exception as e:
                warnings.warn(f"Calibration failed {cid} {out_name}: {e}")

    df_cov   = pd.DataFrame(cov_rows)
    df_spear = pd.DataFrame(spear_rows)
    df_cov.to_csv(OUT / "calibration.csv", index=False)
    df_spear.to_csv(OUT / "spearman.csv", index=False)
    return df_cov, df_spear


# ── plots ──────────────────────────────────────────────────────────────────────

MODEL_COLOR = {
    "GP-M52":      "#377eb8",
    "GP-SE":       "#984ea3",
    "GP-SE-noARD": "#ff7f00",
    "POLY-3-L":    "#e41a1c",
    "NN-M-tanh":   "#a65628",
    "RBF-C":       "#4daf4a",
}
STRATEGY_STYLE = {
    "holdout_80_20": {"label": "Holdout 80/20", "color": "#e41a1c", "marker": "o"},
    "5fold_cv":      {"label": "5-fold CV",     "color": "#377eb8", "marker": "s"},
    "10fold_cv":     {"label": "10-fold CV",    "color": "#4daf4a", "marker": "^"},
    "loocv":         {"label": "LOOCV",         "color": "#984ea3", "marker": "D"},
    "test_set":      {"label": "Test set (ref)","color": "black",   "marker": "*"},
}


def plot_val1(summary: pd.DataFrame) -> None:
    """Box/violin plot of CV RMSE distribution by strategy for top 3 models."""
    raw = pd.read_csv(OUT / "val1_raw.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    for ax_i, cid in enumerate(TOP3):
        ax = axes[ax_i]
        sub = raw[(raw["model"] == cid) & (raw["output"] == "v_out_mean")]
        strats = ["holdout_80_20", "5fold_cv", "10fold_cv", "loocv"]
        data   = [sub[sub["strategy"] == s]["rmse_cv"].dropna().values for s in strats]
        labels = [STRATEGY_STYLE[s]["label"] for s in strats]
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = [STRATEGY_STYLE[s]["color"] for s in strats]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.6)
        # Add test-set reference line
        test_rmse = sub[sub["strategy"] == "test_set"]["rmse_cv"].values
        if len(test_rmse):
            ax.axhline(test_rmse[0], color="black", ls="--", lw=1.5,
                       label=f"Test set: {test_rmse[0]:.3f} V")
            ax.legend(fontsize=8)
        ax.set_title(cid, fontsize=11)
        ax.set_ylabel("RMSE on v_out_mean (V)" if ax_i == 0 else "")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("VAL-1: Validation strategy comparison (v_out_mean)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "val1_strategy_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved: val1_strategy_comparison.png")


def plot_ss1(df: pd.DataFrame) -> None:
    """Learning curves: RMSE vs N for each model (v_out_mean)."""
    sub = df[df["output"] == "v_out_mean"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for cid in LC_MODELS:
        s = sub[sub["model"] == cid].sort_values("N")
        if s.empty:
            continue
        color = MODEL_COLOR.get(cid, "#888")
        ax.plot(s["N"], s["rmse"], marker="o", color=color,
                label=cid, linewidth=2, markersize=5)
    ax.axvline(40, color="gray", ls=":", alpha=0.7,
               label="10d rule (Loeppky 2009)")
    ax.set_xlabel("Training set size N")
    ax.set_ylabel("RMSE on v_out_mean (V, test set)")
    ax.set_title("SS-1: Learning curves — RMSE vs N (v_out_mean)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "ss1_learning_curves.png", dpi=150)
    plt.close(fig)
    print("  Saved: ss1_learning_curves.png")

    # All outputs in one grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for ax_i, out_name in enumerate(OUTPUTS):
        ax = axes[ax_i]
        sub_o = df[df["output"] == out_name]
        for cid in LC_MODELS:
            s = sub_o[sub_o["model"] == cid].sort_values("N")
            if s.empty:
                continue
            ax.plot(s["N"], s["rmse"], marker="o", color=MODEL_COLOR.get(cid, "#888"),
                    label=cid, linewidth=1.5, markersize=4)
        ax.set_title(out_name, fontsize=10)
        ax.set_xlabel("N")
        ax.set_ylabel("RMSE")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    axes[-1].set_visible(False)
    fig.suptitle("SS-1: Learning curves — all outputs", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "ss1_learning_curves_all.png", dpi=150)
    plt.close(fig)
    print("  Saved: ss1_learning_curves_all.png")


def plot_calibration(df_cov: pd.DataFrame, df_spear: pd.DataFrame) -> None:
    """Reliability diagram + Spearman ρ bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Reliability diagram ───────────────────────────────────────────────
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for cid in df_cov["model"].unique():
        sub = df_cov[(df_cov["model"] == cid) & (df_cov["output"] == "v_out_mean")]
        ax.plot(sub["nominal"], sub["empirical"],
                marker="o", label=cid, color=MODEL_COLOR.get(cid, "#888"))
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Reliability diagram (v_out_mean)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.4, 1.0); ax.set_ylim(0.4, 1.0)

    # ── Spearman ρ heatmap ────────────────────────────────────────────────
    ax = axes[1]
    pivot = df_spear.pivot_table(values="spearman_rho",
                                 index="model", columns="output")
    pivot = pivot[OUTPUTS]
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman ρ(σ, |error|)")
    ax.set_xticks(range(len(OUTPUTS)))
    ax.set_xticklabels(OUTPUTS, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for i in range(len(pivot)):
        for j in range(len(OUTPUTS)):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if abs(v) < 0.7 else "white")
    ax.set_title("Spearman ρ(σ_pred, |error|)\npositive = uncertainty is informative", fontsize=10)

    fig.tight_layout()
    fig.savefig(FIG / "calibration_and_spearman.png", dpi=150)
    plt.close(fig)
    print("  Saved: calibration_and_spearman.png")


# ── summary ────────────────────────────────────────────────────────────────────

def write_summary(val1_summary, ss1_df, df_cov, df_spear):
    lines = ["PHASE 4 SUMMARY", "=" * 55, ""]

    # VAL-1: do strategies agree on ranking?
    lines += ["VAL-1 — Validation strategy comparison", "-" * 40]
    for cid in TOP3:
        sub = val1_summary[(val1_summary["model"] == cid) &
                           (val1_summary["output"] == "v_out_mean")]
        for _, row in sub.iterrows():
            lines.append(f"  {cid:15s} {row['strategy']:15s}  "
                         f"mean={row['rmse_mean']:.4f}  std={row['rmse_std']:.4f}")
    lines += [""]

    # SS-1: minimum N recommendation
    lines += ["SS-1 — Learning curve findings", "-" * 40]
    sub = ss1_df[(ss1_df["model"] == "GP-M52") & (ss1_df["output"] == "v_out_mean")]
    sub = sub.sort_values("N")
    for _, row in sub.iterrows():
        lines.append(f"  N={row['N']:3d}  RMSE={row['rmse']:.4f} V")
    lines += [""]

    # Calibration summary
    lines += ["Part F — Calibration", "-" * 40]
    sub_cov = df_cov[df_cov["output"] == "v_out_mean"]
    lines.append(sub_cov.to_string(index=False))
    lines += ["", "Spearman ρ (v_out_mean):"]
    lines.append(df_spear[df_spear["output"] == "v_out_mean"].to_string(index=False))

    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text)
    print("\n" + text)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 4 — VALIDATION STRATEGIES & LEARNING CURVES")
    print("=" * 60)

    print("\n[1] Loading data …")
    X_tr, Y_tr, X_te, Y_te = load_data()
    print(f"  Train (all CCM-valid): {len(X_tr)}, Test: {len(X_te)}")

    # ── VAL-1 ─────────────────────────────────────────────────────────────
    val1_raw_path = OUT / "val1_raw.csv"
    if val1_raw_path.exists():
        print("\n[2] Loading cached VAL-1 …")
        val1_raw     = pd.read_csv(val1_raw_path)
        val1_summary = pd.read_csv(OUT / "val1_summary.csv")
    else:
        print("\n[2] Experiment VAL-1 — validation strategy comparison …")
        val1_raw, val1_summary = run_val1(X_tr, Y_tr, X_te, Y_te)
    print("  VAL-1 done")

    # ── SS-1 ──────────────────────────────────────────────────────────────
    ss1_path = OUT / "ss1_raw.csv"
    if ss1_path.exists():
        print("\n[3] Loading cached SS-1 …")
        ss1_df = pd.read_csv(ss1_path)
    else:
        print("\n[3] Experiment SS-1 — learning curves …")
        ss1_df = run_ss1(X_tr, Y_tr, X_te, Y_te)
    print("  SS-1 done")

    # ── Calibration ───────────────────────────────────────────────────────
    cal_path = OUT / "calibration.csv"
    if cal_path.exists():
        print("\n[4] Loading cached calibration …")
        df_cov   = pd.read_csv(cal_path)
        df_spear = pd.read_csv(OUT / "spearman.csv")
    else:
        print("\n[4] Calibration analysis (Part F) …")
        df_cov, df_spear = run_calibration(X_tr, Y_tr, X_te, Y_te)
    print("  Calibration done")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[5] Generating figures …")
    plot_val1(val1_summary)
    plot_ss1(ss1_df)
    plot_calibration(df_cov, df_spear)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n[6] Summary …")
    write_summary(val1_summary, ss1_df, df_cov, df_spear)

    print(f"\nAll outputs written to {OUT}")


if __name__ == "__main__":
    main()
