#!/usr/bin/env python3
"""
Phase 3 — Model Comparison
===========================

Runs the MC-1 and MC-2 model-comparison analyses.

Experiment MC-1: Full comparison
  - Training data : first 100 CCM-valid rows from train_200.csv
  - Test data     : test_500.csv  (468 CCM-valid points)
  - All 19 model configurations trained on the SAME data
  - Evaluate: RMSE, NRMSE, R², MAE, Max-AE per output variable
  - 5-fold CV on training data
  - Training and prediction times

Experiment MC-2: Hyperparameter sensitivity (top 3 families)
  - Polynomial: sweep degree {1,2,3,4} and regularisation alpha {0.01,0.1,1,10}
  - Kriging: compare SE vs M52 vs M32 kernels (already in MC-1)
  - Neural Network: compare hidden-layer widths and activations

Outputs
-------
  outputs/phase3/mc1_raw.csv           — per-(model,output,fold) metrics
  outputs/phase3/mc1_test_metrics.csv  — test-set metrics for all models
  outputs/phase3/mc1_summary.csv       — ranked model table (Table MC1)
  outputs/phase3/mc2_poly.csv          — polynomial hyperparameter sensitivity
  outputs/phase3/mc2_nn.csv            — NN architecture sensitivity
  outputs/phase3/top_models.txt        — top 3 models selected
  outputs/phase3/figures/              — heatmaps, bar charts, parity plots
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from src.utils.metrics import compute_rmse, compute_nrmse, compute_r2, compute_mae
from src.surrogates.registry import MODEL_REGISTRY, MODEL_FAMILY

# ── config ────────────────────────────────────────────────────────────────────
N_TRAIN    = 100
N_FOLDS    = 5
OUTPUTS    = ["v_out_mean", "i_l_mean", "v_out_ripple", "i_l_ripple", "efficiency"]
INPUT_COLS = ["D", "V_in", "R", "f_sw"]

OUT = ROOT / "outputs" / "phase3"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = ROOT / "outputs" / "phase2" / "train_200.csv"
TEST_CSV  = ROOT / "outputs" / "phase2" / "test_500.csv"


# ── data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and prepare training (N=100 CCM-valid) and test arrays."""
    train_full = pd.read_csv(TRAIN_CSV)
    test_full  = pd.read_csv(TEST_CSV)

    train_ccm = train_full[train_full["ccm_ok"]].reset_index(drop=True)
    test_ccm  = test_full[test_full["ccm_ok"]].reset_index(drop=True)

    # Take first N_TRAIN CCM-valid points
    train_ccm = train_ccm.iloc[:N_TRAIN]

    X_tr = train_ccm[INPUT_COLS].values.astype(float)
    Y_tr = train_ccm[OUTPUTS].values.astype(float)
    X_te = test_ccm[INPUT_COLS].values.astype(float)
    Y_te = test_ccm[OUTPUTS].values.astype(float)

    print(f"  Training : {X_tr.shape[0]} points")
    print(f"  Test     : {X_te.shape[0]} points")
    return X_tr, Y_tr, X_te, Y_te


# ── single model evaluation ────────────────────────────────────────────────────

def evaluate_model(
    config_id: str,
    build_fn,
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_te: np.ndarray,
    Y_te: np.ndarray,
) -> dict:
    """
    Train model on full training set, evaluate on test set, run 5-fold CV.
    Returns a dict with all metrics.
    """
    results = {"config_id": config_id, "family": MODEL_FAMILY[config_id]}

    # ── test-set evaluation ────────────────────────────────────────────────
    test_rows = []
    train_times = []
    pred_times  = []

    for oi, out_name in enumerate(OUTPUTS):
        y_tr = Y_tr[:, oi]
        y_te = Y_te[:, oi]
        try:
            t0    = time.perf_counter()
            model = build_fn(X_tr, y_tr)
            t_train = time.perf_counter() - t0

            t0    = time.perf_counter()
            y_pred = np.array(model.predict(X_te)).ravel()
            t_pred = (time.perf_counter() - t0) / len(X_te) * 1e6  # µs/point

            train_times.append(t_train)
            pred_times.append(t_pred)
            test_rows.append({
                "config_id": config_id, "family": MODEL_FAMILY[config_id],
                "output": out_name,
                "rmse":      compute_rmse(y_te, y_pred),
                "nrmse":     compute_nrmse(y_te, y_pred),
                "r2":        compute_r2(y_te, y_pred),
                "mae":       compute_mae(y_te, y_pred),
                "max_ae":    float(np.max(np.abs(y_te - y_pred))),
                "train_s":   t_train,
                "pred_us":   t_pred,
            })
        except Exception as e:
            warnings.warn(f"  [{config_id}] {out_name} test failed: {e}")
            test_rows.append({
                "config_id": config_id, "family": MODEL_FAMILY[config_id],
                "output": out_name,
                "rmse": np.nan, "nrmse": np.nan, "r2": np.nan,
                "mae": np.nan, "max_ae": np.nan,
                "train_s": np.nan, "pred_us": np.nan,
            })

    # ── 5-fold CV ─────────────────────────────────────────────────────────
    cv_rows = []
    n = len(X_tr)
    rng = np.random.default_rng(42)
    fold_idx = np.array_split(rng.permutation(n), N_FOLDS)

    for fold_i, val_idx in enumerate(fold_idx):
        tr_idx = np.setdiff1d(np.arange(n), val_idx)
        Xf_tr, Xf_val = X_tr[tr_idx], X_tr[val_idx]

        for oi, out_name in enumerate(OUTPUTS):
            yf_tr  = Y_tr[tr_idx,  oi]
            yf_val = Y_tr[val_idx, oi]
            try:
                m      = build_fn(Xf_tr, yf_tr)
                y_pred = np.array(m.predict(Xf_val)).ravel()
                cv_rows.append({
                    "config_id": config_id, "output": out_name, "fold": fold_i,
                    "rmse": compute_rmse(yf_val, y_pred),
                    "r2":   compute_r2(yf_val, y_pred),
                })
            except Exception:
                cv_rows.append({
                    "config_id": config_id, "output": out_name, "fold": fold_i,
                    "rmse": np.nan, "r2": np.nan,
                })

    return {"test": test_rows, "cv": cv_rows}


# ── MC-1 ──────────────────────────────────────────────────────────────────────

def run_mc1(X_tr, Y_tr, X_te, Y_te):
    test_all, cv_all = [], []
    n_configs = len(MODEL_REGISTRY)

    for i, (config_id, family, build_fn) in enumerate(MODEL_REGISTRY):
        t0 = time.perf_counter()
        print(f"  [{i+1:02d}/{n_configs}] {config_id:15s} ({family}) … ", end="", flush=True)
        res = evaluate_model(config_id, build_fn, X_tr, Y_tr, X_te, Y_te)
        elapsed = time.perf_counter() - t0
        test_all.extend(res["test"])
        cv_all.extend(res["cv"])

        # Quick progress: RMSE on v_out_mean
        vout_rows = [r for r in res["test"] if r["output"] == "v_out_mean"]
        rmse_v = vout_rows[0]["rmse"] if vout_rows else float("nan")
        print(f"done {elapsed:.1f}s  v_out RMSE={rmse_v:.4f}")

    df_test = pd.DataFrame(test_all)
    df_cv   = pd.DataFrame(cv_all)
    df_test.to_csv(OUT / "mc1_test_metrics.csv", index=False)
    df_cv.to_csv(OUT / "mc1_raw.csv", index=False)

    # Summary: mean CV RMSE per model (primary sort key) + test RMSE
    cv_summary = (
        df_cv.groupby(["config_id", "output"])["rmse"]
        .mean().reset_index().rename(columns={"rmse": "cv_rmse_mean"})
    )
    mc1_summary = df_test.merge(
        cv_summary, on=["config_id", "output"], how="left"
    )
    mc1_summary.to_csv(OUT / "mc1_summary.csv", index=False)
    return df_test, df_cv, mc1_summary


# ── MC-2: polynomial hyperparameter sensitivity ────────────────────────────────

def run_mc2_poly(X_tr, Y_tr, X_te, Y_te):
    from src.surrogates.registry import _poly
    rows = []
    for degree in [1, 2, 3, 4]:
        for reg in ["ols", "ridge", "lasso"]:
            if reg == "ols" and degree > 2:
                continue   # OLS with high degree almost always overfits badly
            for alpha in ([None] if reg == "ols" else [0.01, 0.1, 1.0, 10.0]):
                kw = {} if alpha is None else {"alpha": alpha}
                build_fn = _poly(degree, reg, **kw) if alpha is not None else _poly(degree, "ols")
                for oi, out_name in enumerate(OUTPUTS):
                    try:
                        m = build_fn(X_tr, Y_tr[:, oi])
                        y_pred = np.array(m.predict(X_te)).ravel()
                        rows.append({
                            "degree": degree, "regulariser": reg,
                            "alpha": alpha if alpha is not None else 0.0,
                            "output": out_name,
                            "rmse": compute_rmse(Y_te[:, oi], y_pred),
                            "r2":   compute_r2(Y_te[:, oi], y_pred),
                        })
                    except Exception:
                        rows.append({"degree": degree, "regulariser": reg,
                                     "alpha": alpha, "output": out_name,
                                     "rmse": np.nan, "r2": np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mc2_poly.csv", index=False)
    return df


def run_mc2_nn(X_tr, Y_tr, X_te, Y_te):
    from src.surrogates.registry import _nn
    configs = [
        ("S",        [32, 16],       "relu", 0.0, 0.0),
        ("M",        [64, 32],       "relu", 0.0, 0.0),
        ("L",        [128, 64, 32],  "relu", 0.1, 0.0),
        ("XL",       [256,128, 64],  "relu", 0.2, 0.0),
        ("M-tanh",   [64, 32],       "tanh", 0.0, 0.0),
        ("M-wd1e-3", [64, 32],       "relu", 0.0, 1e-3),
        ("M-wd1e-4", [64, 32],       "relu", 0.0, 1e-4),
    ]
    rows = []
    for tag, hidden, act, drop, wd in configs:
        build_fn = _nn(hidden, act, drop, wd)
        for oi, out_name in enumerate(OUTPUTS):
            try:
                m = build_fn(X_tr, Y_tr[:, oi])
                y_pred = np.array(m.predict(X_te)).ravel()
                rows.append({
                    "config": tag, "hidden": str(hidden), "activation": act,
                    "dropout": drop, "weight_decay": wd,
                    "output": out_name,
                    "rmse":  compute_rmse(Y_te[:, oi], y_pred),
                    "r2":    compute_r2(Y_te[:, oi], y_pred),
                    "train_s": getattr(m, "_train_time", np.nan),
                })
            except Exception:
                rows.append({"config": tag, "hidden": str(hidden), "activation": act,
                             "dropout": drop, "weight_decay": wd,
                             "output": out_name, "rmse": np.nan, "r2": np.nan,
                             "train_s": np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mc2_nn.csv", index=False)
    return df


# ── plots ──────────────────────────────────────────────────────────────────────

FAMILY_COLOR = {
    "Polynomial": "#e41a1c",
    "Kriging":    "#377eb8",
    "Neural":     "#ff7f00",
    "RBF":        "#4daf4a",
}


def plot_nrmse_heatmap(df_test: pd.DataFrame) -> None:
    """Table MC2: heatmap of NRMSE across all models × outputs."""
    pivot = df_test.pivot_table(values="nrmse", index="config_id", columns="output")
    pivot = pivot[OUTPUTS]   # consistent column order

    # Sort rows by mean NRMSE across outputs
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.15)
    plt.colorbar(im, ax=ax, label="NRMSE")
    ax.set_xticks(range(len(OUTPUTS)))
    ax.set_xticklabels(OUTPUTS, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    # Annotate cells
    for i in range(len(pivot)):
        for j in range(len(OUTPUTS)):
            v = pivot.values[i, j]
            txt = f"{v:.3f}" if np.isfinite(v) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="black" if v < 0.08 else "white")
    ax.set_title("Table MC2: NRMSE heatmap — all models × all outputs", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "mc1_nrmse_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved: mc1_nrmse_heatmap.png")


def plot_rmse_bar(df_test: pd.DataFrame) -> None:
    """Table MC1: bar chart of RMSE on v_out_mean, ranked."""
    sub = df_test[df_test["output"] == "v_out_mean"].copy()
    sub["color"] = sub["family"].map(FAMILY_COLOR)
    sub = sub.sort_values("rmse")

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(sub)), sub["rmse"], color=sub["color"], edgecolor="white")
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub["config_id"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("RMSE on v_out_mean (V)")
    ax.set_title("Table MC1: Model Comparison — RMSE on v_out_mean (N=100 training)", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=c, label=f) for f, c in FAMILY_COLOR.items()]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG / "mc1_rmse_bar_vout.png", dpi=150)
    plt.close(fig)
    print("  Saved: mc1_rmse_bar_vout.png")


def plot_parity(df_test_raw: pd.DataFrame, build_fn_map: dict,
                X_tr, Y_tr, X_te, Y_te, top_ids: list) -> None:
    """Parity plots (predicted vs actual) for top 3 models on v_out_mean."""
    fig, axes = plt.subplots(1, len(top_ids), figsize=(5*len(top_ids), 5))
    if len(top_ids) == 1:
        axes = [axes]
    for ax, cid in zip(axes, top_ids):
        build_fn = build_fn_map[cid]
        try:
            m      = build_fn(X_tr, Y_tr[:, 0])
            y_pred = np.array(m.predict(X_te)).ravel()
            y_true = Y_te[:, 0]
            lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
            ax.scatter(y_true, y_pred, s=8, alpha=0.5, color=FAMILY_COLOR.get(
                MODEL_FAMILY.get(cid, "Polynomial"), "#888"))
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.7)
            rmse = compute_rmse(y_true, y_pred)
            r2   = compute_r2(y_true, y_pred)
            ax.set_title(f"{cid}\nRMSE={rmse:.3f} V, R²={r2:.4f}", fontsize=10)
            ax.set_xlabel("Simulated v_out_mean (V)")
            ax.set_ylabel("Predicted v_out_mean (V)")
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.set_title(f"{cid}\n(failed: {e})", fontsize=8)
    fig.suptitle("Parity plots — top models on v_out_mean (test set)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "mc1_parity_top3.png", dpi=150)
    plt.close(fig)
    print("  Saved: mc1_parity_top3.png")


def plot_training_times(df_test: pd.DataFrame) -> None:
    sub = df_test[df_test["output"] == "v_out_mean"].copy().sort_values("train_s")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.barh(sub["config_id"], sub["train_s"] * 1000,
            color=[FAMILY_COLOR.get(f, "#888") for f in sub["family"]])
    ax.set_xlabel("Training time (ms)")
    ax.set_title("Table MC3: Training times per model (v_out_mean)", fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(FIG / "mc1_training_times.png", dpi=150)
    plt.close(fig)
    print("  Saved: mc1_training_times.png")


# ── top-model selection ───────────────────────────────────────────────────────

def select_top_models(df_test: pd.DataFrame) -> list[str]:
    """Select top 3 model families based on RMSE on v_out_mean (test set)."""
    primary = df_test[df_test["output"] == "v_out_mean"].copy()
    primary = primary.sort_values("rmse").dropna(subset=["rmse"])

    top3_ids = primary["config_id"].head(3).tolist()
    top3_families = primary["family"].head(3).tolist()

    lines = [
        "Top Models by v_out_mean RMSE",
        "=" * 50,
        "",
        "Ranked by RMSE on v_out_mean (test set, N_train=100):",
        "",
        primary[["config_id", "family", "rmse", "nrmse", "r2", "mae"]].head(10).to_string(index=False),
        "",
        f"Top 3 selected: {top3_ids}",
        f"Families: {top3_families}",
        "",
        "These models are used in the subsequent validation and engineering analyses.",
    ]
    text = "\n".join(lines)
    (OUT / "top_models.txt").write_text(text)
    print("\n" + text)
    return top3_ids


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PHASE 3 — MODEL COMPARISON")
    print("=" * 60)

    print("\n[1] Loading data …")
    X_tr, Y_tr, X_te, Y_te = load_data()

    # ── MC-1 ─────────────────────────────────────────────────────────────────
    mc1_test_path = OUT / "mc1_test_metrics.csv"
    if mc1_test_path.exists():
        print("\n[2] Loading cached MC-1 results …")
        df_test   = pd.read_csv(mc1_test_path)
        df_cv     = pd.read_csv(OUT / "mc1_raw.csv")
        mc1_summ  = pd.read_csv(OUT / "mc1_summary.csv")
    else:
        print("\n[2] Experiment MC-1 — full model comparison …")
        df_test, df_cv, mc1_summ = run_mc1(X_tr, Y_tr, X_te, Y_te)

    print("\n  v_out_mean test RMSE ranking:")
    vout = df_test[df_test["output"] == "v_out_mean"].sort_values("rmse")
    print(vout[["config_id", "family", "rmse", "nrmse", "r2", "train_s"]].to_string(index=False))

    # ── MC-2 ─────────────────────────────────────────────────────────────────
    mc2_poly_path = OUT / "mc2_poly.csv"
    mc2_nn_path   = OUT / "mc2_nn.csv"

    if mc2_poly_path.exists():
        print("\n[3] Loading cached MC-2 poly results …")
        df_mc2_poly = pd.read_csv(mc2_poly_path)
    else:
        print("\n[3] MC-2 polynomial hyperparameter sensitivity …")
        df_mc2_poly = run_mc2_poly(X_tr, Y_tr, X_te, Y_te)

    if mc2_nn_path.exists():
        print("\n[4] Loading cached MC-2 NN results …")
        df_mc2_nn = pd.read_csv(mc2_nn_path)
    else:
        print("\n[4] MC-2 neural network architecture sensitivity …")
        df_mc2_nn = run_mc2_nn(X_tr, Y_tr, X_te, Y_te)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[5] Generating figures …")
    plot_nrmse_heatmap(df_test)
    plot_rmse_bar(df_test)
    plot_training_times(df_test)

    # Parity plots for top 3
    top3 = df_test[df_test["output"] == "v_out_mean"].sort_values("rmse")["config_id"].head(3).tolist()
    build_fn_map = {cid: fn for cid, _, fn in MODEL_REGISTRY}
    plot_parity(df_test, build_fn_map, X_tr, Y_tr, X_te, Y_te, top3)

    print("\n[6] Selecting top models …")
    top3_ids = select_top_models(df_test)

    print(f"\nAll outputs written to {OUT}")


if __name__ == "__main__":
    main()
