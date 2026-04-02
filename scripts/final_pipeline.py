#!/usr/bin/env python3
"""
final_pipeline.py
=================
Model comparison, calibration, and learning-curve pipeline.

Data: data/raw/simulation_results_hifi.csv  (100 pts, MATLAB ODE)
      data/raw/doe_inputs.csv

Phase 3 MC-1 : 5-fold CV (seed=42) for 4 main models × 5 outputs
Phase 3 MC-2 : All variant configs (Poly, GP, NN)
Phase 4 LC   : Learning curves N=20,40,60,80,100 for 4 main models
Phase 4 CAL  : GP-SE calibration + Spearman ρ(σ̂, |error|)

Outputs
-------
  outputs/phase3/mc1_raw.csv
  outputs/phase3/mc1_summary.csv
  outputs/phase3/mc2_gp.csv
  outputs/phase3/mc2_nn.csv
  outputs/phase3/mc2_poly.csv
  outputs/phase3/mc2_rbf.csv
  outputs/phase4/calibration.csv
  outputs/phase4/spearman.csv
  outputs/phase4/learning_curves.csv
  thesis_figures/ (8 PNG files)
"""

from __future__ import annotations
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm as sp_norm
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT3 = ROOT / "outputs" / "phase3"
OUT4 = ROOT / "outputs" / "phase4"
OUT3.mkdir(parents=True, exist_ok=True)
OUT4.mkdir(parents=True, exist_ok=True)

# ─── Data ────────────────────────────────────────────────────────────────────
INPUT_COLS  = ["D", "V_in", "R", "f_sw"]
OUTPUT_COLS = ["v_out_mean", "i_l_mean", "v_out_ripple", "i_l_ripple", "efficiency"]


def load_raw():
    sim = pd.read_csv(ROOT / "data/raw/simulation_results_hifi.csv")
    X   = sim[INPUT_COLS].values.astype(float)
    Y   = sim[OUTPUT_COLS].values.astype(float)
    return X, Y


# ─── Metrics ─────────────────────────────────────────────────────────────────
def _rmse(yt, yp):
    return float(np.sqrt(np.mean((yt - yp) ** 2)))

def _nrmse(yt, yp):
    rng = yt.max() - yt.min()
    return _rmse(yt, yp) / rng if rng > 1e-12 else np.nan

def _r2(yt, yp):
    ss = np.sum((yt - yp) ** 2)
    st = np.sum((yt - yt.mean()) ** 2)
    return float(1 - ss / st) if st > 1e-12 else np.nan

def _mae(yt, yp):
    return float(np.mean(np.abs(yt - yp)))


# ─── Model builders ──────────────────────────────────────────────────────────

def build_linear():
    """POLY-1: OLS linear regression with StandardScaler(X)."""
    def _build(X, y):
        pipe = Pipeline([("sc", StandardScaler()), ("ols", LinearRegression())])
        pipe.fit(X, y.ravel())
        return pipe
    return _build


def build_poly(degree: int, reg: str = "ols", alpha: float = 1.0):
    """Polynomial regression with StandardScaler(X).
    reg='ols'|'ridge'|'lasso'; alpha only used for ridge/lasso."""
    def _build(X, y):
        if reg == "ols":
            est = LinearRegression()
        elif reg == "ridge":
            est = Ridge(alpha=alpha)
        else:
            est = Lasso(alpha=alpha, max_iter=10_000)
        pipe = Pipeline([
            ("sc",  StandardScaler()),
            ("pf",  PolynomialFeatures(degree, include_bias=False)),
            ("est", est),
        ])
        pipe.fit(X, y.ravel())
        return pipe
    return _build


def build_gp(corr: str = "squar_exp", ARD: bool = True,
             nugget: float | None = None, n_start: int = 10):
    """Kriging via SMT KRG with StandardScaler(X and y)."""
    def _build(X, y):
        from smt.surrogate_models import KRG
        sc_x = StandardScaler()
        Xs   = sc_x.fit_transform(X)
        ym   = float(y.mean())
        ys   = float(y.std()) + 1e-8
        yn   = (y.ravel() - ym) / ys
        d    = Xs.shape[1]
        nug  = 1e-6 if nugget is None else nugget
        sm   = KRG(
            theta0      = [1e-2] * (d if ARD else 1),
            corr        = corr,
            poly        = "constant",
            nugget      = nug,
            n_start     = n_start,
            print_global= False,
        )
        sm.set_training_values(Xs, yn.reshape(-1, 1))
        sm.train()

        class _W:
            def predict(self_, Xnew):
                Xs2 = sc_x.transform(Xnew)
                return sm.predict_values(Xs2).ravel() * ys + ym
            def predict_std(self_, Xnew):
                Xs2 = sc_x.transform(Xnew)
                var = sm.predict_variances(Xs2).ravel()
                return np.sqrt(np.maximum(var, 0.0)) * ys
        return _W()
    return _build


def build_nn(hidden=(64, 64), activation="relu", weight_decay=0.0,
             dropout=0.0, epochs=500, batch_size=32, seed=42):
    """PyTorch MLP with StandardScaler(X and y).
    Supports optional dropout after each hidden layer."""
    def _build(X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        torch.manual_seed(seed)
        np.random.seed(seed)

        sc_x = StandardScaler()
        Xs   = sc_x.fit_transform(X)
        ym   = float(y.mean())
        ys   = float(y.std()) + 1e-8
        yn   = (y.ravel() - ym) / ys

        Xt  = torch.tensor(Xs, dtype=torch.float32)
        yt_ = torch.tensor(yn, dtype=torch.float32).unsqueeze(1)
        loader = DataLoader(TensorDataset(Xt, yt_),
                            batch_size=batch_size, shuffle=True)

        act_cls = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        layers: list[nn.Module] = []
        in_d = X.shape[1]
        for h in hidden:
            layers += [nn.Linear(in_d, h), act_cls()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        net = nn.Sequential(*layers)

        opt     = torch.optim.Adam(net.parameters(), lr=1e-3,
                                   weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            net.train()
            for xb, yb in loader:
                opt.zero_grad()
                loss_fn(net(xb), yb).backward()
                opt.step()

        class _W:
            def predict(self_, Xnew):
                import torch
                Xs2 = sc_x.transform(Xnew)
                t   = torch.tensor(Xs2, dtype=torch.float32)
                net.eval()
                with torch.no_grad():
                    return net(t).numpy().ravel() * ys + ym
        return _W()
    return _build


def build_rbf(kernel: str, smoothing: float = 1e-4):
    """RBF interpolation via scipy."""
    def _build(X, y):
        from scipy.interpolate import RBFInterpolator
        from scipy.spatial.distance import pdist
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        kw: dict = {"kernel": kernel, "smoothing": smoothing}
        if kernel in ("multiquadric", "inverse_multiquadric", "gaussian"):
            kw["epsilon"] = float(np.mean(pdist(Xs[:min(50, len(Xs))]))) + 1e-8
        interp = RBFInterpolator(Xs, y.ravel(), **kw)

        class _W:
            def predict(self_, Xnew):
                return interp(sc.transform(Xnew))
        return _W()
    return _build


# ─── Config tables ────────────────────────────────────────────────────────────

# MC-1 — the 4 main models
MC1_MAIN = [
    ("POLY-1", build_linear()),
    ("POLY-2", build_poly(2, "ols")),
    ("GP-SE",  build_gp("squar_exp", True, None, 10)),
    ("NN-M",   build_nn([64, 64], "relu", 0.0, 0.0, 500, 32, 42)),
]

# MC-2 Poly variants (POLY-1 and POLY-2 reuse MC-1 results)
MC2_POLY_EXTRA = [
    ("POLY-3",       build_poly(3, "ols")),
    ("POLY-4",       build_poly(4, "ols")),
    ("POLY-3-Ridge", build_poly(3, "ridge", 1.0)),
    ("POLY-4-Ridge", build_poly(4, "ridge", 1.0)),
]

# MC-2 GP variants (GP-SE reuses MC-1 results)
MC2_GP_EXTRA = [
    ("GP-M52",      build_gp("matern52",  True,  None,  10)),
    ("GP-M32",      build_gp("matern32",  True,  None,  10)),
    ("GP-SE-noARD", build_gp("squar_exp", False, None,  10)),
    ("GP-SE-nonug", build_gp("squar_exp", True,  1e-10, 10)),
]

# MC-2 NN variants (NN-M reuses MC-1 results)
MC2_NN_EXTRA = [
    ("NN-S",      build_nn([32, 32],        "relu", 0.0,  0.0,  500, 32, 42)),
    ("NN-L",      build_nn([128, 128, 128], "relu", 0.0,  0.1,  500, 32, 42)),
    ("NN-M-tanh", build_nn([64, 64],        "tanh", 0.0,  0.0,  500, 32, 42)),
    ("NN-M-wd",   build_nn([64, 64],        "relu", 1e-4, 0.0,  500, 32, 42)),
]

# MC-2 RBF variants
MC2_RBF = [
    ("RBF-MQ",  build_rbf("multiquadric")),
    ("RBF-G",   build_rbf("gaussian")),
    ("RBF-TPS", build_rbf("thin_plate_spline")),
    ("RBF-C",   build_rbf("cubic")),
]


# ─── CV runner ────────────────────────────────────────────────────────────────

def run_cv(label: str, build_fn, X: np.ndarray, Y: np.ndarray,
           n_folds: int = 5, seed: int = 42,
           return_preds: bool = False,
           return_std: bool = False):
    """5-fold CV. Returns per-output pooled metrics dict and optionally
    (yt_all, yp_all) arrays of shape (N, n_out), and ys_all for GP."""
    kf   = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    n    = len(X)
    n_out = Y.shape[1]

    yt_all = np.full((n, n_out), np.nan)
    yp_all = np.full((n, n_out), np.nan)
    ys_all = np.full((n, n_out), np.nan) if return_std else None
    fold_rmse = [[] for _ in range(n_out)]

    for fi, (tr, val) in enumerate(kf.split(X)):
        print(f"    [{label}] fold {fi+1}/{n_folds}", end="\r", flush=True)
        for oi in range(n_out):
            try:
                m  = build_fn(X[tr], Y[tr, oi])
                yp = np.array(m.predict(X[val])).ravel()
                yp_all[val, oi] = yp
                yt_all[val, oi] = Y[val, oi]
                fold_rmse[oi].append(_rmse(Y[val, oi], yp))
                if return_std and hasattr(m, "predict_std"):
                    ys_all[val, oi] = m.predict_std(X[val])
            except Exception as e:
                warnings.warn(f"  [{label}] fold {fi} oi {oi}: {e}")
    print()

    metrics = []
    for oi, out in enumerate(OUTPUT_COLS):
        ok = ~(np.isnan(yt_all[:, oi]) | np.isnan(yp_all[:, oi]))
        yt, yp = yt_all[ok, oi], yp_all[ok, oi]
        if len(yt) == 0:
            continue
        metrics.append({
            "output":       out,
            "rmse":         _rmse(yt, yp),
            "nrmse":        _nrmse(yt, yp),
            "r2":           _r2(yt, yp),
            "mae":          _mae(yt, yp),
            "max_ae":       float(np.max(np.abs(yt - yp))),
            "cv_rmse_mean": float(np.mean(fold_rmse[oi])) if fold_rmse[oi] else np.nan,
        })

    if return_std:
        return metrics, yt_all, yp_all, ys_all
    if return_preds:
        return metrics, yt_all, yp_all
    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0_total = time.perf_counter()

    print("=" * 65)
    print("Final Authoritative Pipeline Run")
    print("=" * 65)
    print("Loading data …")
    X, Y = load_raw()
    n = X.shape[0]
    print(f"  {n} samples × {X.shape[1]} inputs × {Y.shape[1]} outputs")
    for oi, out in enumerate(OUTPUT_COLS):
        print(f"  {out}: [{Y[:,oi].min():.4f}, {Y[:,oi].max():.4f}]  "
              f"Δ={Y[:,oi].max()-Y[:,oi].min():.4f}")

    # ── Phase 3 MC-1: 4 main models ──────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("Phase 3 MC-1 — 4 main models (5-fold CV, seed=42)")
    print(f"{'─'*65}")

    mc1_summary_rows: list[dict] = []
    mc1_raw_rows: list[dict] = []
    cv_preds: dict = {}    # config_id → (yt_all, yp_all)
    gp_stds:  dict = {}    # config_id → ys_all

    for config_id, build_fn in MC1_MAIN:
        t0 = time.perf_counter()
        is_gp = config_id.startswith("GP")
        print(f"\n  [{config_id}]")

        if is_gp:
            mets, yt_all, yp_all, ys_all = run_cv(
                config_id, build_fn, X, Y,
                return_preds=True, return_std=True)
            gp_stds[config_id] = ys_all
        else:
            mets, yt_all, yp_all = run_cv(
                config_id, build_fn, X, Y, return_preds=True)

        cv_preds[config_id] = (yt_all, yp_all)
        elapsed = time.perf_counter() - t0

        for row in mets:
            mc1_summary_rows.append({"config_id": config_id, **row})
            mc1_raw_rows.append({
                "config_id": config_id,
                "output":    row["output"],
                "rmse":      row["rmse"],
                "nrmse":     row["nrmse"],
                "r2":        row["r2"],
                "mae":       row["mae"],
            })

        v_row = next((r for r in mets if r["output"] == "v_out_mean"), None)
        if v_row:
            print(f"    v_out_mean: RMSE={v_row['rmse']:.4f}  "
                  f"NRMSE={v_row['nrmse']*100:.3f}%  "
                  f"R²={v_row['r2']:.5f}  ({elapsed:.1f}s)")

    df_mc1 = pd.DataFrame(mc1_summary_rows)
    df_mc1.to_csv(OUT3 / "mc1_summary.csv", index=False)
    pd.DataFrame(mc1_raw_rows).to_csv(OUT3 / "mc1_raw.csv", index=False)
    print(f"\n  Saved → mc1_summary.csv  ({len(df_mc1)} rows)")

    # ── Phase 3 MC-2: Poly variants ──────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("Phase 3 MC-2 — Poly variants")
    print(f"{'─'*65}")

    mc2_poly_rows: list[dict] = []

    # Reuse POLY-1 and POLY-2 from MC-1
    for cid in ["POLY-1", "POLY-2"]:
        for row in mc1_summary_rows:
            if row["config_id"] == cid:
                mc2_poly_rows.append({
                    "config_id": cid,
                    "output":    row["output"],
                    "rmse":      row["rmse"],
                    "nrmse":     row["nrmse"],
                    "r2":        row["r2"],
                    "mae":       row["mae"],
                })

    # Run extra poly configs
    for config_id, build_fn in MC2_POLY_EXTRA:
        t0 = time.perf_counter()
        print(f"\n  [{config_id}]")
        mets = run_cv(config_id, build_fn, X, Y)
        elapsed = time.perf_counter() - t0
        for row in mets:
            mc2_poly_rows.append({
                "config_id": config_id,
                "output":    row["output"],
                "rmse":      row["rmse"],
                "nrmse":     row["nrmse"],
                "r2":        row["r2"],
                "mae":       row["mae"],
            })
        v_row = next((r for r in mets if r["output"] == "v_out_mean"), None)
        if v_row:
            print(f"    v_out_mean: RMSE={v_row['rmse']:.4f}  "
                  f"NRMSE={v_row['nrmse']*100:.3f}%  ({elapsed:.1f}s)")

    pd.DataFrame(mc2_poly_rows).to_csv(OUT3 / "mc2_poly.csv", index=False)
    print(f"\n  Saved → mc2_poly.csv  ({len(mc2_poly_rows)} rows)")

    # ── Phase 3 MC-2: GP variants ────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("Phase 3 MC-2 — GP variants")
    print(f"{'─'*65}")

    mc2_gp_rows: list[dict] = []

    # Reuse GP-SE from MC-1
    for row in mc1_summary_rows:
        if row["config_id"] == "GP-SE":
            mc2_gp_rows.append({
                "config_id": "GP-SE",
                "output":    row["output"],
                "rmse":      row["rmse"],
                "nrmse":     row["nrmse"],
                "r2":        row["r2"],
                "mae":       row["mae"],
            })

    # Run extra GP configs
    for config_id, build_fn in MC2_GP_EXTRA:
        t0 = time.perf_counter()
        is_gp = True
        print(f"\n  [{config_id}]")
        mets, yt_all, yp_all, ys_all = run_cv(
            config_id, build_fn, X, Y, return_preds=True, return_std=True)
        cv_preds[config_id] = (yt_all, yp_all)
        gp_stds[config_id]  = ys_all
        elapsed = time.perf_counter() - t0
        for row in mets:
            mc2_gp_rows.append({
                "config_id": config_id,
                "output":    row["output"],
                "rmse":      row["rmse"],
                "nrmse":     row["nrmse"],
                "r2":        row["r2"],
                "mae":       row["mae"],
            })
        v_row = next((r for r in mets if r["output"] == "v_out_mean"), None)
        if v_row:
            print(f"    v_out_mean: RMSE={v_row['rmse']:.4f}  "
                  f"NRMSE={v_row['nrmse']*100:.3f}%  ({elapsed:.1f}s)")

    pd.DataFrame(mc2_gp_rows).to_csv(OUT3 / "mc2_gp.csv", index=False)
    print(f"\n  Saved → mc2_gp.csv  ({len(mc2_gp_rows)} rows)")

    # ── Phase 3 MC-2: NN variants ────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("Phase 3 MC-2 — NN variants")
    print(f"{'─'*65}")

    mc2_nn_rows: list[dict] = []

    # Reuse NN-M from MC-1
    for row in mc1_summary_rows:
        if row["config_id"] == "NN-M":
            mc2_nn_rows.append({
                "config_id": "NN-M",
                "output":    row["output"],
                "rmse":      row["rmse"],
                "nrmse":     row["nrmse"],
                "r2":        row["r2"],
                "mae":       row["mae"],
            })

    # Run extra NN configs
    for config_id, build_fn in MC2_NN_EXTRA:
        t0 = time.perf_counter()
        print(f"\n  [{config_id}]")
        mets, yt_all, yp_all = run_cv(
            config_id, build_fn, X, Y, return_preds=True)
        cv_preds[config_id] = (yt_all, yp_all)
        elapsed = time.perf_counter() - t0
        for row in mets:
            mc2_nn_rows.append({
                "config_id": config_id,
                "output":    row["output"],
                "rmse":      row["rmse"],
                "nrmse":     row["nrmse"],
                "r2":        row["r2"],
                "mae":       row["mae"],
            })
        v_row = next((r for r in mets if r["output"] == "v_out_mean"), None)
        if v_row:
            print(f"    v_out_mean: RMSE={v_row['rmse']:.4f}  "
                  f"NRMSE={v_row['nrmse']*100:.3f}%  ({elapsed:.1f}s)")

    pd.DataFrame(mc2_nn_rows).to_csv(OUT3 / "mc2_nn.csv", index=False)
    print(f"\n  Saved → mc2_nn.csv  ({len(mc2_nn_rows)} rows)")

    # ── Phase 3 MC-2: RBF variants ───────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("Phase 3 MC-2 — RBF variants")
    print(f"{'─'*65}")

    mc2_rbf_rows: list[dict] = []
    for config_id, build_fn in MC2_RBF:
        t0 = time.perf_counter()
        print(f"\n  [{config_id}]")
        mets = run_cv(config_id, build_fn, X, Y)
        elapsed = time.perf_counter() - t0
        for row in mets:
            mc2_rbf_rows.append({
                "config_id": config_id,
                "output":    row["output"],
                "rmse":      row["rmse"],
                "nrmse":     row["nrmse"],
                "r2":        row["r2"],
                "mae":       row["mae"],
            })
        v_row = next((r for r in mets if r["output"] == "v_out_mean"), None)
        if v_row:
            print(f"    v_out_mean: RMSE={v_row['rmse']:.4f}  "
                  f"NRMSE={v_row['nrmse']*100:.3f}%  ({elapsed:.1f}s)")

    pd.DataFrame(mc2_rbf_rows).to_csv(OUT3 / "mc2_rbf.csv", index=False)
    print(f"\n  Saved → mc2_rbf.csv  ({len(mc2_rbf_rows)} rows)")

    # ── Phase 4: Calibration (GP-SE) ─────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("Phase 4 — Calibration analysis (GP-SE)")
    print(f"{'─'*65}")

    NOMINAL_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    cal_rows: list[dict] = []
    spe_rows: list[dict] = []

    # Use all GP models for Spearman table, but focus on GP-SE for coverage
    for cid in ["GP-SE", "GP-M52", "GP-M32", "GP-SE-noARD"]:
        if cid not in gp_stds:
            print(f"  [{cid}] no std data — skipping")
            continue
        yt_arr = cv_preds[cid][0]
        yp_arr = cv_preds[cid][1]
        ys_arr = gp_stds[cid]

        for oi, out in enumerate(OUTPUT_COLS):
            ok = ~(np.isnan(yt_arr[:,oi]) | np.isnan(yp_arr[:,oi]) | np.isnan(ys_arr[:,oi]))
            yt, yp, ys = yt_arr[ok,oi], yp_arr[ok,oi], ys_arr[ok,oi]
            if len(yt) < 5:
                continue
            err = np.abs(yt - yp)

            # Reliability: nominal vs empirical coverage
            for p in NOMINAL_LEVELS:
                z   = float(sp_norm.ppf((1.0 + p) / 2.0))
                cov = float(np.mean(err <= z * ys))
                cal_rows.append({"model": cid, "output": out,
                                 "nominal": p, "empirical": cov})

            # Spearman ρ(σ̂, |error|)
            valid = ys > 0
            if valid.sum() > 5:
                rho, pval = spearmanr(ys[valid], err[valid])
                spe_rows.append({"model": cid, "output": out,
                                 "spearman_rho": float(rho),
                                 "p_value":      float(pval)})

    pd.DataFrame(cal_rows).to_csv(OUT4 / "calibration.csv", index=False)
    pd.DataFrame(spe_rows).to_csv(OUT4 / "spearman.csv", index=False)
    print(f"  Saved → calibration.csv  ({len(cal_rows)} rows)")
    print(f"  Saved → spearman.csv     ({len(spe_rows)} rows)")

    # ── Phase 4: Learning curves ──────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("Phase 4 — Learning curves (N=20,40,60,80,100)")
    print(f"{'─'*65}")

    N_LC = [20, 40, 60, 80, 100]

    # Use fixed seed to create a permutation, then subsample from first N points
    # 5-fold CV at each N level for consistency with MC-1
    rng_lc = np.random.default_rng(42)
    perm   = rng_lc.permutation(n)
    X_perm = X[perm]
    Y_perm = Y[perm]

    lc_builders = [
        ("POLY-1", build_linear()),
        ("POLY-2", build_poly(2, "ols")),
        ("GP-SE",  build_gp("squar_exp", True, None, 5)),  # n_start=5 for speed
        ("NN-M",   build_nn([64, 64], "relu", 0.0, 0.0, 500, 32, 42)),
    ]

    lc_rows: list[dict] = []
    lc_data_cache: dict = {}   # for figure generation

    for config_id, build_fn in lc_builders:
        print(f"\n  [{config_id}]")
        lc_rows_model = []
        for N in N_LC:
            X_sub = X_perm[:N]
            Y_sub = Y_perm[:N]
            n_folds = min(5, N // 4) if N >= 8 else 2
            mets = run_cv(f"{config_id}-N{N}", build_fn, X_sub, Y_sub,
                          n_folds=n_folds, seed=42)
            for row in mets:
                lc_rows.append({
                    "config_id": config_id,
                    "N": N,
                    "output": row["output"],
                    "nrmse": row["nrmse"],
                    "rmse":  row["rmse"],
                    "r2":    row["r2"],
                })
                lc_rows_model.append({
                    "N": N, "output": row["output"], "nrmse": row["nrmse"]})

            v_row = next((r for r in mets if r["output"] == "v_out_mean"), None)
            if v_row:
                print(f"    N={N:3d}  v_out_mean NRMSE={v_row['nrmse']*100:.3f}%")
        lc_data_cache[config_id] = pd.DataFrame(lc_rows_model)

    pd.DataFrame(lc_rows).to_csv(OUT4 / "learning_curves.csv", index=False)
    print(f"\n  Saved → learning_curves.csv  ({len(lc_rows)} rows)")

    # ── Print results tables ───────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("MAIN RESULTS (for thesis tables):")
    print("─" * 65)
    header = f"{'Model':<12} {'Output':<15} {'RMSE':>10} {'MAE':>10} {'NRMSE(%)':>10} {'R2':>8}"
    print(header)
    print("─" * 65)
    for _, row in df_mc1.sort_values(["config_id", "output"]).iterrows():
        print(f"{row['config_id']:<12} {row['output']:<15} "
              f"{row['rmse']:>10.4f} {row['mae']:>10.4f} "
              f"{row['nrmse']*100:>10.3f} {row['r2']:>8.5f}")

    # Variant results — all MC-2 configs, v_out_mean only
    print(f"\n{'='*65}")
    print("VARIANT RESULTS (for Appendix D)  [v_out_mean NRMSE(%) from 5-fold CV]:")
    print("─" * 65)
    print(f"{'Config':<18} {'v_out_mean NRMSE(%)':>22}")
    print("─" * 40)
    all_mc2 = pd.concat([
        pd.DataFrame(mc2_poly_rows),
        pd.DataFrame(mc2_gp_rows),
        pd.DataFrame(mc2_nn_rows),
        pd.DataFrame(mc2_rbf_rows),
    ], ignore_index=True)
    vout_mc2 = all_mc2[all_mc2["output"] == "v_out_mean"].copy()
    # Deduplicate (POLY-1, POLY-2, NN-M, GP-SE appear in both MC-1 and MC-2)
    vout_mc2 = vout_mc2.drop_duplicates(subset=["config_id"], keep="first")
    for _, row in vout_mc2.iterrows():
        print(f"  {row['config_id']:<16} {row['nrmse']*100:>20.3f}%")

    # Calibration results
    print(f"\n{'='*65}")
    print("CALIBRATION (GP-SE, 5-fold CV):")
    print("─" * 55)
    print(f"{'Output':<16} {'95% Coverage':>16} {'Spearman ρ':>14}")
    print("─" * 55)
    cal_df  = pd.DataFrame(cal_rows)
    spe_df  = pd.DataFrame(spe_rows)
    gp_se_cal = cal_df[(cal_df["model"] == "GP-SE") & (cal_df["nominal"] == 0.95)]
    gp_se_spe = spe_df[spe_df["model"] == "GP-SE"]
    for out in OUTPUT_COLS:
        cov_row = gp_se_cal[gp_se_cal["output"] == out]
        spe_row = gp_se_spe[gp_se_spe["output"] == out]
        cov = cov_row["empirical"].values[0] if len(cov_row) else float("nan")
        rho = spe_row["spearman_rho"].values[0] if len(spe_row) else float("nan")
        print(f"  {out:<14} {cov:>16.4f} {rho:>14.4f}")

    # Learning curves results
    print(f"\n{'='*65}")
    print("LEARNING CURVES (NRMSE % by N):")
    print("─" * 75)
    lc_df = pd.DataFrame(lc_rows)
    header_lc = f"{'Model':<10} {'Output':<15} " + " ".join(f"{'N='+str(N):>8}" for N in N_LC)
    print(header_lc)
    print("─" * 75)
    for cid in ["POLY-1", "POLY-2", "GP-SE", "NN-M"]:
        for out in OUTPUT_COLS:
            sub = lc_df[(lc_df["config_id"] == cid) & (lc_df["output"] == out)].sort_values("N")
            vals = {int(r["N"]): r["nrmse"]*100 for _, r in sub.iterrows()}
            row_str = f"{cid:<10} {out:<15} " + " ".join(
                f"{vals.get(N, float('nan')):>8.3f}" for N in N_LC)
            print(row_str)

    elapsed_total = time.perf_counter() - t0_total
    print(f"\n{'='*65}")
    print(f"Pipeline complete in {elapsed_total/60:.1f} min")

    # ── Generate all 8 figures ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("Generating all 8 thesis figures …")
    print(f"{'='*65}")
    _generate_figures(X, Y, cv_preds, df_mc1, lc_data_cache, all_mc2)


# ─── Figure generation ───────────────────────────────────────────────────────

def _generate_figures(X, Y, cv_preds, df_mc1, lc_data_cache, all_mc2):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    LOCAL = ROOT / "thesis_figures"
    LOCAL.mkdir(parents=True, exist_ok=True)

    OUTPUTS    = OUTPUT_COLS
    OUT_LABELS = [r"$\bar{V}_{out}$", r"$\bar{I}_L$",
                  r"$\Delta V_{out}$", r"$\Delta I_L$", r"$\eta$"]

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
    }

    def save(fig, name):
        fig.savefig(LOCAL / name, dpi=300, bbox_inches="tight")
        print(f"  Saved {name}")

    def _r2_fn(yt, yp):
        ss = np.sum((yt-yp)**2); st = np.sum((yt-yt.mean())**2)
        return (1-ss/st) if st > 1e-12 else np.nan

    # ── Fig 1: mc1_parity_top3.png ───────────────────────────────────────────
    print("── Fig 1: mc1_parity_top3.png")
    oi_v = OUTPUTS.index("v_out_mean")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
        fig.suptitle(r"5-Fold CV Parity Plots — Output: $\bar{V}_{out}$",
                     fontsize=12, y=1.02)
        for ax, (cid, label, color) in zip(axes, FAMILIES):
            if cid not in cv_preds:
                ax.set_title(f"{label}\n(no data)")
                continue
            yt = cv_preds[cid][0][:, oi_v]
            yp = cv_preds[cid][1][:, oi_v]
            ok = ~(np.isnan(yt) | np.isnan(yp))
            yt, yp = yt[ok], yp[ok]
            rv  = float(np.sqrt(np.mean((yt-yp)**2)))
            r2v = _r2_fn(yt, yp)
            lo  = min(yt.min(), yp.min()) * 0.97
            hi  = max(yt.max(), yp.max()) * 1.03
            ax.plot([lo,hi],[lo,hi],"k--",lw=1.2,alpha=0.55)
            ax.scatter(yt, yp, s=22, alpha=0.75, color=color, linewidths=0)
            ax.set_title(f"{label}\nRMSE = {rv:.3g} V,  $R^2$ = {r2v:.4f}", fontsize=9)
            ax.set_xlabel(r"Actual $\bar{V}_{out}$ (V)", fontsize=9)
            if ax is axes[0]:
                ax.set_ylabel(r"Predicted $\bar{V}_{out}$ (V)", fontsize=9)
            ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
            ax.set_aspect("equal", "box")
        fig.tight_layout()
    save(fig, "mc1_parity_top3.png"); plt.close(fig)

    # ── Fig 2: mc1_rmse_bar_vout.png ─────────────────────────────────────────
    print("── Fig 2: mc1_rmse_bar_vout.png")
    CIDS   = [f[0] for f in FAMILIES]
    LABELS = [f[1] for f in FAMILIES]
    COLORS = [f[2] for f in FAMILIES]
    n_out_f = len(OUTPUTS); n_mod = len(CIDS)
    x = np.arange(n_out_f); w = 0.19
    offs = np.linspace(-(n_mod-1)/2, (n_mod-1)/2, n_mod) * w
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))
        for cid, lbl, col, off in zip(CIDS, LABELS, COLORS, offs):
            sub = df_mc1[df_mc1.config_id == cid]
            vals = []
            for out in OUTPUTS:
                r = sub[sub.output == out]
                vals.append(float(r["nrmse"].iloc[0]) * 100 if len(r) else float("nan"))
            bars = ax.bar(x+off, vals, width=w*0.9, label=lbl, color=col,
                          alpha=0.88, edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if np.isfinite(v) and v < 30:
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.08,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(x); ax.set_xticklabels(OUT_LABELS, fontsize=11)
        ax.set_ylabel("NRMSE (%)", fontsize=10)
        ax.set_title("5-Fold CV NRMSE by Output Variable and Model Family", fontsize=11)
        ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=9)
        ax.set_ylim(0, None); fig.tight_layout()
    save(fig, "mc1_rmse_bar_vout.png"); plt.close(fig)

    # ── Fig 3: ss1_learning_curves_all.png ───────────────────────────────────
    print("── Fig 3: ss1_learning_curves_all.png")
    N_LC = [20, 40, 60, 80, 100]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 5, figsize=(17, 4.2))
        fig.suptitle("Learning Curves — NRMSE (%) vs Training Set Size",
                     fontsize=12, y=1.02)
        for oi, (out, lbl, ax) in enumerate(zip(OUTPUTS, OUT_LABELS, axes)):
            for cid, _, color, family_lbl in [
                ("POLY-1", None, "#d6604d", "Linear Regression"),
                ("POLY-2", None, "#4393c3", "Polynomial"),
                ("GP-SE",  None, "#2ca25f", "Kriging"),
                ("NN-M",   None, "#7b2d8b", "Neural Network"),
            ]:
                if cid not in lc_data_cache:
                    continue
                sub = lc_data_cache[cid][lc_data_cache[cid]["output"] == out].sort_values("N")
                ax.plot(sub["N"], sub["nrmse"]*100, marker="o", ms=5,
                        lw=1.8, color=color, label=family_lbl)
            ax.set_title(lbl, fontsize=10)
            ax.set_xlabel("Training size $N$", fontsize=9)
            if oi == 0:
                ax.set_ylabel("NRMSE (%)", fontsize=9)
            ax.set_xticks(N_LC)
            ax.set_xticklabels([str(n) for n in N_LC], fontsize=7)
            ax.set_ylim(bottom=0)
        handles, lbls = axes[0].get_legend_handles_labels()
        fig.legend(handles, lbls, loc="lower center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, -0.07), fontsize=9)
        fig.tight_layout()
    save(fig, "ss1_learning_curves_all.png"); plt.close(fig)

    # ── Fig 4: calibration_and_spearman.png ──────────────────────────────────
    print("── Fig 4: calibration_and_spearman.png")
    cal = pd.read_csv(OUT4 / "calibration.csv")
    spe = pd.read_csv(OUT4 / "spearman.csv")
    gp_cal = cal[cal["model"] == "GP-SE"].sort_values(["output", "nominal"])
    OUT_C = ["#2166ac", "#d6604d", "#4dac26", "#7b2d8b", "#ff7f00"]
    MODEL_ORDER  = ["GP-SE", "GP-M52", "GP-M32", "GP-SE-noARD"]
    MODEL_LABELS = {
        "GP-SE":       "Kriging (SE, ARD)",
        "GP-M52":      r"Kriging (Matérn 5/2)",
        "GP-M32":      r"Kriging (Matérn 3/2)",
        "GP-SE-noARD": "Kriging (SE, isotropic)",
    }
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot([0,1],[0,1],"k--",lw=1.2,alpha=0.55,label="Perfect")
        for out, lbl, c in zip(OUTPUTS, OUT_LABELS, OUT_C):
            sub = gp_cal[gp_cal["output"] == out].sort_values("nominal")
            ax1.plot(sub["nominal"], sub["empirical"], marker="o", ms=5,
                     lw=1.6, color=c, label=lbl)
        ax1.set_xlabel("Nominal coverage", fontsize=10)
        ax1.set_ylabel("Empirical coverage", fontsize=10)
        ax1.set_title("Reliability Diagram — Kriging (GP-SE)", fontsize=10)
        ax1.legend(frameon=False, fontsize=8, loc="upper left")
        ax1.set_xlim(0,1); ax1.set_ylim(0,1)

        avail = [m for m in MODEL_ORDER if m in spe["model"].values]
        if avail:
            pivot = (spe[spe.model.isin(avail)]
                     .pivot_table(index="model", columns="output", values="spearman_rho")
                     .reindex(index=avail, columns=OUTPUTS))
            rho = pivot.values.astype(float)
            im  = ax2.imshow(rho, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax2, shrink=0.85, label=r"Spearman $\rho$")
            ax2.set_xticks(range(len(OUTPUTS)))
            ax2.set_xticklabels(OUT_LABELS, fontsize=9)
            ax2.set_yticks(range(len(avail)))
            ax2.set_yticklabels([MODEL_LABELS.get(m,m) for m in avail], fontsize=8)
            ax2.set_title(r"Spearman $\rho\,(\hat{\sigma},\,|error|)$", fontsize=10)
            for i in range(len(avail)):
                for j in range(len(OUTPUTS)):
                    v = rho[i, j]
                    if not np.isnan(v):
                        ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                                 fontsize=8, color="white" if v > 0.65 else "black")
        fig.tight_layout()
    save(fig, "calibration_and_spearman.png"); plt.close(fig)

    # ── Fig 5: R2_residual_plots.png ─────────────────────────────────────────
    print("── Fig 5: R2_residual_plots.png")
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(4, 5, figsize=(18, 14), squeeze=False)
        fig.suptitle("5-Fold CV Residuals  (Predicted − Actual)",
                     fontsize=13, y=1.005)
        for mi, (cid, label, color) in enumerate(FAMILIES):
            if cid not in cv_preds:
                continue
            yt_a, yp_a = cv_preds[cid]
            for oi, (out, lbl) in enumerate(zip(OUTPUTS, OUT_LABELS)):
                ax  = axes[mi][oi]
                yt  = yt_a[:,oi]; yp = yp_a[:,oi]
                ok  = ~(np.isnan(yt) | np.isnan(yp))
                res = yp[ok] - yt[ok]
                ax.scatter(yt[ok], res, s=16, alpha=0.65, color=color, linewidths=0)
                ax.axhline(0, color="k", lw=0.9, ls="--", alpha=0.6)
                rv  = float(np.sqrt(np.mean(res**2)))
                r2v = _r2_fn(yt[ok], yp[ok])
                ax.text(0.03, 0.97, f"RMSE={rv:.3g}\n$R^2$={r2v:.3f}",
                        transform=ax.transAxes, va="top", ha="left",
                        fontsize=6.5, color=color,
                        bbox=dict(facecolor="white", alpha=0.75,
                                  edgecolor="none", pad=1.5))
                if mi == 0: ax.set_title(lbl, fontsize=10)
                if oi == 0: ax.set_ylabel(f"{label}\nResidual", fontsize=8)
                if mi == 3: ax.set_xlabel("Actual", fontsize=8)
        fig.tight_layout(h_pad=1.2, w_pad=0.6)
    save(fig, "R2_residual_plots.png"); plt.close(fig)

    # ── Fig 6: F3_lhs_pair_plot.png ──────────────────────────────────────────
    print("── Fig 6: F3_lhs_pair_plot.png")
    doe = pd.read_csv(ROOT / "data/raw/doe_inputs.csv")
    doe.columns = ["D", "V_in", "R", "f_sw"]
    doe["f_sw"] = doe["f_sw"] / 1e3
    vlabels = {"D":"Duty cycle $D$","V_in":r"$V_{in}$ (V)",
               "R":r"$R$ ($\Omega$)","f_sw":r"$f_{sw}$ (kHz)"}
    cols = list(doe.columns); nv = len(cols)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nv, nv, figsize=(8.5, 8.5))
        fig.suptitle("100-Point LHS — Input Space Coverage",
                     fontsize=11, y=1.01)
        for row in range(nv):
            for col in range(nv):
                ax = axes[row][col]
                if row == col:
                    ax.hist(doe[cols[row]], bins=14, color="#a8cce0",
                            edgecolor="white", lw=0.4)
                    ax.set_yticks([])
                else:
                    ax.scatter(doe[cols[col]], doe[cols[row]], s=13, alpha=0.75,
                               color="#2166ac", linewidths=0)
                if row == nv-1: ax.set_xlabel(vlabels[cols[col]], fontsize=8)
                else: ax.set_xticklabels([])
                if col == 0 and row != col: ax.set_ylabel(vlabels[cols[row]], fontsize=8)
                elif col != 0: ax.set_yticklabels([])
                ax.tick_params(labelsize=7)
        fig.tight_layout()
    save(fig, "F3_lhs_pair_plot.png"); plt.close(fig)

    # ── Fig 7: F4_output_boxplots.png ────────────────────────────────────────
    print("── Fig 7: F4_output_boxplots.png")
    sim = pd.read_csv(ROOT / "data/raw/simulation_results_hifi.csv")
    present  = [c for c in OUTPUTS if c in sim.columns]
    df_out   = sim[present].dropna().reset_index(drop=True)
    nice = {
        "v_out_mean":  r"$\bar{V}_{out}$ (V)",
        "i_l_mean":    r"$\bar{I}_L$ (A)",
        "v_out_ripple":r"$\Delta V_{out}$ (V)",
        "i_l_ripple":  r"$\Delta I_L$ (A)",
        "efficiency":  r"$\eta$",
    }
    nf = len(present)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nf, nf, figsize=(11, 10))
        fig.suptitle("Output Variable Scatter Matrix — 100-Sample Training Dataset",
                     fontsize=11, y=1.01)
        for row in range(nf):
            for col in range(nf):
                ax = axes[row][col]
                cx = present[col]; cy = present[row]
                if row == col:
                    ax.hist(df_out[cx], bins=14, color="#f4b89a",
                            edgecolor="white", lw=0.4)
                    ax.set_yticks([])
                else:
                    ax.scatter(df_out[cx], df_out[cy], s=10, alpha=0.6,
                               color="#d6604d", linewidths=0)
                    ax.text(0.96, 0.04, f"r={df_out[cx].corr(df_out[cy]):.2f}",
                            transform=ax.transAxes, ha="right", va="bottom",
                            fontsize=6.5, color="#555")
                if row == nf-1: ax.set_xlabel(nice.get(cx, cx), fontsize=8)
                else: ax.set_xticklabels([])
                if col == 0 and row != col: ax.set_ylabel(nice.get(cy, cy), fontsize=8)
                elif col != 0: ax.set_yticklabels([])
                ax.tick_params(labelsize=7)
        fig.tight_layout()
    save(fig, "F4_output_boxplots.png"); plt.close(fig)

    # ── Fig 8: all_variant_comparison.png ────────────────────────────────────
    print("── Fig 8: all_variant_comparison.png")
    CONFIG_ORDER = [
        "POLY-1", "POLY-2", "POLY-3", "POLY-4",
        "POLY-3-Ridge", "POLY-4-Ridge",
        "GP-SE", "GP-M52", "GP-M32", "GP-SE-noARD", "GP-SE-nonug",
        "NN-S", "NN-M", "NN-L", "NN-M-tanh", "NN-M-wd",
        "RBF-MQ", "RBF-G", "RBF-TPS", "RBF-C",
    ]
    SELECTED = {"POLY-2", "GP-SE", "NN-M"}
    DISPLAY  = {
        "POLY-1":       "Linear (OLS)",
        "POLY-2":       "Polynomial deg-2  ★",
        "POLY-3":       "Polynomial deg-3 (OLS)",
        "POLY-4":       "Polynomial deg-4 (OLS)",
        "POLY-3-Ridge": "Polynomial deg-3 (Ridge)",
        "POLY-4-Ridge": "Polynomial deg-4 (Ridge)",
        "GP-SE":        "Kriging GP-SE  ★",
        "GP-M52":       r"Kriging GP-Matérn 5/2",
        "GP-M32":       r"Kriging GP-Matérn 3/2",
        "GP-SE-noARD":  "Kriging GP-SE (isotropic)",
        "GP-SE-nonug":  "Kriging GP-SE (no nugget)",
        "NN-S":         "Neural Net (small 2×32)",
        "NN-M":         "Neural Net (medium 2×64)  ★",
        "NN-L":         "Neural Net (large 3×128+drop)",
        "NN-M-tanh":    "Neural Net (tanh)",
        "NN-M-wd":      "Neural Net (weight decay)",
        "RBF-MQ":       "RBF (Multiquadric)",
        "RBF-G":        "RBF (Gaussian)",
        "RBF-TPS":      "RBF (Thin-plate spline)",
        "RBF-C":        "RBF (Cubic)",
    }

    # Combine all MC-1 and MC-2 for v_out_mean
    mc1_vout = df_mc1[df_mc1.output == "v_out_mean"][["config_id","nrmse"]].copy()
    mc2_vout = all_mc2[all_mc2.output == "v_out_mean"][["config_id","nrmse"]].copy()
    combined = pd.concat([mc1_vout, mc2_vout], ignore_index=True)
    combined = combined.drop_duplicates(subset=["config_id"], keep="first")
    combined = combined[combined.config_id.isin(CONFIG_ORDER)].copy()
    combined["nrmse_pct"] = combined.nrmse * 100
    combined = combined.sort_values("nrmse_pct")

    bar_colors = ["#2ca25f" if c in SELECTED else "#9ecae1"
                  for c in combined.config_id]
    bar_edges  = ["#006d2c" if c in SELECTED else "#4393c3"
                  for c in combined.config_id]
    bar_lws    = [1.8      if c in SELECTED else 0.5
                  for c in combined.config_id]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, max(6, len(combined)*0.45)))
        bars = ax.barh(range(len(combined)), combined.nrmse_pct,
                       color=bar_colors, edgecolor=bar_edges,
                       linewidth=bar_lws, height=0.72)
        for bar, (_, row) in zip(bars, combined.iterrows()):
            ax.text(row.nrmse_pct + 0.04, bar.get_y() + bar.get_height()/2,
                    f"{row.nrmse_pct:.3f}%", va="center", fontsize=7.5)
        ax.set_yticks(range(len(combined)))
        ax.set_yticklabels([DISPLAY.get(c, c) for c in combined.config_id],
                           fontsize=8.5)
        ax.set_xlabel(r"NRMSE (%) on $\bar{V}_{out}$  [5-fold CV]", fontsize=10)
        ax.set_title(r"All-Configuration Comparison — $\bar{V}_{out}$ NRMSE", fontsize=11)
        legend_elems = [
            mpatches.Patch(facecolor="#2ca25f", edgecolor="#006d2c", lw=1.5,
                           label="Selected best-in-family  (★)"),
            mpatches.Patch(facecolor="#9ecae1", edgecolor="#4393c3", lw=0.5,
                           label="Other configurations"),
        ]
        ax.legend(handles=legend_elems, frameon=False, loc="lower right", fontsize=9)
        fig.tight_layout()
    save(fig, "all_variant_comparison.png"); plt.close(fig)

    print("\nAll 8 figures saved to thesis_figures/.")


if __name__ == "__main__":
    main()
