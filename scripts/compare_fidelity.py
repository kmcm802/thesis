"""
compare_fidelity.py
-------------------
Three-way fidelity comparison:
  Averaged-model CSV (100 pts)  vs  Simscape (25 pts)  vs  Kriging surrogate trained on the averaged model.

Comparisons at the 25 Simscape points:
  1. ODE vs Simscape       — modelling / fidelity gap
  2. Surrogate vs ODE      — approximation error
  3. Surrogate vs Simscape — total end-to-end error

Outputs
-------
  outputs/fidelity_gap_summary.csv   per-output RMSE / NRMSE% / MAE for each comparison
  outputs/fidelity_gap_detail.csv    per-point predictions from all three sources
"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from smt.surrogate_models import KRG

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT   = pathlib.Path(__file__).resolve().parents[1]
ODE_PATH = ROOT / "data" / "raw" / "simulation_results_hifi.csv"
SIM_PATH = ROOT / "data" / "raw" / "simscape_validation_results.csv"
OUT_DIR  = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

INPUTS  = ["D", "V_in", "R", "f_sw"]
OUTPUTS = ["v_out_mean", "i_l_mean", "v_out_ripple", "i_l_ripple", "efficiency"]
ODE_SPEED_S_PER_POINT = 0.486   # from the 100-point campaign

# ── load data ─────────────────────────────────────────────────────────────────
ode_full = pd.read_csv(ODE_PATH)          # 100 rows
simscape  = pd.read_csv(SIM_PATH)         # 25 rows, includes wall_time_s

assert len(ode_full) == 100, f"Expected 100 ODE rows, got {len(ode_full)}"
assert len(simscape)  == 25, f"Expected 25 Simscape rows, got {len(simscape)}"

# ── align: every 4th ODE row matches Simscape ────────────────────────────────
ode_sub = ode_full.iloc[::4].reset_index(drop=True)          # rows 0,4,8,...,96

max_input_diff = (ode_sub[INPUTS] - simscape[INPUTS]).abs().max().max()
assert max_input_diff == 0.0, (
    f"Input mismatch between ODE subgrid and Simscape: max diff = {max_input_diff}"
)
print(f"Input alignment verified — max absolute difference: {max_input_diff}")

# ── train Kriging on FULL 100-point ODE dataset ───────────────────────────────
X_all  = ode_full[INPUTS].values.astype(float)
X_eval = simscape[INPUTS].values.astype(float)

scaler_X = StandardScaler()
X_all_s  = scaler_X.fit_transform(X_all)
X_eval_s = scaler_X.transform(X_eval)

surr_preds = {}
print("\nTraining Kriging surrogates …")
for out in OUTPUTS:
    y_all = ode_full[out].values.reshape(-1, 1).astype(float)

    scaler_y = StandardScaler()
    y_all_s  = scaler_y.fit_transform(y_all)

    krg = KRG(
        corr="squar_exp",
        poly="constant",
        n_start=10,
        print_global=False,
    )
    krg.set_training_values(X_all_s, y_all_s)
    krg.train()

    y_pred_s = krg.predict_values(X_eval_s)            # scaled
    y_pred   = scaler_y.inverse_transform(y_pred_s)    # original units
    surr_preds[out] = y_pred.ravel()
    print(f"  {out}: done")

# ── collect predictions at the 25 evaluation points ──────────────────────────
ode_at_eval = ode_sub[OUTPUTS].values     # 25 × 5, ODE truth at eval points
sim_cols = [f"simscape_{name}" for name in OUTPUTS]
if not set(sim_cols).issubset(simscape.columns):
    sim_cols = OUTPUTS
sim_vals = simscape[sim_cols].values

# ── error helper ─────────────────────────────────────────────────────────────
def metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    """RMSE, NRMSE (normalised by ref range, %), MAE."""
    rmse  = np.sqrt(np.mean((pred - ref) ** 2))
    r     = ref.max() - ref.min()
    nrmse = (rmse / r * 100) if r > 0 else np.nan
    mae   = np.mean(np.abs(pred - ref))
    return {"RMSE": rmse, "NRMSE%": nrmse, "MAE": mae}

# ── build summary table ───────────────────────────────────────────────────────
rows = []
for i, out in enumerate(OUTPUTS):
    pred_ode  = ode_at_eval[:, i]
    pred_sim  = sim_vals[:, i]
    pred_surr = surr_preds[out]

    m_fidelity = metrics(pred_ode,  pred_sim)   # ODE vs Simscape  (ref = Simscape)
    m_approx   = metrics(pred_surr, pred_ode)   # Surr vs ODE      (ref = ODE)
    m_total    = metrics(pred_surr, pred_sim)   # Surr vs Simscape (ref = Simscape)

    for comp_label, m in [
        ("ODE_vs_Simscape",      m_fidelity),
        ("Surrogate_vs_ODE",     m_approx),
        ("Surrogate_vs_Simscape",m_total),
    ]:
        rows.append({
            "output":     out,
            "comparison": comp_label,
            **m,
        })

summary_df = pd.DataFrame(rows)

# ── build per-point detail table ──────────────────────────────────────────────
detail_rows = []
for pt in range(25):
    row = {"point_index": pt * 4}   # original ODE row index
    for k in INPUTS:
        row[k] = simscape[INPUTS].iloc[pt][k]
    for i, out in enumerate(OUTPUTS):
        row[f"ode_{out}"]      = ode_at_eval[pt, i]
        row[f"simscape_{out}"] = sim_vals[pt, i]
        row[f"surr_{out}"]     = surr_preds[out][pt]
    detail_rows.append(row)

detail_df = pd.DataFrame(detail_rows)

# ── save ─────────────────────────────────────────────────────────────────────
summary_df.to_csv(OUT_DIR / "fidelity_gap_summary.csv", index=False)
detail_df.to_csv(OUT_DIR  / "fidelity_gap_detail.csv",  index=False)

# ── wall-time comparison ──────────────────────────────────────────────────────
sim_mean_wt   = simscape["wall_time_s"].mean()
sim_std_wt    = simscape["wall_time_s"].std()
speedup       = sim_mean_wt / ODE_SPEED_S_PER_POINT

# ── formatted stdout summary ─────────────────────────────────────────────────
SEP  = "─" * 80
SEP2 = "═" * 80

print(f"\n{SEP2}")
print("  FIDELITY GAP ANALYSIS — THREE-WAY COMPARISON (n = 25 Simscape points)")
print(SEP2)

comparisons = [
    ("ODE_vs_Simscape",       "ODE vs Simscape       (modelling / fidelity gap)"),
    ("Surrogate_vs_ODE",      "Surrogate vs ODE      (approximation error)      "),
    ("Surrogate_vs_Simscape", "Surrogate vs Simscape (total end-to-end error)   "),
]

for comp_key, comp_label in comparisons:
    sub = summary_df[summary_df["comparison"] == comp_key].set_index("output")
    print(f"\n  {comp_label}")
    print(f"  {SEP}")
    header = f"  {'Output':<18}  {'RMSE':>14}  {'NRMSE%':>10}  {'MAE':>14}"
    print(header)
    print(f"  {'-'*18}  {'-'*14}  {'-'*10}  {'-'*14}")
    for out in OUTPUTS:
        r = sub.loc[out]
        print(f"  {out:<18}  {r['RMSE']:>14.6f}  {r['NRMSE%']:>9.3f}%  {r['MAE']:>14.6f}")

print(f"\n{SEP2}")
print("  WALL-TIME COMPARISON")
print(SEP2)
print(f"  Simscape mean wall time : {sim_mean_wt:.3f} s/point  (std {sim_std_wt:.3f} s,  n=25)")
print(f"  ODE      wall time      : {ODE_SPEED_S_PER_POINT:.3f} s/point  (from 100-point campaign)")
print(f"  Simscape / ODE speedup  : {speedup:.1f}×  (ODE is {speedup:.1f}× faster)")
print(f"\n  Outputs written to:")
print(f"    {OUT_DIR / 'fidelity_gap_summary.csv'}")
print(f"    {OUT_DIR / 'fidelity_gap_detail.csv'}")
print(SEP2)
