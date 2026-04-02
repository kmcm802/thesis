# Surrogate Modelling for a DC-DC Boost Converter

BSC Thesis — Killian McMahon — Trinity College Dublin, 2026

## What this is

Code and data for my thesis on comparing four surrogate model families (linear regression, polynomial regression, Kriging, neural network) for predicting steady-state boost converter performance. The converter operates in continuous conduction mode with four input parameters (duty cycle, input voltage, load resistance, switching frequency) and five outputs (mean voltage, mean current, voltage ripple, current ripple, efficiency).

## Structure

- `notebooks/` — Jupyter analysis notebooks numbered 01 through 09, intended to be run in order
- `data/raw/` — 100-point Latin hypercube training set and 25-point Simscape validation set
- `scripts/` — data generation and simulation scripts
- `src/` — MATLAB ODE simulator and Simscape switching model
- `outputs/` — pre-computed results organised by analysis stage, including all figures

## Requirements

- Python 3.11
- MATLAB R2025b with Simscape Electrical (for the ODE simulator and switching model only)

Install Python dependencies:

```
pip install -r requirements.txt
```

The pre-computed outputs are included so the notebooks can be inspected without re-running the full pipeline. The Simscape validation results are also pre-computed in `data/raw/simscape_validation_results.csv` for users without MATLAB.

## Reproducibility

All random seeds are fixed (LHS seed = 42, CV seed = 42). Re-running the notebooks in order will reproduce the thesis results.

## License

MIT
