"""
Model registry for Phase 3 — all surrogate configurations in one place.

Each entry is a (config_id, family, build_fn) tuple where build_fn() returns
a fitted model given (X_train, y_train).  The build_fn interface is:

    model = build_fn(X_train, y_train)
    y_pred = model.predict(X_test)
"""

from __future__ import annotations

import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")


# ── helpers ───────────────────────────────────────────────────────────────────

def _smt_krg(corr: str, ARD: bool, nugget: float | None):
    """Return a factory that builds and trains an SMT KRG model."""
    def build(X, y):
        from smt.surrogate_models import KRG
        d = X.shape[1]
        nug = 1e-6 if nugget is None else nugget
        theta0 = [1e-2] * (d if ARD else 1)
        sm = KRG(
            theta0=theta0, corr=corr, poly="constant",
            nugget=nug, n_start=3, print_global=False,
        )
        t0 = time.perf_counter()
        sm.set_training_values(X, y.reshape(-1, 1))
        sm.train()
        train_time = time.perf_counter() - t0

        # Wrap to expose a .predict() method (SMT uses .predict_values())
        class _KRGWrapper:
            def __init__(self, model, tt):
                self._sm = model
                self._train_time = tt
            def predict(self, X_new):
                return self._sm.predict_values(X_new).ravel()
            def predict_std(self, X_new):
                return np.sqrt(np.maximum(
                    self._sm.predict_variances(X_new).ravel(), 0.0))

        return _KRGWrapper(sm, train_time)
    return build


def _poly(degree: int, regulariser: str = "ols", alpha: float = 1.0):
    """Return a factory for sklearn polynomial + linear model pipeline."""
    def build(X, y):
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        from sklearn.linear_model import LinearRegression, Ridge, Lasso
        from sklearn.pipeline import Pipeline
        reg = {"ols": LinearRegression(), "ridge": Ridge(alpha=alpha),
               "lasso": Lasso(alpha=alpha, max_iter=10000)}[regulariser]
        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("poly",  PolynomialFeatures(degree=degree, include_bias=False)),
            ("model", reg),
        ])
        t0 = time.perf_counter()
        pipe.fit(X, y.ravel())
        pipe._train_time = time.perf_counter() - t0
        return pipe
    return build


def _nn(hidden: list[int], activation: str = "relu",
        dropout: float = 0.0, weight_decay: float = 0.0):
    """Return a factory for a PyTorch MLP trained with early stopping."""
    def build(X, y):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        torch.manual_seed(42)
        np.random.seed(42)

        # ── z-score normalisation ──────────────────────────────────────────
        X_mean, X_std = X.mean(0), X.std(0) + 1e-8
        y_mean, y_std = y.mean(),  y.std()  + 1e-8
        Xn = (X - X_mean) / X_std
        yn = (y.ravel() - y_mean) / y_std

        # ── validation split (10 %) ────────────────────────────────────────
        n     = len(Xn)
        n_val = max(1, int(0.1 * n))
        idx   = np.random.permutation(n)
        X_tr, y_tr = Xn[idx[n_val:]], yn[idx[n_val:]]
        X_vl, y_vl = Xn[idx[:n_val]], yn[idx[:n_val]]

        Xt = torch.tensor(X_tr, dtype=torch.float32)
        yt = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
        Xv = torch.tensor(X_vl, dtype=torch.float32)
        yv = torch.tensor(y_vl, dtype=torch.float32).unsqueeze(1)

        loader = DataLoader(TensorDataset(Xt, yt), batch_size=16, shuffle=True)

        # ── build MLP ─────────────────────────────────────────────────────
        act_fn = {"relu": nn.ReLU(), "tanh": nn.Tanh()}[activation]
        layers: list[nn.Module] = []
        in_dim = X.shape[1]
        for h in hidden:
            layers += [nn.Linear(in_dim, h), act_fn]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        net = nn.Sequential(*layers)

        optimiser = torch.optim.Adam(net.parameters(), lr=1e-3,
                                     weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        best_val, patience_count, best_state = np.inf, 0, None
        t0 = time.perf_counter()
        for epoch in range(1000):
            net.train()
            for xb, yb in loader:
                optimiser.zero_grad()
                loss_fn(net(xb), yb).backward()
                optimiser.step()
            net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(net(Xv), yv).item())
            if val_loss < best_val - 1e-6:
                best_val, patience_count = val_loss, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= 50:
                    break
        if best_state:
            net.load_state_dict(best_state)
        train_time = time.perf_counter() - t0

        # Wrap with a predict method that un-normalises
        class _WrappedNet:
            def __init__(self, model, Xm, Xs, ym, ys, tt):
                self._net = model.eval()
                self._Xm, self._Xs = Xm, Xs
                self._ym, self._ys = ym, ys
                self._train_time    = tt

            def predict(self, X_new):
                import torch
                Xn = (X_new - self._Xm) / self._Xs
                t  = torch.tensor(Xn, dtype=torch.float32)
                with torch.no_grad():
                    out = self._net(t).numpy().ravel()
                return out * self._ys + self._ym

        return _WrappedNet(net, X_mean, X_std, y_mean, y_std, train_time)
    return build


def _rbf(kernel: str, smoothing: float = 1e-4, epsilon: float | None = None):
    """RBF interpolant with z-score normalisation of inputs."""
    def build(X, y):
        from scipy.interpolate import RBFInterpolator
        # Normalise inputs (RBF is sensitive to scale differences)
        X_mean, X_std = X.mean(0), X.std(0) + 1e-8
        Xn = (X - X_mean) / X_std

        kw: dict = {"kernel": kernel, "smoothing": smoothing}
        # epsilon required for multiquadric / gaussian; auto-set to mean NN distance
        if epsilon is not None:
            kw["epsilon"] = epsilon
        elif kernel in ("multiquadric", "inverse_multiquadric", "gaussian"):
            # Reasonable default: mean pairwise distance / sqrt(d)
            from scipy.spatial.distance import pdist
            kw["epsilon"] = float(np.mean(pdist(Xn[:min(50,len(Xn))])))

        t0 = time.perf_counter()
        interp = RBFInterpolator(Xn, y.ravel(), **kw)
        train_time = time.perf_counter() - t0

        class _RBFWrapper:
            def __init__(self, interp, Xm, Xs, tt):
                self._interp = interp
                self._Xm, self._Xs = Xm, Xs
                self._train_time = tt
            def predict(self, X_new):
                return self._interp((X_new - self._Xm) / self._Xs)

        return _RBFWrapper(interp, X_mean, X_std, train_time)
    return build


# ── model registry ────────────────────────────────────────────────────────────

# Each entry: (config_id, family, build_fn)
MODEL_REGISTRY: list[tuple[str, str, callable]] = [
    # ── Polynomial RSM ────────────────────────────────────────────────────────
    ("POLY-1",     "Polynomial", _poly(1, "ols")),
    ("POLY-2",     "Polynomial", _poly(2, "ols")),
    ("POLY-3",     "Polynomial", _poly(3, "ridge", alpha=1.0)),
    ("POLY-4",     "Polynomial", _poly(4, "ridge", alpha=1.0)),
    ("POLY-3-L",   "Polynomial", _poly(3, "lasso", alpha=1e-3)),

    # ── Kriging / GP ──────────────────────────────────────────────────────────
    ("GP-SE",      "Kriging", _smt_krg("squar_exp",  ARD=True,  nugget=None)),
    ("GP-M52",     "Kriging", _smt_krg("matern52",   ARD=True,  nugget=None)),
    ("GP-M32",     "Kriging", _smt_krg("matern32",   ARD=True,  nugget=None)),
    ("GP-SE-noARD","Kriging", _smt_krg("squar_exp",  ARD=False, nugget=None)),
    ("GP-SE-nonug","Kriging", _smt_krg("squar_exp",  ARD=True,  nugget=1e-10)),

    # ── Neural Networks ───────────────────────────────────────────────────────
    ("NN-S",       "Neural",  _nn([32, 16],        "relu", 0.0, 0.0)),
    ("NN-M",       "Neural",  _nn([64, 32],        "relu", 0.0, 0.0)),
    ("NN-L",       "Neural",  _nn([128, 64, 32],   "relu", 0.1, 0.0)),
    ("NN-M-tanh",  "Neural",  _nn([64, 32],        "tanh", 0.0, 0.0)),
    ("NN-M-wd",    "Neural",  _nn([64, 32],        "relu", 0.0, 1e-4)),

    # ── Radial Basis Functions ────────────────────────────────────────────────
    ("RBF-MQ",     "RBF",     _rbf("multiquadric")),
    ("RBF-G",      "RBF",     _rbf("gaussian",     smoothing=1e-4)),
    ("RBF-TPS",    "RBF",     _rbf("thin_plate_spline")),
    ("RBF-C",      "RBF",     _rbf("cubic")),
]

MODEL_IDS   = [m[0] for m in MODEL_REGISTRY]
MODEL_FAMILY = {m[0]: m[1] for m in MODEL_REGISTRY}
