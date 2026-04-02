"""
Radial Basis Function (RBF) surrogate models

Uses scipy.interpolate.RBFInterpolator with four kernel choices:
  multiquadric, gaussian, thin_plate_spline, cubic

References
----------
Hardy (1971) — multiquadric RBF
Buhmann (2003) — comprehensive treatment
Forrester et al. (2008) Ch.3 — practical engineering guide
"""

import numpy as np
import time
import warnings
from scipy.interpolate import RBFInterpolator


class RBFSurrogate:
    """
    RBF interpolant wrapper with a sklearn-compatible fit/predict API.

    Parameters
    ----------
    kernel : str
        RBF kernel type. One of:
        'multiquadric', 'inverse_multiquadric', 'gaussian',
        'linear', 'cubic', 'thin_plate_spline', 'quintic'
    smoothing : float
        Smoothing / regularisation parameter (0 = exact interpolation).
        Small positive values (1e-6) improve numerical conditioning.
    epsilon : float
        Shape parameter for kernels that use it (multiquadric, gaussian, …).
        If None, scipy uses a sensible default.
    """

    def __init__(
        self,
        kernel: str = "thin_plate_spline",
        smoothing: float = 1e-6,
        epsilon: float | None = None,
    ):
        self.kernel    = kernel
        self.smoothing = smoothing
        self.epsilon   = epsilon
        self._interp   = None
        self.train_time_s = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RBFSurrogate":
        """Fit the RBF interpolant to training data."""
        kwargs = {"kernel": self.kernel, "smoothing": self.smoothing}
        if self.epsilon is not None:
            kwargs["epsilon"] = self.epsilon
        t0 = time.perf_counter()
        self._interp = RBFInterpolator(X, y.ravel(), **kwargs)
        self.train_time_s = time.perf_counter() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._interp is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._interp(X)
