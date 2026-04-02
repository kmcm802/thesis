"""
Sampling strategies for Design of Experiments (DoE)

Strategies available:
  random       — uniform random (Monte Carlo)
  lhs          — basic Latin Hypercube (random within strata)
  maximin_lhs  — maximin-optimised LHS (maximise minimum inter-point distance)
  sobol        — scrambled Sobol quasi-random sequence
"""

import numpy as np
import pandas as pd
from pyDOE3 import lhs
from scipy.stats.qmc import Sobol, LatinHypercube
from typing import Dict, Tuple


def generate_lhs_samples(
    param_ranges: Dict[str, Tuple[float, float]],
    n_samples: int,
    criterion: str = 'maximin',
    seed: int = None
) -> pd.DataFrame:
    """
    Generate Latin Hypercube Samples for the design space.

    Parameters
    ----------
    param_ranges : dict
        Dictionary mapping parameter names to (min, max) tuples
        Example: {'D': (0.3, 0.7), 'V_in': (10, 20)}
    n_samples : int
        Number of samples to generate
    criterion : str, optional
        LHS criterion: 'center', 'maximin', 'centermaximin', 'correlation'
        Default is 'maximin' for space-filling design
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame with n_samples rows and one column per parameter

    Examples
    --------
    >>> param_ranges = {
    ...     'D': (0.3, 0.7),
    ...     'V_in': (10, 20),
    ...     'R': (5, 50),
    ... }
    >>> samples = generate_lhs_samples(param_ranges, n_samples=100)
    >>> samples.shape
    (100, 3)
    """
    if seed is not None:
        np.random.seed(seed)

    param_names = list(param_ranges.keys())
    n_params = len(param_names)

    samples_normalized = lhs(n_params, samples=n_samples, criterion=criterion)

    samples = np.zeros_like(samples_normalized)
    for i, param_name in enumerate(param_names):
        min_val, max_val = param_ranges[param_name]
        samples[:, i] = min_val + (max_val - min_val) * samples_normalized[:, i]

    df = pd.DataFrame(samples, columns=param_names)

    return df


def generate_random_samples(
    param_ranges: Dict[str, Tuple[float, float]],
    n_samples: int,
    seed: int = None
) -> pd.DataFrame:
    """
    Generate random uniform samples for the design space.

    Parameters
    ----------
    param_ranges : dict
        Dictionary mapping parameter names to (min, max) tuples
    n_samples : int
        Number of samples to generate
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame with n_samples rows and one column per parameter
    """
    if seed is not None:
        np.random.seed(seed)

    param_names = list(param_ranges.keys())
    samples = {}

    for param_name in param_names:
        min_val, max_val = param_ranges[param_name]
        samples[param_name] = np.random.uniform(min_val, max_val, n_samples)

    return pd.DataFrame(samples)


def train_test_split_doe(
    samples: pd.DataFrame,
    test_fraction: float = 0.2,
    seed: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DoE samples into training and test sets.

    Parameters
    ----------
    samples : pd.DataFrame
        DoE samples to split
    test_fraction : float
        Fraction of samples to use for testing (default: 0.2)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    train_samples : pd.DataFrame
        Training samples
    test_samples : pd.DataFrame
        Test samples
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(samples)
    n_test = int(n_samples * test_fraction)

    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    train_samples = samples.iloc[train_indices].reset_index(drop=True)
    test_samples = samples.iloc[test_indices].reset_index(drop=True)

    return train_samples, test_samples


# ---------------------------------------------------------------------------
# Sobol quasi-random sequence
# ---------------------------------------------------------------------------

def generate_sobol_samples(
    param_ranges: Dict[str, Tuple[float, float]],
    n_samples: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a scrambled Sobol low-discrepancy sequence.

    Uses ``scipy.stats.qmc.Sobol`` (Joe & Kuo 2008 direction numbers).
    n_samples is rounded up to the next power of 2 if necessary for proper
    Sobol construction, then the first n_samples rows are returned.

    Parameters
    ----------
    param_ranges : dict
        Mapping of parameter names → (min, max).
    n_samples : int
        Number of points to return.
    seed : int
        Scrambling seed for reproducibility.

    Returns
    -------
    pd.DataFrame
    """
    param_names = list(param_ranges.keys())
    d = len(param_names)

    sampler = Sobol(d=d, scramble=True, seed=seed)
    # Generate next power-of-2 >= n_samples for proper Sobol sequences
    m = int(np.ceil(np.log2(max(n_samples, 1))))
    raw = sampler.random_base2(m=m)[:n_samples]

    df = pd.DataFrame(raw, columns=param_names)
    for name in param_names:
        lo, hi = param_ranges[name]
        df[name] = lo + (hi - lo) * df[name]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Unified sampling interface
# ---------------------------------------------------------------------------

STRATEGIES = ("random", "lhs", "maximin_lhs", "sobol")


def generate_samples(
    strategy: str,
    param_ranges: Dict[str, Tuple[float, float]],
    n_samples: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Unified sampling entry point.

    Parameters
    ----------
    strategy : str
        One of 'random', 'lhs', 'maximin_lhs', 'sobol'.
    param_ranges : dict
        Parameter bounds: {name: (min, max)}.
    n_samples : int
        Number of design points.
    seed : int
        Random seed (not used for 'sobol' deterministic base, but used for
        scrambling).

    Returns
    -------
    pd.DataFrame
    """
    if strategy == "random":
        return generate_random_samples(param_ranges, n_samples, seed=seed)
    elif strategy == "lhs":
        return generate_lhs_samples(param_ranges, n_samples, criterion=None, seed=seed)
    elif strategy == "maximin_lhs":
        return generate_lhs_samples(param_ranges, n_samples, criterion="maximin", seed=seed)
    elif strategy == "sobol":
        return generate_sobol_samples(param_ranges, n_samples, seed=seed)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {STRATEGIES}.")
