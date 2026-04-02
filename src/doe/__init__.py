"""
Design of Experiments (DoE) module

Provides sampling strategies for exploring the design space:
- Latin Hypercube Sampling (LHS)
- Random Sampling
- Sobol sampling
"""

from .sampling import (
    generate_lhs_samples,
    generate_random_samples,
    generate_sobol_samples,
    generate_samples,
    STRATEGIES,
)

__all__ = [
    "generate_lhs_samples",
    "generate_random_samples",
    "generate_sobol_samples",
    "generate_samples",
    "STRATEGIES",
]
