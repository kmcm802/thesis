"""
Averaged state-space DC-DC boost converter simulator.

Python implementation of the averaged steady-state model used for
surrogate-data generation.

States: x = [i_L, v_C] (inductor current, capacitor voltage)

Averaged state-space:
    dx/dt = A·x + B

    A = [−(R_L + D·R_sw)/L,    −(1−D)/L    ]
        [(1−D)/C,                −1/(R·C)    ]

    B = [(V_in − (1−D)·V_d)/L]
        [0                     ]

Fixed parameters (not function inputs):
    L    = 100e-6 H   (inductor)
    C    = 100e-6 F   (capacitor)
    R_L  = 0.1 Ω      (inductor series resistance)
    R_sw = 0.05 Ω     (switch on-resistance)
    V_d  = 0.7 V      (diode forward voltage)

ODE solver: scipy solve_ivp (RK45), rtol=1e-6, atol=1e-9, max_step=T_sw/20

CCM assumption: all operating points assumed in continuous conduction mode.
"""

import numpy as np
from scipy.integrate import solve_ivp
import time
from dataclasses import dataclass
from typing import Optional


# Fixed parameters
DEFAULT_PARASITICS = {
    "L": 100e-6,
    "C": 100e-6,
    "R_L": 0.1,
    "R_sw": 0.05,
    "V_d": 0.7,
}

# Design space bounds
DESIGN_BOUNDS = {
    "D": (0.2, 0.8),
    "V_in": (8.0, 24.0),
    "R": (5.0, 100.0),
    "f_sw": (20_000.0, 200_000.0),
}


@dataclass
class SimResult:
    """Result of a single boost converter simulation."""
    v_out_mean: float
    i_l_mean: float
    i_l_ripple: float
    v_out_ripple: float
    efficiency: float
    ccm_valid: bool  # True if operating in CCM
    sim_time: float  # wall-clock seconds


def boost_converter_sim(
    D: float,
    V_in: float,
    R: float,
    f_sw: float,
    L: float = 100e-6,
    C: float = 100e-6,
    R_L: float = 0.1,
    R_sw: float = 0.05,
    V_d: float = 0.7,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> SimResult:
    """
    Run averaged state-space ODE simulation of a DC-DC boost converter.

    Parameters
    ----------
    D : float
        Duty cycle (0.2 to 0.8).
    V_in : float
        Input voltage in V (8 to 24).
    R : float
        Load resistance in Ohm (5 to 100).
    f_sw : float
        Switching frequency in Hz (20e3 to 200e3).
    L, C, R_L, R_sw, V_d : float
        Fixed circuit parameters. Override for ideal-case testing.
    rtol, atol : float
        ODE solver tolerances.

    Returns
    -------
    SimResult
        Simulation outputs including CCM validity flag and timing.
    """
    t_start = time.perf_counter()

    T_sw = 1.0 / f_sw

    # State-space matrices
    A = np.array([
        [-(R_L + D * R_sw) / L, -(1 - D) / L],
        [(1 - D) / C, -1 / (R * C)],
    ])
    B_vec = np.array([(V_in - (1 - D) * V_d) / L, 0.0])

    # Integration time: enough for steady state
    tau = R * C
    t_end = max(100 * tau, 500 * T_sw)
    max_step = T_sw / 20

    # Initial conditions
    x0 = np.array([0.0, V_in])

    def ode_rhs(t, x):
        return A @ x + B_vec

    sol = solve_ivp(
        ode_rhs,
        [0, t_end],
        x0,
        method="RK45",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=False,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    # Extract steady-state: last 10 switching cycles
    ss_mask = sol.t >= (t_end - 10 * T_sw)
    if np.sum(ss_mask) < 10:
        raise RuntimeError("Not enough steady-state points")

    i_L_ss = sol.y[0, ss_mask]
    v_C_ss = sol.y[1, ss_mask]

    v_out_mean = np.mean(v_C_ss)
    i_l_mean = np.mean(i_L_ss)

    # Small-ripple expressions
    i_out = v_out_mean / R
    i_l_ripple = (V_in * D * T_sw) / L
    v_out_ripple = (i_out * D * T_sw) / C

    # Efficiency
    P_out = v_out_mean**2 / R
    P_in = V_in * i_l_mean
    if P_in <= 0:
        efficiency = np.nan
    else:
        efficiency = min(P_out / P_in, 1.0)

    # CCM check: I_L_min > 0
    ccm_valid = (i_l_mean - i_l_ripple / 2) > 0

    sim_time = time.perf_counter() - t_start

    return SimResult(
        v_out_mean=v_out_mean,
        i_l_mean=i_l_mean,
        i_l_ripple=i_l_ripple,
        v_out_ripple=v_out_ripple,
        efficiency=efficiency,
        ccm_valid=ccm_valid,
        sim_time=sim_time,
    )


def batch_simulate(
    samples: np.ndarray,
    **kwargs,
) -> list[SimResult]:
    """
    Run simulations for an array of input samples.

    Parameters
    ----------
    samples : np.ndarray, shape (N, 4)
        Columns: [D, V_in, R, f_sw].
    **kwargs
        Passed to boost_converter_sim (e.g. parasitic overrides, tolerances).

    Returns
    -------
    list[SimResult]
        One result per row.
    """
    results = []
    for i in range(samples.shape[0]):
        D, V_in, R, f_sw = samples[i]
        results.append(boost_converter_sim(D, V_in, R, f_sw, **kwargs))
    return results
