from __future__ import annotations

"""
Skeleton for the Maxwell–Ogden + damage constitutive model, adapted from
`Trial_12_22/fun_IMR_damage_Maxwell_Kelvin_KM.m`.

This module is **purely constitutive**:
- It does NOT know about the IMR bubble interior PDEs
- It does NOT call ODE solvers
- It only maps (R, U, r0_grid, state, params) -> (S, Sdot, new_state)

Once this skeleton is filled in, the IMR core can call it in the same way
it currently calls the NHKV constitutive (`nhkv_sedot`).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class MaxwellDamageParams:
    """
    Material parameters for the Maxwell–Ogden + damage model.

    These names mirror the MATLAB code in
    `fun_IMR_damage_Maxwell_Kelvin_KM.m`:
      - GA1, GA2, alpha1, alpha2 : Ogden-type elastic branch A
      - GB1, GB2, beta1, beta2  : Maxwell branch B (non-equilibrium)
      - mu                      : viscosity (for Re)
      - damage_index            : index in the radial grid where damage front is located
      - xi_constant             : special flags for xi (e.g. 0, -0.5, or -1 for “computed”)
      - CaA1, CaA2, CaB1, CaB2  : Cauchy numbers (P_inf / G*)
    """

    GA1: float
    GA2: float
    alpha1: float
    alpha2: float
    GB1: float
    GB2: float
    beta1: float
    beta2: float
    mu: float
    damage_index: int

    # Damage / xi control (see MATLAB global H.xi_constant)
    xi_constant: float = -1.0  # -1: computed from lambda, 0~1 constant, -0.5: 0/1 step, etc.

    # These are convenient to pass explicitly, but could also be derived
    # from GA1,GA2,GB1,GB2 in the IMR core before calling the constitutive model.
    CaA1: float = 1.0
    CaA2: float = 1.0
    CaB1: float = 1.0
    CaB2: float = 1.0


@dataclass
class MaxwellDamageState:
    """
    Internal state of the Maxwell–Ogden + damage model at a given time.

    This is meant to capture the same information as the MATLAB globals:
      - lambda_nv: non-equilibrium stretch in branch B (length MT)
      - xi       : damage variable over the radial grid (0~1, length MT)
      - WA_m     : maximum elastic energy history (if needed)
      - S        : history of stress integral (for numerical Sdot if desired)
      - t        : time history (for numerical Sdot if desired)

    You can extend this dataclass as needed while porting the full model.
    """

    lambda_nv: NDArray[np.float64]          # shape (MT,)
    xi: NDArray[np.float64]                # shape (MT,)

    WA_m: Optional[NDArray[np.float64]] = None  # shape (MT,), maximum WA_0 ever reached
    S_history: Optional[NDArray[np.float64]] = None  # shape (K,), past S values
    t_history: Optional[NDArray[np.float64]] = None  # shape (K,), past times


def initialize_state(MT: int, params: MaxwellDamageParams) -> MaxwellDamageState:
    """
    Initialize the constitutive state at t = 0.

    MATLAB counterpart:
      lambda_nv0 = 1.00001*ones(1,MT);
      xi = ones(MT,1); (no damage before Rmax)
    """
    lambda_nv0 = np.full((MT,), 1.00001, dtype=float)
    xi0 = np.ones((MT,), dtype=float)
    return MaxwellDamageState(lambda_nv=lambda_nv0, xi=xi0)


def evaluate_maxwell_damage(
    R: float,
    U: float,
    Req: float,
    r0_star_list: NDArray[np.float64],
    state: MaxwellDamageState,
    params: MaxwellDamageParams,
    t: float,
    achieved_Rmax: bool,
) -> tuple[float, float, MaxwellDamageState]:
    """
    Single-step evaluation of the Maxwell–Ogden + damage constitutive response.

    Parameters
    ----------
    R : float
        Current bubble radius (dimensionless, consistent with IMR core).
    U : float
        Current bubble wall velocity (dimensionless).
    Req : float
        Dimensionless equilibrium radius (Req_nondim in MATLAB).
    r0_star_list : ndarray
        Radial grid outside the bubble (same as `r0_star_list` in MATLAB code),
        shape (MT,).
    state : MaxwellDamageState
        Current internal state (lambda_nv, xi, etc.).
    params : MaxwellDamageParams
        Material parameters (GA1,GA2,GB1,GB2,alphas,betas,mu,damage_index,...).
    t : float
        Current dimensionless time.
    achieved_Rmax : bool
        Whether the bubble has already reached its first maximum (Rmax).

    Returns
    -------
    S : float
        Stress integral over the material (dimensionless, to be used in KM).
    Sdot : float
        Time derivative of the stress integral (dimensionless).
    new_state : MaxwellDamageState
        Updated internal state after this step.

    Notes
    -----
    This function is **currently a skeleton**. The intention is to port the
    following blocks from `fun_IMR_damage_Maxwell_Kelvin_KM.m` into here:

      - Stretch computation:
            lambda_w   = R/Req;
            lambda_r0  = ( 1 + (1./r0_star_list.^3).*(lambda_w^3-1) ).^(1/3);
      - Maxwell branch (lambda_ne, lambda_nv, s_tt_nv, lambda_nv_dot)
      - Ogden elastic energy WA_0 and WA_m history
      - Damage variable xi (tensile check, damage_index, xi_constant)
      - Stress integral S via trapz over r, and Sdot from S history

    Once implemented, the IMR core can call this function instead of the
    NHKV-specific `nhkv_sedot`.
    """

    # --- TODO: implement Maxwell–Ogden + damage logic here ---
    # For now, we simply return zeros so that wiring can be tested
    # without affecting the current NHKV-based workflow.
    S = 0.0
    Sdot = 0.0
    new_state = state  # no evolution yet
    return S, Sdot, new_state


__all__ = [
    "MaxwellDamageParams",
    "MaxwellDamageState",
    "initialize_state",
    "evaluate_maxwell_damage",
]

