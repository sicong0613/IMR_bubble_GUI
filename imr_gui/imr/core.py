from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

from imr_gui.constitutive.nhkv_model import nhkv_sedot


@dataclass(frozen=True)
class NhkvInputs:
    # Fitting variables (same meaning as MATLAB)
    U0: float  # m/s
    G: float  # Pa
    mu: float  # Pa*s
    # Bubble equilibrium radius (MATLAB uses Req_exp as R0)
    Req: float  # m
    # Total simulation time span (seconds)
    tspan: float  # s

    # Numerical
    NT: int = 500  # match MATLAB default resolution
    rel_tol: float = 1e-8   # BDF needs ~10× tighter rtol than MATLAB ode23tb
    abs_tol: float = 1e-7   # to match MATLAB accuracy (esp. for GMOD lambda_nv)
    solver_method: str = "BDF"  # BDF ≈ MATLAB ode15s; faster than Radau

    # Far-field / material constants (kept consistent with fun_IMR_NHKV.m)
    P_inf: float = 101325.0
    T_inf: float = 298.15
    c_long: float = 1485.0
    rho: float = 998.0
    gamma: float = 5.6e-2
    alpha: float = 0.0  # strain-stiffening (0 => neo-Hookean)

    # Bubble-content / thermodynamic constants (loaded from nhkv.json)
    D0: float = 24.2e-6
    kappa: float = 1.4
    Ru: float = 8.3144598
    A_therm: float = 5.28e-5
    B_therm: float = 1.17e-2
    P_ref: float = 1.17e11
    T_ref: float = 5200.0
    M_vapor: float = 18.01528e-3
    M_air: float = 28.966e-3


@dataclass(frozen=True)
class NhkvOutputs:
    t_sim: NDArray[np.float64]  # s (shifted so t=0 at max R)
    R_sim: NDArray[np.float64]  # m
    U_sim: NDArray[np.float64]  # m/s
    P_sim: NDArray[np.float64]  # Pa
    t_sim_nondim: NDArray[np.float64]
    R_sim_nondim: NDArray[np.float64]
    Rmax_sim: float
    tc: float
    Uc: float
    n_damaged: int = 0  # number of damaged shells (GMOD only; 0 for NHKV)


def _build_jac_sparsity(NT: int):
    """Build a conservative Jacobian sparsity pattern for the LIC ODE system.

    State vector: [R, U, P, Se, Theta(0..NT-1), k(0..NT-1)]

    Without this, scipy's BDF evaluates the dense (4+2*NT)^2 Jacobian via
    finite differences — requiring N+1 RHS calls per update.  With the
    sparsity hint the solver uses graph-coloring and only needs ~30 calls.
    """
    N = 4 + 2 * NT
    S = lil_matrix((N, N), dtype=np.int8)

    BW = 3        # half-bandwidth for FD stencils (conservative)
    BNDRY = 4     # boundary-coupling width (DTheta[-1] uses 3-pt backward diff)

    # -- rows 0-3: R, U, P, Se  -------------------------------------------
    # These couple to each other and to boundary values of Theta / k
    for i in range(4):
        for j in range(4):
            S[i, j] = 1
        for jj in range(max(0, NT - BNDRY), NT):
            S[i, 4 + jj] = 1
            S[i, 4 + NT + jj] = 1

    # -- rows 4..4+NT-1  (Theta)  and  4+NT..4+2*NT-1  (k) ----------------
    for i in range(NT):
        rt = 4 + i
        rk = 4 + NT + i

        # depend on R, U, P, Se
        for j in range(4):
            S[rt, j] = 1
            S[rk, j] = 1

        # local FD stencil in Theta
        for di in range(-BW, BW + 1):
            jj = i + di
            if 0 <= jj < NT:
                S[rt, 4 + jj] = 1
                S[rk, 4 + jj] = 1

        # local FD stencil in k
        for di in range(-BW, BW + 1):
            jj = i + di
            if 0 <= jj < NT:
                S[rt, 4 + NT + jj] = 1
                S[rk, 4 + NT + jj] = 1

        # boundary coupling (pdot/Umix use DTheta[-1], Dk[-1])
        for jj in range(max(0, NT - BNDRY), NT):
            S[rt, 4 + jj] = 1
            S[rt, 4 + NT + jj] = 1
            S[rk, 4 + jj] = 1
            S[rk, 4 + NT + jj] = 1

    return S.tocsc()


def _simulate_lic_with_constitutive(
    inp: NhkvInputs,
    constitutive_fn: Callable[..., float],
    constitutive_params: Any | None = None,
) -> NhkvOutputs:
    """
    Generic LIC IMR solver that accepts a constitutive function to compute
    the elastic stress-integral rate (Sedot).

    The `constitutive_fn` is expected to have signature compatible with
    `nhkv_sedot(R, U, alpha, Ca, Req_nondim, **constitutive_params)`.
    """
    # --- constants for bubble contents (from nhkv.json via NhkvInputs) ---
    D0 = inp.D0
    kappa = inp.kappa
    Ru = inp.Ru
    Rv = Ru / inp.M_vapor
    Ra = Ru / inp.M_air
    A = inp.A_therm
    B = inp.B_therm
    P_ref = inp.P_ref
    T_ref = inp.T_ref

    # --- derived quantities (LIC) ---
    R0 = float(inp.Req)
    Rmax = R0
    Rc = Rmax
    Uc = np.sqrt(inp.P_inf / inp.rho)
    tc = Rmax / Uc

    Pv = P_ref * np.exp(-T_ref / inp.T_inf)
    K_inf = A * inp.T_inf + B

    C_star = inp.c_long / Uc
    We = inp.P_inf * Rc / (2 * inp.gamma)
    Ca = inp.P_inf / inp.G
    Re = inp.P_inf * Rc / (inp.mu * Uc)
    fom = D0 / (Uc * Rc)
    chi = inp.T_inf * K_inf / (inp.P_inf * Rc * Uc)
    A_star = A * inp.T_inf / K_inf
    B_star = B / K_inf
    Pv_star = Pv / inp.P_inf

    Req_nondim = R0 / Rmax  # == 1 for this LIC setup

    NT = int(inp.NT)
    if NT < 20:
        raise ValueError("NT must be >= 20")

    # --- initial conditions ---
    R0_star = 1.0
    U0_star = inp.U0 / Uc
    Theta0 = np.zeros((NT,), dtype=float)

    P0 = Pv + (inp.P_inf + 2 * inp.gamma / R0 - Pv) * ((R0 / Rmax) ** 3)
    P0_star = P0 / inp.P_inf

    alpha = float(inp.alpha)
    Se0 = (
        (3 * alpha - 1) * (5 - 4 * Req_nondim - Req_nondim**4) / (2 * Ca)
        + 2
        * alpha
        * (
            27 / 40
            + 1 / 8 * Req_nondim**8
            + 1 / 5 * Req_nondim**5
            + 1 * Req_nondim**2
            - 2 / Req_nondim
        )
        / Ca
    )

    k0_val = (1 + (Rv / Ra) * (P0_star / Pv_star - 1)) ** (-1)
    k0 = np.full((NT,), k0_val, dtype=float)

    # state: [R, U, P, Se, Theta(1..NT), k(1..NT)]
    x0 = np.concatenate(([R0_star, U0_star, P0_star, Se0], Theta0, k0)).astype(float)

    # grid inside bubble (fixed in y, same as MATLAB)
    deltaY = 1.0 / (NT - 1)
    yk = np.linspace(0.0, 1.0, NT, dtype=float)

    def rhs(_t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        R = x[0]
        U = x[1]
        P = x[2]
        Se = x[3]
        Theta = x[4 : 4 + NT]
        k = x[4 + NT : 4 + 2 * NT]

        # Dirichlet BC for k at wall
        k_wall = (1 + (Rv / Ra) * (P / Pv_star - 1)) ** (-1)
        k = k.copy()
        k[-1] = k_wall

        # mixture fields
        T = (A_star - 1 + np.sqrt(1 + 2 * A_star * Theta)) / A_star
        K_star = A_star * T + B_star
        Rmix = k * Rv + (1 - k) * Ra

        # derivatives (finite differences in spherical coords)
        DTheta = np.empty_like(Theta)
        DTheta[0] = 0.0
        DTheta[1:-1] = (Theta[2:] - Theta[:-2]) / (2 * deltaY)
        DTheta[-1] = (3 * Theta[-1] - 4 * Theta[-2] + Theta[-3]) / (2 * deltaY)

        DDTheta = np.empty_like(Theta)
        DDTheta[0] = 6 * (Theta[1] - Theta[0]) / (deltaY**2)
        DDTheta[1:-1] = (
            (Theta[2:] - 2 * Theta[1:-1] + Theta[:-2]) / (deltaY**2)
            + (2.0 / yk[1:-1]) * DTheta[1:-1]
        )
        DDTheta[-1] = (
            (2 * Theta[-1] - 5 * Theta[-2] + 4 * Theta[-3] - Theta[-4]) / (deltaY**2)
            + (2.0 / yk[-1]) * DTheta[-1]
        )

        Dk = np.empty_like(k)
        Dk[0] = 0.0
        Dk[1:-1] = (k[2:] - k[:-2]) / (2 * deltaY)
        Dk[-1] = (3 * k[-1] - 4 * k[-2] + k[-3]) / (2 * deltaY)

        DDk = np.empty_like(k)
        DDk[0] = 6 * (k[1] - k[0]) / (deltaY**2)
        DDk[1:-1] = (k[2:] - 2 * k[1:-1] + k[:-2]) / (deltaY**2) + (2.0 / yk[1:-1]) * Dk[1:-1]
        DDk[-1] = (
            (2 * k[-1] - 5 * k[-2] + 4 * k[-3] - k[-4]) / (deltaY**2)
            + (2.0 / yk[-1]) * Dk[-1]
        )

        # pressure evolution
        pdot = 3.0 / R * (
            -kappa * P * U
            + (kappa - 1) * chi * DTheta[-1] / R
            + kappa * P * fom * Rv * Dk[-1] / (R * Rmix[-1] * (1 - k[-1]))
        )

        # mixture velocity field
        Umix = ((kappa - 1) * chi / R * DTheta - R * yk * pdot / 3.0) / (kappa * P) + fom / R * (Rv - Ra) / Rmix * Dk

        # Theta evolution
        Theta_prime = (pdot + DDTheta * chi / (R**2)) * (K_star * T / P * (kappa - 1) / kappa) - DTheta * (Umix - yk * U) / R + fom / (R**2) * (Rv - Ra) / Rmix * Dk * DTheta
        Theta_prime[-1] = 0.0

        # k evolution
        k_prime = (
            fom / (R**2) * (DDk + Dk * (-( (Rv - Ra) / Rmix) * Dk - DTheta / np.sqrt(1 + 2 * A_star * Theta) / T))
            - (Umix - U * yk) / R * Dk
        )
        k_prime[-1] = 0.0

        # elastic stress integral evolution (LIC) via constitutive module
        kwargs = constitutive_params or {}
        Sedot = constitutive_fn(
            R=R,
            U=U,
            alpha=alpha,
            Ca=Ca,
            Req_nondim=Req_nondim,
            **kwargs,
        )

        # external pressure (LIC)
        Pext = 0.0
        Pextdot = 0.0

        # Keller-Miksis
        rdot = U
        udot = (
            (1 + U / C_star) * (P - 1 / (We * R) + Se - 4 * U / (Re * R) - 1 - Pext)
            + R / C_star * (pdot + U / (We * R**2) + Sedot + 4 * U**2 / (Re * R**2) - Pextdot)
            - (3 / 2) * (1 - U / (3 * C_star)) * U**2
        ) / ((1 - U / C_star) * R + 4 / (C_star * Re))

        out = np.empty_like(x)
        out[0] = rdot
        out[1] = udot
        out[2] = pdot
        out[3] = Sedot
        out[4 : 4 + NT] = Theta_prime
        out[4 + NT : 4 + 2 * NT] = k_prime
        return out

    t_end_star = float(inp.tspan / tc)

    method = inp.solver_method
    jac_sparsity = _build_jac_sparsity(NT)
    solver_kw: dict = dict(
        rtol=inp.rel_tol,
        atol=inp.abs_tol,
    )
    if method in ("BDF", "Radau"):
        solver_kw["jac_sparsity"] = jac_sparsity

    sol = solve_ivp(
        rhs,
        t_span=(0.0, t_end_star),
        y0=x0,
        method=method,
        **solver_kw,
    )
    if not sol.success or sol.y.shape[1] < 3:
        raise RuntimeError(f"NHKV solver failed: {sol.message}")

    t_nondim = sol.t
    X = sol.y.T
    R_nondim = X[:, 0]
    U_nondim = X[:, 1]
    P_nondim = X[:, 2]

    t_sim = t_nondim * tc
    R_sim = R_nondim * Rc
    U_sim = U_nondim * (Rc / tc)
    P_sim = P_nondim * inp.P_inf

    # Shift time to Rmax point, normalize by simulated Rmax (matches MATLAB)
    Rmax_sim = float(np.max(R_sim))
    Rmax_ind = int(np.argmax(R_sim))
    t_sim_shifted = t_sim - t_sim[Rmax_ind]

    R_sim_nondim = R_sim / Rmax_sim
    t_sim_nondim = t_sim_shifted * Uc / Rmax_sim

    return NhkvOutputs(
        t_sim=t_sim_shifted.astype(float),
        R_sim=R_sim.astype(float),
        U_sim=U_sim.astype(float),
        P_sim=P_sim.astype(float),
        t_sim_nondim=t_sim_nondim.astype(float),
        R_sim_nondim=R_sim_nondim.astype(float),
        Rmax_sim=Rmax_sim,
        tc=float(tc),
        Uc=float(Uc),
    )


def simulate_nhkv_lic(inp: NhkvInputs) -> NhkvOutputs:
    """
    Port of MATLAB `fun_IMR_NHKV.m` for LIC mode using the NHKV constitutive
    law. This is a thin wrapper around the generic `_simulate_lic_with_constitutive`.
    """
    return _simulate_lic_with_constitutive(inp, nhkv_sedot)

