"""Generalized Maxwell-Ogden with Damage (GMOD) LIC solver.

Faithful port of MATLAB ``fun_IMR_damage_Maxwell_Kelvin_KM.m``.
The bubble-interior physics (temperature, vapour-concentration, pressure)
are identical to the NHKV solver; the material model outside the bubble
uses two equilibrium Ogden branches (A), two non-equilibrium Maxwell
branches (B) with viscous dashpot, and a binary damage model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

from imr_gui.imr.core import NhkvOutputs


@dataclass(frozen=True)
class GMODInputs:
    """All inputs for the GMOD (Generalized Maxwell-Ogden + Damage) LIC solver."""

    # --- fitting variables (order matches MATLAB var2fit) ---
    U0: float           # m/s
    GA1: float          # Pa
    GA2: float          # Pa
    alpha1: float       # dimensionless
    alpha2: float       # dimensionless
    GB1: float          # Pa
    GB2: float          # Pa
    beta1: float        # dimensionless
    beta2: float        # dimensionless
    mu: float           # Pa·s
    damage_index: float # 0..MT (rounded to int inside solver)

    # --- geometry / experimental ---
    Req: float          # m
    tspan: float        # s

    # --- numerical ---
    NT: int = 500
    MT: int = 200
    rel_tol: float = 1e-8   # BDF needs ~10× tighter rtol than MATLAB ode23tb
    abs_tol: float = 1e-7   # to resolve lambda_nv (Maxwell branch) dynamics
    solver_method: str = "BDF"  # BDF ≈ MATLAB ode15s; faster than Radau for this problem

    # --- far-field ---
    P_inf: float = 101325.0
    T_inf: float = 298.15
    c_long: float = 1485.0
    rho: float = 998.0
    gamma: float = 5.6e-2

    # --- damage behaviour ---
    xi_constant: float = -1.0   # >=0 → uniform xi; -0.5 → always binary; -1 → auto

    # --- bubble-content constants ---
    D0: float = 24.2e-6
    kappa: float = 1.4
    Ru: float = 8.3144598
    A_therm: float = 5.28e-5
    B_therm: float = 1.17e-2
    P_ref: float = 1.17e11
    T_ref: float = 5200.0
    M_vapor: float = 18.01528e-3
    M_air: float = 28.966e-3

    # --- solver hint ---
    first_step: float = 0.0   # >0 → pass to solve_ivp; 0 → let scipy choose


# -----------------------------------------------------------------------
# internal helpers
# -----------------------------------------------------------------------

class _SolverHistory:
    """Mutable tracker for stress history and damage state (captured in
    the RHS closure, analogous to MATLAB's global ``H``).

    ``last_t`` / ``last_S`` are only updated when the simulation time
    genuinely advances forward.  The ODE solver also calls the RHS at
    the *same* time point with slightly perturbed states when it builds
    the Jacobian via finite differences; those calls must not overwrite
    the stored (S, t) pair or the next real Sdot estimate will be wrong.
    """
    __slots__ = (
        "initialized", "last_t", "last_S", "Sdot_current",
        "achieve_rmax", "WA_m",
        "achieved_lambda_A", "achieved_lambda_B",
    )

    def __init__(self) -> None:
        self.initialized: bool = False
        self.last_t: float = 0.0
        self.last_S: float = 0.0
        self.Sdot_current: float = 0.0
        self.achieve_rmax: bool = False
        self.WA_m: NDArray | None = None
        self.achieved_lambda_A: NDArray | None = None
        self.achieved_lambda_B: NDArray | None = None


def _build_jac_sparsity_gmod(NT: int, MT: int):
    """Jacobian sparsity for state ``[R, U, P, Theta(NT), k(NT), λ_nv(MT)]``."""
    N = 3 + 2 * NT + MT
    S = lil_matrix((N, N), dtype=np.int8)

    BW = 3
    BNDRY = 4
    THR = 3            # Theta block start
    KR = 3 + NT        # k block start
    LR = 3 + 2 * NT   # lambda_nv block start

    # Row 0 (rdot = U)
    S[0, 1] = 1

    # Row 1 (udot) — depends on R, U, P, boundary Θ/k, ALL λ_nv (via S)
    for j in range(3):
        S[1, j] = 1
    for jj in range(max(0, NT - BNDRY), NT):
        S[1, THR + jj] = 1
        S[1, KR + jj] = 1
    for j in range(MT):
        S[1, LR + j] = 1

    # Row 2 (pdot) — depends on R, U, P, boundary Θ/k
    for j in range(3):
        S[2, j] = 1
    for jj in range(max(0, NT - BNDRY), NT):
        S[2, THR + jj] = 1
        S[2, KR + jj] = 1

    # Theta and k rows
    for i in range(NT):
        rt = THR + i
        rk = KR + i
        for j in range(3):
            S[rt, j] = 1
            S[rk, j] = 1
        for di in range(-BW, BW + 1):
            jj = i + di
            if 0 <= jj < NT:
                S[rt, THR + jj] = 1
                S[rk, THR + jj] = 1
                S[rt, KR + jj] = 1
                S[rk, KR + jj] = 1
        for jj in range(max(0, NT - BNDRY), NT):
            S[rt, THR + jj] = 1
            S[rt, KR + jj] = 1
            S[rk, THR + jj] = 1
            S[rk, KR + jj] = 1

    # lambda_nv rows: each depends only on R and lambda_nv[i]
    for i in range(MT):
        row = LR + i
        S[row, 0] = 1
        S[row, LR + i] = 1

    return S.tocsc()


# -----------------------------------------------------------------------
# public API
# -----------------------------------------------------------------------

def simulate_gmod_lic(inp: GMODInputs) -> NhkvOutputs:
    """Run the full GMOD (Generalized Maxwell-Ogden + Damage) LIC simulation.

    Returns the same ``NhkvOutputs`` container used by the NHKV solver so
    the GUI layer can treat both models uniformly.
    """

    # --- bubble-content constants ---
    D0 = inp.D0
    kappa = inp.kappa
    Ru = inp.Ru
    Rv = Ru / inp.M_vapor
    Ra = Ru / inp.M_air
    A = inp.A_therm
    B = inp.B_therm

    # --- derived LIC scales ---
    R0 = float(inp.Req)
    Rc = R0
    Uc = np.sqrt(inp.P_inf / inp.rho)
    tc = R0 / Uc

    Pv = inp.P_ref * np.exp(-inp.T_ref / inp.T_inf)
    K_inf = A * inp.T_inf + B

    C_star = inp.c_long / Uc
    We = inp.P_inf * Rc / (2 * inp.gamma)
    CaA1 = inp.P_inf / inp.GA1
    CaA2 = inp.P_inf / inp.GA2
    CaB1 = inp.P_inf / inp.GB1
    CaB2 = inp.P_inf / inp.GB2
    Re = inp.P_inf * Rc / (inp.mu * Uc)
    fom = D0 / (Uc * Rc)
    chi = inp.T_inf * K_inf / (inp.P_inf * Rc * Uc)
    A_star = A * inp.T_inf / K_inf
    B_star = B / K_inf
    Pv_star = Pv / inp.P_inf

    _EPS_ALPHA = 1e-12
    alpha1 = max(float(inp.alpha1), _EPS_ALPHA)
    alpha2 = max(float(inp.alpha2), _EPS_ALPHA)
    beta1 = max(float(inp.beta1), _EPS_ALPHA)
    beta2 = max(float(inp.beta2), _EPS_ALPHA)
    xi_constant = float(inp.xi_constant)
    damage_idx = max(0, min(int(round(inp.damage_index)), inp.MT))

    Req_nondim = 1.0   # R0/Rmax = 1 for LIC

    NT = int(inp.NT)
    MT = int(inp.MT)
    if NT < 20:
        raise ValueError("NT must be >= 20")
    if MT < 5:
        raise ValueError("MT must be >= 5")

    # --- grids ---
    deltaY = 1.0 / (NT - 1)
    yk = np.linspace(0.0, 1.0, NT, dtype=float)

    temp_arr = np.linspace(0.0, 3.0, MT)
    r0_star = 10.0 ** temp_arr           # reference radial positions
    r0_star3 = r0_star ** 3

    # --- initial conditions ---
    R0_star = 1.0
    U0_star = inp.U0 / Uc
    Theta0 = np.zeros(NT, dtype=float)

    P0 = Pv + (inp.P_inf + 2.0 * inp.gamma / R0 - Pv)
    P0_star = P0 / inp.P_inf

    k0_val = (1 + (Rv / Ra) * (P0_star / Pv_star - 1)) ** (-1)
    k0 = np.full(NT, k0_val, dtype=float)

    lambda_nv0 = np.full(MT, 1.00001, dtype=float)

    # state: [R, U, P, Theta(NT), k(NT), lambda_nv(MT)]
    x0 = np.concatenate(
        ([R0_star, U0_star, P0_star], Theta0, k0, lambda_nv0)
    ).astype(float)

    tracker = _SolverHistory()
    eps_val = float(np.finfo(float).eps)

    use_CaA2 = CaA2 < 1e10
    use_CaB2 = CaB2 < 1e10

    # ------------------------------------------------------------------
    def rhs(t_star: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        R = x[0]
        U = x[1]
        P = x[2]
        Theta = x[3 : 3 + NT]
        k_arr = x[3 + NT : 3 + 2 * NT].copy()
        lnv = x[3 + 2 * NT : 3 + 2 * NT + MT]

        # -- BC for k at bubble wall --
        k_arr[-1] = (1 + (Rv / Ra) * (P / Pv_star - 1)) ** (-1)

        # -- mixture fields inside bubble --
        T_f = (A_star - 1 + np.sqrt(1 + 2 * A_star * Theta)) / A_star
        K_f = A_star * T_f + B_star
        Rmix = k_arr * Rv + (1 - k_arr) * Ra

        # -- spatial derivatives (FD, spherical coords) --
        DTheta = np.empty_like(Theta)
        DTheta[0] = 0.0
        DTheta[1:-1] = (Theta[2:] - Theta[:-2]) / (2 * deltaY)
        DTheta[-1] = (3 * Theta[-1] - 4 * Theta[-2] + Theta[-3]) / (2 * deltaY)

        DDTheta = np.empty_like(Theta)
        DDTheta[0] = 6 * (Theta[1] - Theta[0]) / (deltaY ** 2)
        DDTheta[1:-1] = (
            (Theta[2:] - 2 * Theta[1:-1] + Theta[:-2]) / (deltaY ** 2)
            + (2.0 / yk[1:-1]) * DTheta[1:-1]
        )
        DDTheta[-1] = (
            (2 * Theta[-1] - 5 * Theta[-2] + 4 * Theta[-3] - Theta[-4])
            / (deltaY ** 2)
            + (2.0 / yk[-1]) * DTheta[-1]
        )

        Dk = np.empty_like(k_arr)
        Dk[0] = 0.0
        Dk[1:-1] = (k_arr[2:] - k_arr[:-2]) / (2 * deltaY)
        Dk[-1] = (3 * k_arr[-1] - 4 * k_arr[-2] + k_arr[-3]) / (2 * deltaY)

        DDk = np.empty_like(k_arr)
        DDk[0] = 6 * (k_arr[1] - k_arr[0]) / (deltaY ** 2)
        DDk[1:-1] = (
            (k_arr[2:] - 2 * k_arr[1:-1] + k_arr[:-2]) / (deltaY ** 2)
            + (2.0 / yk[1:-1]) * Dk[1:-1]
        )
        DDk[-1] = (
            (2 * k_arr[-1] - 5 * k_arr[-2] + 4 * k_arr[-3] - k_arr[-4])
            / (deltaY ** 2)
            + (2.0 / yk[-1]) * Dk[-1]
        )

        # -- pressure evolution --
        pdot = (3.0 / R) * (
            -kappa * P * U
            + (kappa - 1) * chi * DTheta[-1] / R
            + kappa * P * fom * Rv * Dk[-1]
            / (R * Rmix[-1] * (1 - k_arr[-1]))
        )

        # -- mixture velocity inside bubble --
        Umix = (
            ((kappa - 1) * chi / R * DTheta - R * yk * pdot / 3.0)
            / (kappa * P)
            + fom / R * (Rv - Ra) / Rmix * Dk
        )

        # -- Theta evolution --
        Theta_prime = (
            (pdot + DDTheta * chi / R ** 2)
            * (K_f * T_f / P * (kappa - 1) / kappa)
            - DTheta * (Umix - yk * U) / R
            + fom / R ** 2 * (Rv - Ra) / Rmix * Dk * DTheta
        )
        Theta_prime[-1] = 0.0

        # -- k evolution --
        k_prime = (
            fom / R ** 2
            * (
                DDk
                + Dk
                * (
                    -((Rv - Ra) / Rmix) * Dk
                    - DTheta / np.sqrt(1 + 2 * A_star * Theta) / T_f
                )
            )
            - (Umix - U * yk) / R * Dk
        )
        k_prime[-1] = 0.0

        # ======= GMOD constitutive model =======

        lambda_w = R / Req_nondim
        lambda_r0 = (1 + (1.0 / r0_star3) * (lambda_w ** 3 - 1)) ** (1.0 / 3.0)
        term_r = lambda_r0 * r0_star

        # non-equilibrium elastic stretch
        lambda_ne = lambda_r0 / lnv
        lne_bp1 = lambda_ne ** beta1
        lne_bp2 = lambda_ne ** beta2

        # deviatoric Cauchy stress in non-eq branch (for viscous flow rule)
        s_tt_nv = (
            1.0 / (CaB1 * 3.0) * (lne_bp1 - lne_bp1 ** (-2))
            + 1.0 / (CaB2 * 3.0) * (lne_bp2 - lne_bp2 ** (-2))
        )
        lambda_nv_dot = s_tt_nv / 2.0 * Re * lnv

        # equilibrium branch
        lam_ap1 = lambda_r0 ** alpha1
        lam_ap2 = lambda_r0 ** alpha2 if use_CaA2 else None

        # -- damage logic --
        if not tracker.achieve_rmax:
            xi = np.ones(MT)
            if U < eps_val:
                tracker.achieve_rmax = True
                tracker.achieved_lambda_A = lambda_r0.copy()
                tracker.achieved_lambda_B = lambda_ne.copy()
                WA_0 = (
                    1.0 / (CaA1 * alpha1) * (lam_ap1 ** (-2) + 2 * lam_ap1 - 3)
                )
                if use_CaA2:
                    WA_0 = WA_0 + 1.0 / (CaA2 * alpha2) * (
                        lam_ap2 ** (-2) + 2 * lam_ap2 - 3
                    )
                tracker.WA_m = WA_0.copy()
        else:
            tensile = lambda_r0 > 1
            xi = np.ones(MT)
            if tensile[0]:
                xi[:damage_idx] = 0.0

            if xi_constant >= 0:
                xi[:] = xi_constant
            elif xi_constant == -0.5:
                xi[:damage_idx] = 0.0
                xi[damage_idx:] = 1.0

        # -- stress integral terms --
        Sint_A = xi * (1.0 / CaA1 * (lam_ap1 ** (-2) - lam_ap1))
        if use_CaA2:
            Sint_A = Sint_A + xi * (1.0 / CaA2 * (lam_ap2 ** (-2) - lam_ap2))

        Sint_B = 1.0 / CaB1 * (lambda_ne ** (-2 * beta1) - lambda_ne ** beta1)
        if use_CaB2:
            Sint_B = Sint_B + 1.0 / CaB2 * (
                lambda_ne ** (-2 * beta2) - lambda_ne ** beta2
            )

        S = float(np.trapz(2.0 * (Sint_A + Sint_B) / term_r, term_r))

        # -- Sdot from finite-difference history --
        # IMPORTANT: the ODE solver calls rhs() multiple times at the *same*
        # t_star when building the Jacobian via finite differences.  We must
        # only update (last_t, last_S) when time genuinely advances; otherwise
        # last_S gets overwritten by a perturbed state value and the next real
        # Sdot estimate becomes incorrect.
        if not tracker.initialized:
            # Very first call: bootstrap the history and return Sdot = 0.
            Sdot = 0.0
            tracker.initialized = True
            tracker.last_t = t_star
            tracker.last_S = S
        elif t_star > tracker.last_t + eps_val:
            # Time has advanced forward: compute Sdot and update history.
            Sdot = (S - tracker.last_S) / (t_star - tracker.last_t)
            tracker.Sdot_current = Sdot
            tracker.last_t = t_star
            tracker.last_S = S
        else:
            # Same or earlier time (Jacobian FD call or step rejection):
            # reuse the cached Sdot without touching last_S / last_t.
            Sdot = tracker.Sdot_current

        # ======= Keller-Miksis (no explicit viscous term) =======
        rdot = U
        udot = (
            (1 + U / C_star) * (P - 1.0 / (We * R) + S - 1)
            + R / C_star * (pdot + U / (We * R ** 2) + Sdot)
            - 1.5 * (1 - U / (3 * C_star)) * U ** 2
        ) / ((1 - U / C_star) * R)

        out = np.empty_like(x)
        out[0] = rdot
        out[1] = udot
        out[2] = pdot
        out[3 : 3 + NT] = Theta_prime
        out[3 + NT : 3 + 2 * NT] = k_prime
        out[3 + 2 * NT : 3 + 2 * NT + MT] = lambda_nv_dot
        return out

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------
    t_end_star = float(inp.tspan / tc)

    method = inp.solver_method
    jac_sparsity = _build_jac_sparsity_gmod(NT, MT)
    solver_kw: dict = dict(
        rtol=inp.rel_tol,
        atol=inp.abs_tol,
    )
    if method in ("BDF", "Radau"):
        solver_kw["jac_sparsity"] = jac_sparsity
    if inp.first_step > 0.0:
        solver_kw["first_step"] = inp.first_step

    sol = solve_ivp(
        rhs,
        t_span=(0.0, t_end_star),
        y0=x0,
        method=method,
        **solver_kw,
    )
    if not sol.success or sol.y.shape[1] < 3:
        raise RuntimeError(f"GMOD solver failed: {sol.message}")

    t_nondim = sol.t
    X = sol.y.T
    R_nondim = X[:, 0]
    U_nondim = X[:, 1]
    P_nondim = X[:, 2]

    t_sim = t_nondim * tc
    R_sim = R_nondim * Rc
    U_sim = U_nondim * (Rc / tc)
    P_sim = P_nondim * inp.P_inf

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


# -----------------------------------------------------------------------
# 1-term variant
# -----------------------------------------------------------------------

@dataclass(frozen=True)
class GMOD1Inputs:
    """All inputs for the 1-term GMOD solver (single Ogden A branch, single Maxwell B branch)."""

    # --- fitting variables ---
    U0: float           # m/s
    GA: float           # Pa  — single equilibrium Ogden modulus
    alpha: float        # dimensionless
    GB: float           # Pa  — single non-equilibrium Maxwell modulus
    beta: float         # dimensionless
    mu: float           # Pa·s
    damage_index: float # 0..MT

    # --- geometry / experimental ---
    Req: float          # m
    tspan: float        # s

    # --- numerical ---
    NT: int = 500
    MT: int = 200
    rel_tol: float = 1e-8
    abs_tol: float = 1e-7
    solver_method: str = "BDF"

    # --- far-field ---
    P_inf: float = 101325.0
    T_inf: float = 298.15
    c_long: float = 1485.0
    rho: float = 998.0
    gamma: float = 5.6e-2

    # --- damage ---
    xi_constant: float = -1.0

    # --- bubble-content constants ---
    D0: float = 24.2e-6
    kappa: float = 1.4
    Ru: float = 8.3144598
    A_therm: float = 5.28e-5
    B_therm: float = 1.17e-2
    P_ref: float = 1.17e11
    T_ref: float = 5200.0
    M_vapor: float = 18.01528e-3
    M_air: float = 28.966e-3


def simulate_gmod1_lic(inp: GMOD1Inputs) -> NhkvOutputs:
    """Run the 1-term GMOD LIC simulation.

    Single equilibrium Ogden branch (A) and single non-equilibrium Maxwell
    branch (B) with viscous dashpot, plus binary damage model.

    Implemented by promoting GMOD1Inputs to the general 2-term GMODInputs
    with the second branch set to near-zero moduli (CaA2/CaB2 >> 1 → inactive).
    This is mathematically exact and keeps the solver numerically stable.

    A small explicit first_step is supplied to avoid overflow in scipy's
    automatic initial-step-size selector (select_initial_step evaluates the
    RHS at a perturbed state that can make lambda_r0 ** alpha1 overflow when
    only one Ogden branch is active).
    """
    Uc = float(np.sqrt(inp.P_inf / inp.rho))
    tc = float(inp.Req) / Uc
    t_end_star = float(inp.tspan) / tc
    # 1e-5 × total nondimensional span → tiny but safe; scipy adapts up quickly
    first_step = t_end_star * 1e-5

    inp2 = GMODInputs(
        U0=inp.U0,
        GA1=inp.GA,   GA2=1e-10,
        alpha1=inp.alpha, alpha2=inp.alpha,
        GB1=inp.GB,   GB2=1e-10,
        beta1=inp.beta,   beta2=inp.beta,
        mu=inp.mu,
        damage_index=inp.damage_index,
        Req=inp.Req, tspan=inp.tspan,
        NT=inp.NT, MT=inp.MT,
        rel_tol=inp.rel_tol, abs_tol=inp.abs_tol,
        solver_method=inp.solver_method,
        P_inf=inp.P_inf, T_inf=inp.T_inf, c_long=inp.c_long,
        rho=inp.rho, gamma=inp.gamma, xi_constant=inp.xi_constant,
        D0=inp.D0, kappa=inp.kappa, Ru=inp.Ru,
        A_therm=inp.A_therm, B_therm=inp.B_therm,
        P_ref=inp.P_ref, T_ref=inp.T_ref,
        M_vapor=inp.M_vapor, M_air=inp.M_air,
        first_step=first_step,
    )
    return simulate_gmod_lic(inp2)


def _simulate_gmod1_standalone(inp: GMOD1Inputs) -> NhkvOutputs:
    """Standalone 1-term solver (single Ogden-A + single Maxwell-B).

    Same physics as simulate_gmod1_lic but with a self-contained RHS.
    Kept for reference / future use when the sparse-Jacobian issue is resolved.
    """

    # --- bubble-content constants ---
    D0 = inp.D0
    kappa = inp.kappa
    Ru = inp.Ru
    Rv = Ru / inp.M_vapor
    Ra = Ru / inp.M_air
    A = inp.A_therm
    B = inp.B_therm

    # --- derived LIC scales ---
    R0 = float(inp.Req)
    Rc = R0
    Uc = np.sqrt(inp.P_inf / inp.rho)
    tc = R0 / Uc

    Pv = inp.P_ref * np.exp(-inp.T_ref / inp.T_inf)
    K_inf = A * inp.T_inf + B

    C_star = inp.c_long / Uc
    We = inp.P_inf * Rc / (2 * inp.gamma)
    CaA = inp.P_inf / inp.GA
    CaB = inp.P_inf / inp.GB
    Re = inp.P_inf * Rc / (inp.mu * Uc)
    fom = D0 / (Uc * Rc)
    chi = inp.T_inf * K_inf / (inp.P_inf * Rc * Uc)
    A_star = A * inp.T_inf / K_inf
    B_star = B / K_inf
    Pv_star = Pv / inp.P_inf

    _EPS_ALPHA = 1e-12
    alpha = max(float(inp.alpha), _EPS_ALPHA)
    beta  = max(float(inp.beta),  _EPS_ALPHA)
    xi_constant = float(inp.xi_constant)
    damage_idx = max(0, min(int(round(inp.damage_index)), inp.MT))

    Req_nondim = 1.0

    NT = int(inp.NT)
    MT = int(inp.MT)
    if NT < 20:
        raise ValueError("NT must be >= 20")
    if MT < 5:
        raise ValueError("MT must be >= 5")

    deltaY = 1.0 / (NT - 1)
    yk = np.linspace(0.0, 1.0, NT, dtype=float)

    temp_arr = np.linspace(0.0, 3.0, MT)
    r0_star  = 10.0 ** temp_arr
    r0_star3 = r0_star ** 3

    # --- initial conditions ---
    R0_star  = 1.0
    U0_star  = inp.U0 / Uc
    Theta0   = np.zeros(NT, dtype=float)

    P0       = Pv + (inp.P_inf + 2.0 * inp.gamma / R0 - Pv)
    P0_star  = P0 / inp.P_inf

    k0_val = (1 + (Rv / Ra) * (P0_star / Pv_star - 1)) ** (-1)
    k0 = np.full(NT, k0_val, dtype=float)

    lambda_nv0 = np.full(MT, 1.00001, dtype=float)

    x0 = np.concatenate(
        ([R0_star, U0_star, P0_star], Theta0, k0, lambda_nv0)
    ).astype(float)

    tracker = _SolverHistory()
    eps_val = float(np.finfo(float).eps)

    # ------------------------------------------------------------------
    def rhs(t_star: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        R     = x[0]
        U     = x[1]
        P     = x[2]
        Theta = x[3 : 3 + NT]
        k_arr = x[3 + NT : 3 + 2 * NT].copy()
        lnv   = x[3 + 2 * NT : 3 + 2 * NT + MT]

        k_arr[-1] = (1 + (Rv / Ra) * (P / Pv_star - 1)) ** (-1)

        T_f  = (A_star - 1 + np.sqrt(1 + 2 * A_star * Theta)) / A_star
        K_f  = A_star * T_f + B_star
        Rmix = k_arr * Rv + (1 - k_arr) * Ra

        DTheta = np.empty_like(Theta)
        DTheta[0]    = 0.0
        DTheta[1:-1] = (Theta[2:] - Theta[:-2]) / (2 * deltaY)
        DTheta[-1]   = (3 * Theta[-1] - 4 * Theta[-2] + Theta[-3]) / (2 * deltaY)

        DDTheta = np.empty_like(Theta)
        DDTheta[0]    = 6 * (Theta[1] - Theta[0]) / (deltaY ** 2)
        DDTheta[1:-1] = (
            (Theta[2:] - 2 * Theta[1:-1] + Theta[:-2]) / (deltaY ** 2)
            + (2.0 / yk[1:-1]) * DTheta[1:-1]
        )
        DDTheta[-1] = (
            (2 * Theta[-1] - 5 * Theta[-2] + 4 * Theta[-3] - Theta[-4])
            / (deltaY ** 2)
            + (2.0 / yk[-1]) * DTheta[-1]
        )

        Dk = np.empty_like(k_arr)
        Dk[0]    = 0.0
        Dk[1:-1] = (k_arr[2:] - k_arr[:-2]) / (2 * deltaY)
        Dk[-1]   = (3 * k_arr[-1] - 4 * k_arr[-2] + k_arr[-3]) / (2 * deltaY)

        DDk = np.empty_like(k_arr)
        DDk[0]    = 6 * (k_arr[1] - k_arr[0]) / (deltaY ** 2)
        DDk[1:-1] = (
            (k_arr[2:] - 2 * k_arr[1:-1] + k_arr[:-2]) / (deltaY ** 2)
            + (2.0 / yk[1:-1]) * Dk[1:-1]
        )
        DDk[-1] = (
            (2 * k_arr[-1] - 5 * k_arr[-2] + 4 * k_arr[-3] - k_arr[-4])
            / (deltaY ** 2)
            + (2.0 / yk[-1]) * Dk[-1]
        )

        pdot = (3.0 / R) * (
            -kappa * P * U
            + (kappa - 1) * chi * DTheta[-1] / R
            + kappa * P * fom * Rv * Dk[-1]
            / (R * Rmix[-1] * (1 - k_arr[-1]))
        )

        Umix = (
            ((kappa - 1) * chi / R * DTheta - R * yk * pdot / 3.0)
            / (kappa * P)
            + fom / R * (Rv - Ra) / Rmix * Dk
        )

        Theta_prime = (
            (pdot + DDTheta * chi / R ** 2)
            * (K_f * T_f / P * (kappa - 1) / kappa)
            - DTheta * (Umix - yk * U) / R
            + fom / R ** 2 * (Rv - Ra) / Rmix * Dk * DTheta
        )
        Theta_prime[-1] = 0.0

        k_prime = (
            fom / R ** 2
            * (
                DDk
                + Dk
                * (
                    -((Rv - Ra) / Rmix) * Dk
                    - DTheta / np.sqrt(1 + 2 * A_star * Theta) / T_f
                )
            )
            - (Umix - U * yk) / R * Dk
        )
        k_prime[-1] = 0.0

        # ======= GMOD1: single Ogden-A + single Maxwell-B =======

        lambda_w   = R / Req_nondim
        lambda_r0  = (1 + (1.0 / r0_star3) * (lambda_w ** 3 - 1)) ** (1.0 / 3.0)
        term_r     = lambda_r0 * r0_star

        # non-equilibrium elastic stretch (single B branch)
        lambda_ne  = lambda_r0 / lnv
        lne_b      = lambda_ne ** beta

        # viscous flow rule (single B branch)
        s_tt_nv       = 1.0 / (CaB * 3.0) * (lne_b - lne_b ** (-2))
        lambda_nv_dot = s_tt_nv / 2.0 * Re * lnv

        # equilibrium Ogden stretch (single A branch)
        lam_a = lambda_r0 ** alpha

        # damage
        if not tracker.achieve_rmax:
            xi = np.ones(MT)
            if U < eps_val:
                tracker.achieve_rmax = True
                tracker.achieved_lambda_A = lambda_r0.copy()
                tracker.achieved_lambda_B = lambda_ne.copy()
                tracker.WA_m = (
                    1.0 / (CaA * alpha) * (lam_a ** (-2) + 2 * lam_a - 3)
                ).copy()
        else:
            xi = np.ones(MT)
            if (lambda_r0 > 1)[0]:
                xi[:damage_idx] = 0.0
            if xi_constant >= 0:
                xi[:] = xi_constant
            elif xi_constant == -0.5:
                xi[:damage_idx] = 0.0
                xi[damage_idx:] = 1.0

        Sint_A = xi * (1.0 / CaA * (lam_a ** (-2) - lam_a))
        Sint_B = 1.0 / CaB * (lambda_ne ** (-2 * beta) - lambda_ne ** beta)

        S = float(np.trapz(2.0 * (Sint_A + Sint_B) / term_r, term_r))

        if not tracker.initialized:
            Sdot = 0.0
            tracker.initialized = True
            tracker.last_t = t_star
            tracker.last_S = S
        elif t_star > tracker.last_t + eps_val:
            Sdot = (S - tracker.last_S) / (t_star - tracker.last_t)
            tracker.Sdot_current = Sdot
            tracker.last_t = t_star
            tracker.last_S = S
        else:
            Sdot = tracker.Sdot_current

        # ======= Keller-Miksis =======
        rdot = U
        udot = (
            (1 + U / C_star) * (P - 1.0 / (We * R) + S - 1)
            + R / C_star * (pdot + U / (We * R ** 2) + Sdot)
            - 1.5 * (1 - U / (3 * C_star)) * U ** 2
        ) / ((1 - U / C_star) * R)

        out = np.empty_like(x)
        out[0] = rdot
        out[1] = udot
        out[2] = pdot
        out[3 : 3 + NT]          = Theta_prime
        out[3 + NT : 3 + 2 * NT] = k_prime
        out[3 + 2 * NT : 3 + 2 * NT + MT] = lambda_nv_dot
        return out

    # ------------------------------------------------------------------
    t_end_star = float(inp.tspan / tc)
    method = inp.solver_method
    jac_sparsity = _build_jac_sparsity_gmod(NT, MT)
    solver_kw: dict = dict(rtol=inp.rel_tol, atol=inp.abs_tol)
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
        raise RuntimeError(f"GMOD1 solver failed: {sol.message}")

    t_nondim = sol.t
    X        = sol.y.T
    R_nondim = X[:, 0]
    U_nondim = X[:, 1]
    P_nondim = X[:, 2]

    t_sim = t_nondim * tc
    R_sim = R_nondim * Rc
    U_sim = U_nondim * (Rc / tc)
    P_sim = P_nondim * inp.P_inf

    Rmax_sim       = float(np.max(R_sim))
    Rmax_ind       = int(np.argmax(R_sim))
    t_sim_shifted  = t_sim - t_sim[Rmax_ind]
    R_sim_nondim   = R_sim / Rmax_sim
    t_sim_nondim   = t_sim_shifted * Uc / Rmax_sim

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
