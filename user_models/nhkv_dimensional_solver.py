from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from imr_gui.constitutive.nhkv_model import nhkv_sedot
from imr_gui.imr import NhkvOutputs
from imr_gui.imr.core import _build_jac_sparsity


def _const(context: dict, name: str, default: float) -> float:
    constants = dict(context.get("constants", {}) or {})
    return float(context.get(name, constants.get(name, default)))


def simulate(params_si: dict, tspan: float, context: dict):
    """NHKV reference solver using SI radius/time state variables.

    This solver is intended as a diagnostic comparison against the built-in
    nondimensional NHKV solver.  The integrated state is

        [R(m), U(m/s), P(Pa), Se(Pa), Theta(...), k(...)]

    and the independent variable is physical time in seconds.  The internal
    thermal variables Theta and k remain dimensionless because they are already
    nondimensional mixture variables in the original IMR gas model.
    """

    # --- parameters and GUI context ---
    U0 = float(params_si.get("U0", 100.0))
    G = float(params_si.get("G", 8.0e6))
    mu = float(params_si.get("mu", 0.226))

    Req = float(context["Req"])
    P_inf = float(context.get("P_inf", 101325.0))
    rho = float(context.get("rho", 998.0))
    gamma = float(context.get("gamma", 5.6e-2))
    c_long = float(context.get("c_long", 1485.0))
    NT = int(context.get("NT", 500))
    bubble_model = str(context.get("bubble_model", "Keller-Miksis"))
    solver_settings = dict(context.get("solver", {}) or {})

    T_inf = _const(context, "T_inf", 298.15)
    D0 = _const(context, "D0", 24.2e-6)
    kappa = _const(context, "kappa", 1.4)
    Ru = _const(context, "Ru", 8.3144598)
    A = _const(context, "A_therm", 5.28e-5)
    B = _const(context, "B_therm", 1.17e-2)
    P_ref = _const(context, "P_ref", 1.17e11)
    T_ref = _const(context, "T_ref", 5200.0)
    M_vapor = _const(context, "M_vapor", 18.01528e-3)
    M_air = _const(context, "M_air", 28.966e-3)
    alpha = _const(context, "alpha", 0.0)

    if NT < 20:
        raise ValueError("NT must be >= 20")
    if Req <= 0.0:
        raise ValueError("Req must be positive")
    if G <= 0.0 or mu <= 0.0:
        raise ValueError("G and mu must be positive")

    # --- characteristic values used only to form dimensionless groups ---
    Rc = Req
    Uc = float(np.sqrt(P_inf / rho)) if rho > 0.0 else 1.0
    tc = Rc / Uc if Uc > 0.0 else 1.0

    Rv = Ru / M_vapor
    Ra = Ru / M_air
    Pv = P_ref * np.exp(-T_ref / T_inf)
    K_inf = A * T_inf + B

    C_star = c_long / Uc
    We = P_inf * Rc / (2.0 * gamma)
    Ca = P_inf / G
    Re = P_inf * Rc / (mu * Uc)
    fom = D0 / (Uc * Rc)
    chi = T_inf * K_inf / (P_inf * Rc * Uc)
    A_star = A * T_inf / K_inf
    B_star = B / K_inf
    Pv_star = Pv / P_inf
    use_rp = bubble_model == "Rayleigh-Plesset"
    Req_nondim = 1.0

    # --- initial conditions in physical units ---
    R_initial = Req
    U_initial = U0
    P_initial = Pv + (P_inf + 2.0 * gamma / Req - Pv)
    P_initial_star = P_initial / P_inf

    Se0_star = (
        (3.0 * alpha - 1.0) * (5.0 - 4.0 * Req_nondim - Req_nondim**4) / (2.0 * Ca)
        + 2.0
        * alpha
        * (
            27.0 / 40.0
            + 1.0 / 8.0 * Req_nondim**8
            + 1.0 / 5.0 * Req_nondim**5
            + Req_nondim**2
            - 2.0 / Req_nondim
        )
        / Ca
    )
    Se_initial = Se0_star * P_inf

    Theta0 = np.zeros((NT,), dtype=float)
    k0_val = (1.0 + (Rv / Ra) * (P_initial_star / Pv_star - 1.0)) ** (-1.0)
    k0 = np.full((NT,), k0_val, dtype=float)
    y0 = np.concatenate(([R_initial, U_initial, P_initial, Se_initial], Theta0, k0)).astype(float)

    deltaY = 1.0 / (NT - 1)
    yk = np.linspace(0.0, 1.0, NT, dtype=float)

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        R_si = y[0]
        U_si = y[1]
        P_si = y[2]
        Se_si = y[3]
        Theta = y[4 : 4 + NT]
        k = y[4 + NT : 4 + 2 * NT]

        R = R_si / Rc
        U = U_si / Uc
        P = P_si / P_inf
        Se = Se_si / P_inf

        k_wall = (1.0 + (Rv / Ra) * (P / Pv_star - 1.0)) ** (-1.0)
        k = k.copy()
        k[-1] = k_wall

        T = (A_star - 1.0 + np.sqrt(1.0 + 2.0 * A_star * Theta)) / A_star
        K_star = A_star * T + B_star
        Rmix = k * Rv + (1.0 - k) * Ra

        DTheta = np.empty_like(Theta)
        DTheta[0] = 0.0
        DTheta[1:-1] = (Theta[2:] - Theta[:-2]) / (2.0 * deltaY)
        DTheta[-1] = (3.0 * Theta[-1] - 4.0 * Theta[-2] + Theta[-3]) / (2.0 * deltaY)

        DDTheta = np.empty_like(Theta)
        DDTheta[0] = 6.0 * (Theta[1] - Theta[0]) / (deltaY**2)
        DDTheta[1:-1] = (
            (Theta[2:] - 2.0 * Theta[1:-1] + Theta[:-2]) / (deltaY**2)
            + (2.0 / yk[1:-1]) * DTheta[1:-1]
        )
        DDTheta[-1] = (
            (2.0 * Theta[-1] - 5.0 * Theta[-2] + 4.0 * Theta[-3] - Theta[-4]) / (deltaY**2)
            + (2.0 / yk[-1]) * DTheta[-1]
        )

        Dk = np.empty_like(k)
        Dk[0] = 0.0
        Dk[1:-1] = (k[2:] - k[:-2]) / (2.0 * deltaY)
        Dk[-1] = (3.0 * k[-1] - 4.0 * k[-2] + k[-3]) / (2.0 * deltaY)

        DDk = np.empty_like(k)
        DDk[0] = 6.0 * (k[1] - k[0]) / (deltaY**2)
        DDk[1:-1] = (
            (k[2:] - 2.0 * k[1:-1] + k[:-2]) / (deltaY**2)
            + (2.0 / yk[1:-1]) * Dk[1:-1]
        )
        DDk[-1] = (
            (2.0 * k[-1] - 5.0 * k[-2] + 4.0 * k[-3] - k[-4]) / (deltaY**2)
            + (2.0 / yk[-1]) * Dk[-1]
        )

        pdot_star = 3.0 / R * (
            -kappa * P * U
            + (kappa - 1.0) * chi * DTheta[-1] / R
            + kappa * P * fom * Rv * Dk[-1] / (R * Rmix[-1] * (1.0 - k[-1]))
        )

        Umix = (
            ((kappa - 1.0) * chi / R * DTheta - R * yk * pdot_star / 3.0) / (kappa * P)
            + fom / R * (Rv - Ra) / Rmix * Dk
        )

        Theta_prime_star = (
            (pdot_star + DDTheta * chi / (R**2)) * (K_star * T / P * (kappa - 1.0) / kappa)
            - DTheta * (Umix - yk * U) / R
            + fom / (R**2) * (Rv - Ra) / Rmix * Dk * DTheta
        )
        Theta_prime_star[-1] = 0.0

        k_prime_star = (
            fom
            / (R**2)
            * (
                DDk
                + Dk
                * (
                    -((Rv - Ra) / Rmix) * Dk
                    - DTheta / np.sqrt(1.0 + 2.0 * A_star * Theta) / T
                )
            )
            - (Umix - U * yk) / R * Dk
        )
        k_prime_star[-1] = 0.0

        Sedot_star = nhkv_sedot(R=R, U=U, alpha=alpha, Ca=Ca, Req_nondim=Req_nondim)

        force_term = P - 1.0 / (We * R) + Se - 4.0 * U / (Re * R) - 1.0
        if use_rp:
            udot_star = (force_term - 1.5 * U**2) / R
        else:
            udot_star = (
                (1.0 + U / C_star) * force_term
                + R
                / C_star
                * (pdot_star + U / (We * R**2) + Sedot_star + 4.0 * U**2 / (Re * R**2))
                - 1.5 * (1.0 - U / (3.0 * C_star)) * U**2
            ) / ((1.0 - U / C_star) * R + 4.0 / (C_star * Re))

        dydt = np.empty_like(y)
        dydt[0] = U_si
        dydt[1] = (Uc / tc) * udot_star
        dydt[2] = (P_inf / tc) * pdot_star
        dydt[3] = (P_inf / tc) * Sedot_star
        dydt[4 : 4 + NT] = Theta_prime_star / tc
        dydt[4 + NT : 4 + 2 * NT] = k_prime_star / tc
        return dydt

    method = str(solver_settings.get("solver_method", "BDF") or "BDF")
    rel_tol = float(solver_settings.get("rel_tol", 1e-8))
    abs_tol_star = float(solver_settings.get("abs_tol", 1e-7))

    atol = np.empty_like(y0)
    atol[0] = max(abs_tol_star * Rc, 1e-15)
    atol[1] = max(abs_tol_star * Uc, 1e-9)
    atol[2] = max(abs_tol_star * P_inf, 1e-3)
    atol[3] = max(abs_tol_star * P_inf, 1e-3)
    atol[4 : 4 + NT] = max(abs_tol_star, 1e-10)
    atol[4 + NT : 4 + 2 * NT] = max(abs_tol_star, 1e-10)

    solver_kw = {"rtol": rel_tol, "atol": atol}
    if method in ("BDF", "Radau"):
        solver_kw["jac_sparsity"] = _build_jac_sparsity(NT)

    sol = solve_ivp(
        rhs,
        t_span=(0.0, float(tspan)),
        y0=y0,
        method=method,
        **solver_kw,
    )
    if not sol.success or sol.y.shape[1] < 3:
        raise RuntimeError(f"NHKV dimensional solver failed: {sol.message}")

    t_sim = sol.t.astype(float)
    R_sim = sol.y[0].astype(float)
    U_sim = sol.y[1].astype(float)
    P_sim = sol.y[2].astype(float)

    Rmax_sim = float(np.max(R_sim))
    peak_idx = int(np.argmax(R_sim))
    t_shifted = t_sim - t_sim[peak_idx]
    denom = Rmax_sim if Rmax_sim > 0.0 else Req

    return NhkvOutputs(
        t_sim=t_shifted,
        R_sim=R_sim,
        U_sim=U_sim,
        P_sim=P_sim,
        t_sim_nondim=(t_shifted * Uc / denom).astype(float),
        R_sim_nondim=(R_sim / denom).astype(float),
        Rmax_sim=Rmax_sim,
        tc=float(tc),
        Uc=float(Uc),
        n_damaged=0,
    )
