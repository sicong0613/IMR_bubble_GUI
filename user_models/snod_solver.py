from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_matrix

from imr_gui.plugin_helpers import make_outputs, shared_ode_solver


class _DamageTracker:
    def __init__(self):
        self.t_rmax_star = float("inf")
        self.damage_mask: NDArray[np.bool_] | None = None


def _as_float(params: dict, name: str) -> float:
    try:
        return float(params[name])
    except KeyError as exc:
        raise KeyError(f"SNOD parameter '{name}' is missing from params_si.") from exc


def _build_jac_sparsity(NT: int, MT: int):
    """Conservative sparsity hint for [R,U,P,Theta,k,lambdaB,lambdaC]."""
    n_theta = 3
    n_k = n_theta + NT
    n_b = n_k + NT
    n_c = n_b + MT
    n_total = n_c + MT
    S = lil_matrix((n_total, n_total), dtype=np.int8)

    bw = 3
    boundary = 4
    theta_tail = range(max(0, NT - boundary), NT)

    for row in range(3):
        for col in range(3):
            S[row, col] = 1
        for j in theta_tail:
            S[row, n_theta + j] = 1
            S[row, n_k + j] = 1
        if row == 1:
            for j in range(MT):
                S[row, n_b + j] = 1
                S[row, n_c + j] = 1

    for i in range(NT):
        for row in (n_theta + i, n_k + i):
            for col in range(3):
                S[row, col] = 1
            for di in range(-bw, bw + 1):
                j = i + di
                if 0 <= j < NT:
                    S[row, n_theta + j] = 1
                    S[row, n_k + j] = 1
            for j in theta_tail:
                S[row, n_theta + j] = 1
                S[row, n_k + j] = 1

    for i in range(MT):
        for row in (n_b + i, n_c + i):
            S[row, 0] = 1
            S[row, 1] = 1
            S[row, row] = 1

    return S.tocsc()


def simulate(params_si: dict, tspan: float, context: dict):
    """Simulate SNOD: damaged Ogden A spring + Ogden B/C Maxwell branches."""
    Req = float(context["Req"])
    NT = int(context.get("NT", 500))
    P_inf = float(context["P_inf"])
    rho = float(context["rho"])
    surface_tension = float(context.get("gamma", 0.056))
    c_long = float(context.get("c_long", 1485.0))
    constants = dict(context.get("constants", {}))

    if Req <= 0.0:
        raise ValueError("SNOD requires Req > 0.")
    if tspan <= 0.0:
        raise ValueError("SNOD requires tspan > 0.")
    if NT < 20:
        raise ValueError("SNOD requires NT >= 20.")

    U0 = _as_float(params_si, "U0")
    GA = _as_float(params_si, "GA")
    GB = _as_float(params_si, "GB")
    GC = _as_float(params_si, "GC")
    alphaA = max(_as_float(params_si, "alphaA"), 1e-12)
    alphaB = max(_as_float(params_si, "alphaB"), 1e-12)
    alphaC = max(_as_float(params_si, "alphaC"), 1e-12)
    muB = _as_float(params_si, "muB")
    muC = _as_float(params_si, "muC")
    damage_Y = _as_float(params_si, "damage_Y")

    if min(GA, GB, GC, muB, muC) <= 0.0:
        raise ValueError("SNOD requires positive GA/GB/GC/muB/muC.")
    if damage_Y <= 1.0:
        raise ValueError("SNOD requires damage_Y > 1.")

    T_inf = float(constants.get("T_inf", 298.15))
    D0 = float(constants.get("D0", 24.2e-6))
    kappa = float(constants.get("kappa", 1.4))
    Ru = float(constants.get("Ru", 8.3144598))
    A = float(constants.get("A_therm", constants.get("A", 5.28e-5)))
    B = float(constants.get("B_therm", constants.get("B", 1.17e-2)))
    P_ref = float(constants.get("P_ref", 1.17e11))
    T_ref = float(constants.get("T_ref", 5200.0))
    M_vapor = float(constants.get("M_vapor", 18.01528e-3))
    M_air = float(constants.get("M_air", 28.966e-3))
    MT = int(constants.get("MT", 100))
    if MT < 5:
        raise ValueError("SNOD requires MT >= 5.")

    Rv = Ru / M_vapor
    Ra = Ru / M_air
    R0 = Req
    Rc = R0
    Uc = float(np.sqrt(P_inf / rho))
    tc = Rc / Uc

    Pv = P_ref * np.exp(-T_ref / T_inf)
    K_inf = A * T_inf + B
    C_star = c_long / Uc
    We = P_inf * Rc / (2.0 * surface_tension)
    CaA = P_inf / GA
    CaB = P_inf / GB
    CaC = P_inf / GC
    ReB = P_inf * Rc / (muB * Uc)
    ReC = P_inf * Rc / (muC * Uc)
    fom = D0 / (Uc * Rc)
    chi = T_inf * K_inf / (P_inf * Rc * Uc)
    A_star = A * T_inf / K_inf
    B_star = B / K_inf
    Pv_star = Pv / P_inf
    Req_star = 1.0

    bubble_model = str(context.get("bubble_model", "Keller-Miksis"))
    use_rp = bubble_model == "Rayleigh-Plesset"
    if bubble_model not in {"Keller-Miksis", "Rayleigh-Plesset"}:
        raise ValueError(f"Unsupported bubble_model for SNOD: {bubble_model!r}")

    delta_y = 1.0 / (NT - 1)
    yk = np.linspace(0.0, 1.0, NT, dtype=float)
    r0_star = 10.0 ** np.linspace(0.0, 2.0, MT, dtype=float)
    r0_star3 = r0_star**3

    R0_star = 1.0
    U0_star = U0 / Uc
    P0 = Pv + (P_inf + 2.0 * surface_tension / R0 - Pv)
    P0_star = P0 / P_inf
    theta0 = np.zeros(NT, dtype=float)
    k0 = np.full(NT, (1.0 + (Rv / Ra) * (P0_star / Pv_star - 1.0)) ** (-1), dtype=float)
    lnvB0 = np.full(MT, 1.00001, dtype=float)
    lnvC0 = np.full(MT, 1.00001, dtype=float)
    y0 = np.concatenate(([R0_star, U0_star, P0_star], theta0, k0, lnvB0, lnvC0)).astype(float)

    tracker = _DamageTracker()
    idx_theta = 3
    idx_k = idx_theta + NT
    idx_b = idx_k + NT
    idx_c = idx_b + MT

    def rhs(t_star: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        R = x[0]
        U = x[1]
        P = x[2]
        theta = x[idx_theta:idx_k]
        k_arr = x[idx_k:idx_b].copy()
        lnvB = x[idx_b:idx_c]
        lnvC = x[idx_c:]

        k_arr[-1] = (1.0 + (Rv / Ra) * (P / Pv_star - 1.0)) ** (-1)
        temp = (A_star - 1.0 + np.sqrt(1.0 + 2.0 * A_star * theta)) / A_star
        k_star = A_star * temp + B_star
        r_mix = k_arr * Rv + (1.0 - k_arr) * Ra

        dtheta = np.empty_like(theta)
        dtheta[0] = 0.0
        dtheta[1:-1] = (theta[2:] - theta[:-2]) / (2.0 * delta_y)
        dtheta[-1] = (3.0 * theta[-1] - 4.0 * theta[-2] + theta[-3]) / (2.0 * delta_y)

        ddtheta = np.empty_like(theta)
        ddtheta[0] = 6.0 * (theta[1] - theta[0]) / (delta_y**2)
        ddtheta[1:-1] = (
            (theta[2:] - 2.0 * theta[1:-1] + theta[:-2]) / (delta_y**2)
            + (2.0 / yk[1:-1]) * dtheta[1:-1]
        )
        ddtheta[-1] = (
            (2.0 * theta[-1] - 5.0 * theta[-2] + 4.0 * theta[-3] - theta[-4]) / (delta_y**2)
            + (2.0 / yk[-1]) * dtheta[-1]
        )

        dk = np.empty_like(k_arr)
        dk[0] = 0.0
        dk[1:-1] = (k_arr[2:] - k_arr[:-2]) / (2.0 * delta_y)
        dk[-1] = (3.0 * k_arr[-1] - 4.0 * k_arr[-2] + k_arr[-3]) / (2.0 * delta_y)

        ddk = np.empty_like(k_arr)
        ddk[0] = 6.0 * (k_arr[1] - k_arr[0]) / (delta_y**2)
        ddk[1:-1] = (
            (k_arr[2:] - 2.0 * k_arr[1:-1] + k_arr[:-2]) / (delta_y**2)
            + (2.0 / yk[1:-1]) * dk[1:-1]
        )
        ddk[-1] = (
            (2.0 * k_arr[-1] - 5.0 * k_arr[-2] + 4.0 * k_arr[-3] - k_arr[-4]) / (delta_y**2)
            + (2.0 / yk[-1]) * dk[-1]
        )

        pdot = 3.0 / R * (
            -kappa * P * U
            + (kappa - 1.0) * chi * dtheta[-1] / R
            + kappa * P * fom * Rv * dk[-1] / (R * r_mix[-1] * (1.0 - k_arr[-1]))
        )
        u_mix = (
            ((kappa - 1.0) * chi / R * dtheta - R * yk * pdot / 3.0) / (kappa * P)
            + fom / R * (Rv - Ra) / r_mix * dk
        )
        theta_dot = (
            (pdot + ddtheta * chi / (R**2)) * (k_star * temp / P * (kappa - 1.0) / kappa)
            - dtheta * (u_mix - yk * U) / R
            + fom / (R**2) * (Rv - Ra) / r_mix * dk * dtheta
        )
        theta_dot[-1] = 0.0

        k_dot = (
            fom
            / (R**2)
            * (ddk + dk * (-((Rv - Ra) / r_mix) * dk - dtheta / np.sqrt(1.0 + 2.0 * A_star * theta) / temp))
            - (u_mix - U * yk) / R * dk
        )
        k_dot[-1] = 0.0

        lambda_w = R / Req_star
        lambda_r0 = (1.0 + (1.0 / r0_star3) * (lambda_w**3 - 1.0)) ** (1.0 / 3.0)
        term_r = lambda_r0 * r0_star
        dlr0_dt = lambda_r0 ** (-2) * R**2 / r0_star3 * U
        x_dot = R**2 * U / term_r**2

        lambda_neB = lambda_r0 / lnvB
        lambda_neC = lambda_r0 / lnvC
        lneB = lambda_neB**alphaB
        lneC = lambda_neC**alphaC

        s_tt_B = 1.0 / (CaB * 3.0) * (lneB - lneB**(-2))
        s_tt_C = 1.0 / (CaC * 3.0) * (lneC - lneC**(-2))
        lnvB_dot = s_tt_B / 2.0 * ReB * lnvB
        lnvC_dot = s_tt_C / 2.0 * ReC * lnvC

        lamA = lambda_r0**alphaA
        if t_star < tracker.t_rmax_star or tracker.damage_mask is None:
            xiA = np.ones(MT, dtype=float)
        else:
            xiA = np.ones(MT, dtype=float)
            xiA[tracker.damage_mask] = 0.0

        Sint_A = xiA * (1.0 / CaA * (lamA**(-2) - lamA))
        Sint_B = 1.0 / CaB * (lambda_neB ** (-2.0 * alphaB) - lambda_neB**alphaB)
        Sint_C = 1.0 / CaC * (lambda_neC ** (-2.0 * alphaC) - lambda_neC**alphaC)
        Sint_integrand = 2.0 * (Sint_A + Sint_B + Sint_C) / term_r
        stress = float(np.trapz(Sint_integrand, term_r))

        coeff_A = xiA * (-alphaA / (CaA * lambda_r0)) * (2.0 * lamA**(-2) + lamA)
        dSintA_dt = coeff_A * dlr0_dt

        dlneB_dt = dlr0_dt / lnvB - lambda_neB * lnvB_dot / lnvB
        dlneC_dt = dlr0_dt / lnvC - lambda_neC * lnvC_dot / lnvC
        coeff_B = (-alphaB / (CaB * lambda_neB)) * (2.0 * lneB**(-2) + lneB)
        coeff_C = (-alphaC / (CaC * lambda_neC)) * (2.0 * lneC**(-2) + lneC)
        dSintB_dt = coeff_B * dlneB_dt
        dSintC_dt = coeff_C * dlneC_dt

        f_dot = (
            2.0 * (dSintA_dt + dSintB_dt + dSintC_dt) / term_r
            - Sint_integrand * x_dot / term_r
        )
        stress_dot = float(np.trapz(f_dot, term_r) + np.trapz(Sint_integrand, x_dot))

        force = P - 1.0 / (We * R) + stress - 1.0
        if use_rp:
            udot = (force - 1.5 * U**2) / R
        else:
            udot = (
                (1.0 + U / C_star) * force
                + R / C_star * (pdot + U / (We * R**2) + stress_dot)
                - 1.5 * (1.0 - U / (3.0 * C_star)) * U**2
            ) / ((1.0 - U / C_star) * R)

        out = np.empty_like(x)
        out[0] = U
        out[1] = udot
        out[2] = pdot
        out[idx_theta:idx_k] = theta_dot
        out[idx_k:idx_b] = k_dot
        out[idx_b:idx_c] = lnvB_dot
        out[idx_c:] = lnvC_dot
        return out

    t_end_star = float(tspan / tc)

    def _rmax_event(_, x: NDArray[np.float64]) -> float:
        return float(x[1])

    _rmax_event.terminal = True  # type: ignore[attr-defined]
    _rmax_event.direction = -1  # type: ignore[attr-defined]

    sol_pre = shared_ode_solver(
        rhs=rhs,
        y0=y0,
        t_span=(0.0, t_end_star),
        context={
            **context,
            "solver": {
                **dict(context.get("solver", {})),
                "rel_tol": 1e-6,
                "abs_tol": 1e-6,
            },
        },
        jac_sparsity=_build_jac_sparsity(NT, MT),
        events=[_rmax_event],
    )
    if len(sol_pre.t_events[0]) > 0:
        t_rmax_star = float(sol_pre.t_events[0][0])
        x_rmax = sol_pre.y_events[0][0]
        R_rmax = float(x_rmax[0])
        lambda_w_rmax = R_rmax / Req_star
        lambda_rmax = (1.0 + (1.0 / r0_star3) * (lambda_w_rmax**3 - 1.0)) ** (1.0 / 3.0)
        tracker.t_rmax_star = t_rmax_star
        tracker.damage_mask = lambda_rmax >= damage_Y

        if damage_Y > 1.0 and float(lambda_rmax[-1]) < damage_Y < float(lambda_rmax[0]):
            r0_y = ((lambda_w_rmax**3 - 1.0) / (damage_Y**3 - 1.0)) ** (1.0 / 3.0)
            idx_ins = int(np.searchsorted(r0_star, r0_y))
            r0_star = np.insert(r0_star, idx_ins, r0_y)
            r0_star3 = r0_star**3
            MT = MT + 1
            lnvB0 = np.insert(lnvB0, idx_ins, 1.00001)
            lnvC0 = np.insert(lnvC0, idx_ins, 1.00001)
            y0 = np.concatenate(([R0_star, U0_star, P0_star], theta0, k0, lnvB0, lnvC0)).astype(float)
            idx_c = idx_b + MT

            lambda_rmax = (1.0 + (1.0 / r0_star3) * (lambda_w_rmax**3 - 1.0)) ** (1.0 / 3.0)
            tracker.damage_mask = lambda_rmax >= damage_Y

    sol = shared_ode_solver(
        rhs=rhs,
        y0=y0,
        t_span=(0.0, t_end_star),
        context=context,
        jac_sparsity=_build_jac_sparsity(NT, MT),
    )
    if sol.y.shape[1] < 3:
        raise RuntimeError("SNOD solver produced too few output points.")

    x_sol = sol.y.T
    t_sim = sol.t * tc
    R_sim = x_sol[:, 0] * Rc
    U_sim = x_sol[:, 1] * Uc
    P_sim = x_sol[:, 2] * P_inf

    return make_outputs(
        t_sim=t_sim,
        R_sim=R_sim,
        U_sim=U_sim,
        P_sim=P_sim,
        tc=tc,
        Uc=Uc,
        n_damaged=0 if tracker.damage_mask is None else int(np.sum(tracker.damage_mask)),
    )
