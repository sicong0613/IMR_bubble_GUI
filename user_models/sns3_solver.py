from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_matrix

from imr_gui.plugin_helpers import make_outputs, shared_ode_solver


def _as_float(params: dict, name: str) -> float:
    try:
        return float(params[name])
    except KeyError as exc:
        raise KeyError(f"SNS3 parameter '{name}' is missing from params_si.") from exc


def _build_jac_sparsity(NT: int, MT: int):
    """Conservative sparsity hint for the SNS3 LIC state vector.

    State:
    [R, U, P, Seq, Theta(0..NT-1), k(0..NT-1), s1(0..MT-1), s2(0..MT-1)]
    """
    n_theta = 4
    n_k = n_theta + NT
    n_s1 = n_k + NT
    n_s2 = n_s1 + MT
    n_total = n_s2 + MT

    S = lil_matrix((n_total, n_total), dtype=np.int8)
    bw = 3
    boundary = 4

    theta_tail = range(max(0, NT - boundary), NT)

    for row in range(4):
        for col in range(4):
            S[row, col] = 1
        for j in theta_tail:
            S[row, n_theta + j] = 1
            S[row, n_k + j] = 1
        if row == 1:
            for j in range(MT):
                S[row, n_s1 + j] = 1
                S[row, n_s2 + j] = 1

    for i in range(NT):
        rows = (n_theta + i, n_k + i)
        for row in rows:
            for col in range(4):
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
        for row in (n_s1 + i, n_s2 + i):
            S[row, 0] = 1
            S[row, 1] = 1
            S[row, row] = 1

    return S.tocsc()


def simulate(params_si: dict, tspan: float, context: dict):
    """Simulate the SNS3 two-nonequilibrium-branch LIC model.

    This is a Python/LIC adaptation of
    `Example_MATLAB/Code_SNS3/qSNS_2branch_JoeyK.m`. The GUI supplies all
    parameters in SI units and passes the selected bubble equation through
    `context["bubble_model"]`.
    """
    Req = float(context["Req"])
    NT = int(context.get("NT", 500))
    P_inf = float(context["P_inf"])
    rho = float(context["rho"])
    gamma = float(context.get("gamma", 0.056))
    c_long = float(context.get("c_long", 1485.0))
    constants = dict(context.get("constants", {}))

    if Req <= 0.0:
        raise ValueError("SNS3 requires Req > 0.")
    if tspan <= 0.0:
        raise ValueError("SNS3 requires tspan > 0.")
    if NT < 20:
        raise ValueError("SNS3 requires NT >= 20.")

    U0 = _as_float(params_si, "U0")
    G = _as_float(params_si, "G")
    G1 = _as_float(params_si, "G1")
    mu1 = _as_float(params_si, "mu1")
    G2 = _as_float(params_si, "G2")
    mu2 = _as_float(params_si, "mu2")

    if min(G, G1, G2, mu1, mu2) <= 0.0:
        raise ValueError("SNS3 requires positive G/G1/G2/mu1/mu2.")

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
    alpha = float(constants.get("alpha", 0.0))

    if MT < 4:
        raise ValueError("SNS3 requires MT >= 4.")

    Rv = Ru / M_vapor
    Ra = Ru / M_air
    R0 = Req
    Rmax = Req
    Rc = Rmax
    Uc = float(np.sqrt(P_inf / rho))
    tc = Rc / Uc

    Pv = P_ref * np.exp(-T_ref / T_inf)
    K_inf = A * T_inf + B
    C_star = c_long / Uc
    We = P_inf * Rc / (2.0 * gamma)
    Ca = P_inf / G
    Ca1 = P_inf / G1
    Ca2 = P_inf / G2
    Re1 = P_inf * Rc / (mu1 * Uc)
    Re2 = P_inf * Rc / (mu2 * Uc)
    De1 = mu1 * Uc / (G1 * Rc)
    De2 = mu2 * Uc / (G2 * Rc)
    fom = D0 / (Uc * Rc)
    chi = T_inf * K_inf / (P_inf * Rc * Uc)
    A_star = A * T_inf / K_inf
    B_star = B / K_inf
    Pv_star = Pv / P_inf
    Req_star = R0 / Rc

    bubble_model = str(context.get("bubble_model", "Keller-Miksis"))
    use_rp = bubble_model == "Rayleigh-Plesset"
    if bubble_model not in {"Keller-Miksis", "Rayleigh-Plesset"}:
        raise ValueError(f"Unsupported bubble_model for SNS3: {bubble_model!r}")

    R0_star = 1.0
    U0_star = U0 / Uc
    P0 = Pv + (P_inf + 2.0 * gamma / R0 - Pv) * ((R0 / Rmax) ** 3)
    P0_star = P0 / P_inf
    Seq0 = (
        (3.0 * alpha - 1.0) * (5.0 - 4.0 * Req_star - Req_star**4) / (2.0 * Ca)
        + 2.0
        * alpha
        * (
            27.0 / 40.0
            + 1.0 / 8.0 * Req_star**8
            + 1.0 / 5.0 * Req_star**5
            + Req_star**2
            - 2.0 / Req_star
        )
        / Ca
    )
    theta0 = np.zeros(NT, dtype=float)
    k0 = np.full(NT, (1.0 + (Rv / Ra) * (P0_star / Pv_star - 1.0)) ** (-1), dtype=float)
    s1_0 = np.zeros(MT, dtype=float)
    s2_0 = np.zeros(MT, dtype=float)
    y0 = np.concatenate(([R0_star, U0_star, P0_star, Seq0], theta0, k0, s1_0, s2_0))

    delta_y = 1.0 / (NT - 1)
    yk = np.linspace(0.0, 1.0, NT, dtype=float)
    r0_star_list = 10.0 ** np.linspace(0.0, 2.0, MT, dtype=float)
    idx_theta = 4
    idx_k = idx_theta + NT
    idx_s1 = idx_k + NT
    idx_s2 = idx_s1 + MT

    def rhs(_t: float, x: NDArray[np.float64]) -> NDArray[np.float64]:
        R = x[0]
        U = x[1]
        P = x[2]
        Seq = x[3]
        theta = x[idx_theta:idx_k]
        k = x[idx_k:idx_s1].copy()
        s1 = x[idx_s1:idx_s2]
        s2 = x[idx_s2:]

        k[-1] = (1.0 + (Rv / Ra) * (P / Pv_star - 1.0)) ** (-1)
        temp = (A_star - 1.0 + np.sqrt(1.0 + 2.0 * A_star * theta)) / A_star
        k_star = A_star * temp + B_star
        r_mix = k * Rv + (1.0 - k) * Ra

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

        dk = np.empty_like(k)
        dk[0] = 0.0
        dk[1:-1] = (k[2:] - k[:-2]) / (2.0 * delta_y)
        dk[-1] = (3.0 * k[-1] - 4.0 * k[-2] + k[-3]) / (2.0 * delta_y)

        ddk = np.empty_like(k)
        ddk[0] = 6.0 * (k[1] - k[0]) / (delta_y**2)
        ddk[1:-1] = (
            (k[2:] - 2.0 * k[1:-1] + k[:-2]) / (delta_y**2)
            + (2.0 / yk[1:-1]) * dk[1:-1]
        )
        ddk[-1] = (
            (2.0 * k[-1] - 5.0 * k[-2] + 4.0 * k[-3] - k[-4]) / (delta_y**2)
            + (2.0 / yk[-1]) * dk[-1]
        )

        pdot = 3.0 / R * (
            -kappa * P * U
            + (kappa - 1.0) * chi * dtheta[-1] / R
            + kappa * P * fom * Rv * dk[-1] / (R * r_mix[-1] * (1.0 - k[-1]))
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
        seq_dot = (
            2.0 * U / R * (3.0 * alpha - 1.0) * (1.0 / lambda_w + 1.0 / lambda_w**4) / Ca
            - 2.0
            * alpha
            * U
            / R
            * (1.0 / lambda_w**8 + 1.0 / lambda_w**5 + 2.0 / lambda_w**2 + 2.0 * lambda_w)
            / Ca
        )

        lambda_out = (1.0 + (1.0 / r0_star_list) ** 3 * (lambda_w**3 - 1.0)) ** (1.0 / 3.0)
        lambda_dot_out = lambda_w**2 * U / (r0_star_list**3 * lambda_out**2)
        term_r = lambda_out * r0_star_list

        s_neq1 = np.trapz(3.0 / term_r * s1, term_r)
        s1_dot = (-4.0 / Re1 * lambda_dot_out / lambda_out - s1) / De1
        s1_dot_term1 = 3.0 * U * R**2 / (term_r**4) * s1
        s1_dot_term2 = 3.0 / term_r * s1_dot
        s_neq1_dot = -np.trapz(s1_dot_term1, term_r) + np.trapz(s1_dot_term2, term_r) - 3.0 / R * s1[0] * U

        s_neq2 = np.trapz(3.0 / term_r * s2, term_r)
        s2_dot = (-4.0 / Re2 * lambda_dot_out / lambda_out - s2) / De2
        s2_dot_term1 = 3.0 * U * R**2 / (term_r**4) * s2
        s2_dot_term2 = 3.0 / term_r * s2_dot
        s_neq2_dot = -np.trapz(s2_dot_term1, term_r) + np.trapz(s2_dot_term2, term_r) - 3.0 / R * s2[0] * U

        rdot = U
        force = P - 1.0 / (We * R) + Seq + s_neq1 + s_neq2 - 1.0
        if use_rp:
            udot = (force - 1.5 * U**2) / R
        else:
            udot = (
                (1.0 + U / C_star) * force
                + R / C_star * (pdot + U / (We * R**2) + seq_dot + s_neq1_dot + s_neq2_dot)
                - 1.5 * (1.0 - U / (3.0 * C_star)) * U**2
            ) / ((1.0 - U / C_star) * R)

        out = np.empty_like(x)
        out[0] = rdot
        out[1] = udot
        out[2] = pdot
        out[3] = seq_dot
        out[idx_theta:idx_k] = theta_dot
        out[idx_k:idx_s1] = k_dot
        out[idx_s1:idx_s2] = s1_dot
        out[idx_s2:] = s2_dot
        return out

    t_end_star = float(tspan / tc)
    jac_sparsity = _build_jac_sparsity(NT, MT)
    sol = shared_ode_solver(
        rhs=rhs,
        y0=y0,
        t_span=(0.0, t_end_star),
        context=context,
        jac_sparsity=jac_sparsity,
    )
    if sol.y.shape[1] < 3:
        raise RuntimeError("SNS3 solver produced too few output points.")

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
    )
