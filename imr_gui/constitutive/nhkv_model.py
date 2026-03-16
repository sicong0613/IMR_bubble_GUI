from __future__ import annotations

import numpy as np


def nhkv_sedot(
    R: float,
    U: float,
    alpha: float,
    Ca: float,
    Req_nondim: float,
) -> float:
    """
    Neo-Hookean / NHKV elastic stress-integral evolution (Sedot) for LIC,
    factored out from the IMR core.

    This corresponds to the analytic expression used in the MATLAB
    `fun_IMR_NHKV.m` implementation when alpha is the strain-stiffening
    parameter and Ca the Cauchy number (P_inf / G).
    """
    Rst = R / Req_nondim
    term1 = (3 * alpha - 1) * (1.0 / Rst + 1.0 / (Rst**4)) / Ca
    term2 = alpha * (1.0 / (Rst**8) + 1.0 / (Rst**5) + 2.0 / (Rst**2) + 2.0 * Rst) / Ca
    return 2.0 * U / R * (term1 - term2)


__all__ = ["nhkv_sedot"]

