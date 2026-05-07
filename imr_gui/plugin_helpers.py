from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_ivp

from imr_gui.imr import NhkvOutputs


def shared_ode_solver(
    *,
    rhs: Callable,
    y0,
    t_span: tuple[float, float],
    context: dict,
    jac_sparsity=None,
    events=None,
    first_step: float | None = None,
    max_step: float | None = None,
    dense_output: bool = False,
    **kwargs: Any,
):
    """Run scipy.solve_ivp using the GUI solver settings from plugin context.

    This helper is intentionally small: it centralizes method/tolerance handling
    and failure checking, while leaving state definitions, RHS physics, events,
    and post-processing under the plugin solver's control.
    """
    solver = dict(context.get("solver", {}))
    method = solver.get("solver_method", "BDF")
    rel_tol = float(solver.get("rel_tol", 1e-8))
    abs_tol = float(solver.get("abs_tol", 1e-7))

    solve_kwargs: dict[str, Any] = {
        "method": method,
        "rtol": rel_tol,
        "atol": abs_tol,
        "dense_output": dense_output,
    }
    if jac_sparsity is not None and method in ("BDF", "Radau"):
        solve_kwargs["jac_sparsity"] = jac_sparsity
    if events is not None:
        solve_kwargs["events"] = events
    if first_step is not None and float(first_step) > 0.0:
        solve_kwargs["first_step"] = float(first_step)
    if max_step is not None and float(max_step) > 0.0:
        solve_kwargs["max_step"] = float(max_step)
    solve_kwargs.update(kwargs)

    sol = solve_ivp(rhs, t_span=t_span, y0=y0, **solve_kwargs)
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    return sol


def make_outputs(
    *,
    t_sim,
    R_sim,
    U_sim,
    P_sim,
    Rmax_sim: float | None = None,
    tc: float | None = None,
    Uc: float | None = None,
    n_damaged: int = 0,
) -> NhkvOutputs:
    """Build the GUI's shared output container from dimensional arrays."""
    t = np.asarray(t_sim, dtype=float).reshape(-1)
    R = np.asarray(R_sim, dtype=float).reshape(-1)
    U = np.asarray(U_sim, dtype=float).reshape(-1)
    P = np.asarray(P_sim, dtype=float).reshape(-1)

    if not (t.size == R.size == U.size == P.size):
        raise ValueError("t_sim, R_sim, U_sim, and P_sim must have the same length.")
    if t.size < 3:
        raise ValueError("Simulation output must contain at least 3 points.")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(R))):
        raise ValueError("Simulation output contains non-finite t/R values.")

    rmax = float(np.max(R)) if Rmax_sim is None else float(Rmax_sim)
    i_peak = int(np.argmax(R))
    t_shifted = t - t[i_peak]

    if Uc is None:
        Uc = 1.0
    if tc is None:
        tc = 1.0

    denom = rmax if rmax != 0.0 else 1.0
    return NhkvOutputs(
        t_sim=t_shifted.astype(float),
        R_sim=R.astype(float),
        U_sim=U.astype(float),
        P_sim=P.astype(float),
        t_sim_nondim=(t_shifted * float(Uc) / denom).astype(float),
        R_sim_nondim=(R / denom).astype(float),
        Rmax_sim=rmax,
        tc=float(tc),
        Uc=float(Uc),
        n_damaged=int(n_damaged),
    )

