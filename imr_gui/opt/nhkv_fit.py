"""Model-agnostic fitting engine.

The caller provides a ``make_sim`` callable that maps
``(params_si_dict, tspan) -> NhkvOutputs``; the fitting engine
handles the optimizer loop, progress reporting and stop logic.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, differential_evolution


# -----------------------------------------------------------------------
# data classes
# -----------------------------------------------------------------------

@dataclass
class FitConfig:
    """Model-agnostic fitting configuration."""
    t_exp: NDArray[np.float64]
    R_exp: NDArray[np.float64]
    make_sim: Callable[[dict, float], object]   # (params_si, tspan) -> outputs
    param_names: List[str]                      # ordered list of parameter names
    tspan_factor: float = 1.2


@dataclass
class FitProgress:
    """Emitted after each optimizer iteration."""
    nfev: int
    best_err: float
    best_params: dict
    t_sim: NDArray[np.float64] | None = None
    R_sim: NDArray[np.float64] | None = None
    Rmax_sim: float | None = None
    tc: float | None = None
    step_size: float | None = None
    status: str = ""


@dataclass
class FitResult:
    """Final fitting result (model-agnostic)."""
    best_params: dict
    lsq_err: float
    nfev: int
    t_sim: NDArray[np.float64] | None = None
    R_sim: NDArray[np.float64] | None = None
    Rmax_sim: float | None = None
    tc: float | None = None


# keep backward-compat aliases used by FitWorker
NhkvFitConfig = FitConfig
NhkvFitResult = FitResult


class _UserStop(Exception):
    pass


# -----------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------

def _eval_and_sim(
    params_si: dict,
    cfg: FitConfig,
) -> tuple[float, object | None]:
    t_exp = cfg.t_exp
    R_exp = cfg.R_exp
    if t_exp.size < 3 or R_exp.size < 3:
        return 1e10, None

    tspan = (t_exp[-1] - t_exp[0]) * cfg.tspan_factor
    if tspan <= 0:
        tspan = t_exp[-1] - t_exp[0] if t_exp[-1] > t_exp[0] else 1e-6

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            out = cfg.make_sim(params_si, tspan)
    except Exception:
        return 1e10, None

    t_sim = out.t_sim
    R_sim = out.R_sim
    if t_sim.size < 3 or R_sim.size < 3:
        return 1e10, None

    try:
        R_sim_interp = np.interp(t_exp, t_sim, R_sim,
                                 left=R_sim[0], right=R_sim[-1])
    except Exception:
        return 1e10, None

    err = float(np.sum(((R_exp - R_sim_interp) * 1e6) ** 2))
    return (err if np.isfinite(err) else 1e10), out


# -----------------------------------------------------------------------
# public API
# -----------------------------------------------------------------------

def fit_nhkv_to_experiment(
    cfg: FitConfig,
    bounds_si: Dict[str, Tuple[float, float]],
    fit_flags: Dict[str, bool] | None = None,
    scales: Dict[str, str] | None = None,
    initial_values: Dict[str, float] | None = None,
    progress_callback: Callable[[FitProgress], None] | None = None,
    stop_flag: Callable[[], bool] | None = None,
    method: str = "Nelder-Mead",
) -> FitResult:
    """Fit arbitrary parameter set to experimental R(t).

    Works with any model — the ``cfg.make_sim`` callable encapsulates
    which solver and inputs type to use.
    """
    param_names = cfg.param_names
    if fit_flags is None:
        fit_flags = {n: True for n in param_names}
    if scales is None:
        scales = {}
    if initial_values is None:
        initial_values = {}

    _default_bounds = {n: (1e-30, 1e30) for n in param_names}

    # --- separate active vs fixed -----------------------------------------
    active: list[tuple[str, str]] = []
    opt_bounds: list[tuple[float, float]] = []
    fixed: dict[str, float] = {}

    for name in param_names:
        if fit_flags.get(name, True):
            sc = scales.get(name, "lin")
            lb, ub = bounds_si.get(name, _default_bounds[name])
            if sc == "log":
                lb = max(lb, 1e-30)
                ub = max(ub, 1e-30)
                lb, ub = np.log10(lb), np.log10(ub)
            opt_bounds.append((float(min(lb, ub)), float(max(lb, ub))))
            active.append((name, sc))
        else:
            fixed[name] = initial_values.get(name, 1.0)

    if not active:
        raise ValueError("At least one parameter must be selected for fitting.")

    # --- optimizer-space ↔ SI ---------------------------------------------
    def _to_si_dict(theta_opt: NDArray) -> dict:
        p: dict[str, float] = dict(fixed)
        for i, (nm, sc) in enumerate(active):
            v = float(theta_opt[i])
            p[nm] = 10.0 ** v if sc == "log" else v
        return p

    # --- objective + tracking ---------------------------------------------
    tracker: dict = {
        "nfev": 0, "best_err": float("inf"),
        "best_params": None, "best_out": None,
    }

    def obj(theta_opt):
        if stop_flag and stop_flag():
            return 1e10
        tracker["nfev"] += 1
        params_si = _to_si_dict(theta_opt)
        err, out = _eval_and_sim(params_si, cfg)
        if err < tracker["best_err"]:
            tracker["best_err"] = err
            tracker["best_params"] = dict(params_si)
            tracker["best_out"] = out
            _emit_progress(status="improved")
        return err

    def _emit_progress(step_size: float | None = None, status: str = ""):
        if progress_callback is None or tracker["best_params"] is None:
            return
        out = tracker["best_out"]
        t_sim = R_sim = Rmax = tc_v = None
        if out is not None:
            t_sim, R_sim = out.t_sim, out.R_sim
            Rmax, tc_v = out.Rmax_sim, out.tc
        progress_callback(FitProgress(
            nfev=tracker["nfev"],
            best_err=tracker["best_err"],
            best_params=dict(tracker["best_params"]),
            t_sim=t_sim, R_sim=R_sim,
            Rmax_sim=Rmax, tc=tc_v,
            step_size=step_size, status=status,
        ))

    # --- initial guess evaluation -----------------------------------------
    init_params = {n: initial_values.get(n, 1.0) for n in param_names}
    init_err, init_out = _eval_and_sim(init_params, cfg)
    tracker["nfev"] = 1
    if init_err < tracker["best_err"]:
        tracker["best_err"] = init_err
        tracker["best_params"] = dict(init_params)
        tracker["best_out"] = init_out
    _emit_progress(status="initial")

    if stop_flag and stop_flag():
        bp = tracker["best_params"] or init_params
        bo = tracker["best_out"]
        return FitResult(
            best_params=bp, lsq_err=tracker["best_err"], nfev=tracker["nfev"],
            t_sim=bo.t_sim if bo else None, R_sim=bo.R_sim if bo else None,
            Rmax_sim=bo.Rmax_sim if bo else None, tc=bo.tc if bo else None,
        )

    # --- run optimizer ----------------------------------------------------
    result = None

    if method.lower() in ("nelder-mead", "powell"):
        x0 = []
        for name, sc in active:
            val = initial_values.get(name, 1.0)
            if sc == "log":
                val = np.log10(max(val, 1e-30))
            x0.append(val)
        x0 = np.array(x0)
        lb_arr = np.array([b[0] for b in opt_bounds])
        ub_arr = np.array([b[1] for b in opt_bounds])
        x0 = np.clip(x0, lb_arr, ub_arr)

        cb_state = {"prev_xk": x0.copy(), "prev_best": tracker["best_err"],
                    "iter": 0}

        def _local_cb(xk):
            if stop_flag and stop_flag():
                raise _UserStop()
            cb_state["iter"] += 1
            step = float(np.linalg.norm(xk - cb_state["prev_xk"]))
            cur_best = tracker["best_err"]
            status = "successful" if cur_best < cb_state["prev_best"] else "refine"
            cb_state["prev_xk"] = xk.copy()
            cb_state["prev_best"] = cur_best
            _emit_progress(step_size=step, status=status)

        try:
            result = minimize(
                obj, x0, method=method,
                bounds=opt_bounds,
                callback=_local_cb,
                options={"maxfev": 500, "xatol": 1e-4, "fatol": 1e-1,
                         "adaptive": True},
            )
        except _UserStop:
            result = None

    elif method.lower() == "differential-evolution":
        def _de_cb(xk, convergence=0):
            if stop_flag and stop_flag():
                return True
            _emit_progress()
            return False

        try:
            result = differential_evolution(
                obj, bounds=opt_bounds,
                strategy="best1bin",
                maxiter=30, popsize=5,
                tol=1e-3, polish=False,
                updating="deferred", workers=1,
                callback=_de_cb,
            )
        except _UserStop:
            result = None
    else:
        raise ValueError(f"Unknown optimisation method: {method!r}")

    # --- best result ------------------------------------------------------
    if result is not None and hasattr(result, "x"):
        res_params = _to_si_dict(result.x)
        res_err = float(result.fun)
        if tracker["best_params"] is not None and tracker["best_err"] < res_err:
            best_params = tracker["best_params"]
            best_err = tracker["best_err"]
        else:
            best_params = res_params
            best_err = res_err
    elif tracker["best_params"] is not None:
        best_params = tracker["best_params"]
        best_err = tracker["best_err"]
    else:
        raise RuntimeError("Fitting produced no valid result.")

    best_out = tracker.get("best_out")
    if best_out is None and not (stop_flag and stop_flag()):
        _, best_out = _eval_and_sim(best_params, cfg)

    return FitResult(
        best_params=best_params,
        lsq_err=best_err,
        nfev=tracker["nfev"],
        t_sim=best_out.t_sim if best_out else None,
        R_sim=best_out.R_sim if best_out else None,
        Rmax_sim=best_out.Rmax_sim if best_out else None,
        tc=best_out.tc if best_out else None,
    )
