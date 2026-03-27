"""Model-agnostic fitting engine.

The caller provides a ``make_sim`` callable that maps
``(params_si_dict, tspan) -> NhkvOutputs``; the fitting engine
handles the optimizer loop, progress reporting and stop logic.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, differential_evolution, dual_annealing, basinhopping

try:
    import cma as _cma
    _HAS_CMA = True
except ImportError:
    _HAS_CMA = False


# -----------------------------------------------------------------------
# public constants
# -----------------------------------------------------------------------

AVAILABLE_METHODS: list[str] = [
    "Nelder-Mead",
    "Powell",
    "Pattern Search",
    "Differential Evolution",
    "CMA-ES",
    "Dual Annealing",
    "Basin Hopping",
]

DE_STRATEGIES: list[str] = [
    "best1bin", "best1exp", "rand1exp", "randtobest1exp",
    "currenttobest1exp", "best2exp", "rand2exp", "randtobest1bin",
    "currenttobest1bin", "best2bin", "rand2bin", "rand1bin",
]


# -----------------------------------------------------------------------
# data classes
# -----------------------------------------------------------------------

@dataclass
class OptConfig:
    """Optimizer settings, passed from the Settings dialog to the fit engine."""
    method: str = "Nelder-Mead"

    # ---- shared ----
    n_workers: int = 1          # parallel workers (DE and Pattern Search)
    max_fev: int = 500          # max function evaluations (Nelder-Mead / Powell)
    x_tol: float = 1e-4        # parameter-space convergence tolerance
    f_tol: float = 1e-1        # function-value convergence tolerance

    # ---- Nelder-Mead / Powell ----
    nm_adaptive: bool = True    # adaptive simplex (recommended for >2 params)

    # ---- Differential Evolution ----
    de_strategy: str = "best1bin"
    de_maxiter: int = 30
    de_popsize: int = 5
    de_mutation: float = 0.7
    de_recombination: float = 0.7

    # ---- CMA-ES ----
    cma_sigma0: float = 0.3     # initial step-size (in normalised space)
    cma_maxfev: int = 1000

    # ---- Dual Annealing ----
    da_maxfev: int = 1000
    da_initial_temp: float = 5230.0
    da_restart_temp: float = 2e-5

    # ---- Basin Hopping ----
    bh_n_iter: int = 50
    bh_stepsize: float = 0.5

    # ---- Pattern Search ----
    ps_complete_poll: bool = False  # False = opportunistic (MATLAB default)
    ps_mesh_contraction: float = 0.5
    ps_mesh_expansion: float = 2.0
    ps_initial_mesh: float = 1.0    # relative to param range in optimizer space
    ps_search_pts: int = 0          # random search points before each poll step


@dataclass
class FitConfig:
    """Model-agnostic fitting configuration."""
    t_exp: NDArray[np.float64]
    R_exp: NDArray[np.float64]
    make_sim: Callable[[dict, float], object]   # (params_si, tspan) -> outputs
    param_names: List[str]                      # ordered list of parameter names
    tspan_factor: float = 1.2
    mp_make_sim: Callable | None = None         # picklable version for DE multiprocessing


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
    sim_out: object | None = None  # full NhkvOutputs from the best-fit simulation


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

    # Mean squared error in µm² — dividing by n makes it comparable across
    # experiments with different data densities.
    n = max(t_exp.size, 1)
    err = float(np.sum(((R_exp - R_sim_interp) * 1e6) ** 2) / n)
    return (err if np.isfinite(err) else 1e10), out


# -----------------------------------------------------------------------
# picklable DE objective (for ProcessPoolExecutor / workers=N path)
# -----------------------------------------------------------------------

class _DEObjFn:
    """Module-level picklable wrapper used by DE with workers > 1.

    Replicates the core logic of _eval_and_sim without holding any
    Qt or closure references so it survives multiprocessing pickle.
    """

    def __init__(self, active, fixed, mp_make_sim, t_exp, R_exp, tspan_factor):
        self.active = active
        self.fixed = fixed
        self.mp_make_sim = mp_make_sim
        self.t_exp = t_exp
        self.R_exp = R_exp
        self.tspan_factor = tspan_factor

    def _to_si(self, theta_opt):
        p: dict = dict(self.fixed)
        for i, (nm, sc) in enumerate(self.active):
            v = float(theta_opt[i])
            p[nm] = 10.0 ** v if sc == "log" else v
        return p

    def __call__(self, theta_opt):
        import warnings
        t_exp, R_exp = self.t_exp, self.R_exp
        if t_exp.size < 3 or R_exp.size < 3:
            return 1e10
        tspan = (t_exp[-1] - t_exp[0]) * self.tspan_factor
        if tspan <= 0:
            tspan = max(t_exp[-1] - t_exp[0], 1e-6)
        params_si = self._to_si(theta_opt)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                out = self.mp_make_sim(params_si, tspan)
        except Exception:
            return 1e10
        t_sim, R_sim = out.t_sim, out.R_sim
        if t_sim.size < 3 or R_sim.size < 3:
            return 1e10
        try:
            R_interp = np.interp(t_exp, t_sim, R_sim, left=R_sim[0], right=R_sim[-1])
        except Exception:
            return 1e10
        n = max(t_exp.size, 1)
        err = float(np.sum(((R_exp - R_interp) * 1e6) ** 2) / n)
        return err if np.isfinite(err) else 1e10


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
    method: str = "Nelder-Mead",          # kept for back-compat
    opt_config: OptConfig | None = None,
) -> FitResult:
    """Fit arbitrary parameter set to experimental R(t).

    Works with any model — the ``cfg.make_sim`` callable encapsulates
    which solver and inputs type to use.

    ``opt_config`` takes precedence over the legacy ``method`` argument.
    """
    if opt_config is None:
        opt_config = OptConfig(method=method)

    param_names = cfg.param_names
    if fit_flags is None:
        fit_flags = {n: True for n in param_names}
    if scales is None:
        scales = {}
    if initial_values is None:
        initial_values = {}

    _default_bounds = {n: (1e-30, 1e30) for n in param_names}

    # --- separate active vs fixed ----------------------------------------
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

    # --- optimizer-space ↔ SI --------------------------------------------
    def _to_si_dict(theta_opt: NDArray) -> dict:
        p: dict[str, float] = dict(fixed)
        for i, (nm, sc) in enumerate(active):
            v = float(theta_opt[i])
            p[nm] = 10.0 ** v if sc == "log" else v
        return p

    # --- objective + tracking --------------------------------------------
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
            if _emit_on_improve:
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

    # --- initial guess evaluation ----------------------------------------
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
            sim_out=bo,
        )

    # --- build initial guess vector in optimizer space -------------------
    x0 = np.array([
        np.log10(max(initial_values.get(nm, 1.0), 1e-30)) if sc == "log"
        else initial_values.get(nm, 1.0)
        for nm, sc in active
    ])
    lb_arr = np.array([b[0] for b in opt_bounds])
    ub_arr = np.array([b[1] for b in opt_bounds])
    x0 = np.clip(x0, lb_arr, ub_arr)

    # --- run optimizer ---------------------------------------------------
    result = None
    algo = opt_config.method.lower().replace(" ", "-").replace("_", "-")

    # NM/Powell/Pattern Search use a per-iteration emit; obj() should NOT
    # emit independently (avoids duplicate lines with no step_size).
    _emit_on_improve = algo not in ("nelder-mead", "powell", "pattern-search")

    # ---- Nelder-Mead / Powell ----
    if algo in ("nelder-mead", "powell"):
        cb_state = {"prev_xk": x0.copy(), "prev_best": tracker["best_err"],
                    "iter": 0}

        def _local_cb(xk):
            if stop_flag and stop_flag():
                raise _UserStop()
            cb_state["iter"] += 1
            step = float(np.linalg.norm(xk - cb_state["prev_xk"]))
            cur_best = tracker["best_err"]
            status = "improved" if cur_best < cb_state["prev_best"] else "refine"
            cb_state["prev_xk"] = xk.copy()
            cb_state["prev_best"] = cur_best
            _emit_progress(step_size=step, status=status)

        nm_opts: dict = {
            "maxfev": opt_config.max_fev,
            "xatol": opt_config.x_tol,
            "fatol": opt_config.f_tol,
        }
        if algo == "nelder-mead":
            nm_opts["adaptive"] = opt_config.nm_adaptive
        try:
            result = minimize(
                obj, x0,
                method=opt_config.method,
                bounds=opt_bounds,
                callback=_local_cb,
                options=nm_opts,
            )
        except _UserStop:
            result = None

    # ---- Pattern Search (GPS with GPSPositiveBasis2N) ----
    elif algo == "pattern-search":
        # Generalized Pattern Search — equivalent to MATLAB patternsearch with
        # GPSPositiveBasis2N poll method.  Operates in optimizer space (each axis
        # already log- or lin-scaled); mesh steps are scaled to each parameter's
        # optimizer-space range so all axes are treated uniformly.
        #
        # Parallel mode (n_workers > 1, cfg.mp_make_sim set):
        #   The entire poll batch (2N points) is evaluated simultaneously via
        #   ProcessPoolExecutor.  Sequential opportunistic early-exit is replaced
        #   by selecting the best result from the batch.  Requires mp_make_sim
        #   (picklable callable) just like DE multiprocessing.

        from concurrent.futures import ProcessPoolExecutor

        ps_scale = ub_arr - lb_arr
        ps_scale[ps_scale == 0] = 1.0

        delta = float(opt_config.ps_initial_mesh)
        x_curr = x0.copy()
        f_curr = tracker["best_err"]   # reuse the already-computed initial eval

        rng = np.random.default_rng()
        n_active = len(active)
        poll_dirs = [(i, s) for i in range(n_active) for s in (+1.0, -1.0)]
        last_success: int | None = None

        n_workers = max(1, opt_config.n_workers)
        _mode = {"use_mp": n_workers > 1 and cfg.mp_make_sim is not None}

        if _mode["use_mp"]:
            ps_obj = _DEObjFn(
                active=active, fixed=fixed,
                mp_make_sim=cfg.mp_make_sim,
                t_exp=cfg.t_exp, R_exp=cfg.R_exp,
                tspan_factor=cfg.tspan_factor,
            )

        def _batch_eval(pts: list) -> list[tuple]:
            """Return [(x, err), ...] for *pts*, using multiprocessing if enabled."""
            if _mode["use_mp"]:
                errs = list(_ps_pool.map(ps_obj, pts))
                tracker["nfev"] += len(pts)
                return list(zip(pts, errs))
            out = []
            for xp in pts:
                out.append((xp, obj(xp)))
            return out

        _ps_pool = (
            ProcessPoolExecutor(max_workers=n_workers) if _mode["use_mp"] else None
        )
        try:
            while tracker["nfev"] < opt_config.max_fev and delta > opt_config.x_tol:
                if stop_flag and stop_flag():
                    break

                improved = False

                # --- optional random search step ---
                if opt_config.ps_search_pts > 0:
                    search_pts = [
                        lb_arr + rng.random(n_active) * (ub_arr - lb_arr)
                        for _ in range(opt_config.ps_search_pts)
                    ]
                    results = _batch_eval(search_pts)
                    best_x, best_f = min(results, key=lambda r: r[1])
                    if best_f < f_curr:
                        # Re-evaluate in main process to get simulation output
                        p_si = _to_si_dict(best_x)
                        err_main, out_main = _eval_and_sim(p_si, cfg)
                        if err_main < tracker["best_err"]:
                            tracker["best_err"] = err_main
                            tracker["best_params"] = dict(p_si)
                            tracker["best_out"] = out_main
                            if not _mode["use_mp"]:
                                tracker["nfev"] += 1  # obj() already counted for mp
                        x_curr = best_x.copy()
                        f_curr = best_f
                        improved = True

                # --- poll step ---
                skip_poll = improved and not opt_config.ps_complete_poll and not _mode["use_mp"]
                if not skip_poll:
                    # Direction ordering: last successful first, then shuffled rest
                    if last_success is not None:
                        order = [last_success] + [
                            k for k in range(len(poll_dirs)) if k != last_success
                        ]
                    else:
                        order = list(range(len(poll_dirs)))
                        rng.shuffle(order)

                    if _mode["use_mp"]:
                        # Parallel: evaluate all directions at once
                        trial_pts = []
                        for k in order:
                            i, sign = poll_dirs[k]
                            x_try = x_curr.copy()
                            x_try[i] += sign * delta * ps_scale[i]
                            trial_pts.append(np.clip(x_try, lb_arr, ub_arr))
                        results = _batch_eval(trial_pts)
                        # Detect silent worker failure: all results are the
                        # sentinel error value — workers likely cannot import
                        # the simulation module in the subprocess.
                        if all(r[1] >= 9e9 for r in results):
                            _mode["use_mp"] = False   # fall back to sequential
                        best_idx = int(np.argmin([r[1] for r in results]))
                        best_x, best_f = results[best_idx]
                        if best_f < f_curr:
                            p_si = _to_si_dict(best_x)
                            err_main, out_main = _eval_and_sim(p_si, cfg)
                            if err_main < tracker["best_err"]:
                                tracker["best_err"] = err_main
                                tracker["best_params"] = dict(p_si)
                                tracker["best_out"] = out_main
                            x_curr = best_x.copy()
                            f_curr = best_f
                            last_success = order[best_idx]
                            improved = True
                        else:
                            last_success = None
                    else:
                        # Sequential: opportunistic or complete
                        for k in order:
                            i, sign = poll_dirs[k]
                            x_try = x_curr.copy()
                            x_try[i] += sign * delta * ps_scale[i]
                            x_try = np.clip(x_try, lb_arr, ub_arr)
                            f_try = obj(x_try)
                            if f_try < f_curr:
                                x_curr = x_try.copy()
                                f_curr = f_try
                                last_success = k
                                improved = True
                                if not opt_config.ps_complete_poll:
                                    break
                            if tracker["nfev"] >= opt_config.max_fev:
                                break

                # --- mesh update ---
                if improved:
                    delta = min(delta * opt_config.ps_mesh_expansion,
                                opt_config.ps_initial_mesh * 4.0)
                else:
                    last_success = None
                    delta *= opt_config.ps_mesh_contraction

                # delta == 0 means floating-point underflow (too many consecutive
                # failures) — no trial point can differ from x_curr, stop now.
                if delta == 0.0:
                    break

                _emit_progress(step_size=delta,
                               status="improved" if improved else "refine")
        finally:
            if _ps_pool is not None:
                _ps_pool.shutdown(wait=False)

        result = None   # tracker holds the best

    # ---- Differential Evolution ----
    elif algo == "differential-evolution":
        n_workers = max(1, opt_config.n_workers)

        de_kwargs = dict(
            bounds=opt_bounds,
            strategy=opt_config.de_strategy,
            maxiter=opt_config.de_maxiter,
            popsize=opt_config.de_popsize,
            mutation=opt_config.de_mutation,
            recombination=opt_config.de_recombination,
            tol=opt_config.f_tol,
            polish=False,
            updating="deferred",
        )

        try:
            if n_workers > 1 and cfg.mp_make_sim is not None:
                # True multiprocessing via scipy's internal ProcessPoolExecutor.
                # Objective must be picklable — use _DEObjFn (module-level class).
                # Progress is reported from the callback (main process only).
                de_obj = _DEObjFn(
                    active=active, fixed=fixed,
                    mp_make_sim=cfg.mp_make_sim,
                    t_exp=cfg.t_exp, R_exp=cfg.R_exp,
                    tspan_factor=cfg.tspan_factor,
                )

                def _de_cb_mp(xk, convergence=0):
                    if stop_flag and stop_flag():
                        return True
                    # Evaluate current best in main process to get sim output.
                    params_si = _to_si_dict(xk)
                    err, out = _eval_and_sim(params_si, cfg)
                    if err < tracker["best_err"]:
                        tracker["best_err"] = err
                        tracker["best_params"] = dict(params_si)
                        tracker["best_out"] = out
                    _emit_progress()
                    return False

                result = differential_evolution(
                    de_obj, workers=n_workers, callback=_de_cb_mp, **de_kwargs,
                )
                tracker["nfev"] = result.nfev
            else:
                # Single-worker path — closure obj() tracks improvements directly.
                def _de_cb(xk, convergence=0):
                    if stop_flag and stop_flag():
                        return True
                    _emit_progress()
                    return False

                result = differential_evolution(
                    obj, workers=1, callback=_de_cb, **de_kwargs,
                )
        except _UserStop:
            result = None

    # ---- CMA-ES ----
    elif algo == "cma-es":
        if not _HAS_CMA:
            raise RuntimeError(
                "CMA-ES requires the 'cma' package: pip install cma"
            )
        # Normalise x0 to [0,1] space to make sigma0 universal
        span = ub_arr - lb_arr
        span[span == 0] = 1.0

        def _obj_norm(theta_norm):
            return obj(lb_arr + theta_norm * span)

        x0_norm = (x0 - lb_arr) / span
        bounds_norm = [[0.0] * len(active), [1.0] * len(active)]

        cma_opts = _cma.CMAOptions()
        cma_opts["bounds"] = bounds_norm
        cma_opts["maxfevals"] = opt_config.cma_maxfev
        cma_opts["tolx"] = opt_config.x_tol
        cma_opts["tolfun"] = opt_config.f_tol
        cma_opts["verbose"] = -9   # silent

        try:
            es = _cma.CMAEvolutionStrategy(x0_norm, opt_config.cma_sigma0, cma_opts)
            while not es.stop():
                if stop_flag and stop_flag():
                    break
                solutions = es.ask()
                fitvals = [_obj_norm(s) for s in solutions]
                es.tell(solutions, fitvals)
                _emit_progress(status="improved" if es.result.fbest < tracker["best_err"] else "refine")
        except Exception:
            pass
        result = None   # tracker holds the best

    # ---- Dual Annealing ----
    elif algo == "dual-annealing":
        try:
            result = dual_annealing(
                obj,
                bounds=opt_bounds,
                x0=x0,
                maxfun=opt_config.da_maxfev,
                initial_temp=opt_config.da_initial_temp,
                restart_temp_ratio=opt_config.da_restart_temp,
                minimizer_kwargs={"method": "L-BFGS-B", "bounds": opt_bounds},
                callback=lambda x, f, ctx: bool(stop_flag and stop_flag()),
            )
        except _UserStop:
            result = None

    # ---- Basin Hopping ----
    elif algo == "basin-hopping":
        def _bh_cb(x, f, accepted):
            if stop_flag and stop_flag():
                raise _UserStop()
            _emit_progress()

        try:
            result = basinhopping(
                obj,
                x0,
                niter=opt_config.bh_n_iter,
                stepsize=opt_config.bh_stepsize,
                minimizer_kwargs={"method": "L-BFGS-B", "bounds": opt_bounds,
                                  "options": {"maxfun": opt_config.max_fev}},
                callback=_bh_cb,
            )
        except _UserStop:
            result = None

    else:
        raise ValueError(f"Unknown optimisation method: {opt_config.method!r}")

    # --- best result -----------------------------------------------------
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
    if best_out is None:
        _, best_out = _eval_and_sim(best_params, cfg)

    return FitResult(
        best_params=best_params,
        lsq_err=best_err,
        nfev=tracker["nfev"],
        t_sim=best_out.t_sim if best_out else None,
        R_sim=best_out.R_sim if best_out else None,
        Rmax_sim=best_out.Rmax_sim if best_out else None,
        tc=best_out.tc if best_out else None,
        sim_out=best_out,
    )
