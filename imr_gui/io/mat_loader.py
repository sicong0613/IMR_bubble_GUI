from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat


@dataclass(frozen=True)
class ExperimentData:
    t: NDArray[np.float64]  # seconds
    R: NDArray[np.float64]  # meters
    source_path: str
    t_key: str
    R_key: str
    P_inf: float | None = None  # Pa
    rho: float | None = None  # kg/m^3
    R_eq: float | None = None  # m (equilibrium radius)


def _is_numeric_array(x) -> bool:
    """True if *x* can be squeezed into a 1-D float array."""
    try:
        arr = np.asarray(x)
        if arr.dtype.kind == "O":
            return False
        arr = np.squeeze(arr)
        return arr.ndim == 1 and arr.size > 0
    except Exception:
        return False


def _as_1d_float(x) -> NDArray[np.float64]:
    arr = np.asarray(x)
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {arr.shape}")
    return arr.astype(float)


_T_PREFERRED = ["t", "t_exp", "time", "time_exp"]
_T_PREFIXES  = ["t_", "time"]

_R_PREFERRED = ["R", "R1", "R_exp", "R1_exp", "radius", "radius_exp"]
_R_PREFIXES  = ["r_", "r1", "radius"]


def _pick_key_numeric(
    namespace: dict,
    preferred: list[str],
    prefix_hints: list[str],
) -> str | None:
    """Find the best key for a numeric 1-D array in *namespace*.

    1. Exact match from *preferred* (case-insensitive).
    2. Prefix match from *prefix_hints* (case-insensitive), only if the value
       is a numeric array (rejects structs, cell arrays, etc.).
    """
    lower_map = {k.lower(): k for k in namespace}

    for p in preferred:
        real_key = lower_map.get(p.lower())
        if real_key is not None and _is_numeric_array(namespace[real_key]):
            return real_key

    for k in namespace:
        lk = k.lower()
        if any(lk.startswith(pfx) for pfx in prefix_hints):
            if _is_numeric_array(namespace[k]):
                return k

    return None


def _flatten_namespace(m: dict) -> dict:
    """Flatten top-level mat_struct objects into *prefix.field* entries.

    If a .mat file saves ``struct_best_fit`` containing fields ``t_exp``
    and ``R1_exp``, this returns ``{"struct_best_fit.t_exp": <array>, ...}``
    alongside the original top-level entries.
    """
    flat: dict = {}
    for k, v in m.items():
        if k.startswith("__"):
            continue
        flat[k] = v
        if hasattr(v, "_fieldnames"):
            for field in v._fieldnames:
                flat[f"{k}.{field}"] = getattr(v, field)
        elif isinstance(v, np.ndarray) and v.dtype.kind == "O" and v.ndim == 0:
            inner = v.item()
            if hasattr(inner, "_fieldnames"):
                for field in inner._fieldnames:
                    flat[f"{k}.{field}"] = getattr(inner, field)
    return flat


def _get_scalar_from(namespace: dict, candidates: list[str]) -> float | None:
    for name in candidates:
        if name in namespace:
            try:
                val = namespace[name]
                arr = np.asarray(val)
                if arr.dtype.kind == "O":
                    continue
                arr = arr.astype(float).ravel()
                if arr.size == 0:
                    continue
                return float(arr[0])
            except Exception:
                continue
    return None


def _find_rmax_time(t: NDArray[np.float64],
                    R: NDArray[np.float64],
                    n_pts: int = 7) -> float:
    """Return the sub-sample peak time by fitting a parabola to the *n_pts*
    points nearest to max(R), then solving for the vertex analytically.

    Mirrors MATLAB's ``Rmax_fit`` (which uses 4 points and index space);
    here we use *n_pts* points and operate directly in time space.

    Falls back to the raw argmax time if the fit is ill-conditioned
    (e.g. parabola opens upward, or fitted peak lies outside the window).
    """
    if R.size < 3:
        return float(t[np.argmax(R)])

    i_max = int(np.argmax(R))
    half  = n_pts // 2
    i_lo  = max(0, i_max - half)
    i_hi  = min(R.size - 1, i_max + half)

    # need at least 3 points for a degree-2 fit
    if (i_hi - i_lo + 1) < 3:
        return float(t[i_max])

    t_win = t[i_lo : i_hi + 1]
    R_win = R[i_lo : i_hi + 1]

    # polyfit returns [a, b, c] for  R = a*t^2 + b*t + c
    p = np.polyfit(t_win, R_win, 2)
    a, b = p[0], p[1]

    # parabola must open downward (a < 0) to have a maximum
    if a >= 0:
        return float(t[i_max])

    t_peak = -b / (2.0 * a)

    # sanity check: vertex must lie within the fitting window
    if not (t_win[0] <= t_peak <= t_win[-1]):
        return float(t[i_max])

    return float(t_peak)


def find_rmax_value(t: NDArray[np.float64],
                    R: NDArray[np.float64],
                    n_pts: int = 7) -> float:
    """Return the sub-sample peak R by evaluating the parabolic fit at its vertex.

    Uses the same windowed polyfit as ``_find_rmax_time``.  Falls back to
    ``max(R)`` if the fit is ill-conditioned.
    """
    if R.size < 3:
        return float(np.max(R))

    i_max = int(np.argmax(R))
    half  = n_pts // 2
    i_lo  = max(0, i_max - half)
    i_hi  = min(R.size - 1, i_max + half)

    if (i_hi - i_lo + 1) < 3:
        return float(R[i_max])

    t_win = t[i_lo : i_hi + 1]
    R_win = R[i_lo : i_hi + 1]

    p = np.polyfit(t_win, R_win, 2)
    a, b = p[0], p[1]

    if a >= 0:
        return float(R[i_max])

    t_peak = -b / (2.0 * a)

    if not (t_win[0] <= t_peak <= t_win[-1]):
        return float(R[i_max])

    return float(np.polyval(p, t_peak))


def load_experiment_mat(path: str) -> ExperimentData:
    """Load experimental data from a ``.mat`` file.

    Handles both flat layouts (``t``, ``R`` at top level) and nested
    MATLAB structs (fields inside a ``struct_best_fit``-like object).
    """
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    ns = _flatten_namespace(m)

    t_key = _pick_key_numeric(ns, _T_PREFERRED, _T_PREFIXES)
    R_key = _pick_key_numeric(ns, _R_PREFERRED, _R_PREFIXES)

    if t_key is None or R_key is None:
        avail = [k for k in ns if not k.startswith("__")]
        raise ValueError(
            f"Could not find time / radius arrays in .mat file.\n"
            f"Available keys: {avail}"
        )

    t = _as_1d_float(ns[t_key])
    R = _as_1d_float(ns[R_key])
    if t.shape[0] != R.shape[0]:
        raise ValueError(
            f"t and R must have same length.  "
            f"Got len(t)={t.shape[0]} (key={t_key!r}), "
            f"len(R)={R.shape[0]} (key={R_key!r})"
        )

    order = np.argsort(t)
    t = t[order]
    R = R[order]

    # Shift time so that the interpolated R-peak sits at t = 0,
    # matching MATLAB's fun_read_data / Rmax_fit convention.
    t_peak = _find_rmax_time(t, R, n_pts=7)
    t = t - t_peak

    P_inf = _get_scalar_from(ns, ["Pinf", "P_inf", "pinf"])
    rho   = _get_scalar_from(ns, ["rho", "density"])
    R_eq  = _get_scalar_from(ns, ["R_eq", "Req", "R1_eq"])

    return ExperimentData(
        t=t,
        R=R,
        source_path=path,
        t_key=t_key,
        R_key=R_key,
        P_inf=P_inf,
        rho=rho,
        R_eq=R_eq,
    )
