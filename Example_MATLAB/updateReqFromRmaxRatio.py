from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

try:
    import mat73
except ImportError:  # pragma: no cover - optional dependency
    mat73 = None


# Edit these values, then run this file directly.
MAT_FILES = [r"C:\Users\49531\Box\Individual_folder_Sicong_Wang\Projects\PDMS\2026_03_16\0_Rt_data\R_data_15_50_11_converted.mat"
,r"C:\Users\49531\Box\Individual_folder_Sicong_Wang\Projects\PDMS\2026_03_16\0_Rt_data\R_data_15_50_45_converted.mat"
,r"C:\Users\49531\Box\Individual_folder_Sicong_Wang\Projects\PDMS\2026_03_16\0_Rt_data\R_data_15_51_35_converted.mat"
,r"C:\Users\49531\Box\Individual_folder_Sicong_Wang\Projects\PDMS\2026_03_16\0_Rt_data\R_data_15_52_26_converted.mat"
    # r"Example_MATLAB\code_Rmax_as_beginning\data\05mW_01min_01.mat",
]
RMAX_OVER_REQ = 1.75
OUTPUT_DIR = r"C:\Users\49531\Box\Individual_folder_Sicong_Wang\Projects\PDMS"


def load_mat(path: Path) -> dict:
    try:
        return loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        if mat73 is None:
            raise
        return mat73.loadmat(path)


def clean_mat_for_savemat(mat: dict) -> dict:
    return {
        key: value
        for key, value in mat.items()
        if not key.startswith("__")
    }


def read_r_field(mat: dict) -> np.ndarray:
    if "R" not in mat:
        raise KeyError("MAT file does not contain field 'R'")
    r = np.asarray(mat["R"], dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size == 0:
        raise ValueError("R field contains no finite values")
    return r


def estimate_rmax_quartic_5pt(r: np.ndarray) -> tuple[float, int, str]:
    idx = int(np.argmax(r))
    raw_max = float(r[idx])
    if idx < 2 or idx > r.size - 3:
        return raw_max, idx, "raw_max_edge"

    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = r[idx - 2: idx + 3].astype(float)
    try:
        coeff = np.polyfit(x, y, deg=4)
    except Exception:
        return raw_max, idx, "raw_max_polyfit_failed"

    if not np.all(np.isfinite(coeff)):
        return raw_max, idx, "raw_max_nonfinite_polyfit"

    deriv = np.polyder(coeff)
    candidates = [-2.0, 0.0, 2.0]
    for root in np.roots(deriv):
        if abs(root.imag) < 1e-10:
            xr = float(root.real)
            if -2.0 <= xr <= 2.0:
                candidates.append(xr)

    values = [(float(np.polyval(coeff, xr)), xr) for xr in candidates]
    values = [(val, xr) for val, xr in values if np.isfinite(val)]
    if not values:
        return raw_max, idx, "raw_max_no_valid_vertex"

    rmax, x_at_max = max(values, key=lambda item: item[0])
    if rmax < raw_max:
        return raw_max, idx, "raw_max_quartic_lower"
    method = f"quartic_5pt_x={x_at_max:.6g}"
    return float(rmax), idx, method


def update_one_file(path: Path, output_dir: Path, ratio: float) -> dict:
    if ratio <= 0.0:
        raise ValueError("RMAX_OVER_REQ must be positive")

    mat = load_mat(path)
    r = read_r_field(mat)
    rmax, idx_max, method = estimate_rmax_quartic_5pt(r)
    req = rmax / ratio

    out = clean_mat_for_savemat(mat)
    # The GUI loader recognizes R_eq, Req, and R1_eq. Set all three so the
    # updated files are unambiguous for both old and new workflows.
    out["R_eq"] = float(req)
    out["Req"] = float(req)
    out["R1_eq"] = float(req)
    out["Rmax_from_R"] = float(rmax)
    out["Rmax_over_Req_target"] = float(ratio)
    out["Rmax_raw_index"] = int(idx_max)
    out["Rmax_method"] = method

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / path.name
    savemat(out_path, out)

    return {
        "input": str(path),
        "output": str(out_path),
        "Rmax": rmax,
        "Req": req,
        "Rmax_over_Req": ratio,
        "Rmax_raw_index": idx_max,
        "Rmax_method": method,
    }


def main() -> int:
    if not MAT_FILES:
        raise SystemExit("Edit MAT_FILES in this script before running.")
    output_dir = Path(OUTPUT_DIR)
    rows = []
    for item in MAT_FILES:
        path = Path(item)
        try:
            row = update_one_file(path, output_dir, float(RMAX_OVER_REQ))
            rows.append(row)
            print(
                f"updated {path.name}: Rmax={row['Rmax']:.8g}, "
                f"Req={row['Req']:.8g}, method={row['Rmax_method']}"
            )
        except Exception as exc:
            print(f"failed {path}: {exc}")

    print(f"Done. Wrote {len(rows)} file(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
