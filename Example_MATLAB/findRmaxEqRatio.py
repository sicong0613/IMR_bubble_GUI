from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat

try:
    import mat73
except ImportError:  # pragma: no cover - optional dependency
    mat73 = None


# Edit these values, then run this file directly.
MAT_FILES = [r"C:\Users\49531\Box\Individual_folder_Sicong_Wang\Projects\PDMS\2026_03_16\0_Rt_data\R_data_15_34_31_converted.mat"
,r"C:\Users\49531\Box\Individual_folder_Sicong_Wang\Projects\PDMS\2026_03_16\0_Rt_data\R_data_15_35_30_converted.mat"
,r"C:\Users\49531\Box\Individual_folder_Sicong_Wang\Projects\PDMS\2026_03_16\0_Rt_data\R_data_15_36_25_converted.mat"
,r"C:\Users\49531\Box\Individual_folder_Sicong_Wang\Projects\PDMS\2026_03_16\0_Rt_data\R_data_15_43_22_converted.mat"
,r"C:\Users\49531\Box\Individual_folder_Sicong_Wang\Projects\PDMS\2026_03_16\0_Rt_data\R_data_15_44_22_converted.mat"
,r"C:\Users\49531\Box\Individual_folder_Sicong_Wang\Projects\PDMS\2026_03_16\0_Rt_data\R_data_15_44_53_converted.mat"
    # r"Example_MATLAB\code_Rmax_as_beginning\data\05mW_01min_01.mat",
]
OUTPUT_CSV = "Rmax_Req_ratio.csv"
REQ_TAIL_POINTS = 15


def load_mat(path: Path) -> dict:
    try:
        return loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        if mat73 is None:
            raise
        return mat73.loadmat(path)


def read_r_field(path: Path) -> np.ndarray:
    mat = load_mat(path)
    if "R" not in mat:
        raise KeyError("MAT file does not contain field 'R'")
    r = np.asarray(mat["R"], dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size == 0:
        raise ValueError("R field contains no finite values")
    return r


def estimate_req(r: np.ndarray, n_tail: int = 15) -> tuple[float, int]:
    n = min(int(n_tail), int(r.size))
    if n <= 0:
        raise ValueError("R field is empty")
    return float(np.mean(r[-n:])), n


def estimate_rmax_quadratic_3pt(r: np.ndarray) -> tuple[float, int, str]:
    idx = int(np.argmax(r))
    raw_max = float(r[idx])
    if idx <= 0 or idx >= r.size - 1:
        return raw_max, idx, "raw_max_edge"

    y = r[idx - 1: idx + 2].astype(float)
    x = np.array([-1.0, 0.0, 1.0])
    try:
        a, b, c = np.polyfit(x, y, deg=2)
    except Exception:
        return raw_max, idx, "raw_max_polyfit_failed"

    if not np.isfinite(a) or abs(a) < 1e-300:
        return raw_max, idx, "raw_max_degenerate_quadratic"

    x_vertex = -b / (2.0 * a)
    if a >= 0.0 or x_vertex < -1.0 or x_vertex > 1.0:
        return raw_max, idx, "raw_max_vertex_outside"

    rmax = float(a * x_vertex**2 + b * x_vertex + c)
    if not np.isfinite(rmax):
        return raw_max, idx, "raw_max_nonfinite_vertex"
    return rmax, idx, "quadratic_3pt"


def iter_paths_from_list_file(path: Path) -> Iterable[Path]:
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        yield Path(text.strip("\"'"))


def collect_input_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for list_file in args.list or []:
        paths.extend(iter_paths_from_list_file(Path(list_file)))
    paths.extend(Path(p) for p in args.inputs)
    return paths


def analyse_file(path: Path, n_tail: int) -> dict:
    r = read_r_field(path)
    req, n_req = estimate_req(r, n_tail=n_tail)
    rmax, idx_max, method = estimate_rmax_quadratic_3pt(r)
    ratio = rmax / req if req != 0.0 else np.nan
    return {
        "file": str(path),
        "name": path.name,
        "n_R": int(r.size),
        "Req": req,
        "Req_n_tail": int(n_req),
        "Rmax": rmax,
        "Rmax_raw_index": int(idx_max),
        "Rmax_method": method,
        "Rmax_over_Req": ratio,
        "error": "",
    }


def write_csv(rows: list[dict], output: Path) -> None:
    fieldnames = [
        "file",
        "name",
        "n_R",
        "Req",
        "Req_n_tail",
        "Rmax",
        "Rmax_raw_index",
        "Rmax_method",
        "Rmax_over_Req",
        "error",
    ]
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read R from MAT files, estimate Req from the last N R values, "
            "estimate Rmax by 3-point quadratic interpolation, and export Rmax/Req."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="MAT files to analyse. You can also provide paths via --list.",
    )
    parser.add_argument(
        "--list",
        action="append",
        help="Text file containing one MAT path per line. Can be used multiple times.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="Rmax_Req_ratio.csv",
        help="Output CSV path. Default: Rmax_Req_ratio.csv",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=15,
        help="Number of final finite R values used to compute Req. Default: 15.",
    )
    args = parser.parse_args()

    paths = [Path(p) for p in MAT_FILES] if MAT_FILES else collect_input_paths(args)
    if not paths:
        parser.error("No MAT files provided. Edit MAT_FILES in this script or use input paths/--list.")
    output = Path(args.output if (args.inputs or args.list) else OUTPUT_CSV)
    n_tail = int(args.tail if (args.inputs or args.list) else REQ_TAIL_POINTS)

    rows: list[dict] = []
    for path in paths:
        try:
            rows.append(analyse_file(path, n_tail=n_tail))
        except Exception as exc:
            rows.append({
                "file": str(path),
                "name": path.name,
                "n_R": "",
                "Req": "",
                "Req_n_tail": "",
                "Rmax": "",
                "Rmax_raw_index": "",
                "Rmax_method": "",
                "Rmax_over_Req": "",
                "error": str(exc),
            })

    write_csv(rows, output)
    print(f"Wrote {len(rows)} row(s) to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
