from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

try:
    import mat73
except ImportError:  # pragma: no cover - optional dependency
    mat73 = None


# Edit these values, then run this file directly.
# All .mat files directly inside INPUT_DIR will be processed.
INPUT_DIR = r"C:\Users\49531\Box\Individual_folder_Sicong_Wang\Projects\PDMS"
OUTPUT_DIR = r"C:\Users\49531\Box\Individual_folder_Sicong_Wang\Projects\PDMS\Sweep_result_with_legend"
OVERWRITE = False

# A legend is generated from every keyword that appears in the file name.
# Example: file name contains "Req" -> read variable Req from the .mat file
# and write legend = "Req = X".
KEYWORDS = [
    "Req",
    "U0",
    "Rmax",
    "G1",
    "G2",
    "G3",
    "GA",
    "GB",
    "GC",
    "G",
    "muB",
    "muC",
    "mu",
    "alphaA",
    "alphaB",
    "alphaC",
    "alpha",
    "beta",
    "lambda_Y",
    "damage_Y",
]

# Optional fallback variable names. The left side is the keyword in the file
# name; the list is searched in order inside the .mat file.
ALIASES = {
    "Req": ["Req", "R_eq", "Req_m", "Req_um"],
    "Rmax": ["Rmax", "Rmax_exp", "Rmax_sim", "Rmax_um"],
}

UNIT_SCALE = {
    # Store Req/Rmax in the legend as micrometers if the .mat value is in SI.
    "Req": ("um", 1e6),
    "Rmax": ("um", 1e6),
}

GREEK_LEGEND_NAMES = {
    "alpha": r"\alpha",
    "alphaA": r"\alphaA",
    "alphaB": r"\alphaB",
    "alphaC": r"\alphaC",
    "beta": r"\beta",
    "gamma": r"\gamma",
    "lambda": r"\lambda",
    "lambda_Y": r"\lambda_Y",
    "mu": r"\mu",
    "muB": r"\muB",
    "muC": r"\muC",
}


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


def discover_files() -> list[Path]:
    if not INPUT_DIR:
        raise ValueError("Set INPUT_DIR before running this script.")
    input_dir = Path(INPUT_DIR)
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"INPUT_DIR is not a folder: {input_dir}")
    return sorted(input_dir.glob("*.mat"))


def matched_keywords(path: Path) -> list[str]:
    name = path.stem.lower()
    # Longer keywords first avoids matching "G" before "G1" in display order.
    keywords = sorted(KEYWORDS, key=len, reverse=True)
    return [
        key
        for key in keywords
        if re.search(rf"(^|[^A-Za-z0-9]){re.escape(key)}($|[^A-Za-z0-9])", name, re.IGNORECASE)
    ]


def scalar_from_struct_best_fit(mat: dict, variable_names: list[str]):
    if "struct_best_fit" not in mat:
        return None, ""
    target_names = {name.lower() for name in variable_names}
    entries = np.asarray(mat["struct_best_fit"], dtype=object).reshape(-1)
    for entry in entries:
        if not hasattr(entry, "name") or not hasattr(entry, "value"):
            continue
        name = str(getattr(entry, "name")).strip()
        if name.lower() not in target_names:
            continue
        try:
            value = float(np.asarray(getattr(entry, "value")).reshape(-1)[0])
        except Exception:
            continue
        if np.isfinite(value):
            return value, f"struct_best_fit.{name}"
    return None, ""


def scalar_from_filename(path: Path, keyword: str):
    pattern = (
        rf"(^|[^A-Za-z0-9]){re.escape(keyword)}[^A-Za-z0-9]*"
        rf"([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
    )
    match = re.search(pattern, path.stem, re.IGNORECASE)
    if not match:
        return None, ""
    try:
        value = float(match.group(2))
    except Exception:
        return None, ""
    if not np.isfinite(value):
        return None, ""
    return value, "filename"


def scalar_from_mat(mat: dict, variable_names: list[str]):
    lower_to_key = {str(key).lower(): key for key in mat.keys()}
    for variable_name in variable_names:
        real_key = lower_to_key.get(variable_name.lower())
        if real_key is None:
            continue
        arr = np.asarray(mat[real_key])
        if arr.size == 0:
            continue
        try:
            values = np.asarray(arr, dtype=float).reshape(-1)
        except Exception:
            continue
        values = values[np.isfinite(values)]
        if values.size:
            return float(values[0]), str(real_key)
    value, source = scalar_from_struct_best_fit(mat, variable_names)
    if value is not None:
        return value, source
    return None, ""


def format_legend_value(keyword: str, value: float) -> str:
    unit, scale = UNIT_SCALE.get(keyword, ("", 1.0))
    legend_name = GREEK_LEGEND_NAMES.get(keyword, keyword)
    shown = value * scale
    if unit:
        return f"{legend_name} = {shown:.6g} {unit}"
    return f"{legend_name} = {shown:.6g}"


def build_legend(path: Path, mat: dict) -> tuple[str, list[str]]:
    parts: list[str] = []
    missing: list[str] = []
    for keyword in matched_keywords(path):
        variable_names = ALIASES.get(keyword, [keyword])
        value, _source_key = scalar_from_mat(mat, variable_names)
        if value is None:
            value, _source_key = scalar_from_filename(path, keyword)
        if value is None:
            missing.append(keyword)
            continue
        parts.append(format_legend_value(keyword, value))
    return ", ".join(parts), missing


def add_legend_to_file(path: Path, output_dir: Path) -> tuple[bool, str]:
    mat = load_mat(path)
    legend, missing = build_legend(path, mat)
    if not legend:
        detail = "no keyword matched"
        if missing:
            detail = "matched keyword(s), but variable(s) missing: " + ", ".join(missing)
        return False, f"{path.name}: skipped ({detail})"

    out_data = clean_mat_for_savemat(mat)
    out_data["legend"] = legend

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = path if OVERWRITE else output_dir / path.name
    savemat(out_path, out_data, do_compression=False)
    return True, f"{path.name}: legend = {legend}"


def main():
    files = discover_files()
    output_dir = Path(OUTPUT_DIR)
    ok = 0
    for path in files:
        try:
            changed, message = add_legend_to_file(path, output_dir)
            ok += int(changed)
            print(message)
        except Exception as exc:
            print(f"{path.name}: failed ({exc})")
    print(f"Done. Wrote legend to {ok}/{len(files)} file(s).")


if __name__ == "__main__":
    main()
