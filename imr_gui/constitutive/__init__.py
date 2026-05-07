from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
import sys
from typing import Callable, List


@dataclass(frozen=True)
class UnitOption:
    label: str
    factor: float


@dataclass(frozen=True)
class ConstitutiveParameter:
    name: str
    label: str
    units: List[UnitOption]
    default: float
    lb: float
    ub: float
    scale: str
    fit_default: bool


@dataclass(frozen=True)
class ConstitutiveModel:
    id: str
    display_name: str
    description: str
    parameters: List[ConstitutiveParameter]
    constants: dict  # non-fittable physics constants from JSON
    solver_entrypoint: str | None = None  # "module:function" for user plugins


def _model_from_json_data(data: dict) -> ConstitutiveModel:
    params: List[ConstitutiveParameter] = []
    for p in data.get("parameters", []):
        units_raw = p.get("units")
        units: List[UnitOption] = []
        if isinstance(units_raw, list):
            for u in units_raw:
                units.append(
                    UnitOption(
                        label=str(u.get("label", "")),
                        factor=float(u.get("factor", 1.0)),
                    )
                )
        else:
            # Backwards-compatible fallback: a single string unit
            unit_label = p.get("unit", "")
            units.append(UnitOption(label=unit_label, factor=1.0))

        params.append(
            ConstitutiveParameter(
                name=p["name"],
                label=p.get("label", p["name"]),
                units=units,
                default=float(p.get("default", 0.0)),
                lb=float(p.get("lb", 0.0)),
                ub=float(p.get("ub", 0.0)),
                scale=p.get("scale", "lin"),
                fit_default=bool(p.get("fit_default", True)),
            )
        )
    raw_const = data.get("constants", {})
    constants = {k: v for k, v in raw_const.items()
                 if not k.startswith("_") and isinstance(v, (int, float))}

    return ConstitutiveModel(
        id=data.get("id", "NHKV"),
        display_name=data.get("display_name", "NHKV"),
        description=data.get("description", ""),
        parameters=params,
        constants=constants,
        solver_entrypoint=data.get("solver_entrypoint"),
    )


def _load_model_json(filename: str) -> ConstitutiveModel:
    """Load a constitutive model from a JSON file in this package."""
    data = json.loads(files(__package__).joinpath(filename).read_text(encoding="utf-8"))
    return _model_from_json_data(data)


def _load_model_json_path(path: Path) -> ConstitutiveModel:
    data = json.loads(path.read_text(encoding="utf-8"))
    return _model_from_json_data(data)


def load_nhkv_model() -> ConstitutiveModel:
    return _load_model_json("nhkv.json")


def load_nhkv_rmax_model() -> ConstitutiveModel:
    return _load_model_json("nhkv_rmax.json")


def load_gmod1_model() -> ConstitutiveModel:
    return _load_model_json("gmod1.json")


def load_gmod2_model() -> ConstitutiveModel:
    return _load_model_json("gmod.json")


BUILTIN_MODELS = {
    "NHKV": load_nhkv_model,
    "NHKV (Rmax)": load_nhkv_rmax_model,
    "GMOD1": load_gmod1_model,
    "GMOD2": load_gmod2_model,
}


def _candidate_user_model_dirs() -> list[Path]:
    roots = [
        Path.cwd() / "user_models",
        Path(__file__).resolve().parents[2] / "user_models",
    ]
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except Exception:
            resolved = root
        if resolved not in seen:
            out.append(root)
            seen.add(resolved)
    return out


def discover_user_models() -> dict[str, Callable[[], ConstitutiveModel]]:
    models: dict[str, Callable[[], ConstitutiveModel]] = {}
    for root in _candidate_user_model_dirs():
        if not root.is_dir():
            continue
        root_str = str(root.resolve())
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        for path in sorted(root.glob("*.json")):
            try:
                model = _load_model_json_path(path)
            except Exception:
                continue
            if not model.id or not model.solver_entrypoint:
                continue
            models[model.id] = (lambda p=path: _load_model_json_path(p))
    return models


def load_available_models() -> dict[str, Callable[[], ConstitutiveModel]]:
    models = dict(BUILTIN_MODELS)
    models.update(discover_user_models())
    return models


AVAILABLE_MODELS = load_available_models()


__all__ = [
    "UnitOption", "ConstitutiveParameter", "ConstitutiveModel",
    "load_nhkv_model", "load_nhkv_rmax_model",
    "load_gmod1_model", "load_gmod2_model",
    "BUILTIN_MODELS", "AVAILABLE_MODELS", "discover_user_models", "load_available_models",
]

