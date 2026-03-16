from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from typing import List


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


def _load_model_json(filename: str) -> ConstitutiveModel:
    """Load a constitutive model from a JSON file in this package."""
    data = json.loads(files(__package__).joinpath(filename).read_text(encoding="utf-8"))
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
    )


def load_nhkv_model() -> ConstitutiveModel:
    return _load_model_json("nhkv.json")


def load_gmod_model() -> ConstitutiveModel:
    return _load_model_json("gmod.json")


AVAILABLE_MODELS = {
    "NHKV": load_nhkv_model,
    "GMOD": load_gmod_model,
}


__all__ = [
    "UnitOption", "ConstitutiveParameter", "ConstitutiveModel",
    "load_nhkv_model", "load_gmod_model", "AVAILABLE_MODELS",
]

