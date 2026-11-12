"""Model registry: name -> model class. Add a model = one module + one entry here."""

from __future__ import annotations

from pdm_agents.models.lgbm import LGBMModel
from pdm_agents.models.sequence import SequenceModel

MODEL_REGISTRY: dict[str, type] = {
    "lgbm": LGBMModel,
    "sequence": SequenceModel,
}


def get_model_class(name: str) -> type:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown model {name!r}; known: {sorted(MODEL_REGISTRY)}") from exc
