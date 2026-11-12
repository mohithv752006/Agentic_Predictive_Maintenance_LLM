"""Configuration loading: YAML files + environment variables + environment detection.

All tunable values live in ``server/configs/*.yaml``. Strings of the form
``${VAR}`` or ``${VAR:default}`` are substituted from the environment, which is
populated from ``server/.env`` locally or from Colab Secrets inside Colab.
"""

from __future__ import annotations

import copy
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

SERVER_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = SERVER_ROOT / "configs"
ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")

# Keys under ``paths`` that hold data/outputs (moved to Google Drive in Colab).
DATA_PATH_KEYS = ("raw", "parquet", "artifacts", "reports", "runtime")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Strict):
    raw: Path
    parquet: Path
    artifacts: Path
    reports: Path
    runtime: Path
    knowledge_base: Path


class LoggingConfig(_Strict):
    level: str = "INFO"
    json_format: bool = Field(default=False, alias="json")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class SplitConfig(_Strict):
    serial_fractions: dict[str, float]
    time_fractions: dict[str, float]

    @field_validator("serial_fractions", "time_fractions")
    @classmethod
    def _check_fractions(cls, v: dict[str, float]) -> dict[str, float]:
        if set(v) != {"train", "val", "test"}:
            raise ValueError("split fractions need exactly train/val/test keys")
        if abs(sum(v.values()) - 1.0) > 1e-6:
            raise ValueError("split fractions must sum to 1")
        return v


class DataConfig(_Strict):
    source: Literal["backblaze", "huggingface"] = "backblaze"
    drive_stats_page: str
    huggingface_dataset: str
    download_workers: int = 4
    years: list[int]
    drive_models: Literal["auto"] | list[str] = "auto"
    top_k_models: int = 4
    holdout_models: int = 1
    min_failures_per_model: int = 30
    window_days: int = 30
    min_window_obs: int = 10
    neg_pos_ratio: int = 20
    negative_stride_days: int = 7
    eval_max_windows: int = 400_000
    split: SplitConfig
    duckdb_memory_limit: str = "6GB"
    duckdb_threads: int = 8
    smart_attributes: list[int]
    smart_value_types: list[Literal["raw", "normalized"]]

    @property
    def smart_columns(self) -> list[str]:
        """Harmonized SMART column names kept from the raw CSVs."""
        return [f"smart_{a}_{t}" for a in self.smart_attributes for t in self.smart_value_types]


class LLMConfig(_Strict):
    base_url: str = ""
    api_key: str = ""
    model: str = "Qwen/Qwen3.5-4B"
    temperature: float = 0.1
    max_tokens: int = 800
    timeout_s: float = 60
    max_retries: int = 2
    enable_thinking: bool = False
    health_timeout_s: float = 10


class EmbeddingsConfig(_Strict):
    model: str
    device: str = "cpu"


class RagConfig(_Strict):
    case_top_k: int = 5
    same_model_only: bool = True
    knowledge_top_k: int = 3
    case_index_max_negatives: int = 50_000


class LGBMConfig(_Strict):
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 40
    num_threads: int = 8


class SequenceConfig(_Strict):
    kind: Literal["cnn", "lstm"] = "cnn"
    hidden: int = 64
    epochs: int = 15
    batch_size: int = 256
    learning_rate: float = 1e-3
    patience: int = 3
    max_windows: int = 150_000  # cap on train/val windows materialized as tensors


class CalibrationConfig(_Strict):
    method: Literal["isotonic", "platt"] = "isotonic"


class FusionConfig(_Strict):
    method: Literal["stacking", "log_odds", "majority_vote"] = "stacking"


class ModelsConfig(_Strict):
    lgbm: LGBMConfig
    sequence: SequenceConfig
    calibration: CalibrationConfig
    fusion: FusionConfig


class DecisionConfig(_Strict):
    cost_ratios: list[float]
    default_cost_ratio: float = 50
    escalate_if_uncertainty_above: float = 0.35
    escalate_if_agent_disagreement_above: float = 0.4
    fixed_threshold: float = 0.5


class DriftConfig(_Strict):
    psi_warn: float = 0.1
    psi_alert: float = 0.25
    bins: int = 10
    recalibration_window_days: int = 60
    sample_drives: int = 200
    min_family_weight: float = 0.1


class EvaluationConfig(_Strict):
    llm_sample_size: int = 500
    llm_sample_max_positives: int = 150
    bootstrap_samples: int = 1000
    far_levels: list[float] = [0.01, 0.05]
    coverage_levels: list[float] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ece_bins: int = 10
    llm_concurrency: int = 4


class ApiConfig(_Strict):
    cors_origins: list[str]
    page_size: int = 50


class FeatureFlags(_Strict):
    use_rag: bool = True
    use_case_retrieval: bool = True
    use_knowledge_retrieval: bool = True
    use_calibration: bool = True
    use_cost_policy: bool = True
    use_drift: bool = True
    use_planner: bool = True


class AgentSpec(_Strict):
    """One entry of ``agents.yaml``."""

    class_path: str = Field(alias="class")
    enabled: bool = True
    kind: Literal["numeric", "context", "retrieval", "global"]
    description: str = ""
    attributes: list[str] = []

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ActionSpec(_Strict):
    """One entry of ``actions.yaml``."""

    description: str = ""
    fixed_cost: float
    exposure_horizon: int | None = None
    exposure_fraction: float = 0.0
    delegates: bool = False


class AppConfig(_Strict):
    """Fully resolved configuration (default.yaml + agents.yaml + actions.yaml + overrides)."""

    env: Literal["auto", "local", "colab"] = "auto"
    resolved_env: Literal["local", "colab"] = "local"
    colab_base_dir: str
    base_dir: Path = SERVER_ROOT
    paths: PathsConfig
    logging: LoggingConfig
    data: DataConfig
    horizons: list[int]
    llm: LLMConfig
    embeddings: EmbeddingsConfig
    rag: RagConfig
    models: ModelsConfig
    decision: DecisionConfig
    drift: DriftConfig
    evaluation: EvaluationConfig
    api: ApiConfig
    features: FeatureFlags
    seeds: list[int]
    agents: dict[str, AgentSpec]
    actions: dict[str, ActionSpec]
    experiment: dict[str, Any] = {}

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    def enabled_agents(self) -> list[str]:
        """Names of enabled agents, honouring the RAG feature flags."""
        names = []
        for name, spec in self.agents.items():
            if not spec.enabled:
                continue
            if spec.kind == "retrieval":
                if not self.features.use_rag:
                    continue
                if name == "case_retrieval" and not self.features.use_case_retrieval:
                    continue
                if name == "knowledge_retrieval" and not self.features.use_knowledge_retrieval:
                    continue
            names.append(name)
        return names

    def numeric_agents(self) -> list[str]:
        """Enabled agents that own a SMART attribute family (and a family model)."""
        return [n for n in self.enabled_agents() if self.agents[n].kind == "numeric"]


def is_colab() -> bool:
    """True when running inside a Google Colab runtime."""
    if os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU") is not None:
        return True
    try:
        import google.colab  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    return True


def _load_colab_secrets(names: list[str]) -> None:
    """Copy selected Colab Secrets into ``os.environ`` (no-op outside Colab)."""
    try:
        from google.colab import userdata  # type: ignore[import-not-found]
    except ImportError:
        return
    for name in names:
        if os.environ.get(name):
            continue
        try:
            value = userdata.get(name)
        except Exception:  # secret missing or access not granted
            continue
        if value:
            os.environ[name] = value


def _interpolate(value: Any) -> Any:
    """Recursively substitute ``${VAR[:default]}`` in strings."""
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    if isinstance(value, str):

        def repl(m: re.Match[str]) -> str:
            return os.environ.get(m.group(1), m.group(2) or "")

        return ENV_PATTERN.sub(repl, value)
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` recursively updated with ``override`` (inputs untouched)."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(
    overrides: dict[str, Any] | None = None,
    config_dir: Path | None = None,
    env_file: Path | None = None,
) -> AppConfig:
    """Load and validate the full configuration.

    Args:
        overrides: nested dict deep-merged over the YAML (used by experiments,
            tests and runtime settings).
        config_dir: directory holding ``default.yaml``, ``agents.yaml``, ``actions.yaml``.
        env_file: ``.env`` file to load (defaults to ``server/.env`` if present).
    """
    config_dir = config_dir or CONFIG_DIR
    env_file = env_file or SERVER_ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)

    raw = _read_yaml(config_dir / "default.yaml")
    raw["agents"] = _read_yaml(config_dir / "agents.yaml").get("agents", {})
    raw["actions"] = _read_yaml(config_dir / "actions.yaml").get("actions", {})
    if overrides:
        raw = deep_merge(raw, overrides)

    requested_env = os.environ.get("PDM_ENV") or raw.get("env", "auto")
    if requested_env == "auto":
        resolved = "colab" if is_colab() else "local"
    else:
        resolved = requested_env
    if resolved == "colab":
        _load_colab_secrets(["LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "HF_TOKEN"])

    raw = _interpolate(raw)
    base_dir = Path(raw["colab_base_dir"]) if resolved == "colab" else SERVER_ROOT
    paths = dict(raw["paths"])
    for key in DATA_PATH_KEYS:
        p = Path(paths[key])
        paths[key] = p if p.is_absolute() else base_dir / p
    kb = Path(paths["knowledge_base"])
    paths["knowledge_base"] = kb if kb.is_absolute() else SERVER_ROOT / kb
    raw["paths"] = paths
    raw["resolved_env"] = resolved
    raw["base_dir"] = base_dir

    cfg = AppConfig.model_validate(raw)
    _validate_cross_refs(cfg)
    return cfg


def _validate_cross_refs(cfg: AppConfig) -> None:
    """Check references between config sections that pydantic cannot see."""
    for name, action in cfg.actions.items():
        if action.exposure_horizon is not None and action.exposure_horizon not in cfg.horizons:
            raise ValueError(
                f"action {name!r} uses exposure_horizon={action.exposure_horizon}, "
                f"not one of horizons {cfg.horizons}"
            )
    if not any(not a.delegates for a in cfg.actions.values()):
        raise ValueError("actions.yaml needs at least one non-delegating action")
    if cfg.decision.default_cost_ratio <= 0:
        raise ValueError("default_cost_ratio must be positive")


def ensure_dirs(cfg: AppConfig) -> None:
    """Create data/output directories if missing."""
    for key in DATA_PATH_KEYS:
        getattr(cfg.paths, key).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Process-wide default configuration (cached)."""
    return load_config()
