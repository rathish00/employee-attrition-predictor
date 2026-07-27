"""Central configuration for the attrition prediction pipeline.

All paths, hyperparameters, and column definitions live here — nothing
downstream hardcodes a path string or a magic number. Load once via
``Config.load()`` and pass it through the pipeline.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class Paths:
    """Filesystem locations, resolved relative to the project root."""

    raw_data: Path
    cleaned_data: Path
    model_file: Path
    schema_file: Path
    eda_output_dir: Path
    model_output_dir: Path

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "Paths":
        return cls(**{k: PROJECT_ROOT / v for k, v in d.items()})


@dataclass(frozen=True)
class ModelConfig:
    """Training hyperparameters and evaluation settings."""

    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    random_forest_params: dict[str, Any] = field(default_factory=dict)
    gradient_boosting_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    """Top-level config object for the whole pipeline."""

    paths: Paths
    model: ModelConfig
    target_column: str
    id_columns: list[str]
    constant_columns: list[str]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> "Config":
        """Load config from a YAML file.

        Raises:
            FileNotFoundError: if the config file doesn't exist.
            KeyError: if a required top-level section is missing.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found at {path}. "
                "Expected a config.yaml at the project root."
            )
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        for required in ("paths", "model", "target_column", "id_columns", "constant_columns"):
            if required not in raw:
                raise KeyError(f"config.yaml is missing required section: '{required}'")

        return cls(
            paths=Paths.from_dict(raw["paths"]),
            model=ModelConfig(**raw["model"]),
            target_column=raw["target_column"],
            id_columns=raw["id_columns"],
            constant_columns=raw["constant_columns"],
        )


def get_config() -> Config:
    """Convenience accessor honoring an ATTRITION_CONFIG env override (useful for tests/CI)."""
    override = os.environ.get("ATTRITION_CONFIG")
    return Config.load(override) if override else Config.load()
