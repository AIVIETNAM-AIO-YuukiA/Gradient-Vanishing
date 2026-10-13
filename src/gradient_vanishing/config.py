"""Versioned JSON configuration for the notebook-equivalent experiments."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping


class ConfigError(ValueError):
    """An experiment configuration cannot be used as supplied."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConfigError(f"Cannot read JSON configuration {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConfigError(f"JSON configuration {path} must be an object")
    return value


def _merge(target: dict[str, Any], patch: Mapping[str, Any], prefix: str = "") -> None:
    for key, value in patch.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if key not in target:
            raise ConfigError(f"Unknown configuration key: {name}")
        if isinstance(target[key], dict):
            if not isinstance(value, dict):
                raise ConfigError(f"{name} must be an object")
            _merge(target[key], value, name)
        else:
            target[key] = value


def _fail(name: str, rule: str) -> None:
    raise ConfigError(f"{name} {rule}")


def _integer(value: Any, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        _fail(name, f"must be an integer >= {minimum}")


def _number(value: Any, name: str, minimum: float = 0, *, strict: bool = False) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or (value <= minimum if strict else value < minimum):
        _fail(name, f"must be a finite number {'>' if strict else '>='} {minimum}")


def _list(value: Any, name: str, check: Any, *, unique: bool = True) -> None:
    if not isinstance(value, list) or not value:
        _fail(name, "must be a nonempty list")
    for item in value:
        check(item, name)
    if unique and len(value) != len(set(value)):
        _fail(name, "must contain unique values")


def _choice(value: Any, name: str, choices: tuple[str, ...]) -> None:
    if not isinstance(value, str) or value not in choices:
        _fail(name, f"must be one of {choices}")


def _validate(config: dict[str, Any]) -> None:
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        _fail("schema_version", "must be 1")
    _list(config["sequence_lengths"], "sequence_lengths", lambda x, n: _integer(x, n, 1))
    _list(config["model_names"], "model_names", lambda x, n: _choice(x, n, ("Vanilla RNN", "LSTM", "GRU")))
    _list(config["seeds"], "seeds", lambda x, n: _integer(x, n))
    training = config["training"]
    for key in ("updates", "tuning_updates", "batch_size", "evaluation_interval", "evaluation_size"):
        _integer(training[key], f"training.{key}", 1)
    _list(training["gradient_capture_updates"], "training.gradient_capture_updates", lambda x, n: _integer(x, n), unique=True)
    if any(update >= training["updates"] for update in training["gradient_capture_updates"]):
        _fail("training.gradient_capture_updates", "must be below training.updates (zero-based)")
    for key, value in config["split_seeds"].items():
        _integer(value, f"split_seeds.{key}")
    hyper = config["default_hyperparameters"]
    _number(hyper["learning_rate"], "default_hyperparameters.learning_rate", strict=True)
    _integer(hyper["hidden_size"], "default_hyperparameters.hidden_size", 1)
    _choice(hyper["initialization"], "default_hyperparameters.initialization", ("default", "orthogonal", "xavier"))
    _choice(config["main_gate_mode"], "main_gate_mode", ("default", "zero", "one", "chrono"))
    tuning = config["tuning"]
    for key in ("trials", "pruner_startup_trials"):
        _integer(tuning[key], f"tuning.{key}", 1)
    for key in ("sampler_seed", "pruner_warmup_steps"):
        _integer(tuning[key], f"tuning.{key}")
    for key in ("learning_rate_min", "learning_rate_max"):
        _number(tuning[key], f"tuning.{key}", strict=True)
    if tuning["learning_rate_min"] >= tuning["learning_rate_max"]:
        _fail("tuning.learning_rate_min", "must be less than tuning.learning_rate_max")
    if type(tuning["learning_rate_log"]) is not bool:
        _fail("tuning.learning_rate_log", "must be a boolean")
    _list(tuning["hidden_sizes"], "tuning.hidden_sizes", lambda x, n: _integer(x, n, 1))
    _list(tuning["initializations"], "tuning.initializations", lambda x, n: _choice(x, n, ("default", "orthogonal", "xavier")))
    ablation = config["ablation"]
    _list(ablation["sequence_lengths"], "ablation.sequence_lengths", lambda x, n: _integer(x, n, 1))
    _list(ablation["gate_modes"], "ablation.gate_modes", lambda x, n: _choice(x, n, ("zero", "one", "chrono")))
    _number(ablation["validation_threshold"], "ablation.validation_threshold", strict=True)
    gradients = config["gradients"]
    for key in ("initial_seeds", "sweep_seeds", "sweep_sequence_length"):
        _integer(gradients[key], f"gradients.{key}", 1)
    _list(gradients["sweep_biases"], "gradients.sweep_biases", lambda x, n: _number(x, n, -math.inf), unique=True)
    _choice(config["device"], "device", ("auto", "cpu", "cuda"))
    if not isinstance(config["output_dir"], str) or not config["output_dir"].strip():
        _fail("output_dir", "must be a nonempty string")


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load defaults, apply a partial JSON file, then command overrides.

    Nested objects merge by key. The returned dictionary is independent of the
    package resource and of subsequent calls.
    """
    default_path = files("gradient_vanishing").joinpath("configs/default.json")
    config = _read_json(Path(str(default_path)))
    if path is not None:
        _merge(config, _read_json(Path(path)))
    if overrides is not None:
        if not isinstance(overrides, Mapping):
            raise ConfigError("Command overrides must be an object")
        _merge(config, overrides)
    _validate(config)
    return config


def prepare_runtime(config: Mapping[str, Any]) -> None:
    """Set the notebook's CUDA determinism prerequisite before Torch import.

    Call this at a future experiment entry point before importing model modules.
    Torch deterministic algorithms and seeds are configured by that entry point.
    """
    if config["device"] in ("auto", "cuda"):
        if "torch" in sys.modules:
            raise ConfigError("CUDA runtime must be prepared before importing torch")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and print effective experiment configuration")
    parser.add_argument("config", nargs="?", help="partial JSON configuration file")
    args = parser.parse_args()
    try:
        print(json.dumps(load_config(args.config), indent=2))
    except ConfigError as error:
        parser.exit(2, f"Configuration error: {error}\n")


if __name__ == "__main__":
    main()
