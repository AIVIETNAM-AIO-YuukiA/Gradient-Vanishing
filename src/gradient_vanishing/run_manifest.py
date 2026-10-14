"""Atomic saved-result and run-manifest persistence, without experiment imports."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping


NOTEBOOK_REFERENCE_SHA256 = "71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_directory(root: str | Path, stage: str) -> tuple[str, Path]:
    """Create one unique run directory without consuming experiment RNG."""
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    run_id = f"{stage}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{time.time_ns()}-{os.getpid()}"
    path = base / run_id
    path.mkdir(exist_ok=False)
    return run_id, path


def _json_safe(value: Any) -> Any:
    """Use the existing diagnostic string convention for nonfinite floats."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def atomic_json(path: str | Path, value: Any) -> None:
    """Replace a JSON file only after a complete, strict serialization."""
    target = Path(path)
    payload = json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         prefix=f".{target.name}.", suffix=".tmp",
                                         dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_entry(run_dir: Path, file: Path, kind: str) -> dict[str, Any]:
    return {"kind": kind, "path": file.relative_to(run_dir).as_posix(),
            "sha256": file_sha256(file), "bytes": file.stat().st_size}


def source_revision() -> str | None:
    try:
        from gradient_vanishing._build_info import SOURCE_REVISION
    except ImportError:
        SOURCE_REVISION = None
    if SOURCE_REVISION is not None:
        return SOURCE_REVISION
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                            text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def notebook_file_hash() -> str | None:
    notebook = Path(__file__).resolve().parents[2] / "notebooks" / "GradientVanishingAnalysis.ipynb"
    return file_sha256(notebook) if notebook.exists() else None


def environment_metadata() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in ("torch", "optuna", "numpy", "matplotlib"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    torch = sys.modules.get("torch")
    return {"python_version": sys.version.split()[0], "dependencies": versions,
            "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
            "cuda_available": torch.cuda.is_available() if torch is not None else None,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()
                if torch is not None else None,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def base_manifest(run_id: str, stage: str, command: list[str], config: Mapping[str, Any],
                  overrides: Mapping[str, Any], scope: str) -> dict[str, Any]:
    if stage == "gradients":
        seeds = list(range(config["gradients"]["initial_seeds"]))
        models = list(config["model_names"])
        lengths = list(config["sequence_lengths"])
    elif stage == "ablation":
        seeds = list(config["seeds"])
        models = ["Vanilla RNN", "LSTM", "GRU"]
        lengths = list(config["ablation"]["sequence_lengths"])
    elif stage == "sweep":
        seeds = list(range(config["gradients"]["sweep_seeds"]))
        models = ["LSTM", "GRU"]
        lengths = [config["gradients"]["sweep_sequence_length"]]
    elif stage == "main":
        seeds = list(config["seeds"])
        models = list(config["model_names"])
        lengths = list(config["sequence_lengths"])
    else:
        seeds = models = lengths = None  # Report inputs, not its CLI config, define these.
    return {
        "schema_version": 1, "run_id": run_id, "experiment_type": stage,
        "stage": stage, "command": list(command), "status": "running",
        "started_at": utc_now(), "completed_at": None,
        "effective_configuration": dict(config), "configuration_overrides": dict(overrides),
        "configuration_file": None,
        "scope": scope, "seeds": seeds,
        "model_names": models,
        "sequence_lengths": lengths,
        "initialization_methods": list(config["tuning"]["initializations"]),
        "gate_mode": config["main_gate_mode"],
        "ablation_gate_modes": list(config["ablation"]["gate_modes"]),
        "ablation_sequence_lengths": list(config["ablation"]["sequence_lengths"]),
        "retention_bias_values": list(config["gradients"]["sweep_biases"]),
        "initial_gradient_seed_count": config["gradients"]["initial_seeds"],
        "sweep_seed_count": config["gradients"]["sweep_seeds"],
        "gate_bias_parameters": {"LSTM": "W_f.bias = b", "GRU": "W_z.bias = -b"},
        "default_hyperparameters": dict(config["default_hyperparameters"]),
        "training_updates": config["training"]["updates"],
        "tuning_updates": config["training"]["tuning_updates"],
        "optuna_trials": config["tuning"]["trials"],
        "requested_device": config["device"], "resolved_device": None,
        "dtype": None, "environment": environment_metadata(),
        "git_commit": source_revision(),
        "notebook_reference_sha256": NOTEBOOK_REFERENCE_SHA256,
        "notebook_file_sha256": notebook_file_hash(),
        "artifacts": [], "input_sources": [], "selected_parameters": None,
        "error": None,
    }
