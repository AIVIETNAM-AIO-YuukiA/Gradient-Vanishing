"""Notebook hidden-state gradient measurements and numeric diagnostics.

The measured quantity at step t is mean_b ||dL/dh[t,b,:]||_2, where L is
mean-reduced MSE of the final prediction. It omits LSTM cell-state paths.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from gradient_vanishing.config import load_config
from gradient_vanishing.data import generate_memorization_task
from gradient_vanishing.models import TrackerBase, build_model
from gradient_vanishing.training import NOTEBOOK_SHA256, TrainingResult, train_model


MEASUREMENT = "mean over batch of L2 norm across hidden units of d(final-step mean MSE)/dh_t"


@dataclass
class HiddenGradientMeasurement:
    gradients: tuple[torch.Tensor, ...]  # t=0 through T-1, each (batch, hidden)
    norms: list[float]
    loss: float


@dataclass
class InitialGradientResult:
    model: str
    sequence_length: int
    seeds: list[int]
    configuration: dict[str, Any]
    hyperparameters: dict[str, Any]
    gate_mode: str | int | float
    device: str
    dtype: str
    python_version: str
    torch_version: str
    timestep: list[int]
    temporal_distance: list[int]
    raw_norms: list[list[float]]
    normalized_curves: list[list[float]]
    geometric_mean: list[float]
    decay_mean: float
    decay_sample_std: float
    fit_timesteps: list[int]
    zero_ratio_count: int
    nonfinite_ratio_count: int
    model_training_mode: bool
    final_torch_rng_sha256: str
    measurement: str = MEASUREMENT
    notebook_sha256: str = NOTEBOOK_SHA256


@dataclass
class TrainingGradientResult:
    run: TrainingResult
    capture_updates: list[int]
    raw_norms: dict[int, list[float]]
    log10_relative_norms: dict[int, list[float]]
    timestep: list[int]
    temporal_distance: list[int]
    final_torch_rng_sha256: str
    measurement: str = MEASUREMENT


def measure_hidden_gradients(
    model: TrackerBase, x: torch.Tensor, y: torch.Tensor,
) -> HiddenGradientMeasurement:
    """Run the notebook's forward/MSE/backward and read retained hidden grads.

    The model's mode and parameters are left unchanged. No clipping is applied.
    Call with a fresh model or zero existing parameter gradients first.
    """
    loss = F.mse_loss(model(x), y)
    loss.backward()
    gradients = tuple(state.grad.detach().clone() for state in model.h_states)
    return HiddenGradientMeasurement(gradients, model.get_gradient_norms(), loss.item())


def decay_factor_from_curves(curves: np.ndarray, sequence_length: int) -> tuple[float, float]:
    """Notebook's per-seed log-linear fit on t=T//2,...,T-2, then mean/std."""
    t = np.arange(sequence_length // 2, sequence_length - 1)
    rates = [np.exp(-np.polyfit(t, np.log(np.clip(curve[t], 1e-300, None)), 1)[0])
             for curve in curves]
    return float(np.mean(rates)), float(np.std(rates, ddof=1))


def measure_init_gradients(
    name: str,
    sequence_length: int,
    config: Mapping[str, Any] | None = None,
    *,
    gate_mode: str | int | float | None = None,
    n_seeds: int | None = None,
    hyperparameters: Mapping[str, Any] | None = None,
    device: str | torch.device | None = None,
) -> InitialGradientResult:
    """Measure float64 initial curves with the notebook's global RNG order."""
    effective_config = copy.deepcopy(config if config is not None else load_config())
    params = dict(hyperparameters) if hyperparameters is not None else {
        "lr": effective_config["default_hyperparameters"]["learning_rate"],
        "hidden_size": effective_config["default_hyperparameters"]["hidden_size"],
        "init_method": effective_config["default_hyperparameters"]["initialization"],
    }
    mode = effective_config["main_gate_mode"] if gate_mode is None else gate_mode
    count = effective_config["gradients"]["initial_seeds"] if n_seeds is None else n_seeds
    requested_device = effective_config["device"] if device is None else device
    selected_device = torch.device(
        "cuda" if requested_device == "auto" and torch.cuda.is_available() else
        "cpu" if requested_device == "auto" else requested_device
    )
    if count < 2:
        raise ValueError("At least two seeds are needed for sample standard deviation")
    if sequence_length < 5:
        raise ValueError("Sequence length must provide at least two decay-fit timesteps")
    torch.use_deterministic_algorithms(True)

    raw_norms: list[list[float]] = []
    curves: list[np.ndarray] = []
    for seed in range(count):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = build_model(name, params, mode, sequence_length,
                            dtype=torch.float64, device=selected_device)
        x, y = generate_memorization_task(effective_config["training"]["batch_size"],
                                          sequence_length, device=selected_device)
        measurement = measure_hidden_gradients(model, x.double(), y.double())
        norms = np.array(measurement.norms)
        raw_norms.append(measurement.norms)
        with np.errstate(divide="ignore", invalid="ignore"):
            curves.append(norms / norms[-1])

    seed_curves = np.array(curves)
    geometric_mean = np.exp(np.log(np.clip(seed_curves, 1e-300, None)).mean(0))
    decay_mean, decay_std = decay_factor_from_curves(seed_curves, sequence_length)
    return InitialGradientResult(
        model=name, sequence_length=sequence_length, seeds=list(range(count)),
        configuration=effective_config, hyperparameters=params, gate_mode=mode,
        device=str(selected_device), dtype="torch.float64",
        python_version=sys.version.split()[0], torch_version=torch.__version__,
        timestep=list(range(sequence_length)),
        temporal_distance=list(range(sequence_length - 1, -1, -1)),
        raw_norms=raw_norms, normalized_curves=seed_curves.tolist(),
        geometric_mean=geometric_mean.tolist(), decay_mean=decay_mean,
        decay_sample_std=decay_std,
        fit_timesteps=list(range(sequence_length // 2, sequence_length - 1)),
        zero_ratio_count=int(np.count_nonzero(seed_curves == 0)),
        nonfinite_ratio_count=int(np.count_nonzero(~np.isfinite(seed_curves))),
        model_training_mode=model.training,
        final_torch_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
    )


def training_log10_relative_norms(norms: list[float]) -> list[float]:
    """Notebook heatmap numbers: log10(clip(g/(g_last+1e-30), 1e-10, inf))."""
    values = np.asarray(norms)
    return np.log10(np.clip(values / (values[-1] + 1e-30), 1e-10, None)).tolist()


def capture_training_gradients(
    name: str,
    config: Mapping[str, Any],
    sequence_length: int,
    seed: int,
    *,
    updates: int | None = None,
    capture_updates: list[int] | tuple[int, ...] | None = None,
    gate_mode: str | int | float | None = None,
    hyperparameters: Mapping[str, Any] | None = None,
    device: str | torch.device | None = None,
) -> TrainingGradientResult:
    """Use the training loop's pre-clipping captures; add heatmap numbers."""
    effective_updates = config["training"]["updates"] if updates is None else updates
    captures = list(config["training"]["gradient_capture_updates"] if capture_updates is None
                    else capture_updates)
    if any(index < 0 or index >= effective_updates for index in captures):
        raise ValueError("capture updates must be valid zero-based update indices")
    run = train_model(name, config, sequence_length, seed, epochs=effective_updates,
                      gate_init=gate_mode, record_grad_epochs=captures,
                      hyperparameters=hyperparameters, device=device)
    norms = run.history["grad_norms"]
    return TrainingGradientResult(
        run=run, capture_updates=captures, raw_norms=norms,
        log10_relative_norms={index: training_log10_relative_norms(norms[index])
                              for index in captures},
        timestep=list(range(sequence_length)),
        temporal_distance=list(range(sequence_length - 1, -1, -1)),
        final_torch_rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
    )


def _json_numbers(value: Any) -> Any:
    """Represent IEEE nonfinite values explicitly in strict JSON output."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_numbers(item) for item in value]
    return value


def save_diagnostic_result(
    path: str | Path, result: InitialGradientResult | TrainingGradientResult,
) -> None:
    """Save numeric diagnostics and provenance without training or plotting."""
    if isinstance(result, InitialGradientResult):
        record = {"kind": "initial_hidden_gradients", **asdict(result)}
    else:
        record = {
            "kind": "training_hidden_gradients",
            "metadata": result.run.metadata,
            "history": result.run.history,
            "capture_updates": result.capture_updates,
            "raw_norms": result.raw_norms,
            "log10_relative_norms": result.log10_relative_norms,
            "timestep": result.timestep,
            "temporal_distance": result.temporal_distance,
            "model_training_mode": result.run.model.training,
            "final_torch_rng_sha256": result.final_torch_rng_sha256,
            "measurement": result.measurement,
        }
    Path(path).write_text(json.dumps(_json_numbers(record), indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")
