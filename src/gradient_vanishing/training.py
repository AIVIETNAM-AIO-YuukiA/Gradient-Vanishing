"""One seeded notebook-equivalent training run (notebook cell 13)."""

from __future__ import annotations

import copy
import random
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np
import torch
from torch.nn import functional as F

from gradient_vanishing.config import load_config
from gradient_vanishing.data import generate_memorization_task, get_eval_sets
from gradient_vanishing.models import TrackerBase, build_model


NOTEBOOK_SHA256 = "71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44"


class Trial(Protocol):
    def report(self, value: float, step: int) -> None: ...
    def should_prune(self) -> bool: ...


@dataclass
class TrainingResult:
    model: TrackerBase
    history: dict[str, Any]
    test_mse: float | None
    metadata: dict[str, Any]


@torch.no_grad()
def evaluate(model: TrackerBase, x: torch.Tensor, y: torch.Tensor) -> float:
    """Return MSE, including the notebook's unconditional train-mode side effect."""
    model.eval()
    loss = F.mse_loss(model(x), y).item()
    model.train()
    return loss


def train_model(
    name: str,
    config: Mapping[str, Any] | None,
    T: int,
    seed: int,
    epochs: int | None = None,
    gate_init: str | int | float | None = None,
    record_grad_epochs: tuple[int, ...] | list[int] = (),
    trial: Trial | None = None,
    *,
    hyperparameters: Mapping[str, Any] | None = None,
    device: str | torch.device | None = None,
    evaluate_test: bool = False,
) -> TrainingResult:
    """Train one final model; update indices in ``record_grad_epochs`` are zero-based.

    ``hyperparameters`` uses the notebook keys ``lr``, ``hidden_size`` and
    ``init_method`` so a later tuning stage can pass its selected values.
    """
    effective_config = copy.deepcopy(config if config is not None else load_config())
    training = effective_config["training"]
    params = dict(hyperparameters) if hyperparameters is not None else {
        "lr": effective_config["default_hyperparameters"]["learning_rate"],
        "hidden_size": effective_config["default_hyperparameters"]["hidden_size"],
        "init_method": effective_config["default_hyperparameters"]["initialization"],
    }
    epochs = training["updates"] if epochs is None else epochs
    gate_init = effective_config["main_gate_mode"] if gate_init is None else gate_init
    requested_device = effective_config["device"] if device is None else device
    selected_device = torch.device(
        "cuda" if requested_device == "auto" and torch.cuda.is_available() else
        "cpu" if requested_device == "auto" else requested_device
    )
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if evaluate_test and trial is not None:
        raise ValueError("test evaluation is unavailable during a trial")

    torch.use_deterministic_algorithms(True)
    # Notebook set_seed runs before constructing the model or requesting sets.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = build_model(name, params, gate_init, T, device=selected_device)
    if name == "GRU":
        initial_gate = {"parameter": "W_z.bias", "retention_bias_sign": -1,
                        "bias": model.W_z.bias.detach().tolist(), "paired_input_bias": None}
    elif name == "LSTM":
        initial_gate = {"parameter": "W_f.bias", "retention_bias_sign": 1,
                        "bias": model.W_f.bias.detach().tolist(),
                        "paired_input_bias": model.W_i.bias.detach().tolist() if gate_init == "chrono" else None}
    else:
        initial_gate = None
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    x_val, y_val = get_eval_sets(T, effective_config, device=selected_device)["val"]

    history: dict[str, Any] = {"train_loss": [], "val_epochs": [], "val_loss": [], "grad_norms": {}}
    for epoch in range(epochs):
        optimizer.zero_grad()
        x, y = generate_memorization_task(training["batch_size"], T, device=selected_device)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        if epoch in record_grad_epochs:
            history["grad_norms"][epoch] = model.get_gradient_norms()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        history["train_loss"].append(loss.item())
        if (epoch + 1) % training["evaluation_interval"] == 0 or epoch == epochs - 1:
            validation_loss = evaluate(model, x_val, y_val)
            history["val_epochs"].append(epoch + 1)
            history["val_loss"].append(validation_loss)
            if trial is not None:
                trial.report(validation_loss, epoch + 1)
                if trial.should_prune():
                    from optuna.exceptions import TrialPruned
                    raise TrialPruned()
    history["final_val"] = history["val_loss"][-1]

    test_mse = None
    if evaluate_test:
        # The notebook's test driver explicitly enters eval mode before evaluate.
        test_x, test_y = get_eval_sets(T, effective_config, device=selected_device)["test"]
        model.eval()
        test_mse = evaluate(model, test_x, test_y)
    metadata = {
        "configuration": effective_config,
        "hyperparameters": params,
        "model": name,
        "sequence_length": T,
        "seed": seed,
        "updates": epochs,
        "gate_mode": gate_init,
        "initial_gate": initial_gate,
        "gradient_capture_updates": list(record_grad_epochs),
        "batch_size": training["batch_size"],
        "evaluation_interval": training["evaluation_interval"],
        "evaluation_size": training["evaluation_size"],
        "test_evaluated": evaluate_test,
        "split_seeds": copy.deepcopy(effective_config["split_seeds"]),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "device": str(selected_device),
        "dtype": str(next(model.parameters()).dtype),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "notebook_sha256": NOTEBOOK_SHA256,
    }
    return TrainingResult(model, history, test_mse, metadata)
