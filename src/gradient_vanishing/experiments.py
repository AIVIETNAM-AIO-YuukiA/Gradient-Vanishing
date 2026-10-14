"""One notebook-equivalent Optuna study (notebook cell 15)."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from gradient_vanishing.config import load_config
from gradient_vanishing.training import NOTEBOOK_SHA256, train_model


@dataclass
class StudyResult:
    study: optuna.Study
    record: dict[str, Any]

    @property
    def best_params(self) -> dict[str, Any]:
        return self.study.best_params


class NoCompletedTrialError(ValueError):
    """Optuna has no best trial; ``result`` retains the partial study record."""

    def __init__(self, message: str, result: StudyResult) -> None:
        super().__init__(message)
        self.result = result


def suggest_hyperparameters(trial: optuna.Trial, config: Mapping[str, Any]) -> dict[str, Any]:
    """Suggest the notebook's common parameters in its original call order."""
    tuning = config["tuning"]
    return {
        "lr": trial.suggest_float("lr", tuning["learning_rate_min"],
                                  tuning["learning_rate_max"], log=tuning["learning_rate_log"]),
        "hidden_size": trial.suggest_categorical("hidden_size", tuning["hidden_sizes"]),
        "init_method": trial.suggest_categorical("init_method", tuning["initializations"]),
    }


def tune_model(
    name: str,
    sequence_length: int,
    config: Mapping[str, Any] | None = None,
    *,
    n_trials: int | None = None,
    tuning_updates: int | None = None,
    device: str | torch.device | None = None,
) -> StudyResult:
    """Run one TPE/median-pruner study and return its Optuna and numeric results.

    No study name is passed to Optuna: the notebook lets Optuna generate one.
    Trial failures follow Optuna's default behavior; uncaught exceptions stop
    ``optimize`` just as they do in the notebook.
    """
    effective_config = copy.deepcopy(config if config is not None else load_config())
    tuning = effective_config["tuning"]
    budget = tuning["trials"] if n_trials is None else n_trials
    updates = effective_config["training"]["tuning_updates"] if tuning_updates is None else tuning_updates
    if budget < 1 or updates < 1:
        raise ValueError("Trial and update budgets must be positive")
    tuning["trials"] = budget
    effective_config["training"]["tuning_updates"] = updates

    def objective(trial: optuna.Trial) -> float:
        params = suggest_hyperparameters(trial, effective_config)
        result = train_model(
            name, effective_config, sequence_length,
            seed=effective_config["split_seeds"]["tuning_offset"] + trial.number,
            epochs=updates, trial=trial, hyperparameters=params, device=device,
        )
        return result.history["final_val"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=tuning["sampler_seed"]),
        pruner=MedianPruner(n_startup_trials=tuning["pruner_startup_trials"],
                            n_warmup_steps=tuning["pruner_warmup_steps"]),
    )
    study.optimize(objective, n_trials=budget)
    distributions = study.trials[0].distributions
    selected_device = str(device) if device is not None else effective_config["device"]
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    record = {
        "schema_version": 1,
        "configuration": effective_config,
        "model": name,
        "sequence_length": sequence_length,
        "study_name": study.study_name,
        "direction": "minimize",
        "search_space": {
            "lr": {"kind": "float", "low": distributions["lr"].low,
                   "high": distributions["lr"].high, "log": distributions["lr"].log},
            "hidden_size": {"kind": "categorical", "choices": list(distributions["hidden_size"].choices)},
            "init_method": {"kind": "categorical", "choices": list(distributions["init_method"].choices)},
        },
        "sampler": {"type": "TPESampler", "seed": tuning["sampler_seed"]},
        "pruner": {"type": "MedianPruner", "n_startup_trials": tuning["pruner_startup_trials"],
                   "n_warmup_steps": tuning["pruner_warmup_steps"]},
        "trial_budget": budget,
        "updates_per_trial": updates,
        "trial_seed_offset": effective_config["split_seeds"]["tuning_offset"],
        "gate_mode": effective_config["main_gate_mode"],
        "rng_order": "suggest parameters, then reseed Python/NumPy/Torch per trial before model construction",
        "objective": "final post-update validation mean MSE",
        "trials": [
            {"number": trial.number,
             "seed": effective_config["split_seeds"]["tuning_offset"] + trial.number,
             "params": dict(trial.params), "value": trial.value,
             "state": trial.state.name.lower(),
             "intermediate_values": dict(trial.intermediate_values)}
            for trial in study.trials
        ],
        "best_trial_number": None,
        "best_value": None,
        "best_params": None,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "optuna_version": optuna.__version__,
        "device": str(selected_device),
        "dtype": "torch.float32",
        "notebook_sha256": NOTEBOOK_SHA256,
    }
    try:
        best = study.best_trial
    except ValueError as error:
        raise NoCompletedTrialError(str(error), StudyResult(study, record)) from error
    record["best_trial_number"] = best.number
    record["best_value"] = best.value
    record["best_params"] = dict(best.params)
    return StudyResult(study, record)


def save_study_result(path: str | Path, result: StudyResult) -> None:
    """Write the lightweight study record without training or a run manifest."""
    Path(path).write_text(json.dumps(result.record, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")
