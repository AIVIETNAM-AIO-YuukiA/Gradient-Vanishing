"""Notebook-equivalent tuning, main comparison, and ablation (cells 15, 17, 27)."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from gradient_vanishing.config import load_config
from gradient_vanishing.data import get_eval_sets
from gradient_vanishing.gradients import measure_init_gradients, training_log10_relative_norms
from gradient_vanishing.training import NOTEBOOK_SHA256, evaluate, train_model


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


@dataclass
class ComparisonResult:
    record: dict[str, Any]


class ComparisonInterruptedError(RuntimeError):
    """A comparison stopped; ``result`` retains completed and failed work."""

    def __init__(self, message: str, result: ComparisonResult) -> None:
        super().__init__(message)
        self.result = result


def run_main_comparison(
    config: Mapping[str, Any] | None = None,
    *,
    device: str | torch.device | None = None,
) -> ComparisonResult:
    """Run the notebook's length → model → default/tuned → seed comparison.

    Pass a validated reduced configuration for CPU checks. The checked-in
    configuration schedules the original nine studies and 90 final runs.
    """
    effective_config = copy.deepcopy(config if config is not None else load_config())
    selected_device = str(device) if device is not None else effective_config["device"]
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    lengths = effective_config["sequence_lengths"]
    names = effective_config["model_names"]
    seeds = effective_config["seeds"]
    training = effective_config["training"]
    captures = training["gradient_capture_updates"]
    default_params = {
        "lr": effective_config["default_hyperparameters"]["learning_rate"],
        "hidden_size": effective_config["default_hyperparameters"]["hidden_size"],
        "init_method": effective_config["default_hyperparameters"]["initialization"],
    }
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "configuration": effective_config,
        "model_order": list(names),
        "sequence_length_order": list(lengths),
        "condition_order": ["default", "tuned"],
        "seed_order": list(seeds),
        "results": {}, "best_params": {}, "studies": {},
        "gradient_heatmap": None,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "optuna_version": optuna.__version__,
        "device": selected_device,
        "dtype": "torch.float32",
        "notebook_sha256": NOTEBOOK_SHA256,
    }
    result = ComparisonResult(record)
    phase = "setup"
    length: int | None = None
    name: str | None = None
    condition: str | None = None
    seed: int | None = None
    try:
        for length in lengths:
            key = str(length)
            record["results"][key], record["best_params"][key], record["studies"][key] = {}, {}, {}
            # The notebook requests the fixed test pair before this length's studies.
            test_x, test_y = get_eval_sets(length, effective_config, device=selected_device)["test"]
            for name in names:
                record["results"][key][name] = {}
                phase, condition, seed = "tuning", None, None
                study = tune_model(name, length, effective_config, device=selected_device)
                record["studies"][key][name] = study.record
                tuned_params = dict(study.best_params)
                record["best_params"][key][name] = tuned_params
                for condition, params in (("default", default_params), ("tuned", tuned_params)):
                    bucket = {
                        "hyperparameters": dict(params),
                        "tuning_study_name": study.study.study_name if condition == "tuned" else None,
                        "tuning_best_trial_number": study.record["best_trial_number"] if condition == "tuned" else None,
                        "runs": [], "test_mean": None, "test_sample_std": None,
                    }
                    record["results"][key][name][condition] = bucket
                    for seed in seeds:
                        phase = "training"
                        run_record = {
                            "model": name, "sequence_length": length,
                            "seed": seed, "status": "running", "condition": condition,
                            "hyperparameters": dict(params),
                            "split_seeds": copy.deepcopy(effective_config["split_seeds"]),
                            "tuning_study_name": bucket["tuning_study_name"],
                            "tuning_best_trial_number": bucket["tuning_best_trial_number"],
                        }
                        bucket["runs"].append(run_record)
                        run = train_model(
                            name, effective_config, length, seed,
                            epochs=training["updates"], gate_init=effective_config["main_gate_mode"],
                            record_grad_epochs=captures, hyperparameters=params,
                            device=selected_device,
                        )
                        phase = "test_evaluation"
                        run_record.update({"history": run.history,
                                           "final_val": run.history["final_val"],
                                           "metadata": run.metadata})
                        test_mse = evaluate(run.model, test_x, test_y)
                        run_record.update({
                            "status": "complete", "test_mse": test_mse,
                            "model_training_mode_after_test": run.model.training,
                        })
                    tests = np.asarray([run["test_mse"] for run in bucket["runs"]], dtype=float)
                    bucket["test_mean"] = float(np.mean(tests))
                    bucket["test_sample_std"] = float(np.std(tests, ddof=1)) if len(tests) > 1 else None

        max_length = max(lengths)
        record["gradient_heatmap"] = {
            "sequence_length": max_length,
            "model_order": list(names),
            "capture_updates": list(captures),
            "log10_relative_norms": {
                str(index): [training_log10_relative_norms(
                    record["results"][str(max_length)][model]["tuned"]["runs"][0]["history"]["grad_norms"][index]
                ) for model in names]
                for index in captures
            },
        }
        record["status"] = "complete"
    except Exception as error:
        record["status"] = "incomplete"
        record["failure"] = {"phase": phase, "sequence_length": length,
                             "model": name, "condition": condition, "seed": seed,
                             "error_type": type(error).__name__, "message": str(error)}
        if isinstance(error, NoCompletedTrialError) and length is not None and name is not None:
            record["studies"][str(length)][name] = error.result.record
        if seed is not None and condition is not None and phase != "tuning":
            record["results"][str(length)][name][condition]["runs"][-1]["status"] = "failed"
        raise ComparisonInterruptedError(str(error), result) from error
    finally:
        record["final_torch_rng_sha256"] = hashlib.sha256(
            torch.get_rng_state().numpy().tobytes()).hexdigest()
    return result


def save_main_comparison(path: str | Path, result: ComparisonResult) -> None:
    """Write raw comparison results and provenance without retraining."""
    Path(path).write_text(json.dumps(result.record, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


@dataclass
class AblationResult:
    record: dict[str, Any]


class AblationInterruptedError(RuntimeError):
    """An ablation stopped; ``result`` retains completed and failed work."""

    def __init__(self, message: str, result: AblationResult) -> None:
        super().__init__(message)
        self.result = result


def epochs_to_threshold(history: Mapping[str, Any], threshold: float) -> int | None:
    """First checked validation update strictly below the notebook threshold."""
    return next((epoch for epoch, value in zip(history["val_epochs"], history["val_loss"])
                 if value < threshold), None)


def run_gate_ablation(
    config: Mapping[str, Any] | None = None,
    *,
    device: str | torch.device | None = None,
) -> AblationResult:
    """Run notebook cell 27's seven conditions, independently of main/tuning."""
    effective_config = copy.deepcopy(config if config is not None else load_config())
    selected_device = str(device) if device is not None else effective_config["device"]
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    ablation = effective_config["ablation"]
    training = effective_config["training"]
    params = {
        "lr": effective_config["default_hyperparameters"]["learning_rate"],
        "hidden_size": effective_config["default_hyperparameters"]["hidden_size"],
        "init_method": effective_config["default_hyperparameters"]["initialization"],
    }
    conditions = [("Vanilla RNN", "—", "default")]
    conditions.extend((name, mode, mode) for name in ("LSTM", "GRU")
                      for mode in ablation["gate_modes"])
    record: dict[str, Any] = {
        "schema_version": 1, "status": "running", "configuration": effective_config,
        "sequence_length_order": list(ablation["sequence_lengths"]),
        "condition_order": [{"model": name, "label": label, "gate_mode": mode}
                            for name, label, mode in conditions],
        "seed_order": list(effective_config["seeds"]),
        "threshold": ablation["validation_threshold"],
        "gradient_seed_count": effective_config["gradients"]["initial_seeds"],
        "hyperparameters": params, "results": {},
        "python_version": sys.version.split()[0], "torch_version": torch.__version__,
        "optuna_version": optuna.__version__, "device": selected_device,
        "training_dtype": "torch.float32", "initial_gradient_dtype": "torch.float64",
        "notebook_sha256": NOTEBOOK_SHA256,
    }
    result = AblationResult(record)
    length: int | None = None
    name: str | None = None
    label: str | None = None
    seed: int | None = None
    phase = "setup"
    try:
        for length in ablation["sequence_lengths"]:
            key = str(length)
            record["results"][key] = []
            # Cell 27 requests the fixed test set once before all conditions.
            phase, name, label, seed = "fixed_test_set", None, None, None
            test_x, test_y = get_eval_sets(length, effective_config, device=selected_device)["test"]
            for name, label, gate_mode in conditions:
                gate_parameter = ("W_z.bias" if name == "GRU" else
                                  "W_f.bias" if name == "LSTM" else None)
                retention_bias_sign = (-1 if name == "GRU" else
                                       1 if name == "LSTM" else None)
                condition = {
                    "model": name, "sequence_length": length, "label": label,
                    "gate_mode": gate_mode, "gate_parameter": gate_parameter,
                    "retention_bias_sign": retention_bias_sign,
                    "hyperparameters": dict(params),
                    "threshold": ablation["validation_threshold"],
                    "runs": [], "initial_gradients": None,
                    "test_mean": None, "test_sample_std": None,
                    "threshold_success_count": 0, "threshold_mean_success_update": None,
                    "decay_mean": None, "decay_sample_std": None, "status": "running",
                }
                record["results"][key].append(condition)
                for seed in effective_config["seeds"]:
                    phase = "training"
                    run_record = {
                        "model": name, "sequence_length": length, "label": label,
                        "gate_mode": gate_mode, "seed": seed,
                        "hyperparameters": dict(params),
                        "split_seeds": copy.deepcopy(effective_config["split_seeds"]),
                        "status": "running",
                    }
                    condition["runs"].append(run_record)
                    run = train_model(name, effective_config, length, seed,
                                      epochs=training["updates"], gate_init=gate_mode,
                                      hyperparameters=params, device=selected_device)
                    run_record.update({"history": run.history, "metadata": run.metadata,
                                       "final_val": run.history["final_val"],
                                       "initial_gate": run.metadata["initial_gate"],
                                       "first_threshold_update": epochs_to_threshold(
                                           run.history, ablation["validation_threshold"])})
                    phase = "test_evaluation"
                    test_mse = evaluate(run.model, test_x, test_y)
                    run_record.update({"test_mse": test_mse, "status": "complete",
                                       "model_training_mode_after_test": run.model.training})
                phase, seed = "initial_gradients", None
                initial = measure_init_gradients(
                    name, length, effective_config, gate_mode=gate_mode,
                    n_seeds=effective_config["gradients"]["initial_seeds"],
                    hyperparameters=params, device=selected_device,
                )
                condition["initial_gradients"] = asdict(initial)
                condition["decay_mean"] = initial.decay_mean
                condition["decay_sample_std"] = initial.decay_sample_std
                test_values = np.asarray([run["test_mse"] for run in condition["runs"]], dtype=float)
                condition["test_mean"] = float(np.mean(test_values))
                condition["test_sample_std"] = (float(np.std(test_values, ddof=1))
                                                if len(test_values) > 1 else None)
                successful = [run["first_threshold_update"] for run in condition["runs"]
                              if run["first_threshold_update"] is not None]
                condition["threshold_success_count"] = len(successful)
                condition["threshold_mean_success_update"] = (
                    float(np.mean(successful)) if successful else None)
                condition["status"] = "complete"
        record["status"] = "complete"
    except Exception as error:
        record["status"] = "incomplete"
        record["failure"] = {"phase": phase, "sequence_length": length,
                             "model": name, "label": label, "seed": seed,
                             "error_type": type(error).__name__, "message": str(error)}
        if phase in ("training", "test_evaluation"):
            record["results"][str(length)][-1]["runs"][-1]["status"] = "failed"
        raise AblationInterruptedError(str(error), result) from error
    finally:
        record["final_torch_rng_sha256"] = hashlib.sha256(
            torch.get_rng_state().numpy().tobytes()).hexdigest()
    return result


def save_gate_ablation(path: str | Path, result: AblationResult) -> None:
    """Write the raw ablation and its provenance as strict JSON."""
    Path(path).write_text(json.dumps(result.record, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")
