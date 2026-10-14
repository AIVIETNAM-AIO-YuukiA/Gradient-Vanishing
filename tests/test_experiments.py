"""Optuna study contract and reduced direct-notebook comparison."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest
import torch

import gradient_vanishing.experiments as experiments
import gradient_vanishing.training as training
from gradient_vanishing.config import load_config
from gradient_vanishing.data import make_fixed_set


ROOT = Path(__file__).resolve().parents[1]


def _small_config(*, trials=3, updates=3):
    return load_config(overrides={
        "training": {"batch_size": 8, "evaluation_size": 32, "tuning_updates": updates},
        "tuning": {"trials": trials},
        "device": "cpu",
    })


def test_suggestions_use_notebook_search_space_and_order():
    class RecordingTrial:
        def __init__(self):
            self.calls = []

        def suggest_float(self, name, low, high, *, log):
            self.calls.append((name, low, high, log))
            return 0.001

        def suggest_categorical(self, name, choices):
            self.calls.append((name, list(choices)))
            return choices[0]

    trial = RecordingTrial()
    params = experiments.suggest_hyperparameters(trial, load_config())
    assert trial.calls == [
        ("lr", 0.0001, 0.05, True),
        ("hidden_size", [16, 32, 64]),
        ("init_method", ["orthogonal", "xavier", "default"]),
    ]
    assert params == {"lr": 0.001, "hidden_size": 16, "init_method": "orthogonal"}


def test_objective_passes_suggestions_seed_and_final_validation_to_training(monkeypatch):
    calls = []

    def fake_train(name, config, length, seed, epochs, trial, hyperparameters, device):
        calls.append((name, length, seed, epochs, hyperparameters, device))
        return SimpleNamespace(history={"val_loss": [0.1, 0.5], "final_val": 0.5})

    monkeypatch.setattr(experiments, "train_model", fake_train)
    result = experiments.tune_model("GRU", 10, _small_config(trials=2), device="cpu")
    assert [call[2] for call in calls] == [1000, 1001]
    assert all(call[:2] == ("GRU", 10) and call[3] == 3 and call[5] == "cpu"
               for call in calls)
    assert [call[4] for call in calls] == [trial["params"] for trial in result.record["trials"]]
    assert [trial["value"] for trial in result.record["trials"]] == [0.5, 0.5]
    assert result.record["direction"] == "minimize"
    assert result.record["best_trial_number"] == 0  # Optuna keeps the first tied best trial
    assert set(result.best_params) == {"lr", "hidden_size", "init_method"}
    assert result.record["sampler"] == {"type": "TPESampler", "seed": 42}
    assert result.record["pruner"] == {"type": "MedianPruner",
                                       "n_startup_trials": 5, "n_warmup_steps": 50}
    assert result.record["gate_mode"] == "one"
    assert all(trial["state"] == "complete" for trial in result.record["trials"])


def test_pruned_and_failed_trials_are_recorded_without_changing_optuna_selection(monkeypatch):
    config = _small_config(trials=3, updates=10)
    config["tuning"]["pruner_startup_trials"] = 1
    config["tuning"]["pruner_warmup_steps"] = 0

    def fake_train(name, config, length, seed, epochs, trial, hyperparameters, device):
        if trial.number == 2:
            return SimpleNamespace(history={"final_val": float("nan")})
        value = 0.1 if trial.number == 0 else 1.0
        trial.report(value, 10)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        return SimpleNamespace(history={"final_val": value})

    monkeypatch.setattr(experiments, "train_model", fake_train)
    result = experiments.tune_model("Vanilla RNN", 10, config, device="cpu")
    assert [trial["state"] for trial in result.record["trials"]] == ["complete", "pruned", "fail"]
    assert result.record["trials"][0]["intermediate_values"] == {10: 0.1}
    assert result.record["trials"][1]["intermediate_values"] == {10: 1.0}
    assert result.record["best_trial_number"] == 0
    assert result.record["best_value"] == 0.1


def test_all_failed_trials_have_no_best_trial(monkeypatch):
    monkeypatch.setattr(experiments, "train_model",
                        lambda *args, **kwargs: SimpleNamespace(history={"final_val": float("nan")}))
    with pytest.raises(experiments.NoCompletedTrialError) as caught:
        experiments.tune_model("Vanilla RNN", 10, _small_config(trials=1), device="cpu")
    assert caught.value.result.record["trials"][0]["state"] == "fail"
    assert caught.value.result.record["best_trial_number"] is None
    assert caught.value.result.record["best_params"] is None


def test_all_pruned_trials_keep_partial_provenance(monkeypatch, tmp_path):
    def prune(*args, **kwargs):
        raise optuna.exceptions.TrialPruned()

    monkeypatch.setattr(experiments, "train_model", prune)
    with pytest.raises(experiments.NoCompletedTrialError) as caught:
        experiments.tune_model("GRU", 10, _small_config(trials=2), device="cpu")
    result = caught.value.result
    assert [trial["state"] for trial in result.record["trials"]] == ["pruned", "pruned"]
    assert [trial["seed"] for trial in result.record["trials"]] == [1000, 1001]
    path = tmp_path / "partial.json"
    experiments.save_study_result(path, result)
    assert len(json.loads(path.read_text(encoding="utf-8"))["trials"]) == 2


def test_budget_overrides_are_recorded_without_mutating_input(monkeypatch):
    original = _small_config(trials=3, updates=3)
    monkeypatch.setattr(experiments, "train_model",
                        lambda *args, **kwargs: SimpleNamespace(history={"final_val": 0.25}))
    result = experiments.tune_model("Vanilla RNN", 10, original,
                                    n_trials=1, tuning_updates=2, device="cpu")
    assert result.record["configuration"]["tuning"]["trials"] == 1
    assert result.record["configuration"]["training"]["tuning_updates"] == 2
    assert original["tuning"]["trials"] == 3
    assert original["training"]["tuning_updates"] == 3
    with pytest.raises(ValueError, match="budgets"):
        experiments.tune_model("Vanilla RNN", 10, original, n_trials=0, device="cpu")


def test_real_objective_never_uses_test_split(monkeypatch):
    def validation_only(length, config, *, device):
        validation = make_fixed_set(config["training"]["evaluation_size"], length,
                                    config["split_seeds"]["validation"], device=device)
        return {"val": validation}

    monkeypatch.setattr(training, "get_eval_sets", validation_only)
    result = experiments.tune_model("Vanilla RNN", 2, _small_config(trials=1, updates=1),
                                    device="cpu")
    assert result.record["trials"][0]["state"] == "complete"
    assert result.record["best_value"] >= 0


@pytest.fixture(scope="module")
def direct_notebook_study():
    script = """
import json
from scripts.capture_notebook_baseline import _load_notebook, _execute_definitions
notebook = _load_notebook()
ns = _execute_definitions(notebook)
ns['BATCH_SIZE'] = 8
ns['N_EVAL'] = 32
ns['TUNE_EPOCHS'] = 3
source = ''.join(notebook['cells'][15]['source'])
exec(compile(source, 'GradientVanishingAnalysis.ipynb:cell-15', 'exec'), ns)
created = []
original_create = ns['optuna'].create_study
def capture_study(*args, **kwargs):
    study = original_create(*args, **kwargs)
    created.append(study)
    return study
ns['optuna'].create_study = capture_study
try:
    selected = ns['tune_model']('Vanilla RNN', 10, n_trials=3)
finally:
    ns['optuna'].create_study = original_create
study = created[0]
print('DIRECT_STUDY_JSON=' + json.dumps({
    'trials': [{'number': t.number, 'params': t.params, 'value': t.value,
                'state': t.state.name.lower(), 'intermediate_values': t.intermediate_values}
               for t in study.trials],
    'best_params': selected, 'best_value': study.best_value,
    'best_trial_number': study.best_trial.number,
}))
"""
    completed = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                               capture_output=True, text=True, check=True)
    return json.loads(next(line.split("=", 1)[1] for line in completed.stdout.splitlines()
                           if line.startswith("DIRECT_STUDY_JSON=")))


def test_reduced_study_matches_direct_original_notebook(direct_notebook_study, tmp_path):
    config = _small_config()
    result = experiments.tune_model("Vanilla RNN", 10, config, device="cpu")
    original = direct_notebook_study
    assert len(result.record["trials"]) == 3
    for actual, expected in zip(result.record["trials"], original["trials"]):
        assert actual["number"] == expected["number"]
        assert actual["seed"] == 1000 + actual["number"]
        assert actual["params"] == expected["params"]
        assert actual["state"] == expected["state"]
        assert actual["value"] == pytest.approx(expected["value"], rel=1e-5, abs=1e-6)
        assert actual["intermediate_values"] == {
            int(step): pytest.approx(value, rel=1e-5, abs=1e-6)
            for step, value in expected["intermediate_values"].items()}
    assert result.best_params == original["best_params"]
    assert result.record["best_value"] == pytest.approx(original["best_value"], rel=1e-5, abs=1e-6)
    assert result.record["best_trial_number"] == original["best_trial_number"]

    path = tmp_path / "study.json"
    experiments.save_study_result(path, result)
    restored = json.loads(path.read_text(encoding="utf-8"))
    assert restored["configuration"] == config
    assert restored["search_space"] == result.record["search_space"]
    assert restored["study_name"] == result.study.study_name
    assert restored["python_version"] == sys.version.split()[0]
    assert restored["torch_version"] == torch.__version__
    assert restored["optuna_version"] == optuna.__version__
    assert restored["notebook_sha256"] == "71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44"


def test_reduced_study_is_reproducible_under_same_conditions():
    config = _small_config(trials=2, updates=2)
    first = experiments.tune_model("LSTM", 2, config, device="cpu")
    second = experiments.tune_model("LSTM", 2, config, device="cpu")
    assert [(trial["params"], trial["value"], trial["state"])
            for trial in first.record["trials"]] == [
                (trial["params"], trial["value"], trial["state"])
                for trial in second.record["trials"]]
    assert first.best_params == second.best_params
    assert first.record["best_value"] == second.record["best_value"]


def test_notebook_and_baseline_fixture_hashes_are_unchanged():
    assert hashlib.sha256((ROOT / "notebooks/GradientVanishingAnalysis.ipynb").read_bytes()).hexdigest() == "71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44"
    assert hashlib.sha256((ROOT / "tests/fixtures/notebook_baseline.json").read_bytes()).hexdigest() == "76e69564a5530b0a92e5cbac32b0b5ade492d516bd0e091c734c9e918b3dd336"
