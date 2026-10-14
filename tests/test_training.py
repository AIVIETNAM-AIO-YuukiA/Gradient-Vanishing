"""Training contract against the directly executed notebook oracle."""

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from gradient_vanishing.config import load_config
from gradient_vanishing.data import get_eval_sets
from gradient_vanishing.training import evaluate, train_model
import gradient_vanishing.training as training_module


FIXTURE = json.loads((Path(__file__).parent / "fixtures/notebook_baseline.json").read_text(encoding="utf-8"))
CASES = {case["id"]: case for case in FIXTURE["cases"]}


def _close(actual, expected):
    if isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _close(a, e)
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6), (actual, expected)
    else:
        assert actual == expected


@pytest.mark.parametrize("name,case_id", [
    ("Vanilla RNN", "vanilla_rnn_training_t10"),
    ("LSTM", "lstm_training_t10"),
    ("GRU", "gru_training_t10"),
])
def test_three_update_notebook_training_probe(name, case_id, monkeypatch):
    reference = CASES[case_id]["expected"]
    inputs = CASES[case_id]["input"]
    config = load_config()
    observed_batches = []
    original_generate = training_module.generate_memorization_task

    def record_batch(*args, **kwargs):
        x, y = original_generate(*args, **kwargs)
        observed_batches.append((x, y))
        return x, y

    monkeypatch.setattr(training_module, "generate_memorization_task", record_batch)
    result = train_model(name, config, 10, 0, epochs=3,
                         gate_init="one", record_grad_epochs=(0, 2), device="cpu")
    assert len(observed_batches) == 3
    for x, y in observed_batches:
        assert list(x.shape) == inputs["training_batch_shape"]
        assert list(y.shape) == [inputs["training_batch_shape"][1], 1]
        assert torch.equal(x[0], y)
        assert torch.count_nonzero(x[1:]) == 0
    for key in ("train_loss", "val_epochs", "val_loss", "final_val"):
        _close(result.history[key], reference["history"][key])
    for key, norms in reference["history"]["grad_norms"].items():
        _close(result.history["grad_norms"][int(key)], norms)
    assert result.test_mse is None
    assert result.model.training is reference["mode_after_train"]

    test_x, test_y = get_eval_sets(10, config, device="cpu")["test"]
    val_x, val_y = get_eval_sets(10, config, device="cpu")["val"]
    assert list(val_x.shape) == inputs["validation_shape"]
    assert list(val_y.shape) == [inputs["validation_shape"][1], 1]
    assert list(test_x.shape) == inputs["test_shape"]
    assert list(test_y.shape) == [inputs["test_shape"][1], 1]
    assert test_x[0, :4, 0].tolist() == CASES[case_id]["input"]["test_x_first_values"]
    result.model.eval()
    assert result.model.training is reference["mode_before_evaluate"]
    state = torch.get_rng_state().clone()
    _close(evaluate(result.model, test_x, test_y), reference["test_mse"])
    assert result.model.training is reference["mode_after_evaluate"]
    assert torch.equal(torch.get_rng_state(), state)
    with torch.no_grad():
        prediction = result.model(test_x[:, :4, :])
    assert list(prediction.shape) == reference["test_predictions_first_four"]["shape"]
    _close(prediction.tolist(), reference["test_predictions_first_four"]["values"])


def test_validation_schedule_and_trial_report_and_prune():
    class Trial:
        def __init__(self, prune=False):
            self.reports = []
            self.prune = prune

        def report(self, value, step):
            self.reports.append((value, step))

        def should_prune(self):
            return self.prune

    config = load_config(overrides={"training": {"batch_size": 4, "evaluation_size": 8}})
    trial = Trial()
    result = train_model("Vanilla RNN", config, 2, 4, epochs=11, trial=trial, device="cpu")
    assert result.history["val_epochs"] == [10, 11]
    assert trial.reports == list(zip(result.history["val_loss"], [10, 11]))
    assert result.test_mse is None
    with pytest.raises(ValueError, match="test evaluation"):
        train_model("Vanilla RNN", config, 2, 4, epochs=1, trial=Trial(),
                    evaluate_test=True, device="cpu")
    with pytest.raises(__import__("optuna").exceptions.TrialPruned):
        train_model("Vanilla RNN", config, 2, 4, epochs=11, trial=Trial(True), device="cpu")


def test_metadata_is_json_serializable_and_has_effective_settings(tmp_path):
    config = load_config(overrides={"training": {"batch_size": 4, "evaluation_size": 8}})
    result = train_model("GRU", config, 2, 19, epochs=1, device="cpu")
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(result.metadata), encoding="utf-8")
    restored = json.loads(path.read_text(encoding="utf-8"))
    assert restored["configuration"] == config
    assert restored["hyperparameters"] == {"lr": 0.001, "hidden_size": 32, "init_method": "default"}
    assert restored["seed"] == 19
    assert restored["updates"] == 1
    assert restored["device"] == "cpu" and restored["dtype"] == "torch.float32"
    assert restored["python_version"] and restored["torch_version"]
    assert restored["deterministic_algorithms"] is True
    assert restored["notebook_sha256"] == CASES["gru_training_t10"]["source_notebook_sha256"]


def test_fresh_batches_and_clip_before_every_adam_step(monkeypatch):
    config = load_config(overrides={"training": {"batch_size": 4, "evaluation_size": 8}})
    events = []
    samples = []
    original_generate = training_module.generate_memorization_task
    original_clip = torch.nn.utils.clip_grad_norm_
    original_step = torch.optim.Adam.step

    def generate(*args, **kwargs):
        x, y = original_generate(*args, **kwargs)
        samples.append(x[0, :, 0].clone())
        events.append("sample")
        return x, y

    def clip(*args, **kwargs):
        parameters = list(args[0])
        assert parameters and all(parameter.grad is not None for parameter in parameters)
        assert kwargs["max_norm"] == 5.0
        events.append("clip")
        return original_clip(parameters, *args[1:], **kwargs)

    def step(optimizer, *args, **kwargs):
        events.append("step")
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(training_module, "generate_memorization_task", generate)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    monkeypatch.setattr(torch.optim.Adam, "step", step)
    train_model("Vanilla RNN", config, 2, 8, epochs=3, device="cpu")
    assert events == ["sample", "clip", "step"] * 3
    assert all(not torch.equal(samples[0], sample) for sample in samples[1:])


def test_evaluate_uses_no_grad_and_leaves_training_mode():
    class Observer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.grad_enabled = None
            self.mode_during_forward = None

        def forward(self, x):
            self.grad_enabled = torch.is_grad_enabled()
            self.mode_during_forward = self.training
            return x

    model = Observer()
    model.eval()
    assert evaluate(model, torch.ones(2, 1), torch.zeros(2, 1)) == 1.0
    assert model.grad_enabled is False
    assert model.mode_during_forward is False
    assert model.training is True


def test_seeded_runs_reproduce_final_parameters_and_torch_rng():
    config = load_config(overrides={"training": {"batch_size": 4, "evaluation_size": 8}})
    first = train_model("GRU", config, 2, 23, epochs=2, device="cpu", evaluate_test=True)
    first_rng = torch.get_rng_state().clone()
    first_parameters = [value.detach().clone() for value in first.model.parameters()]
    second = train_model("GRU", config, 2, 23, epochs=2, device="cpu", evaluate_test=True)
    assert torch.equal(torch.get_rng_state(), first_rng)
    assert first.history == second.history
    assert first.test_mse == second.test_mse
    assert all(torch.equal(a, b) for a, b in zip(first_parameters, second.model.parameters()))


def test_notebook_and_oracle_files_remain_untouched():
    root = Path(__file__).resolve().parents[1]
    notebook = root / "Notebooks/GradientVanishingAnalysis.ipynb"
    fixture = root / "tests/fixtures/notebook_baseline.json"
    assert hashlib.sha256(notebook.read_bytes()).hexdigest() == "71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44"
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == "edc601e85522254471424940869598836cfea940a8d8711580bf8f0eca0818b9"
