"""Hidden-gradient mathematical checks and original-notebook regressions."""

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from gradient_vanishing.config import load_config
from gradient_vanishing.data import generate_memorization_task
from gradient_vanishing.gradients import (
    capture_training_gradients, decay_factor_from_curves, measure_hidden_gradients,
    measure_init_gradients, save_diagnostic_result, training_log10_relative_norms,
)
from gradient_vanishing.models import (
    GRU_Tracker, LSTM_Tracker, TrackerBase, VanillaRNN_Tracker, build_model,
)


ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "tests/fixtures/notebook_baseline.json").read_text(encoding="utf-8"))["cases"]


def _close(actual, expected, tolerance):
    np.testing.assert_allclose(actual, expected, **tolerance)


@pytest.mark.parametrize("case", [case for case in CASES if case["kind"] == "model"],
                         ids=lambda case: case["id"])
def test_stored_fixture_hidden_gradients_and_full_bptt(case):
    settings = case["configuration"]
    expected = case["expected"]
    dtype = getattr(torch, case["precision"])
    torch.manual_seed(case["seed"])
    model = build_model(settings["model"], settings["hyperparameters"],
                        settings["gate_mode"], settings["sequence_length"],
                        dtype=dtype, device="cpu")
    rng_digest = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    assert rng_digest == expected["rng_after_model_sha256"]
    x, y = generate_memorization_task(settings["batch_size"],
                                      settings["sequence_length"], device="cpu")
    x = x.to(dtype).requires_grad_()
    observation = measure_hidden_gradients(model, x, y.to(dtype))
    tolerance = case["comparison"]["computed_values"]
    _close(observation.norms, expected["hidden_gradient_norms"], tolerance)
    assert observation.loss == pytest.approx(expected["mse"],
                                             rel=tolerance["rtol"], abs=tolerance["atol"])
    assert len(observation.gradients) == settings["sequence_length"]
    for gradient, norm in zip(observation.gradients, observation.norms):
        assert gradient.shape == (settings["batch_size"], settings["hyperparameters"]["hidden_size"])
        assert gradient.dtype == dtype
        _close(gradient.norm(p=2, dim=1).mean().item(), norm, tolerance)
    assert observation.norms[0] > 0
    assert model.training is expected["training_mode_after_backward"]
    assert x.grad is not None and x.grad[0].abs().sum() > 0
    if isinstance(model, LSTM_Tracker):
        assert all(not cell.requires_grad for cell in model.c_states)


def test_two_step_rnn_hidden_derivatives_match_independent_chain_rule():
    model = VanillaRNN_Tracker(1, 1, 1).double()
    with torch.no_grad():
        model.W_ih.weight.fill_(1)
        model.W_ih.bias.zero_()
        model.W_hh.weight.fill_(0.5)
        model.W_ho.weight.fill_(1)
        model.W_ho.bias.zero_()
    x = torch.tensor([[[0.4]], [[0.0]]], dtype=torch.float64, requires_grad=True)
    result = measure_hidden_gradients(model, x, torch.zeros(1, 1, dtype=torch.float64))
    h0 = math.tanh(0.4)
    h1 = math.tanh(0.5 * h0)
    last_derivative = 2 * h1
    first_derivative = last_derivative * (1 - h1 * h1) * 0.5
    assert result.gradients[0].item() == pytest.approx(first_derivative, abs=1e-14)
    assert result.gradients[1].item() == pytest.approx(last_derivative, abs=1e-14)
    assert result.norms == pytest.approx([abs(first_derivative), abs(last_derivative)], abs=1e-14)
    assert x.grad[0].item() == pytest.approx(first_derivative * (1 - h0 * h0), abs=1e-14)


def test_hidden_norm_is_batch_mean_of_per_sample_l2_norms():
    model = TrackerBase()
    hidden = torch.zeros(2, 2, requires_grad=True)
    hidden.grad = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    model.h_states = [hidden]
    assert model.get_gradient_norms() == [3.5]  # (5 + 2) / 2


@pytest.mark.parametrize("cls", [VanillaRNN_Tracker, LSTM_Tracker, GRU_Tracker])
def test_first_hidden_state_receives_full_sequence_gradient(cls):
    torch.manual_seed(17)
    model = cls(1, 3, 1).double()
    x = torch.tensor([[[0.25]], [[0.0]], [[0.0]], [[0.0]]],
                     dtype=torch.float64, requires_grad=True)
    measurement = measure_hidden_gradients(model, x, torch.zeros(1, 1, dtype=torch.float64))
    assert len(measurement.gradients) == 4
    assert measurement.gradients[0].abs().sum() > 0
    assert x.grad[0].abs().sum() > 0


def test_normalization_geometric_mean_and_decay_fit_use_notebook_indices():
    curve_a = np.array([1/16, 1/8, 1/4, 1/2, 1.0])
    curve_b = np.array([1/81, 1/27, 1/9, 1/3, 1.0])
    curves = np.stack([curve_a, curve_b])
    geometric = np.exp(np.log(np.clip(curves, 1e-300, None)).mean(0))
    _close(geometric, np.sqrt(curve_a * curve_b), {"rtol": 1e-12, "atol": 1e-12})
    mean, std = decay_factor_from_curves(curves, 5)
    assert mean == pytest.approx(5/12, abs=1e-12)
    assert std == pytest.approx(np.std([0.5, 1/3], ddof=1), abs=1e-12)
    assert training_log10_relative_norms([1e-20, 1.0]) == pytest.approx([-10.0, 0.0])


@pytest.fixture(scope="module")
def direct_notebook_initial_curves():
    # Cell 23's definitions are executed verbatim. Its plotting tail is omitted.
    script = """
import json
from scripts.capture_notebook_baseline import _load_notebook, _execute_definitions
notebook = _load_notebook()
ns = _execute_definitions(notebook)
source = ''.join(notebook['cells'][23]['source']).split('T_GRAD =')[0]
exec(compile(source, 'GradientVanishingAnalysis.ipynb:cell-23', 'exec'), ns)
torch = ns['torch']
output = {}
for name in ns['MODEL_NAMES']:
    mean, curves = ns['measure_init_gradients'](name, 10, n_seeds=2)
    decay = ns['decay_factor_from_curves'](curves, 10)
    output[name] = {'mean': mean.tolist(), 'curves': curves.tolist(),
                    'decay': decay, 'rng': torch.get_rng_state().tolist()}
print('DIRECT_REFERENCE_JSON=' + json.dumps(output))
"""
    completed = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                               capture_output=True, text=True, check=True)
    return json.loads(next(line.split("=", 1)[1] for line in completed.stdout.splitlines()
                           if line.startswith("DIRECT_REFERENCE_JSON=")))


@pytest.mark.parametrize("name", ["Vanilla RNN", "LSTM", "GRU"])
def test_initial_curves_match_direct_notebook_execution(name, direct_notebook_initial_curves):
    result = measure_init_gradients(name, 10, load_config(), n_seeds=2, device="cpu")
    expected = direct_notebook_initial_curves[name]
    tolerance = {"rtol": 1e-8, "atol": 1e-10}
    _close(result.normalized_curves, expected["curves"], tolerance)
    _close(result.geometric_mean, expected["mean"], tolerance)
    _close([result.decay_mean, result.decay_sample_std], expected["decay"], tolerance)
    assert torch.get_rng_state().tolist() == expected["rng"]
    assert result.seeds == [0, 1]
    assert result.fit_timesteps == [5, 6, 7, 8]
    assert result.timestep == list(range(10))
    assert result.temporal_distance == list(range(9, -1, -1))
    assert result.zero_ratio_count == result.nonfinite_ratio_count == 0
    assert result.dtype == "torch.float64" and result.device == "cpu"
    assert result.model_training_mode is True
    assert result.final_torch_rng_sha256 == hashlib.sha256(
        bytes(expected["rng"])).hexdigest()


@pytest.mark.parametrize("name,case_id", [
    ("Vanilla RNN", "vanilla_rnn_training_t10"),
    ("LSTM", "lstm_training_t10"),
    ("GRU", "gru_training_t10"),
])
def test_training_capture_matches_stored_fixture(name, case_id):
    reference = next(case for case in CASES if case["id"] == case_id)
    result = capture_training_gradients(name, load_config(), 10, 0,
                                        updates=3, capture_updates=(0, 2), device="cpu")
    assert result.capture_updates == [0, 2]
    for index, expected in reference["expected"]["history"]["grad_norms"].items():
        _close(result.raw_norms[int(index)], expected,
               reference["comparison"]["computed_values"])
        assert len(result.log10_relative_norms[int(index)]) == 10
    assert result.run.model.training is True
    assert result.run.metadata["gradient_capture_updates"] == [0, 2]


def test_training_gradient_read_precedes_parameter_clipping(monkeypatch):
    events = []
    read = TrackerBase.get_gradient_norms
    clip = torch.nn.utils.clip_grad_norm_

    def record_read(model):
        events.append("read")
        return read(model)

    def record_clip(*args, **kwargs):
        events.append("clip")
        return clip(*args, **kwargs)

    monkeypatch.setattr(TrackerBase, "get_gradient_norms", record_read)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_clip)
    config = load_config(overrides={"training": {"batch_size": 4, "evaluation_size": 8}})
    capture_training_gradients("Vanilla RNN", config, 5, 0,
                               updates=1, capture_updates=(0,), device="cpu")
    assert events == ["read", "clip"]


def test_zero_and_nonfinite_ratios_are_reported_without_redefinition(monkeypatch, tmp_path):
    from gradient_vanishing import gradients

    original = gradients.measure_hidden_gradients
    calls = 0

    def synthetic(model, x, y):
        nonlocal calls
        calls += 1
        observation = original(model, x, y)
        observation.norms = [0.0] * 4 + ([0.0] if calls == 1 else [1.0])
        return observation

    monkeypatch.setattr(gradients, "measure_hidden_gradients", synthetic)
    config = load_config(overrides={"training": {"batch_size": 4}})
    with np.errstate(divide="ignore", invalid="ignore"):
        result = measure_init_gradients("Vanilla RNN", 5, config, n_seeds=2, device="cpu")
    assert result.zero_ratio_count == 4
    assert result.nonfinite_ratio_count == 5
    assert all(math.isnan(value) for value in result.normalized_curves[0])
    assert result.geometric_mean[0] != result.geometric_mean[0]  # unchanged NaN propagation
    path = tmp_path / "nonfinite.json"
    save_diagnostic_result(path, result)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["normalized_curves"][0] == ["NaN"] * 5
    assert record["nonfinite_ratio_count"] == 5


def test_diagnostic_json_round_trip_includes_numeric_arrays_and_provenance(tmp_path):
    config = load_config(overrides={"training": {"batch_size": 4, "evaluation_size": 8}})
    initial = measure_init_gradients("GRU", 5, config, n_seeds=2, device="cpu")
    path = tmp_path / "initial.json"
    save_diagnostic_result(path, initial)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["kind"] == "initial_hidden_gradients"
    assert record["seeds"] == [0, 1]
    assert record["configuration"] == config
    assert record["python_version"] == sys.version.split()[0]
    assert record["torch_version"] == torch.__version__
    assert record["raw_norms"] == initial.raw_norms
    assert record["fit_timesteps"] == [2, 3]
    assert record["measurement"] == initial.measurement
    assert record["notebook_sha256"] == "71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44"

    training = capture_training_gradients("GRU", config, 5, 0,
                                          updates=1, capture_updates=(0,), device="cpu")
    path = tmp_path / "training.json"
    save_diagnostic_result(path, training)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["kind"] == "training_hidden_gradients"
    assert record["metadata"]["seed"] == 0
    assert record["metadata"]["dtype"] == "torch.float32"
    assert record["capture_updates"] == [0]
    assert record["raw_norms"]["0"] == training.raw_norms[0]
    assert record["log10_relative_norms"]["0"] == training.log10_relative_norms[0]


def test_source_and_fixture_hashes_remain_unchanged():
    assert hashlib.sha256((ROOT / "Notebooks/GradientVanishingAnalysis.ipynb").read_bytes()).hexdigest() == "71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44"
    assert hashlib.sha256((ROOT / "tests/fixtures/notebook_baseline.json").read_bytes()).hexdigest() == "76e69564a5530b0a92e5cbac32b0b5ade492d516bd0e091c734c9e918b3dd336"
