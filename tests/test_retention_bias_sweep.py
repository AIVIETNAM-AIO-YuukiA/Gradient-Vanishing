"""Notebook cell 28 numeric sweep: schedule, gate semantics, and direct oracle."""

import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

import gradient_vanishing.experiments as experiments
import gradient_vanishing.gradients as gradient_module
from gradient_vanishing.config import load_config
from gradient_vanishing.models import GRU_Tracker, apply_gate_init


ROOT = Path(__file__).resolve().parents[1]
BIAS_GRID = [-4.0 + 0.5 * index for index in range(17)]


def _reduced_config():
    return load_config(overrides={
        "gradients": {"sweep_sequence_length": 5, "sweep_seeds": 2},
        "training": {"batch_size": 8}, "device": "cpu",
    })


def test_default_sweep_grid_order_sign_and_no_training_or_test_access(monkeypatch):
    config = load_config()
    calls = []

    @dataclass
    class Initial:
        decay_mean: float = 0.7
        decay_sample_std: float = 0.01
        normalized_curves: tuple = ((1.0,), (1.0,))

    def gradients(name, length, configuration, *, gate_mode, n_seeds,
                  hyperparameters, device):
        calls.append((name, length, gate_mode, n_seeds, dict(hyperparameters), device))
        return Initial()

    def unexpected(*args, **kwargs):
        raise AssertionError("sweep must not train or use evaluation sets")

    monkeypatch.setattr(experiments, "measure_init_gradients", gradients)
    monkeypatch.setattr(experiments, "train_model", unexpected)
    monkeypatch.setattr(experiments, "evaluate", unexpected)
    monkeypatch.setattr(experiments, "get_eval_sets", unexpected)
    record = experiments.run_retention_bias_sweep(config, device="cpu").record
    assert record["status"] == "complete"
    assert record["model_order"] == ["LSTM", "GRU"]
    assert record["bias_order"] == BIAS_GRID
    assert len(calls) == 34
    assert [(call[0], call[2]) for call in calls] == [
        (name, bias) for name in ("LSTM", "GRU") for bias in BIAS_GRID]
    assert all(call[1] == 100 and call[3] == 5 for call in calls)
    assert all(call[4] == {"lr": 0.001, "hidden_size": 32,
                                "init_method": "default"} for call in calls)
    for name, sign, parameter in (("LSTM", 1, "W_f.bias"), ("GRU", -1, "W_z.bias")):
        points = record["results"][name]
        assert len(points) == 17
        for bias, point in zip(BIAS_GRID, points):
            assert point["retention_bias"] == bias
            assert point["theoretical_retention"] == pytest.approx(1 / (1 + math.exp(-bias)))
            assert point["gate_parameter"] == parameter
            assert point["retention_bias_sign"] == sign
            assert point["assigned_gate_bias"] == [sign * bias] * 32
            assert point["seeds"] == list(range(5))
            assert point["decay_mean"] == 0.7
    assert record["theoretical_reference_line"] == [[0.0, 0.0], [1.0, 1.0]]


def test_custom_gru_positive_retention_bias_reduces_update_gate_activation():
    torch.manual_seed(23)
    model = GRU_Tracker(1, 1, 1)
    with torch.no_grad():
        model.W_z.weight.zero_()
    reset_before = model.W_r.bias.detach().clone()
    activations = []
    for bias in (-2.0, 0.0, 2.0):
        apply_gate_init(model, bias, 5)
        assert model.W_z.bias.item() == -bias
        activations.append(torch.sigmoid(model.W_z.bias).item())
        assert torch.equal(model.W_r.bias, reset_before)
    assert activations[0] > activations[1] > activations[2]
    assert [1 - value for value in activations] == pytest.approx(
        [1 / (1 + math.exp(-bias)) for bias in (-2.0, 0.0, 2.0)])


def test_partial_sweep_is_serializable(monkeypatch, tmp_path):
    calls = []

    @dataclass
    class Initial:
        decay_mean: float = 0.8
        decay_sample_std: float = 0.02

    def gradients(*args, **kwargs):
        calls.append(kwargs["gate_mode"])
        if len(calls) == 3:
            raise RuntimeError("measurement failed")
        return Initial()

    monkeypatch.setattr(experiments, "measure_init_gradients", gradients)
    with pytest.raises(experiments.SweepInterruptedError) as caught:
        experiments.run_retention_bias_sweep(_reduced_config())
    record = caught.value.result.record
    assert record["status"] == "incomplete"
    assert record["failure"]["model"] == "LSTM"
    assert record["failure"]["retention_bias"] == -3.0
    assert [point["status"] for point in record["results"]["LSTM"]] == [
        "complete", "complete", "failed"]
    path = tmp_path / "partial.json"
    experiments.save_retention_bias_sweep(path, caught.value.result)
    assert json.loads(path.read_text(encoding="utf-8"))["failure"]["error_type"] == "RuntimeError"


@pytest.fixture(scope="module")
def direct_notebook_sweep():
    script = """
import ast, hashlib, json, numpy as np
from scripts.capture_notebook_baseline import _load_notebook, _execute_definitions
notebook = _load_notebook()
ns = _execute_definitions(notebook)
import torch
source = ''.join(notebook['cells'][23]['source'])
tree = ast.parse(source)
definitions = ast.Module(body=[node for node in tree.body
                               if isinstance(node, ast.FunctionDef) and node.name in
                               {'measure_init_gradients', 'decay_factor_from_curves'}], type_ignores=[])
exec(compile(definitions, 'GradientVanishingAnalysis.ipynb:cell-23', 'exec'), ns)
ns['BATCH_SIZE'] = 8
original_build = ns['build_model']
original_mse = ns['F'].mse_loss
models, measurements, points = [], [], []
def capture_build(name, config, gate_init, length, *args, **kwargs):
    model = original_build(name, config, gate_init, length, *args, **kwargs)
    digest = hashlib.sha256()
    for parameter_name, parameter in model.state_dict().items():
        digest.update(parameter_name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    models.append({'name': name, 'bias': gate_init,
                   'parameter_sha256': digest.hexdigest(),
                   'gate': (model.W_f.bias.detach().tolist() if name == 'LSTM' else
                            model.W_z.bias.detach().tolist()),
                   'reset': model.W_r.bias.detach().tolist() if name == 'GRU' else None})
    return model
def capture_mse(prediction, target, *args, **kwargs):
    loss = original_mse(prediction, target, *args, **kwargs)
    measurements.append({'prediction': prediction.detach().flatten().tolist(), 'loss': loss.item()})
    return loss
ns['build_model'] = capture_build
ns['F'].mse_loss = capture_mse
try:
    for name in ('LSTM', 'GRU'):
        for bias in np.linspace(-4, 4, 17):
            mean, curves = ns['measure_init_gradients'](name, 5, gate_init=float(bias), n_seeds=2)
            decay, std = ns['decay_factor_from_curves'](curves, 5)
            points.append({'name': name, 'bias': float(bias), 'mean': mean.tolist(),
                           'curves': curves.tolist(), 'decay': decay, 'std': std})
finally:
    ns['F'].mse_loss = original_mse
print('DIRECT_SWEEP_JSON=' + json.dumps({
    'points': points, 'models': models, 'measurements': measurements,
    'rng_sha256': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
}))
"""
    completed = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    return json.loads(next(line.split("=", 1)[1] for line in completed.stdout.splitlines()
                           if line.startswith("DIRECT_SWEEP_JSON=")))


def test_all_biases_match_direct_notebook(direct_notebook_sweep, monkeypatch, tmp_path):
    models, measurements = [], []
    original_build = gradient_module.build_model
    original_mse = gradient_module.F.mse_loss

    def capture_build(name, config, gate_mode, length, *args, **kwargs):
        model = original_build(name, config, gate_mode, length, *args, **kwargs)
        digest = hashlib.sha256()
        for parameter_name, parameter in model.state_dict().items():
            digest.update(parameter_name.encode())
            digest.update(parameter.detach().cpu().numpy().tobytes())
        models.append({"name": name, "bias": gate_mode,
                       "parameter_sha256": digest.hexdigest(),
                       "gate": (model.W_f.bias.detach().tolist() if name == "LSTM" else
                                model.W_z.bias.detach().tolist()),
                       "reset": model.W_r.bias.detach().tolist() if name == "GRU" else None})
        return model

    def capture_mse(prediction, target, *args, **kwargs):
        loss = original_mse(prediction, target, *args, **kwargs)
        measurements.append({"prediction": prediction.detach().flatten().tolist(),
                             "loss": loss.item()})
        return loss

    monkeypatch.setattr(gradient_module, "build_model", capture_build)
    monkeypatch.setattr(gradient_module.F, "mse_loss", capture_mse)
    record = experiments.run_retention_bias_sweep(_reduced_config()).record
    expected = direct_notebook_sweep
    assert record["status"] == "complete"
    assert record["final_torch_rng_sha256"] == expected["rng_sha256"]
    points = [point for name in ("LSTM", "GRU") for point in record["results"][name]]
    assert len(points) == len(expected["points"]) == 34
    assert len(models) == len(expected["models"]) == 68
    for index, (point, reference) in enumerate(zip(points, expected["points"])):
        assert (point["model"], point["retention_bias"]) == (reference["name"], reference["bias"])
        np.testing.assert_allclose(point["initial_gradients"]["normalized_curves"],
                                   reference["curves"], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(point["initial_gradients"]["geometric_mean"],
                                   reference["mean"], rtol=1e-8, atol=1e-10)
        assert point["decay_mean"] == pytest.approx(reference["decay"], rel=1e-8, abs=1e-10)
        assert point["decay_sample_std"] == pytest.approx(reference["std"], rel=1e-8, abs=1e-10)
        assert point["initial_gradients"]["timestep"] == list(range(5))
        assert point["initial_gradients"]["fit_timesteps"] == [2, 3]
        for seed_index in (0, 1):
            offset = index * 2 + seed_index
            actual_model, original_model = models[offset], expected["models"][offset]
            assert (actual_model["name"], actual_model["bias"]) == (
                original_model["name"], original_model["bias"])
            assert actual_model["parameter_sha256"] == original_model["parameter_sha256"]
            assert actual_model["gate"] == original_model["gate"]
            if point["model"] == "GRU":
                assert actual_model["reset"] == original_model["reset"]
            actual_mse, original_measure = measurements[offset], expected["measurements"][offset]
            assert actual_mse["prediction"] == pytest.approx(
                original_measure["prediction"], rel=1e-8, abs=1e-10)
            assert actual_mse["loss"] == pytest.approx(original_measure["loss"], rel=1e-8, abs=1e-10)
    path = tmp_path / "sweep.json"
    experiments.save_retention_bias_sweep(path, experiments.SweepResult(record))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["configuration"] == _reduced_config()
    assert saved["device"] == "cpu" and saved["dtype"] == "torch.float64"
    assert saved["python_version"] == sys.version.split()[0]
    assert saved["torch_version"] == torch.__version__
