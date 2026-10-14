"""Gate ablation cell 27: schedule, threshold semantics, and direct notebook check."""

import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import gradient_vanishing.experiments as experiments
import gradient_vanishing.training as training_module
from gradient_vanishing.config import load_config
from gradient_vanishing.models import GRU_Tracker, apply_gate_init


ROOT = Path(__file__).resolve().parents[1]


def _reduced_config():
    return load_config(overrides={
        "seeds": [0, 1], "ablation": {"sequence_lengths": [5]},
        "training": {"updates": 2, "batch_size": 8, "evaluation_size": 32,
                     "gradient_capture_updates": [0, 1]},
        "gradients": {"initial_seeds": 2}, "device": "cpu",
    })


def test_threshold_uses_only_checked_validation_and_strict_cutoff():
    assert experiments.epochs_to_threshold(
        {"val_epochs": [10, 20, 30], "val_loss": [0.02, 0.01, 0.009]}, 0.01) == 30
    assert experiments.epochs_to_threshold(
        {"val_epochs": [10, 20], "val_loss": [0.02, 0.01]}, 0.01) is None


def test_positive_gru_retention_bias_negates_update_gate_bias():
    torch.manual_seed(7)
    model = GRU_Tracker(1, 1, 1)
    with torch.no_grad():
        model.W_z.weight.zero_()
    reset_bias = model.W_r.bias.detach().clone()
    apply_gate_init(model, "zero", 5)
    zero_retention = 1 - torch.sigmoid(model.W_z.bias).item()
    apply_gate_init(model, "one", 5)
    one_retention = 1 - torch.sigmoid(model.W_z.bias).item()
    assert model.W_z.bias.item() == -1.0
    assert torch.equal(model.W_r.bias, reset_bias)
    assert zero_retention == pytest.approx(0.5)
    assert one_retention == pytest.approx(1 / (1 + math.exp(-1)))
    assert one_retention > zero_retention


def test_default_condition_matrix_and_fixed_test_reuse(monkeypatch):
    config = load_config()
    events = []
    pairs = {}

    def sets(length, configuration, *, device):
        pair = (object(), object())
        pairs[length] = pair
        events.append(("set", length))
        return {"test": pair}

    def train(name, configuration, length, seed, *, epochs, gate_init,
              hyperparameters, device):
        events.append(("train", length, name, gate_init, seed, epochs, dict(hyperparameters)))
        history = {"val_epochs": [10, 300], "val_loss": [0.01, 0.005],
                   "final_val": 0.005, "train_loss": [0.1], "grad_norms": {}}
        return SimpleNamespace(model=SimpleNamespace(training=True, seed=seed),
                               history=history, metadata={"seed": seed, "initial_gate": None})

    def evaluate(model, x, y):
        events.append(("test", model.seed, x, y))
        return 0.2 + model.seed / 100

    @dataclass
    class Initial:
        decay_mean: float = 0.8
        decay_sample_std: float = 0.01
        normalized_curves: tuple = ((1.0,), (1.0,))

    def gradients(name, length, configuration, *, gate_mode, n_seeds,
                  hyperparameters, device):
        events.append(("gradient", length, name, gate_mode, n_seeds))
        return Initial()

    monkeypatch.setattr(experiments, "get_eval_sets", sets)
    monkeypatch.setattr(experiments, "train_model", train)
    monkeypatch.setattr(experiments, "evaluate", evaluate)
    monkeypatch.setattr(experiments, "measure_init_gradients", gradients)
    record = experiments.run_gate_ablation(config, device="cpu").record
    conditions = record["results"]["100"]
    assert record["status"] == "complete"
    assert [(x["model"], x["label"], x["gate_mode"]) for x in conditions] == [
        ("Vanilla RNN", "—", "default"),
        ("LSTM", "zero", "zero"), ("LSTM", "one", "one"),
        ("LSTM", "chrono", "chrono"), ("GRU", "zero", "zero"),
        ("GRU", "one", "one"), ("GRU", "chrono", "chrono")]
    assert len([event for event in events if event[0] == "train"]) == 35
    assert len([event for event in events if event[0] == "gradient"]) == 7
    assert events[0] == ("set", 100)
    assert all(event[5] == 300 for event in events if event[0] == "train")
    assert all(event[4] == 10 for event in events if event[0] == "gradient")
    assert all(event[2] is pairs[100][0] and event[3] is pairs[100][1]
               for event in events if event[0] == "test")
    for condition in conditions:
        assert [run["seed"] for run in condition["runs"]] == [0, 1, 2, 3, 4]
        assert condition["threshold_success_count"] == 5
        assert condition["threshold_mean_success_update"] == 300.0
        assert condition["test_mean"] == pytest.approx(0.22)
        assert condition["test_sample_std"] == pytest.approx(
            float(np.std([0.2 + seed / 100 for seed in range(5)], ddof=1)))


def test_failed_condition_retains_partial_provenance(monkeypatch, tmp_path):
    config = _reduced_config()
    monkeypatch.setattr(experiments, "get_eval_sets", lambda *args, **kwargs:
                        {"test": (object(), object())})

    def fail_second_seed(name, configuration, length, seed, **kwargs):
        if seed == 1:
            raise RuntimeError("training failed")
        return SimpleNamespace(model=SimpleNamespace(training=True),
                               history={"val_epochs": [2], "val_loss": [0.2],
                                        "final_val": 0.2}, metadata={"initial_gate": None})

    monkeypatch.setattr(experiments, "train_model", fail_second_seed)
    monkeypatch.setattr(experiments, "evaluate", lambda *args: 0.3)
    with pytest.raises(experiments.AblationInterruptedError) as caught:
        experiments.run_gate_ablation(config)
    record = caught.value.result.record
    assert record["status"] == "incomplete"
    assert record["failure"]["phase"] == "training"
    runs = record["results"]["5"][0]["runs"]
    assert [run["status"] for run in runs] == ["complete", "failed"]
    assert runs[1]["model"] == "Vanilla RNN"
    assert runs[1]["split_seeds"] == config["split_seeds"]
    path = tmp_path / "partial.json"
    experiments.save_gate_ablation(path, caught.value.result)
    assert json.loads(path.read_text(encoding="utf-8"))["failure"]["seed"] == 1


def test_failed_fixed_set_has_accurate_failure_context(monkeypatch):
    def fail_set(*args, **kwargs):
        raise RuntimeError("fixed set failed")

    monkeypatch.setattr(experiments, "get_eval_sets", fail_set)
    with pytest.raises(experiments.AblationInterruptedError) as caught:
        experiments.run_gate_ablation(_reduced_config())
    assert caught.value.result.record["failure"] == {
        "phase": "fixed_test_set", "sequence_length": 5, "model": None,
        "label": None, "seed": None, "error_type": "RuntimeError",
        "message": "fixed set failed",
    }


def test_failed_initial_gradients_do_not_report_training_seed(monkeypatch):
    monkeypatch.setattr(experiments, "get_eval_sets", lambda *args, **kwargs:
                        {"test": (object(), object())})
    monkeypatch.setattr(experiments, "train_model", lambda *args, **kwargs:
                        SimpleNamespace(model=SimpleNamespace(training=True),
                                        history={"val_epochs": [2], "val_loss": [0.2],
                                                 "final_val": 0.2},
                                        metadata={"initial_gate": None}))
    monkeypatch.setattr(experiments, "evaluate", lambda *args: 0.3)

    def fail_gradients(*args, **kwargs):
        raise RuntimeError("gradient failed")

    monkeypatch.setattr(experiments, "measure_init_gradients", fail_gradients)
    with pytest.raises(experiments.AblationInterruptedError) as caught:
        experiments.run_gate_ablation(_reduced_config())
    failure = caught.value.result.record["failure"]
    assert failure["phase"] == "initial_gradients"
    assert failure["seed"] is None
    assert failure["model"] == "Vanilla RNN"


@pytest.fixture(scope="module")
def direct_notebook_ablation():
    script = """
import ast, hashlib, json
from scripts.capture_notebook_baseline import _load_notebook, _execute_definitions
notebook = _load_notebook()
ns = _execute_definitions(notebook)
import torch
for index, names in ((19, {'mean_std'}), (23, {'measure_init_gradients', 'decay_factor_from_curves'})):
    tree = ast.parse(''.join(notebook['cells'][index]['source']))
    definitions = ast.Module(body=[node for node in tree.body
                                   if isinstance(node, ast.FunctionDef) and node.name in names],
                             type_ignores=[])
    exec(compile(definitions, f'GradientVanishingAnalysis.ipynb:cell-{index}', 'exec'), ns)
ns['ABLATION_T'] = [5]
ns['SEEDS'] = [0, 1]
ns['EPOCHS'] = 2
ns['BATCH_SIZE'] = 8
ns['N_EVAL'] = 32
original_train = ns['train_model']
original_build = ns['build_model']
original_evaluate = ns['evaluate']
original_gradients = ns['measure_init_gradients']
runs, tests, gradients, initial = [], [], [], []
inside_training = [False]
def capture_build(name, config, gate_init, length, *args, **kwargs):
    model = original_build(name, config, gate_init, length, *args, **kwargs)
    if inside_training[0]:
        initial.append({'name': name, 'gate_mode': gate_init,
                        'gate': (model.W_z.bias.detach().tolist() if name == 'GRU' else
                                 model.W_f.bias.detach().tolist() if name == 'LSTM' else None),
                        'other_gate': (model.W_r.bias.detach().tolist() if name == 'GRU' else
                                       model.W_i.bias.detach().tolist() if name == 'LSTM' else None)})
    return model
def capture_train(*args, **kwargs):
    inside_training[0] = True
    try:
        model, hist = original_train(*args, **kwargs)
    finally:
        inside_training[0] = False
    with torch.no_grad():
        prediction = model(torch.zeros(args[2], 1, 1)).item()
    runs.append({'name': args[0], 'gate_mode': kwargs.get('gate_init', 'one'),
                 'seed': args[3], 'history': hist, 'prediction': prediction})
    return model, hist
def capture_evaluate(model, x, y):
    value = original_evaluate(model, x, y)
    if not inside_training[0]:
        tests.append(value)
    return value
def capture_gradients(name, length, config=None, gate_init='one', n_seeds=10):
    mean, curves = original_gradients(name, length, config or ns['DEFAULT_CONFIG'],
                                     gate_init, n_seeds=2)
    gradients.append({'name': name, 'gate_mode': gate_init,
                      'curves': curves.tolist(), 'mean': mean.tolist()})
    return mean, curves
ns['train_model'] = capture_train
ns['build_model'] = capture_build
ns['evaluate'] = capture_evaluate
ns['measure_init_gradients'] = capture_gradients
ns['display'] = lambda value: None
source = ''.join(notebook['cells'][27]['source'])
source = source.replace('ABLATION_T  = [100]', 'ABLATION_T  = [5]')
exec(compile(source, 'GradientVanishingAnalysis.ipynb:cell-27', 'exec'), ns)
print('DIRECT_ABLATION_JSON=' + json.dumps({
    'runs': runs, 'tests': tests, 'gradients': gradients, 'initial': initial,
    'rows': ns['ablation_rows'],
    'rng_sha256': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
}))
"""
    completed = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    return json.loads(next(line.split("=", 1)[1] for line in completed.stdout.splitlines()
                           if line.startswith("DIRECT_ABLATION_JSON=")))


def test_reduced_ablation_matches_direct_notebook(direct_notebook_ablation, tmp_path, monkeypatch):
    initial = []
    predictions = []
    real_build = training_module.build_model
    real_train = experiments.train_model

    def capture_build(name, params, gate_mode, length, *args, **kwargs):
        model = real_build(name, params, gate_mode, length, *args, **kwargs)
        initial.append({"name": name, "gate_mode": gate_mode,
                        "gate": (model.W_z.bias.detach().tolist() if name == "GRU" else
                                 model.W_f.bias.detach().tolist() if name == "LSTM" else None),
                        "other_gate": (model.W_r.bias.detach().tolist() if name == "GRU" else
                                       model.W_i.bias.detach().tolist() if name == "LSTM" else None)})
        return model

    def capture_train(*args, **kwargs):
        run = real_train(*args, **kwargs)
        with torch.no_grad():
            predictions.append(run.model(torch.zeros(args[2], 1, 1)).item())
        return run

    monkeypatch.setattr(training_module, "build_model", capture_build)
    monkeypatch.setattr(experiments, "train_model", capture_train)
    result = experiments.run_gate_ablation(_reduced_config()).record
    original = direct_notebook_ablation
    conditions = result["results"]["5"]
    assert result["status"] == "complete"
    assert len(conditions) == len(original["rows"]) == 7
    assert result["final_torch_rng_sha256"] == original["rng_sha256"]
    for index, condition in enumerate(conditions):
        row = original["rows"][index]
        assert (condition["model"], condition["label"]) == (row["Model"], row["Gate init"])
        for seed_index, run in enumerate(condition["runs"]):
            direct_index = index * 2 + seed_index
            reference = original["runs"][direct_index]
            assert initial[direct_index]["name"] == original["initial"][direct_index]["name"]
            for gate in ("gate", "other_gate"):
                if initial[direct_index][gate] is None:
                    assert original["initial"][direct_index][gate] is None
                else:
                    assert initial[direct_index][gate] == pytest.approx(
                        original["initial"][direct_index][gate], rel=1e-5, abs=1e-6)
            if condition["model"] == "Vanilla RNN":
                assert run["initial_gate"] is None
                assert condition["gate_parameter"] is None
            else:
                assert run["initial_gate"]["parameter"] == condition["gate_parameter"]
                assert run["initial_gate"]["retention_bias_sign"] == condition["retention_bias_sign"]
                assert run["initial_gate"]["bias"] == pytest.approx(
                    original["initial"][direct_index]["gate"], rel=1e-5, abs=1e-6)
                if condition["model"] == "LSTM" and condition["gate_mode"] == "chrono":
                    assert run["initial_gate"]["paired_input_bias"] == pytest.approx(
                        original["initial"][direct_index]["other_gate"], rel=1e-5, abs=1e-6)
            assert predictions[direct_index] == pytest.approx(reference["prediction"], rel=1e-5, abs=1e-6)
            assert run["seed"] == reference["seed"]
            assert run["test_mse"] == pytest.approx(original["tests"][direct_index], rel=1e-5, abs=1e-6)
            for key in ("train_loss", "val_loss"):
                assert run["history"][key] == pytest.approx(reference["history"][key], rel=1e-5, abs=1e-6)
            assert run["history"]["val_epochs"] == reference["history"]["val_epochs"]
            assert run["first_threshold_update"] is None
            assert run["model_training_mode_after_test"] is True
        expected = original["gradients"][index]
        assert condition["initial_gradients"]["timestep"] == [0, 1, 2, 3, 4]
        assert condition["initial_gradients"]["temporal_distance"] == [4, 3, 2, 1, 0]
        assert condition["initial_gradients"]["fit_timesteps"] == [2, 3]
        assert condition["initial_gradients"]["dtype"] == "torch.float64"
        np.testing.assert_allclose(condition["initial_gradients"]["normalized_curves"],
                                   expected["curves"], rtol=1e-5, atol=1e-6)
        assert condition["decay_mean"] == pytest.approx(
            float(np.mean([np.exp(-np.polyfit(np.arange(2, 4),
                                             np.log(np.clip(curve[2:4], 1e-300, None)), 1)[0])
                           for curve in expected["curves"]])), rel=1e-5, abs=1e-6)
    path = tmp_path / "ablation.json"
    experiments.save_gate_ablation(path, experiments.AblationResult(result))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["configuration"] == _reduced_config()
    assert saved["results"]["5"][0]["gate_mode"] == "default"
