"""Notebook cell 17 comparison order and a reduced independent reference run."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import gradient_vanishing.experiments as experiments
from gradient_vanishing.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def _reduced_config(*, seeds=(0, 1)):
    return load_config(overrides={
        "sequence_lengths": [10], "model_names": ["Vanilla RNN"], "seeds": list(seeds),
        "training": {"updates": 3, "tuning_updates": 2,
                     "gradient_capture_updates": [0, 2]},
        "tuning": {"trials": 2}, "device": "cpu",
    })


def test_default_grid_schedule_fixed_test_reuse_and_summary(monkeypatch):
    config = load_config()
    events = []
    tests = {}

    def fake_sets(length, configuration, *, device):
        pair = (object(), object())
        tests[length] = pair
        events.append(("fixed_test", length))
        return {"test": pair}

    def fake_tune(name, length, configuration, *, device):
        events.append(("tune", length, name))
        params = {"lr": 0.02, "hidden_size": 16, "init_method": "xavier"}
        return SimpleNamespace(best_params=params, study=SimpleNamespace(study_name="study"),
                               record={"best_trial_number": 1, "best_params": params})

    def fake_train(name, configuration, length, seed, *, epochs, gate_init,
                   record_grad_epochs, hyperparameters, device):
        events.append(("train", length, name, seed, dict(hyperparameters), epochs,
                       gate_init, tuple(record_grad_epochs)))
        history = {"train_loss": [0.2], "val_epochs": [epochs], "val_loss": [0.3],
                   "final_val": 0.3,
                   "grad_norms": {index: [1.0] * length for index in record_grad_epochs}}
        return SimpleNamespace(model=SimpleNamespace(training=True, seed=seed),
                               history=history, metadata={"seed": seed, "updates": epochs})

    def fake_evaluate(model, x, y):
        events.append(("test", model.seed, x, y))
        return 0.1 + model.seed / 100

    monkeypatch.setattr(experiments, "get_eval_sets", fake_sets)
    monkeypatch.setattr(experiments, "tune_model", fake_tune)
    monkeypatch.setattr(experiments, "train_model", fake_train)
    monkeypatch.setattr(experiments, "evaluate", fake_evaluate)
    result = experiments.run_main_comparison(config, device="cpu").record

    assert result["status"] == "complete"
    assert len([event for event in events if event[0] == "tune"]) == 9
    assert len([event for event in events if event[0] == "train"]) == 90
    assert len([event for event in events if event[0] == "test"]) == 90
    assert [event[:3] for event in events[:3]] == [
        ("fixed_test", 10), ("tune", 10, "Vanilla RNN"), ("train", 10, "Vanilla RNN")]
    first = [event for event in events if event[0] == "train"][:10]
    assert [event[3] for event in first] == [0, 1, 2, 3, 4] * 2
    assert all(event[5:] == (300, "one", (0, 50, 150, 299)) for event in first)
    assert [event[4] for event in first[:5]] == [
        {"lr": 0.001, "hidden_size": 32, "init_method": "default"}] * 5
    assert [event[4] for event in first[5:]] == [
        {"lr": 0.02, "hidden_size": 16, "init_method": "xavier"}] * 5
    for length in config["sequence_lengths"]:
        pair = tests[length]
        assert sum(event[0] == "fixed_test" and event[1] == length for event in events) == 1
        assert sum(event[0] == "test" and event[2] is pair[0] and event[3] is pair[1]
                   for event in events) == 30
    bucket = result["results"]["10"]["Vanilla RNN"]["tuned"]
    assert bucket["tuning_best_trial_number"] == 1
    assert bucket["test_mean"] == pytest.approx(np.mean([0.1 + seed / 100 for seed in range(5)]))
    assert bucket["test_sample_std"] == pytest.approx(
        np.std([0.1 + seed / 100 for seed in range(5)], ddof=1))
    assert list(bucket["runs"][0]["history"]["grad_norms"]) == [0, 50, 150, 299]
    assert result["gradient_heatmap"]["sequence_length"] == 100


def test_partial_result_is_serializable_and_keeps_completed_runs(monkeypatch, tmp_path):
    config = _reduced_config()
    params = {"lr": 0.02, "hidden_size": 16, "init_method": "xavier"}
    monkeypatch.setattr(experiments, "get_eval_sets", lambda *args, **kwargs:
                        {"test": (object(), object())})
    monkeypatch.setattr(experiments, "tune_model", lambda *args, **kwargs:
                        SimpleNamespace(best_params=params, study=SimpleNamespace(study_name="s"),
                                        record={"best_trial_number": 0}))

    def train(name, configuration, length, seed, **kwargs):
        if seed == 1:
            raise RuntimeError("training failed")
        return SimpleNamespace(model=SimpleNamespace(training=True),
                               history={"final_val": 0.2, "grad_norms": {}}, metadata={})

    monkeypatch.setattr(experiments, "train_model", train)
    monkeypatch.setattr(experiments, "evaluate", lambda *args: 0.3)
    with pytest.raises(experiments.ComparisonInterruptedError) as caught:
        experiments.run_main_comparison(config)
    result = caught.value.result
    assert result.record["status"] == "incomplete"
    assert result.record["failure"]["phase"] == "training"
    runs = result.record["results"]["10"]["Vanilla RNN"]["default"]["runs"]
    assert [run["status"] for run in runs] == ["complete", "failed"]
    assert runs[1]["model"] == "Vanilla RNN"
    assert runs[1]["sequence_length"] == 10
    assert runs[1]["split_seeds"] == config["split_seeds"]
    path = tmp_path / "partial.json"
    experiments.save_main_comparison(path, result)
    assert json.loads(path.read_text(encoding="utf-8"))["failure"]["seed"] == 1


@pytest.fixture(scope="module")
def direct_notebook_comparison():
    script = """
import hashlib, json
from scripts.capture_notebook_baseline import _load_notebook, _execute_definitions
notebook = _load_notebook()
ns = _execute_definitions(notebook)
import torch
ns['SEQ_LENS'] = [10]
ns['MODEL_NAMES'] = ['Vanilla RNN']
ns['SEEDS'] = [0, 1]
ns['EPOCHS'] = 3
ns['TUNE_EPOCHS'] = 2
ns['N_TRIALS'] = 2
for index in (15, 17):
    source = ''.join(notebook['cells'][index]['source'])
    exec(compile(source, f'GradientVanishingAnalysis.ipynb:cell-{index}', 'exec'), ns)
study = ns['best_params'][10]['Vanilla RNN']
runs = ns['results'][10]['Vanilla RNN']
print('DIRECT_COMPARISON_JSON=' + json.dumps({
    'best_params': study,
    'runs': runs,
    'rng_sha256': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
}))
"""
    completed = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    return json.loads(next(line.split("=", 1)[1] for line in completed.stdout.splitlines()
                           if line.startswith("DIRECT_COMPARISON_JSON=")))


def test_reduced_comparison_matches_direct_notebook_and_fixture(direct_notebook_comparison, tmp_path):
    result = experiments.run_main_comparison(_reduced_config()).record
    original = direct_notebook_comparison
    assert result["status"] == "complete"
    assert result["best_params"]["10"]["Vanilla RNN"] == original["best_params"]
    assert result["final_torch_rng_sha256"] == original["rng_sha256"]
    assert result["studies"]["10"]["Vanilla RNN"]["trial_budget"] == 2
    assert result["studies"]["10"]["Vanilla RNN"]["updates_per_trial"] == 2
    for condition in ("default", "tuned"):
        actual = result["results"]["10"]["Vanilla RNN"][condition]
        expected = original["runs"][condition]
        assert [run["seed"] for run in actual["runs"]] == [run["seed"] for run in expected]
        for run, reference in zip(actual["runs"], expected):
            assert run["test_mse"] == pytest.approx(reference["test_mse"], rel=1e-5, abs=1e-6)
            assert run["final_val"] == pytest.approx(reference["hist"]["final_val"], rel=1e-5, abs=1e-6)
            for key in ("train_loss", "val_loss"):
                assert run["history"][key] == pytest.approx(reference["hist"][key], rel=1e-5, abs=1e-6)
            assert run["history"]["val_epochs"] == reference["hist"]["val_epochs"]
            for update in (0, 2):
                assert run["history"]["grad_norms"][update] == pytest.approx(
                    reference["hist"]["grad_norms"][str(update)], rel=1e-5, abs=1e-6)
            assert run["model_training_mode_after_test"] is True
            assert run["metadata"]["split_seeds"] == _reduced_config()["split_seeds"]
            assert run["metadata"]["updates"] == 3
            assert run["metadata"]["gate_mode"] == "one"
            assert run["tuning_study_name"] == (result["studies"]["10"]["Vanilla RNN"]["study_name"]
                                                   if condition == "tuned" else None)
        values = [run["test_mse"] for run in expected]
        assert actual["test_mean"] == pytest.approx(float(np.mean(values)), rel=1e-5, abs=1e-6)
        assert actual["test_sample_std"] == pytest.approx(float(np.std(values, ddof=1)), rel=1e-5, abs=1e-6)

    tuned_first = original["runs"]["tuned"][0]["hist"]["grad_norms"]
    for update in (0, 2):
        norms = np.asarray(tuned_first[str(update)], dtype=float)
        expected = np.log10(np.clip(norms / (norms[-1] + 1e-30), 1e-10, None))
        assert result["gradient_heatmap"]["log10_relative_norms"][str(update)][0] == pytest.approx(
            expected.tolist(), rel=1e-5, abs=1e-6)

    fixture = json.loads((ROOT / "tests/fixtures/notebook_baseline.json").read_text(encoding="utf-8"))
    # The independent three-update default seed-0 probe precedes the notebook main comparison.
    baseline = next(case for case in fixture["cases"]
                    if case["id"] == "vanilla_rnn_training_t10")["expected"]
    default_zero = result["results"]["10"]["Vanilla RNN"]["default"]["runs"][0]
    assert default_zero["history"]["train_loss"] == pytest.approx(
        baseline["history"]["train_loss"], rel=1e-5, abs=1e-6)
    path = tmp_path / "comparison.json"
    experiments.save_main_comparison(path, experiments.ComparisonResult(result))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["configuration"] == _reduced_config()
    assert saved["notebook_sha256"] == "71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44"
    assert saved["python_version"] == sys.version.split()[0]
    assert saved["torch_version"] == torch.__version__
