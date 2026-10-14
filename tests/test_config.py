import json
import subprocess
import sys

import pytest

from gradient_vanishing.config import ConfigError, load_config


def test_default_matches_notebook_constants():
    config = load_config()
    assert config["schema_version"] == 1
    assert config["sequence_lengths"] == [10, 30, 100]
    assert config["model_names"] == ["Vanilla RNN", "LSTM", "GRU"]
    assert config["seeds"] == [0, 1, 2, 3, 4]
    assert config["training"] == {"updates": 300, "tuning_updates": 150, "batch_size": 128, "evaluation_interval": 10, "evaluation_size": 2048, "gradient_capture_updates": [0, 50, 150, 299]}
    assert config["split_seeds"] == {"validation": 10000, "test": 20000, "tuning_offset": 1000}
    assert config["default_hyperparameters"] == {"learning_rate": 0.001, "hidden_size": 32, "initialization": "default"}
    assert config["main_gate_mode"] == "one"
    assert config["tuning"] == {"trials": 20, "learning_rate_min": 0.0001, "learning_rate_max": 0.05, "learning_rate_log": True, "hidden_sizes": [16, 32, 64], "initializations": ["orthogonal", "xavier", "default"], "sampler_seed": 42, "pruner_startup_trials": 5, "pruner_warmup_steps": 50}
    assert config["ablation"] == {"sequence_lengths": [100], "gate_modes": ["zero", "one", "chrono"], "validation_threshold": 0.01}
    assert config["gradients"] == {"initial_seeds": 10, "sweep_seeds": 5, "sweep_sequence_length": 100, "sweep_biases": [-4.0 + 0.5 * i for i in range(17)]}


def test_file_and_command_overrides_have_precedence(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"training": {"updates": 40, "gradient_capture_updates": [0, 39]}}), encoding="utf-8")
    result = load_config(path, {"training": {"updates": 20, "gradient_capture_updates": [0, 19]}})
    assert result["training"]["updates"] == 20
    assert result["training"]["batch_size"] == 128
    assert load_config(path)["training"]["updates"] == 40
    assert load_config()["training"]["updates"] == 300


@pytest.mark.parametrize("patch, message", [
    ({"unknown": 1}, "unknown"),
    ({"training": {"mystery": 1}}, "training.mystery"),
    ({"training": {"updates": True}}, "training.updates"),
    ({"training": []}, "training"),
    ({"seeds": "0,1"}, "seeds"),
    ({"training": {"batch_size": 0}}, "training.batch_size"),
    ({"training": {"updates": 2, "gradient_capture_updates": [0, 299]}}, "training.gradient_capture_updates"),
    ({"tuning": {"learning_rate_min": 0.1}}, "tuning.learning_rate_min"),
    ({"sequence_lengths": [10, 10]}, "sequence_lengths"),
    ({"main_gate_mode": "reset"}, "main_gate_mode"),
    ({"schema_version": 2}, "schema_version"),
])
def test_rejects_invalid_configuration(patch, message):
    with pytest.raises(ConfigError, match=message):
        load_config(overrides=patch)


def test_rejects_invalid_json_file(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{bad", encoding="utf-8")
    with pytest.raises(ConfigError, match="JSON"):
        load_config(path)


def test_import_and_config_check_do_not_load_training_or_plotting():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import gradient_vanishing; from gradient_vanishing.config import load_config; load_config(); print(sorted(set(sys.modules) & {'torch', 'matplotlib', 'optuna'}))"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "[]"


def test_cuda_determinism_environment_is_set_before_torch_import():
    result = subprocess.run(
        [sys.executable, "-c", "import os,sys; from gradient_vanishing.config import load_config, prepare_runtime; prepare_runtime(load_config(overrides={'device':'cuda'})); print(os.environ['CUBLAS_WORKSPACE_CONFIG'], 'torch' in sys.modules)"],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == ":4096:8 False"
