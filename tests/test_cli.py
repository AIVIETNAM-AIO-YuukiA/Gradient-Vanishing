"""Command boundary, persistence, and provenance checks for Ticket 12."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from gradient_vanishing import cli
from gradient_vanishing.reporting import load_saved_result
from gradient_vanishing.run_manifest import atomic_json, base_manifest
from gradient_vanishing.config import load_config


def _reduced(tmp_path):
    path = tmp_path / "reduced.json"
    path.write_text(json.dumps({
        "sequence_lengths": [5], "model_names": ["Vanilla RNN"], "seeds": [0],
        "device": "cpu", "training": {"updates": 1, "tuning_updates": 1,
            "batch_size": 2, "evaluation_interval": 1, "evaluation_size": 4,
            "gradient_capture_updates": [0]},
        "tuning": {"trials": 1, "hidden_sizes": [2], "initializations": ["default"]},
        "ablation": {"sequence_lengths": [5], "gate_modes": ["zero"]},
        "gradients": {"initial_seeds": 2, "sweep_seeds": 2,
            "sweep_sequence_length": 5, "sweep_biases": [0.0]},
    }), encoding="utf-8")
    return path


def test_help_validation_and_import_are_safe(tmp_path):
    env = {**os.environ, "PYTHONPATH": "src"}
    help_run = subprocess.run([sys.executable, "-m", "gradient_vanishing", "--help"],
                              capture_output=True, text=True, env=env, check=False)
    assert help_run.returncode == 0
    assert all(stage in help_run.stdout for stage in ("main", "gradients", "ablation", "sweep", "report"))
    import_run = subprocess.run([sys.executable, "-c", "import gradient_vanishing.cli; import sys; "
                                 "print('torch' in sys.modules)"], capture_output=True,
                                text=True, env=env, check=False)
    assert import_run.stdout.strip() == "False"
    bad = tmp_path / "bad.json"
    bad.write_text('{"training": {"updates": 0}}', encoding="utf-8")
    assert cli.main(["gradients", "--config", str(bad), "--output-dir", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()
    assert cli.main(["unknown"]) != 0


def test_precedence_full_guard_and_atomic_manifest(tmp_path, monkeypatch):
    reduced = _reduced(tmp_path)
    seen = []

    def fake_run(stage, config):
        seen.append((stage, config))
        return [("initial_gradients", "initial_gradient_Vanilla_RNN_T5.json", {
            "kind": "initial_hidden_gradients", "model": "Vanilla RNN", "sequence_length": 5,
            "normalized_curves": [[1.0] * 5], "geometric_mean": [1.0] * 5,
            "decay_mean": 1.0, "timestep": list(range(5)), "measurement": "test",
        })]

    monkeypatch.setattr(cli, "_execute_experiment", fake_run)
    out = tmp_path / "runs"
    assert cli.main(["gradients", "--config", str(reduced), "--set", "training.batch_size=3",
                     "--output-dir", str(out)]) == 0
    assert seen[0][1]["training"]["batch_size"] == 3
    run = next(out.iterdir())
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and manifest["stage"] == "gradients"
    assert manifest["configuration_overrides"] == {"training": {"batch_size": 3},
                                                      "output_dir": str(out)}
    assert manifest["effective_configuration"]["training"]["batch_size"] == 3
    assert manifest["configuration_file"]["sha256"] == hashlib.sha256(reduced.read_bytes()).hexdigest()
    assert manifest["scope"] == "reduced_or_modified"
    assert manifest["seeds"] == [0, 1]
    assert manifest["model_names"] == ["Vanilla RNN"]
    assert manifest["sequence_lengths"] == [5]
    assert manifest["started_at"] and manifest["completed_at"]
    assert manifest["notebook_reference_sha256"]
    artifact = manifest["artifacts"][0]
    result = run / artifact["path"]
    assert artifact["sha256"] == hashlib.sha256(result.read_bytes()).hexdigest()
    assert load_saved_result(result, "initial_gradients")["decay_mean"] == 1.0
    assert cli.main(["main", "--output-dir", str(tmp_path / "full")]) == 2
    assert not (tmp_path / "full").exists()


@pytest.mark.parametrize("stage", ["main", "ablation", "sweep"])
def test_reduced_stage_dispatch_and_result_persistence(tmp_path, monkeypatch, stage):
    reduced = _reduced(tmp_path)
    invoked = []

    def fake_run(selected, config):
        invoked.append(selected)
        record = {"status": "complete", "device": "cpu",
                  "hyperparameters": {"lr": 0.01}, "value": 0.125}
        if selected == "main":
            record["studies"] = {"5": {"Vanilla RNN": {
                "schema_version": 1, "model": "Vanilla RNN", "sequence_length": 5,
                "trials": [{"number": 0, "params": {"lr": 0.01}, "value": 0.125,
                            "state": "complete"}], "best_trial_number": 0,
                "search_space": {"lr": {"kind": "float"}}}}}
        return [(stage, f"{stage}.json", record)]

    monkeypatch.setattr(cli, "_execute_experiment", fake_run)
    out = tmp_path / "runs"
    assert cli.main([stage, "--config", str(reduced), "--output-dir", str(out)]) == 0
    assert invoked == [stage]
    run = next(out.iterdir())
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    if stage == "ablation":
        assert manifest["model_names"] == ["Vanilla RNN", "LSTM", "GRU"]
        assert manifest["sequence_lengths"] == [5]
    if stage == "sweep":
        assert manifest["model_names"] == ["LSTM", "GRU"]
        assert manifest["seeds"] == [0, 1]
    assert manifest["selected_parameters"] == {"lr": 0.01}
    assert json.loads((run / manifest["artifacts"][0]["path"]).read_text(encoding="utf-8"))["value"] == 0.125
    if stage == "main":
        assert len(manifest["artifacts"]) == 2
        study = run / manifest["artifacts"][1]["path"]
        assert load_saved_result(study, "study")["best_trial_number"] == 0


def test_partial_and_failed_runs_preserve_status_and_records(tmp_path, monkeypatch):
    reduced = _reduced(tmp_path)

    class Interrupted(RuntimeError):
        def __init__(self):
            super().__init__("stopped during training")
            self.result = type("Result", (), {"record": {"status": "incomplete",
                "results": {"5": {"Vanilla RNN": {"default": {"runs": [
                    {"seed": 0, "status": "failed"}]}}}}}})()

    def partial(stage, config):
        raise Interrupted()
        yield  # retain generator behavior

    monkeypatch.setattr(cli, "_execute_experiment", partial)
    out = tmp_path / "runs"
    assert cli.main(["main", "--config", str(reduced), "--output-dir", str(out)]) == 1
    run = next(out.iterdir())
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "incomplete"
    assert manifest["error"]["type"] == "Interrupted"
    assert json.loads((run / "results" / "main.json").read_text(encoding="utf-8"))["results"]["5"]["Vanilla RNN"]["default"]["runs"][0]["status"] == "failed"

    def failed(stage, config):
        raise RuntimeError("setup failed")
        yield

    monkeypatch.setattr(cli, "_execute_experiment", failed)
    assert cli.main(["main", "--config", str(reduced), "--output-dir", str(out)]) == 1
    runs = list(out.iterdir())
    assert len(runs) == 2
    failed_run = next(path for path in runs if path != run)
    failure = json.loads((failed_run / "manifest.json").read_text(encoding="utf-8"))
    assert failure["status"] == "failed" and failure["artifacts"] == []


def test_real_reduced_gradients_then_report_without_training(tmp_path, monkeypatch):
    reduced = _reduced(tmp_path)
    out = tmp_path / "runs"
    assert cli.main(["gradients", "--config", str(reduced), "--output-dir", str(out)]) == 0
    gradient_run = next(out.iterdir())
    gradient_manifest = json.loads((gradient_run / "manifest.json").read_text(encoding="utf-8"))
    result = gradient_run / gradient_manifest["artifacts"][0]["path"]
    saved = load_saved_result(result, "initial_gradients")
    assert len(saved["normalized_curves"]) == 2
    assert saved["model"] == "Vanilla RNN"

    def forbidden(*args, **kwargs):
        raise AssertionError("report command executed an experiment")

    monkeypatch.setattr(cli, "_execute_experiment", forbidden)
    assert cli.main(["report", "--config", str(reduced), "--output-dir", str(out),
                     "--source", f"initial_gradients={result}"]) == 0
    runs = list(out.iterdir())
    assert len(runs) == 2
    report_run = next(path for path in runs if path != gradient_run)
    report_manifest = json.loads((report_run / "manifest.json").read_text(encoding="utf-8"))
    assert report_manifest["status"] == "complete"
    assert report_manifest["scope"] == "report_from_saved_results"
    assert report_manifest["seeds"] is None
    assert report_manifest["input_sources"][0]["model"] == "Vanilla RNN"
    assert (report_run / "reports" / "initial_gradient_curves_T5.png").exists()
    assert hashlib.sha256(result.read_bytes()).hexdigest() == gradient_manifest["artifacts"][0]["sha256"]


def test_manifest_serialization_preserves_torch_rng_and_numeric_values(tmp_path):
    state = torch.get_rng_state().clone()
    config = load_config(overrides={"device": "cpu", "sequence_lengths": [5]})
    manifest = base_manifest("run", "gradients", ["gradient-vanishing", "gradients"],
                             config, {"sequence_lengths": [5]}, "reduced_or_modified")
    target = tmp_path / "manifest.json"
    atomic_json(target, manifest)
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["environment"]["dependencies"]["torch"]
    assert saved["environment"]["deterministic_algorithms"] is not None
    assert saved["notebook_reference_sha256"] == manifest["notebook_reference_sha256"]
    assert torch.equal(torch.get_rng_state(), state)
    value = 0.2858848571777344
    atomic_json(target, {"exact_float": value, "nonfinite": float("nan")})
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "exact_float": value, "nonfinite": "NaN"}
    assert torch.equal(torch.get_rng_state(), state)
