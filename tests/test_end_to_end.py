"""Small real CLI pipeline; the full empirical matrix is deliberately excluded."""

import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from gradient_vanishing.reporting import load_saved_result, main_summary_rows


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "ticket13_reduced_cpu.json"


def _run(stage, config, root, *extra):
    command = [sys.executable, "-m", "gradient_vanishing", stage,
               "--config", str(config), "--output-dir", str(root), *extra]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)
    assert completed.returncode == 0, (command, completed.stdout, completed.stderr)
    directory = Path(completed.stdout.strip().splitlines()[-1])
    assert directory.is_dir()
    return directory


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_science_equal(left, right):
    """Optuna's generated study name is metadata; compare all other fields."""
    if isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            if key not in ("study_name", "tuning_study_name"):
                _assert_science_equal(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_science_equal(a, b)
    elif isinstance(left, float):
        assert left == pytest.approx(right, rel=1e-5, abs=1e-6)
    else:
        assert left == right


def test_real_reduced_pipeline_round_trip_and_repeatability(tmp_path):
    """Exercise tune -> train -> evaluate -> save -> report with real inputs."""
    config = _read(CONFIG)
    assert config["sequence_lengths"] == [10]
    assert config["model_names"] == ["Vanilla RNN", "LSTM", "GRU"]
    assert config["tuning"]["trials"] == 2 and config["training"]["updates"] == 3
    root = tmp_path / "runs"
    main = _run("main", CONFIG, root)
    repeated = _run("main", CONFIG, root)
    gradients = _run("gradients", CONFIG, root)
    ablation = _run("ablation", CONFIG, root)
    sweep = _run("sweep", CONFIG, root)

    for directory in (main, repeated, gradients, ablation, sweep):
        manifest = _read(directory / "manifest.json")
        assert manifest["status"] == "complete"
        assert manifest["scope"] == "reduced_or_modified"
        assert manifest["effective_configuration"]["training"]["updates"] == 3
        assert manifest["environment"]["dependencies"]["torch"]
        assert re.fullmatch(r"[0-9a-f]{40}", manifest["git_commit"])
        assert manifest["resolved_device"] == "cpu"
        assert manifest["notebook_reference_sha256"] == (
            "71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44")
        for item in manifest["artifacts"]:
            source = directory / item["path"]
            assert source.is_file()
            assert hashlib.sha256(source.read_bytes()).hexdigest() == item["sha256"]

    first = load_saved_result(main / "results" / "main.json", "main")
    second = load_saved_result(repeated / "results" / "main.json", "main")
    _assert_science_equal(first, second)
    assert first["model_order"] == config["model_names"]
    assert first["seed_order"] == [0, 1]
    assert all(len(first["studies"]["10"][name]["trials"]) == 2 for name in first["model_order"])
    assert all(len(first["results"]["10"][name][condition]["runs"]) == 2
               for name in first["model_order"] for condition in ("default", "tuned"))
    study = main / "results" / "study_vanilla_rnn_T10.json"
    assert load_saved_result(study, "study")["best_trial_number"] in (0, 1)

    gradient_files = [gradients / "results" / f"initial_gradient_{slug}_T10.json"
                      for slug in ("vanilla_rnn", "lstm", "gru")]
    assert all(load_saved_result(path, "initial_gradients")["seeds"] == [0, 1]
               for path in gradient_files)
    gate_record = load_saved_result(ablation / "results" / "gate_ablation.json", "ablation")
    assert len(gate_record["condition_order"]) == 7
    sweep_record = load_saved_result(sweep / "results" / "retention_bias_sweep.json", "sweep")
    assert sweep_record["bias_order"] == [-1.0, 0.0, 1.0]
    assert sweep_record["model_order"] == ["LSTM", "GRU"]

    sources = {"main": main / "results" / "main.json",
               "study": study,
               "ablation": ablation / "results" / "gate_ablation.json",
               "sweep": sweep / "results" / "retention_bias_sweep.json"}
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in [*sources.values(), *gradient_files]}
    flags = [item for kind, path in sources.items() for item in ("--source", f"{kind}={path}")]
    flags += [item for path in gradient_files for item in ("--source", f"initial_gradients={path}")]
    report = _run("report", CONFIG, root, *flags)
    report_manifest = _read(report / "manifest.json")
    index = _read(report / "reports" / "report_index.json")
    assert report_manifest["scope"] == "report_from_saved_results"
    assert report_manifest["status"] == "complete"
    assert len(index["sources"]) == 7 and not index["skipped"]
    assert {entry["path"]: entry["sha256"] for entry in index["sources"]} == before
    assert {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*sources.values(), *gradient_files]} == before
    assert (report / "reports" / "retention_bias_sweep_T10.png").is_file()
    assert (report / "reports" / "retention_bias_sweep_T10.svg").is_file()
    assert (report / "reports" / "optuna_trials.csv").is_file()
    with (report / "reports" / "main_summary.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    expected_rows = main_summary_rows(first)
    assert len(csv_rows) == len(expected_rows) == 3
    for actual, expected in zip(csv_rows, expected_rows):
        assert actual == {key: str(value) for key, value in expected.items()}
    with (report / "reports" / "retention_bias_values.csv").open(encoding="utf-8", newline="") as handle:
        sweep_rows = list(csv.DictReader(handle))
    assert len(sweep_rows) == 6
    assert float(sweep_rows[0]["Measured initial decay r"]) == (
        sweep_record["results"]["LSTM"][0]["decay_mean"])
