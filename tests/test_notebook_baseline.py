"""The immutable oracle comes from executing the original notebook, not package code."""

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "capture_notebook_baseline.py"
FIXTURE = ROOT / "tests" / "fixtures" / "notebook_baseline.json"
NOTEBOOK_HASH = "71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44"


def _canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    committed = json.loads(FIXTURE.read_text(encoding="utf-8"))
    destination = tmp_path_factory.mktemp("notebook_baseline") / "fresh.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(destination)],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    fresh = json.loads(destination.read_text(encoding="utf-8"))
    return committed, fresh


def _compare(observed, reference, tolerance, *, exact=False):
    if isinstance(reference, dict):
        assert observed.keys() == reference.keys()
        for key in reference:
            parameter_field = key in {"initial_parameters", "target_gate_weight", "target_gate_bias",
                                      "reset_gate_bias", "input_gate_bias"}
            _compare(observed[key], reference[key], tolerance, exact=exact or parameter_field)
    elif isinstance(reference, list):
        assert len(observed) == len(reference)
        for actual, expected in zip(observed, reference):
            _compare(actual, expected, tolerance, exact=exact)
    elif type(reference) is float and not exact:
        assert math.isclose(observed, reference, rel_tol=tolerance["rtol"], abs_tol=tolerance["atol"]), (observed, reference)
    else:
        assert observed == reference


def test_original_notebook_capture_creates_verifiable_cases(baseline):
    committed, fresh = baseline
    assert fresh["schema_version"] == 1
    assert fresh["source"]["notebook_sha256"] == NOTEBOOK_HASH
    assert {case["kind"] for case in fresh["cases"]} == {"data", "model", "gate_initialization", "training"}
    assert {case["id"] for case in fresh["cases"]} == {case["id"] for case in committed["cases"]}


def test_fixture_schema_provenance_and_checksums(baseline):
    committed, fresh = baseline
    assert hashlib.sha256((ROOT / committed["source"]["notebook"]).read_bytes()).hexdigest() == NOTEBOOK_HASH
    assert committed["source"]["source_snapshot_commit"] == "72418d9241b81e8ffefc73c917eab6663413cb10"
    assert committed["source"]["saved_notebook_outputs"] == "historical_unverified_not_used"
    assert committed["source"]["executed_cell_indices"] == fresh["source"]["executed_cell_indices"] == [3, 5, 7, 9, 11, 13]
    assert committed["environment"]["device"] == "cpu"
    assert committed["environment"]["torch_deterministic_algorithms"] is True
    assert committed["environment"]["dependency_versions"] == fresh["environment"]["dependency_versions"]
    for field in ("python", "torch_runtime_version", "torch_cuda_runtime", "device",
                  "torch_deterministic_algorithms", "cublas_workspace_config",
                  "cuda_visible_devices", "torch_threads"):
        assert committed["environment"][field] == fresh["environment"][field], field
    for document in (committed, fresh):
        assert document["integrity_sha256"] == _canonical_digest({k: v for k, v in document.items() if k != "integrity_sha256"})
        for case in document["cases"]:
            assert case["integrity_sha256"] == _canonical_digest({k: v for k, v in case.items() if k != "integrity_sha256"})
            assert case["environment_sha256"] == _canonical_digest(document["environment"])
            assert case["source_notebook_sha256"] == NOTEBOOK_HASH
            assert case["source_cells"]
            assert case["source_snapshot_commit"] == committed["source"]["source_snapshot_commit"]
            assert case["capture_invocation"] == document["invocation"]
            assert case["status"] == "executed_verified"
            assert case["device"] == "cpu"
            assert case["seed"] >= 0
            assert case["rng_order"] and case["provenance"]
            assert case["input"] and case["expected"]
            assert case["comparison"]["initial_parameters"] == "exact array equality"
    assert (ROOT / "Notebooks" / "GradientVanishingAnalysis.ipynb").stat().st_size > 0


def test_fresh_execution_matches_committed_numerical_oracle(baseline):
    committed, fresh = baseline
    for reference, observed in zip(committed["cases"], fresh["cases"]):
        assert reference["id"] == observed["id"]
        assert reference["configuration"] == observed["configuration"]
        assert reference["precision"] == observed["precision"]
        tolerance = reference["comparison"]["computed_values"]
        assert tolerance == ({"rtol": 1e-8, "atol": 1e-10} if reference["precision"] == "float64"
                             else {"rtol": 1e-5, "atol": 1e-6})
        _compare(observed["input"], reference["input"], tolerance)
        _compare(observed["expected"], reference["expected"], tolerance)
    assert len(committed["cases"]) == len(fresh["cases"]) == len({case["id"] for case in fresh["cases"]})


def test_data_model_and_training_observations_are_complete(baseline):
    committed, _ = baseline
    cases = {case["id"]: case for case in committed["cases"]}
    data = cases["data_t10_seed7"]["expected"]
    assert data["x"]["shape"] == [10, 4, 1]
    assert data["y"]["shape"] == [4, 1]
    assert data["x"]["values"][0] == data["y"]["values"]
    assert all(step == [[0.0]] * 4 for step in data["x"]["values"][1:])
    assert data["fixed_sets"]["val"]["x_sha256"] != data["fixed_sets"]["test"]["x_sha256"]
    for model in ("vanilla_rnn", "lstm", "gru"):
        for length in (1, 10, 30):
            case = cases[f"{model}_t{length}_float32"]
            assert case["expected"]["prediction"]["shape"] == [4, 1]
            assert len(case["expected"]["hidden_gradient_norms"]) == length
            assert case["expected"]["hidden_gradient_norms"][0] > 0  # full BPTT reaches earliest state
            assert math.isfinite(case["expected"]["mse"])
        assert cases[f"{model}_t30_float64"]["precision"] == "float64"
        training = cases[f"{model}_training_t10"]["expected"]
        assert len(training["history"]["train_loss"]) == 3
        assert training["history"]["val_epochs"] == [3]
        assert set(training["history"]["grad_norms"]) == {"0", "2"}
        assert training["mode_before_evaluate"] is False
        assert training["mode_after_evaluate"] is True  # preserved notebook side effect
    assert cases["gru_gate_one"]["expected"]["target_gate_bias"]["values"] == [-1.0] * 32
    assert cases["gru_gate_zero"]["expected"]["target_gate_bias"]["values"] == [0.0] * 32
    assert cases["gru_gate_default"]["expected"]["reset_gate_bias"] == cases["gru_gate_one"]["expected"]["reset_gate_bias"]
    forget = cases["lstm_gate_chrono"]["expected"]["target_gate_bias"]["values"]
    input_gate = cases["lstm_gate_chrono"]["expected"]["input_gate_bias"]["values"]
    assert input_gate == [-value for value in forget]


def test_capture_refuses_to_overwrite_existing_fixture(tmp_path):
    destination = tmp_path / "fresh.json"
    destination.write_text("keep", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(destination)],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "Refusing to overwrite" in result.stderr
    assert destination.read_text(encoding="utf-8") == "keep"
