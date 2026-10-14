"""Execute selected original notebook cells to capture pre-extraction CPU behavior.

This script does not contain a copy of the data, model, or training algorithms.
It loads and executes their definitions from the SHA-256-checked notebook.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Notebooks" / "GradientVanishingAnalysis.ipynb"
SOURCE_SHA256 = "71ec17cdb7fffcc39fef12bfe4977f75117ceedaa90c03109e79f51d98059f44"
SOURCE_COMMIT = "72418d9241b81e8ffefc73c917eab6663413cb10"
DEFINITION_CELLS = [3, 5, 7, 9, 11, 13]
FLOAT32_TOLERANCE = {"rtol": 1e-5, "atol": 1e-6}
FLOAT64_TOLERANCE = {"rtol": 1e-8, "atol": 1e-10}


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_digest(value: Any) -> str:
    return _digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _tensor(tensor: Any) -> dict[str, Any]:
    detached = tensor.detach().cpu().contiguous()
    return {"shape": list(detached.shape), "dtype": str(detached.dtype), "values": detached.tolist()}


def _tensor_digest(tensor: Any) -> str:
    return _digest(tensor.detach().cpu().contiguous().numpy().tobytes())


def _record(case_id: str, kind: str, cells: list[int], *, seed: int, precision: str,
            configuration: dict[str, Any], rng_order: str, input_data: dict[str, Any],
            expected: dict[str, Any]) -> dict[str, Any]:
    case = {
        "id": case_id,
        "kind": kind,
        "status": "executed_verified",
        "source_notebook_sha256": SOURCE_SHA256,
        "source_cells": cells,
        "source_snapshot_commit": SOURCE_COMMIT,
        "configuration": configuration,
        "seed": seed,
        "rng_order": rng_order,
        "device": "cpu",
        "precision": precision,
        "input": input_data,
        "expected": expected,
        "comparison": {
            "initial_parameters": "exact array equality",
            "computed_values": FLOAT64_TOLERANCE if precision == "float64" else FLOAT32_TOLERANCE,
            "reason": "Pinned CPU environment; exact initial parameters and approved float32/float64 tolerances from refactoring_spec.md",
        },
        "provenance": "direct execution of original notebook definitions; no saved notebook outputs used",
    }
    return case


def _load_notebook() -> dict[str, Any]:
    data = NOTEBOOK.read_bytes()
    if _digest(data) != SOURCE_SHA256:
        raise RuntimeError("Original notebook SHA-256 differs from the audited source")
    snapshot = subprocess.check_output(
        ["git", "show", f"{SOURCE_COMMIT}:Notebooks/GradientVanishingAnalysis.ipynb"], cwd=ROOT
    )
    if _digest(snapshot) != SOURCE_SHA256:
        raise RuntimeError("Committed source snapshot does not match the audited notebook")
    return json.loads(data)


def _versions() -> dict[str, str]:
    versions = {}
    for line in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        name, required = line.split("==", 1)
        actual = importlib.metadata.version(name)
        if actual != required:
            raise RuntimeError(f"Reference environment mismatch: {name}=={required} required, found {actual}")
        versions[name] = actual
    return versions


def _execute_definitions(notebook: dict[str, Any]) -> dict[str, Any]:
    # A CPU environment choice, set before the original cell imports Torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if "torch" in sys.modules:
        raise RuntimeError("Capture must start before Torch is imported")
    namespace: dict[str, Any] = {"__name__": "__notebook_baseline__"}
    for index in DEFINITION_CELLS:
        cell = notebook["cells"][index]
        if cell["cell_type"] != "code":
            raise RuntimeError(f"Expected code in notebook cell {index}")
        source = "".join(cell["source"])
        exec(compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)
    if namespace["device"].type != "cpu":
        raise RuntimeError("Reference capture requires CPU execution")
    return namespace


def capture() -> dict[str, Any]:
    notebook = _load_notebook()
    versions = _versions()
    ns = _execute_definitions(notebook)
    torch = ns["torch"]
    F = ns["F"]
    cases = []
    config = dict(ns["DEFAULT_CONFIG"])

    # Data: global training RNG, and separate fixed-set generators.
    ns["set_seed"](7)
    before = _tensor_digest(torch.get_rng_state())
    x, y = ns["generate_memorization_task"](4, 10)
    after = _tensor_digest(torch.get_rng_state())
    fixed = ns["get_eval_sets"](10)
    cases.append(_record(
        "data_t10_seed7", "data", [3, 5, 7], seed=7, precision="float32",
        configuration={"batch_size": 4, "sequence_length": 10, "validation_seed": ns["VAL_SEED"],
                       "test_seed": ns["TEST_SEED"], "fixed_size": ns["N_EVAL"]},
        rng_order="set_seed(7), generate_memorization_task(4,10) on global Torch RNG, then get_eval_sets(10) with independent generators",
        input_data={"shape": [10, 4, 1], "dtype": "torch.float32", "rng_before_sha256": before},
        expected={"x": _tensor(x), "y": _tensor(y), "rng_after_sha256": after,
                  "memoryless_baseline": ns["MEMORYLESS_BASELINE"],
                  "fixed_sets": {split: {"x_shape": list(pair[0].shape), "y_shape": list(pair[1].shape),
                                          "x_first_values": pair[0][0, :4, 0].tolist(),
                                          "y_first_values": pair[1][:4, 0].tolist(),
                                          "x_sha256": _tensor_digest(pair[0]), "y_sha256": _tensor_digest(pair[1])}
                                 for split, pair in fixed.items()}},
    ))

    for name in ns["MODEL_NAMES"]:
        for sequence_length, dtype in ((1, torch.float32), (10, torch.float32),
                                       (30, torch.float32), (30, torch.float64)):
            # Independent cases begin from the same seed. Construction and gate
            # initialization precede batch generation, exactly as in train_model.
            seed = 7
            ns["set_seed"](seed)
            rng_before = _tensor_digest(torch.get_rng_state())
            model = ns["build_model"](name, config, ns["MAIN_GATE_INIT"], sequence_length, dtype=dtype)
            parameters = {key: _tensor(value) for key, value in model.state_dict().items()}
            rng_after_model = _tensor_digest(torch.get_rng_state())
            x, y = ns["generate_memorization_task"](4, sequence_length)
            x = x.to(dtype=dtype).requires_grad_()  # diagnostic observation only
            y = y.to(dtype=dtype)
            gate_outputs: dict[str, list[Any]] = {"W_z": [], "W_r": [], "W_h": []}
            handles = []
            if name == "GRU":
                for gate_name in gate_outputs:
                    def observe(_module: Any, _inputs: Any, output: Any, *, key: str = gate_name) -> None:
                        gate_outputs[key].append(output.detach().cpu())
                    handles.append(getattr(model, gate_name).register_forward_hook(observe))
            prediction = model(x)
            for handle in handles:
                handle.remove()
            loss = F.mse_loss(prediction, y)
            loss.backward()
            expected: dict[str, Any] = {
                "initial_parameters": parameters,
                "rng_after_model_sha256": rng_after_model,
                "prediction": _tensor(prediction),
                "mse": loss.item(),
                "hidden_gradient_norms": model.get_gradient_norms(),
                "input_gradient_first_step": _tensor(x.grad[0]),
                "input_gradient_last_step": _tensor(x.grad[-1]),
                "first_parameter_gradient": _tensor(next(model.parameters()).grad),
                "training_mode_after_backward": model.training,
            }
            if name == "GRU":
                expected["gate_observations"] = {
                    gate_name: {"first_preactivation": _tensor(values[0]),
                                "last_preactivation": _tensor(values[-1]),
                                "first_activation": _tensor(torch.sigmoid(values[0]) if gate_name != "W_h" else torch.tanh(values[0]))}
                    for gate_name, values in gate_outputs.items()
                }
            if name == "LSTM":
                expected["cell_state_observations"] = {
                    "first": _tensor(model.c_states[0]), "last": _tensor(model.c_states[-1]),
                    "stored_states_detached": all(not cell.requires_grad for cell in model.c_states),
                }
            cases.append(_record(
                f"{name.lower().replace(' ', '_')}_t{sequence_length}_{str(dtype).split('.')[-1]}",
                "model", [3, 5, 7, 9, 11], seed=seed, precision=str(dtype).split(".")[-1],
                configuration={"model": name, "sequence_length": sequence_length, "batch_size": 4,
                               "hyperparameters": config, "gate_mode": ns["MAIN_GATE_INIT"],
                               "input_requires_grad": True},
                rng_order="set_seed(7), build_model on CPU, apply weight/gate init, then generate batch from global Torch RNG; no reseeding before forward/backward",
                input_data={"x": _tensor(x), "y": _tensor(y), "rng_before_sha256": rng_before},
                expected=expected,
            ))

    for model_name, modes in (("GRU", ["default", "zero", "one", "chrono", 2.0]),
                              ("LSTM", ["chrono"])):
        for mode in modes:
            ns["set_seed"](13)
            model = ns["build_model"](model_name, config, mode, 30)
            gate = model.W_z if model_name == "GRU" else model.W_f
            expected = {"target_gate_weight": _tensor(gate.weight), "target_gate_bias": _tensor(gate.bias),
                        "rng_after_model_sha256": _tensor_digest(torch.get_rng_state())}
            if model_name == "GRU":
                expected["reset_gate_bias"] = _tensor(model.W_r.bias)
            else:
                expected["input_gate_bias"] = _tensor(model.W_i.bias)
            cases.append(_record(
                f"{model_name.lower()}_gate_{str(mode).replace('.', '_')}",
                "gate_initialization", [3, 5, 9, 11], seed=13, precision="float32",
                configuration={"model": model_name, "sequence_length": 30,
                               "hyperparameters": config, "retention_bias_mode": mode},
                rng_order="set_seed(13), constructor initialization, apply_weight_init(default), apply_gate_init(mode,30), move to CPU",
                input_data={"sequence_length": 30, "hidden_size": config["hidden_size"]},
                expected=expected,
            ))

    for name in ns["MODEL_NAMES"]:
        ns["_EVAL_SETS"].clear()
        model, history = ns["train_model"](
            name, config, 10, seed=0, epochs=3, gate_init=ns["MAIN_GATE_INIT"],
            record_grad_epochs=(0, 2),
        )
        test_x, test_y = ns["get_eval_sets"](10)["test"]
        mode_after_train = model.training
        model.eval()
        mode_before_evaluate = model.training
        test_mse = ns["evaluate"](model, test_x, test_y)
        with torch.no_grad():
            test_predictions = model(test_x[:,:4,:])
        cases.append(_record(
            f"{name.lower().replace(' ', '_')}_training_t10", "training",
            [3, 5, 7, 9, 11, 13], seed=0, precision="float32",
            configuration={"model": name, "sequence_length": 10, "updates": 3,
                           "batch_size": ns["BATCH_SIZE"], "evaluation_size": ns["N_EVAL"],
                           "evaluation_interval": ns["EVAL_EVERY"], "hyperparameters": config,
                           "gate_mode": ns["MAIN_GATE_INIT"], "gradient_capture_updates": [0, 2]},
            rng_order="train_model reseeds to 0; build_model and Adam precede fixed validation creation; per update: zero_grad, fresh global-RNG batch, forward, backward, pre-clipping gradient capture, clip, step, post-step evaluation",
            input_data={"training_batch_shape": [10, ns["BATCH_SIZE"], 1],
                        "validation_shape": [10, ns["N_EVAL"], 1],
                        "test_shape": list(test_x.shape),
                        "test_x_first_values": test_x[0, :4, 0].tolist()},
            expected={"history": history, "test_mse": test_mse,
                      "test_predictions_first_four": _tensor(test_predictions),
                      "mode_after_train": mode_after_train,
                      "mode_before_evaluate": mode_before_evaluate,
                      "mode_after_evaluate": model.training},
        ))

    environment = {"python": sys.version.split()[0], "platform": platform.platform(),
                   "processor": platform.processor(), "dependency_versions": versions,
                   "torch_runtime_version": torch.__version__,
                   "torch_cuda_runtime": torch.version.cuda,
                   "device": str(ns["device"]),
                   "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "torch_threads": torch.get_num_threads()}
    for case in cases:
        case["environment_sha256"] = _canonical_digest(environment)
    result = {
        "schema_version": 1,
        "source": {"notebook": "Notebooks/GradientVanishingAnalysis.ipynb",
                   "notebook_sha256": SOURCE_SHA256, "source_snapshot_commit": SOURCE_COMMIT,
                   "executed_cell_indices": DEFINITION_CELLS,
                   "saved_notebook_outputs": "historical_unverified_not_used"},
        "environment": environment,
        "capture_method": "Selected original code cells executed in index order; case drivers call those notebook functions/classes directly",
        "cases": cases,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new JSON fixture path; existing files are never overwritten")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing fixture: {args.output}")
    result = capture()
    invocation = {"argv": [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output)],
                  "working_directory": str(Path.cwd())}
    result["invocation"] = invocation
    for case in result["cases"]:
        case["capture_invocation"] = invocation
        case["integrity_sha256"] = _canonical_digest({key: value for key, value in case.items()
                                                       if key != "integrity_sha256"})
    result["integrity_sha256"] = _canonical_digest({key: value for key, value in result.items()
                                                    if key != "integrity_sha256"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"Captured {len(result['cases'])} directly executed notebook cases in {args.output}")


if __name__ == "__main__":
    main()
