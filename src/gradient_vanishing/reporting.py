"""Read saved numeric results and render notebook-equivalent reports.

This module imports no experiment, model, training, or gradient code.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gradient_vanishing.config import load_config


MODEL_ORDER = ("Vanilla RNN", "LSTM", "GRU")
MODEL_COLORS = {name: f"C{index}" for index, name in enumerate(MODEL_ORDER)}
SOURCE_KINDS = {"main", "ablation", "sweep", "initial_gradients", "study"}
_TABLE_FILES = {"main_summary.csv", "gate_ablation.csv", "retention_bias_values.csv", "optuna_trials.csv"}
_FIGURE_FILE = re.compile(
    r"(?:main_learning_curves|training_gradient_heatmap_T\d+|"
    r"retention_bias_sweep_T\d+|initial_gradient_curves_T\d+)\.(?:png|svg)\Z"
)


class ReportInputError(ValueError):
    """A saved result cannot support the requested notebook report."""


def load_saved_result(path: str | Path, kind: str) -> dict[str, Any]:
    """Read one saved JSON record and check its report-level schema."""
    source = Path(path)
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReportInputError(f"Cannot read {source}: {error}") from error
    if not isinstance(record, dict):
        raise ReportInputError(f"{source} must contain a JSON object")
    required = {
        "main": ("schema_version", "configuration", "sequence_length_order", "model_order",
                 "seed_order", "results", "best_params"),
        "ablation": ("schema_version", "configuration", "sequence_length_order", "condition_order",
                     "seed_order", "threshold", "results"),
        "sweep": ("schema_version", "configuration", "sequence_length", "model_order", "bias_order",
                  "results", "theoretical_reference_line"),
        "initial_gradients": ("model", "sequence_length", "normalized_curves", "geometric_mean",
                              "decay_mean", "timestep", "measurement"),
        "study": ("schema_version", "model", "sequence_length", "trials", "best_trial_number", "search_space"),
    }
    if kind not in required:
        raise ReportInputError(f"Unsupported result kind: {kind}")
    missing = [key for key in required[kind] if key not in record]
    if missing:
        raise ReportInputError(f"{source} is missing {kind} fields: {', '.join(missing)}")
    if kind != "initial_gradients" and record["schema_version"] != 1:
        raise ReportInputError(f"{source} has unsupported {kind} schema version")
    if kind == "initial_gradients" and record.get("kind") != "initial_hidden_gradients":
        raise ReportInputError(f"{source} is not an initial hidden-gradient record")
    return record


def _complete_runs(bucket: Mapping[str, Any] | None, seeds: Sequence[int]) -> list[dict[str, Any]] | None:
    if bucket is None or not isinstance(bucket.get("runs"), list):
        return None
    runs = bucket["runs"]
    if len(runs) != len(seeds) or [run.get("seed") for run in runs] != list(seeds):
        return None
    if any(run.get("status") != "complete" or "test_mse" not in run for run in runs):
        return None
    return runs


def _mean_spread(values: Sequence[float], precision: int = 4) -> tuple[float, float | None, str]:
    numbers = np.asarray(values, dtype=float)
    if not len(numbers) or not np.isfinite(numbers).all():
        raise ReportInputError("A complete metric must contain finite values")
    mean = float(np.mean(numbers))
    spread = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None
    displayed = f"{mean:.{precision}f} ± {spread:.{precision}f}" if spread is not None else f"{mean:.{precision}f} ± NA"
    return mean, spread, displayed


def main_summary_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Cell 19: one length/model row; sample spread is never inferred from one seed."""
    rows = []
    seeds = record["seed_order"]
    for length in record["sequence_length_order"]:
        for name in record["model_order"]:
            buckets = record["results"].get(str(length), {}).get(name, {})
            default = _complete_runs(buckets.get("default"), seeds)
            tuned = _complete_runs(buckets.get("tuned"), seeds)
            default_mean, _, default_text = _mean_spread([run["test_mse"] for run in default]) if default else (None, None, "")
            tuned_mean, _, tuned_text = _mean_spread([run["test_mse"] for run in tuned]) if tuned else (None, None, "")
            params = record.get("best_params", {}).get(str(length), {}).get(name)
            rows.append({
                "T": length, "Model": name,
                "Test MSE (mặc định)": default_text,
                "Test MSE (sau tuning)": tuned_text,
                "Chênh lệch (mặc định − tuned)": (
                    f"{default_mean - tuned_mean:+.4f}" if default_mean is not None and tuned_mean is not None else ""),
                "Tham số tuned": json.dumps(params, ensure_ascii=False, sort_keys=True) if params else "",
                "Status": "complete" if default and tuned else "incomplete",
                "Default seeds present": ",".join(str(run.get("seed")) for run in buckets.get("default", {}).get("runs", [])),
                "Tuned seeds present": ",".join(str(run.get("seed")) for run in buckets.get("tuned", {}).get("runs", [])),
            })
    return rows


def learning_curve_data(record: Mapping[str, Any]) -> dict[tuple[str, int, str], tuple[list[int], list[float]]]:
    """Cell 21: aligned mean validation MSE only, preserving the missing band."""
    curves = {}
    for condition in ("default", "tuned"):
        for length in record["sequence_length_order"]:
            for name in record["model_order"]:
                bucket = record["results"].get(str(length), {}).get(name, {}).get(condition)
                runs = _complete_runs(bucket, record["seed_order"])
                if runs is None:
                    raise ReportInputError(f"Missing complete learning curve: T={length}, {name}, {condition}")
                epochs = runs[0]["history"]["val_epochs"]
                if any(run["history"]["val_epochs"] != epochs for run in runs):
                    raise ReportInputError(f"Misaligned validation checks: T={length}, {name}, {condition}")
                values = np.asarray([run["history"]["val_loss"] for run in runs], dtype=float)
                if values.ndim != 2 or values.shape[1] != len(epochs) or not np.isfinite(values).all():
                    raise ReportInputError(f"Invalid validation values: T={length}, {name}, {condition}")
                curves[(condition, length, name)] = (list(epochs), np.mean(values, axis=0).tolist())
    return curves


def training_heatmap_data(record: Mapping[str, Any]) -> dict[str, Any]:
    """Cell 25: tuned first seed, raw norms divided by last norm, then log10."""
    length = max(record["sequence_length_order"])
    updates = record["configuration"]["training"]["gradient_capture_updates"]
    names = record["model_order"]
    matrices = {}
    for update in updates:
        rows = []
        for name in names:
            bucket = record["results"].get(str(length), {}).get(name, {}).get("tuned")
            runs = _complete_runs(bucket, record["seed_order"])
            if runs is None:
                raise ReportInputError(f"Missing tuned gradient run: T={length}, {name}")
            norms = runs[0]["history"]["grad_norms"]
            values = np.asarray(norms.get(str(update), norms.get(update)), dtype=float)
            if values.shape != (length,):
                raise ReportInputError(f"Invalid gradient profile: T={length}, {name}, update={update}")
            rows.append(np.log10(np.clip(values / (values[-1] + 1e-30), 1e-10, None)))
        matrices[str(update)] = np.stack(rows).tolist()
    stored = record.get("gradient_heatmap")
    if stored is not None:
        for update in updates:
            reference = np.asarray(stored["log10_relative_norms"][str(update)], dtype=float)
            if not np.allclose(matrices[str(update)], reference, rtol=1e-8, atol=1e-10, equal_nan=True):
                raise ReportInputError(f"Saved heatmap disagrees with raw gradients at update {update}")
    return {"sequence_length": length, "model_order": list(names),
            "capture_updates": list(updates), "matrices": matrices,
            "vmin": -10, "vmax": 0}


def ablation_summary_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Cell 27: raw test spread, first checked threshold, initial decay."""
    rows = []
    seeds = record["seed_order"]
    threshold = record["threshold"]
    for length in record["sequence_length_order"]:
        available = record["results"].get(str(length), [])
        for requested in record["condition_order"]:
            matches = [item for item in available if item.get("model") == requested["model"]
                       and item.get("label") == requested["label"]]
            if len(matches) > 1:
                raise ReportInputError(f"Duplicate ablation condition: {requested}")
            item = matches[0] if matches else None
            runs = _complete_runs(item, seeds)
            complete = bool(item and item.get("status") == "complete" and runs and item.get("decay_mean") is not None)
            test_text = _mean_spread([run["test_mse"] for run in runs])[2] if complete else ""
            speeds = [run.get("first_threshold_update") for run in runs] if complete else []
            successful = [speed for speed in speeds if speed is not None]
            speed_text = (f"{np.mean(successful):.0f} ({len(successful)}/{len(seeds)} seed đạt)"
                          if successful else f"không đạt (0/{len(seeds)})") if complete else ""
            if complete:
                spread = item.get("decay_sample_std")
                decay_text = f"{item['decay_mean']:.3f} ± {spread:.3f}" if spread is not None else f"{item['decay_mean']:.3f} ± NA"
            else:
                decay_text = ""
            rows.append({"T": length, "Model": requested["model"], "Gate init": requested["label"],
                         "Test MSE": test_text,
                         f"Epoch đến val MSE < {threshold}": speed_text,
                         "Hệ số suy giảm r (khởi tạo)": decay_text,
                         "Status": "complete" if complete else "incomplete",
                         "Seeds present": ",".join(str(run.get("seed")) for run in item.get("runs", [])) if item else ""})
    return rows


def sweep_plot_data(record: Mapping[str, Any]) -> dict[str, Any]:
    """Cell 28: sigmoid(b) versus initial hidden-gradient decay, model order."""
    output = {"sequence_length": record["sequence_length"], "models": {},
              "reference_line": record["theoretical_reference_line"]}
    for name in record["model_order"]:
        points = record["results"].get(name, [])
        if len(points) != len(record["bias_order"]):
            raise ReportInputError(f"Missing bias values for {name}")
        if any(point.get("status") != "complete" or point.get("retention_bias") != bias
               for point, bias in zip(points, record["bias_order"])):
            raise ReportInputError(f"Incomplete or misordered bias sweep for {name}")
        if any(not math.isclose(point["theoretical_retention"], 1 / (1 + math.exp(-bias)),
                                rel_tol=1e-12, abs_tol=1e-12)
               for point, bias in zip(points, record["bias_order"])):
            raise ReportInputError(f"Incorrect theoretical retention values for {name}")
        output["models"][name] = {
            "bias": list(record["bias_order"]),
            "retention": [point["theoretical_retention"] for point in points],
            "decay": [point["decay_mean"] for point in points],
        }
    return output


def initial_gradient_plot_data(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Cell 23: per-seed and geometric mean ratios by backward distance."""
    if not records:
        raise ReportInputError("Initial-gradient plot needs at least one saved record")
    length = records[0]["sequence_length"]
    by_name = {}
    for record in records:
        name = record["model"]
        if name in by_name or name not in MODEL_ORDER or record["sequence_length"] != length:
            raise ReportInputError("Initial-gradient records must have unique models and one length")
        curves = np.asarray(record["normalized_curves"], dtype=float)
        mean = np.asarray(record["geometric_mean"], dtype=float)
        if curves.ndim != 2 or curves.shape[1] != length or mean.shape != (length,):
            raise ReportInputError(f"Invalid initial-gradient shape for {name}")
        by_name[name] = {"seeds": curves[:, ::-1].tolist(), "geometric_mean": mean[::-1].tolist(),
                         "decay_mean": record["decay_mean"], "measurement": record["measurement"]}
    return {"sequence_length": length, "distance": list(range(length)),
            "models": {name: by_name[name] for name in MODEL_ORDER if name in by_name}}


def study_trial_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Optuna trial observations, including failed and pruned trials."""
    return [{"Model": record["model"], "T": record["sequence_length"],
             "Study": record.get("study_name", ""),
             "Direction": record.get("direction", ""),
             "Trial": trial["number"], "Seed": trial["seed"],
             "Parameters": json.dumps(trial["params"], sort_keys=True),
             "Final validation MSE": trial["value"] if trial["value"] is not None else "",
             "Intermediate validation MSE": json.dumps(trial.get("intermediate_values", {}), sort_keys=True),
             "State": trial["state"],
             "Selected best": trial["number"] == record["best_trial_number"]}
            for trial in record["trials"]]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ReportInputError(f"No rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig: Any, directory: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("png", "svg"):
        path = directory / f"{stem}.{suffix}"
        metadata = {"Date": None} if suffix == "svg" else None
        fig.savefig(path, dpi=180, bbox_inches="tight", metadata=metadata)
        paths.append(path.name)
    plt.close(fig)
    return paths


def _draw_learning_curves(record: Mapping[str, Any], directory: Path) -> list[str]:
    curves = learning_curve_data(record)
    lengths = record["sequence_length_order"]
    fig, axes = plt.subplots(2, len(lengths), figsize=(6 * len(lengths), 9),
                             sharex=True, sharey=True, squeeze=False)
    fig.suptitle(f"Validation MSE by training update (mean of {len(record['seed_order'])} seeds)")
    for row, condition in enumerate(("default", "tuned")):
        for col, length in enumerate(lengths):
            axis = axes[row, col]
            for name in record["model_order"]:
                x, y = curves[(condition, length, name)]
                axis.plot(x, y, color=MODEL_COLORS[name], linewidth=2, label=name)
            axis.axhline(1 / 3, color="red", linestyle=":", label="Theoretical memoryless MSE (1/3)")
            axis.set_yscale("log")
            axis.set_title(f"{condition} · T={length}")
            axis.grid(True, which="both", alpha=0.3)
            if row == 1:
                axis.set_xlabel("Training update")
            if col == 0:
                axis.set_ylabel("Validation MSE (log scale)")
    axes[0, 0].legend(fontsize="small")
    fig.tight_layout()
    return _save_figure(fig, directory, "main_learning_curves")


def _draw_heatmap(record: Mapping[str, Any], directory: Path) -> list[str]:
    data = training_heatmap_data(record)
    updates = data["capture_updates"]
    fig, axes = plt.subplots(len(updates), 1, figsize=(16, 2.2 * len(updates) + 1),
                             sharex=True, squeeze=False)
    for row, update in enumerate(updates):
        axis = axes[row, 0]
        matrix = np.asarray(data["matrices"][str(update)], dtype=float)
        image = axis.imshow(matrix, cmap="magma", vmin=-10, vmax=0, aspect="auto", interpolation="nearest")
        axis.set_yticks(range(len(data["model_order"])), labels=data["model_order"])
        axis.set_title(f"T={data['sequence_length']}, notebook epoch {update} · float32, display floor 1e-10")
        fig.colorbar(image, ax=axis, label="log10 relative hidden-gradient norm")
    step = max(data["sequence_length"] // 10, 1)
    axes[-1, 0].set_xticks(range(0, data["sequence_length"], step))
    axes[-1, 0].set_xlabel("Timestep t")
    fig.tight_layout()
    return _save_figure(fig, directory, f"training_gradient_heatmap_T{data['sequence_length']}")


def _draw_initial_gradients(records: Sequence[Mapping[str, Any]], directory: Path) -> list[str]:
    data = initial_gradient_plot_data(records)
    fig, axis = plt.subplots(figsize=(10, 5))
    for name, series in data["models"].items():
        for curve in series["seeds"]:
            axis.semilogy(data["distance"], curve, color=MODEL_COLORS[name], alpha=0.15, linewidth=1)
        axis.semilogy(data["distance"], series["geometric_mean"], color=MODEL_COLORS[name],
                      linewidth=2.5, label=f"{name} (r ≈ {series['decay_mean']:.3f}/step)")
    axis.set_xlabel("Temporal distance from final step (T−1−t)")
    axis.set_ylabel("Initial hidden-gradient norm / final-step norm (log scale)")
    axis.set_title(f"Initial hidden-state gradients, T={data['sequence_length']} (float64)")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    return _save_figure(fig, directory, f"initial_gradient_curves_T{data['sequence_length']}")


def _draw_sweep(record: Mapping[str, Any], directory: Path) -> list[str]:
    data = sweep_plot_data(record)
    fig, axis = plt.subplots(figsize=(10, 5))
    for name, marker in (("LSTM", "o-"), ("GRU", "s-")):
        series = data["models"][name]
        axis.plot(series["retention"], series["decay"], marker,
                  label=name, color=MODEL_COLORS[name])
    line = np.asarray(data["reference_line"], dtype=float)
    axis.plot(line[:, 0], line[:, 1], "k--", alpha=0.5, label="Theoretical gate path: y=x")
    axis.set_xlabel("Theoretical retention = sigmoid(b) (LSTM f; GRU 1−z)")
    axis.set_ylabel("Measured initial hidden-gradient decay per step r")
    axis.set_title(f"Numeric gate bias and initial-gradient decay, T={data['sequence_length']}")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    return _save_figure(fig, directory, f"retention_bias_sweep_T{data['sequence_length']}")


def _source_metadata(path: Path, kind: str, record: Mapping[str, Any]) -> dict[str, Any]:
    configuration = record.get("configuration")
    scope = "full_configuration" if configuration == load_config() else "reduced_or_modified"
    status = record.get("status", "complete")
    if kind == "study" and record.get("best_trial_number") is None:
        status = "incomplete"
    return {
        "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "kind": kind, "status": status, "scope": scope,
        "model": record.get("model"), "sequence_length": record.get("sequence_length"),
        "model_order": record.get("model_order"),
        "sequence_length_order": record.get("sequence_length_order"),
        "seeds": record.get("seed_order", record.get("seeds")),
        "hyperparameters": record.get("hyperparameters"),
        "gate_mode": record.get("gate_mode"),
        "device": record.get("device"), "dtype": record.get("dtype"),
        "training_dtype": record.get("training_dtype"),
        "initial_gradient_dtype": record.get("initial_gradient_dtype"),
        "python_version": record.get("python_version"), "torch_version": record.get("torch_version"),
        "optuna_version": record.get("optuna_version"),
        "notebook_sha256": record.get("notebook_sha256"),
        "measurement": record.get("measurement"),
        "fit_timesteps": record.get("fit_timesteps"),
        "zero_ratio_count": record.get("zero_ratio_count"),
        "nonfinite_ratio_count": record.get("nonfinite_ratio_count"),
        "source_designation": record.get("source_designation", "experiment_result"),
        "study_name": record.get("study_name"),
        "study_direction": record.get("direction"),
    }


def _clear_previous_artifacts(directory: Path) -> None:
    """Remove only files recorded by an earlier report in this directory."""
    index_path = directory / "report_index.json"
    if not index_path.exists():
        return
    try:
        previous = json.loads(index_path.read_text(encoding="utf-8"))
        names = [entry["path"] for entry in previous["artifacts"]]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ReportInputError(f"Cannot read prior report index {index_path}: {error}") from error
    if any(not isinstance(name, str) or Path(name).name != name
           or (name not in _TABLE_FILES and not _FIGURE_FILE.fullmatch(name)) for name in names):
        raise ReportInputError(f"Prior report index contains an unmanaged artifact: {index_path}")
    for name in names:
        (directory / name).unlink(missing_ok=True)
    index_path.unlink()


def render_reports(sources: Mapping[str, str | Path | Sequence[str | Path]], output_dir: str | Path) -> dict[str, Any]:
    """Regenerate available tables and figures solely from saved result JSON.

    Keys are ``main``, ``ablation``, ``sweep``, ``initial_gradients`` (a
    sequence of paths), and ``study``. Missing inputs are listed as skipped.
    No experiment function is imported or invoked.
    """
    unknown = set(sources) - SOURCE_KINDS
    if unknown:
        raise ReportInputError(f"Unsupported source kinds: {', '.join(sorted(unknown))}")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _clear_previous_artifacts(directory)
    index: dict[str, Any] = {"schema_version": 1, "sources": [], "artifacts": [], "skipped": []}

    def artifact(kind: str, paths: Sequence[str], source_ids: Sequence[int]) -> None:
        index["artifacts"].extend({"kind": kind, "path": path, "source_ids": list(source_ids)}
                                  for path in paths)

    with plt.rc_context({"svg.hashsalt": "gradient-vanishing-report", "font.family": "DejaVu Sans"}):
        for kind in ("main", "ablation", "sweep", "initial_gradients", "study"):
            if kind not in sources:
                index["skipped"].append({"kind": kind, "reason": "source result not supplied"})
                continue
            paths = sources[kind] if kind == "initial_gradients" else [sources[kind]]
            if kind == "initial_gradients" and isinstance(paths, (str, Path)):
                paths = [paths]
            if not paths:
                index["skipped"].append({"kind": kind, "reason": "no source result paths supplied"})
                continue
            records = []
            source_ids = []
            for candidate in paths:
                path = Path(candidate)
                record = load_saved_result(path, kind)
                source_ids.append(len(index["sources"]))
                index["sources"].append(_source_metadata(path, kind, record))
                records.append(record)
            if kind == "main":
                rows = main_summary_rows(records[0])
                filename = "main_summary.csv"
                _write_csv(directory / filename, rows)
                artifact(kind, [filename], source_ids)
                for figure_kind, draw in (("main_learning_curves", _draw_learning_curves),
                                          ("training_gradient_heatmap", _draw_heatmap)):
                    try:
                        artifact(figure_kind, draw(records[0], directory), source_ids)
                    except (ReportInputError, KeyError, TypeError, ValueError) as error:
                        index["skipped"].append({"kind": figure_kind, "reason": str(error)})
            elif kind == "ablation":
                filename = "gate_ablation.csv"
                _write_csv(directory / filename, ablation_summary_rows(records[0]))
                artifact(kind, [filename], source_ids)
            elif kind == "sweep":
                rows = []
                record = records[0]
                for name in record["model_order"]:
                    available = {point["retention_bias"]: point
                                 for point in record["results"].get(name, [])}
                    for bias in record["bias_order"]:
                        point = available.get(bias)
                        rows.append({"Model": name, "T": record["sequence_length"],
                                     "Retention bias b": bias,
                                     "Theoretical sigmoid(b)": (point["theoretical_retention"] if point else ""),
                                     "Measured initial decay r": (point.get("decay_mean", "") if point else ""),
                                     "Status": point["status"] if point else "missing"})
                filename = "retention_bias_values.csv"
                _write_csv(directory / filename, rows)
                artifact(kind, [filename], source_ids)
                try:
                    artifact("retention_bias_sweep", _draw_sweep(record, directory), source_ids)
                except (ReportInputError, KeyError, TypeError, ValueError) as error:
                    index["skipped"].append({"kind": "retention_bias_sweep", "reason": str(error)})
            elif kind == "initial_gradients":
                supplied_models = {record["model"] for record in records}
                missing_models = [name for name in MODEL_ORDER if name not in supplied_models]
                if missing_models:
                    index["skipped"].append({"kind": "initial_gradient_models",
                                             "reason": f"source result not supplied for: {', '.join(missing_models)}"})
                artifact(kind, _draw_initial_gradients(records, directory), source_ids)
            elif kind == "study":
                filename = "optuna_trials.csv"
                _write_csv(directory / filename, study_trial_rows(records[0]))
                artifact(kind, [filename], source_ids)
    (directory / "report_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                                  encoding="utf-8")
    return index
