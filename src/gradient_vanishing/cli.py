"""Thin command boundary for existing notebook-equivalent experiment stages."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Mapping

from gradient_vanishing.config import ConfigError, load_config, prepare_runtime
from gradient_vanishing.run_manifest import (
    artifact_entry, atomic_json, base_manifest, environment_metadata, file_sha256,
    new_run_directory, utc_now,
)


STAGES = ("main", "gradients", "ablation", "sweep", "report")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gradient-vanishing",
                                     description="Run notebook-equivalent experiments or report saved results")
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in STAGES:
        sub = subparsers.add_parser(stage, help={
            "main": "Optuna tune, then compare default and tuned models",
            "gradients": "Measure initial hidden-state gradients",
            "ablation": "Run gate-initialization ablation",
            "sweep": "Run numeric retention-bias sweep",
            "report": "Render tables and figures from saved JSON only",
        }[stage])
        sub.add_argument("--config", type=Path, help="Partial JSON configuration; overrides package defaults")
        sub.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                         help="Override a config key, e.g. training.updates=3; repeatable")
        sub.add_argument("--output-dir", type=Path, help="Root for a new, uniquely named run directory")
        if stage == "report":
            sub.add_argument("--source", action="append", required=True, metavar="KIND=PATH",
                             help="Saved result: main, ablation, sweep, study, or initial_gradients; repeatable")
        else:
            sub.add_argument("--allow-full", action="store_true",
                             help="Explicitly allow the notebook-size experiment matrix")
    return parser


def _overrides(settings: list[str], output_dir: Path | None) -> dict[str, Any]:
    patch: dict[str, Any] = {}
    for setting in settings:
        if "=" not in setting:
            raise ConfigError(f"Expected KEY=JSON override: {setting}")
        dotted, raw = setting.split("=", 1)
        keys = dotted.split(".")
        if not all(key and key.isidentifier() for key in keys):
            raise ConfigError(f"Invalid override key: {dotted}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ConfigError(f"Invalid JSON override {dotted}: {error}") from error
        target = patch
        for key in keys[:-1]:
            nested = target.setdefault(key, {})
            if not isinstance(nested, dict):
                raise ConfigError(f"Conflicting overrides for {dotted}")
            target = nested
        target[keys[-1]] = value
    if output_dir is not None:
        patch["output_dir"] = str(output_dir)
    return patch


def _is_full_stage(stage: str, config: Mapping[str, Any]) -> bool:
    """Recognize the original matrix budget, even if device/output differ."""
    original = load_config()
    if stage == "main":
        return (config["sequence_lengths"] == original["sequence_lengths"]
                and config["model_names"] == original["model_names"]
                and config["seeds"] == original["seeds"]
                and config["training"]["updates"] >= original["training"]["updates"]
                and config["training"]["tuning_updates"] >= original["training"]["tuning_updates"]
                and config["tuning"]["trials"] >= original["tuning"]["trials"])
    if stage == "gradients":
        return (config["sequence_lengths"] == original["sequence_lengths"]
                and config["model_names"] == original["model_names"]
                and config["gradients"]["initial_seeds"] >= original["gradients"]["initial_seeds"])
    if stage == "ablation":
        return (config["ablation"]["sequence_lengths"] == original["ablation"]["sequence_lengths"]
                and config["ablation"]["gate_modes"] == original["ablation"]["gate_modes"]
                and config["seeds"] == original["seeds"]
                and config["training"]["updates"] >= original["training"]["updates"]
                and config["gradients"]["initial_seeds"] >= original["gradients"]["initial_seeds"])
    if stage == "sweep":
        return (config["gradients"]["sweep_biases"] == original["gradients"]["sweep_biases"]
                and config["gradients"]["sweep_sequence_length"] == original["gradients"]["sweep_sequence_length"]
                and config["gradients"]["sweep_seeds"] >= original["gradients"]["sweep_seeds"])
    return False


def _execute_experiment(stage: str, config: Mapping[str, Any]) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Yield completed raw records; imports occur after runtime preparation."""
    if stage == "gradients":
        from gradient_vanishing.gradients import measure_init_gradients
        for length in config["sequence_lengths"]:
            for name in config["model_names"]:
                result = measure_init_gradients(name, length, config)
                slug = name.lower().replace(" ", "_")
                yield "initial_gradients", f"initial_gradient_{slug}_T{length}.json", {
                    "kind": "initial_hidden_gradients", **asdict(result)}
    else:
        from gradient_vanishing.experiments import (
            run_gate_ablation, run_main_comparison, run_retention_bias_sweep,
        )
        if stage == "main":
            yield "main", "main.json", run_main_comparison(config).record
        elif stage == "ablation":
            yield "ablation", "gate_ablation.json", run_gate_ablation(config).record
        elif stage == "sweep":
            yield "sweep", "retention_bias_sweep.json", run_retention_bias_sweep(config).record
        else:
            raise ValueError(f"Unsupported experiment stage: {stage}")


def _report_sources(settings: list[str]) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for setting in settings:
        if "=" not in setting:
            raise ConfigError(f"Expected KIND=PATH source: {setting}")
        kind, raw = setting.split("=", 1)
        if kind not in ("main", "ablation", "sweep", "study", "initial_gradients") or not raw:
            raise ConfigError(f"Unsupported or empty report source: {setting}")
        if kind == "initial_gradients":
            sources.setdefault(kind, []).append(Path(raw))
        elif kind in sources:
            raise ConfigError(f"Duplicate report source kind: {kind}")
        else:
            sources[kind] = Path(raw)
    return sources


def _record_artifact(run_dir: Path, manifest: dict[str, Any], kind: str,
                     filename: str, record: dict[str, Any]) -> None:
    destination = run_dir / "results" / filename
    destination.parent.mkdir(exist_ok=True)
    atomic_json(destination, record)
    manifest["artifacts"].append(artifact_entry(run_dir, destination, kind))
    manifest["resolved_device"] = record.get("device", manifest["resolved_device"])
    manifest["dtype"] = record.get("dtype", record.get("training_dtype", manifest["dtype"]))
    manifest["selected_parameters"] = record.get("best_params", record.get("hyperparameters"))
    if kind == "main":
        for length, models in record.get("studies", {}).items():
            for name, study in models.items():
                study_file = destination.parent / f"study_{name.lower().replace(' ', '_')}_T{length}.json"
                atomic_json(study_file, study)
                manifest["artifacts"].append(artifact_entry(run_dir, study_file, "study"))


def _execute_report(run_dir: Path, manifest: dict[str, Any], sources: dict[str, Any]) -> None:
    from gradient_vanishing.reporting import render_reports
    report_dir = run_dir / "reports"
    index = render_reports(sources, report_dir)
    manifest["input_sources"] = index["sources"]
    for item in index["artifacts"]:
        file = report_dir / item["path"]
        manifest["artifacts"].append(artifact_entry(run_dir, file, item["kind"]))
    manifest["artifacts"].append(artifact_entry(run_dir, report_dir / "report_index.json", "report_index"))
    if any(source["status"] != "complete" for source in index["sources"]):
        manifest["status"] = "incomplete"
    else:
        manifest["status"] = "complete"


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    try:
        patch = _overrides(args.set, args.output_dir)
        config = load_config(args.config, patch)
        sources = _report_sources(args.source) if args.stage == "report" else None
        full = _is_full_stage(args.stage, config)
        if full and args.stage != "report" and not args.allow_full:
            raise ConfigError("Notebook-size matrix requires --allow-full; provide explicit reduced config otherwise")
        if args.stage != "report":
            prepare_runtime(config)
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    try:
        run_id, run_dir = new_run_directory(config["output_dir"], args.stage)
        command = ["gradient-vanishing", *(argv if argv is not None else sys.argv[1:])]
        manifest = base_manifest(run_id, args.stage, command, config, patch,
                                 "report_from_saved_results" if args.stage == "report" else
                                 "full_configuration" if full else "reduced_or_modified")
        if args.config is not None:
            manifest["configuration_file"] = {"path": str(args.config.resolve()),
                                               "sha256": file_sha256(args.config)}
        atomic_json(run_dir / "manifest.json", manifest)
        try:
            if args.stage == "report":
                _execute_report(run_dir, manifest, sources)
            else:
                for kind, filename, record in _execute_experiment(args.stage, config):
                    _record_artifact(run_dir, manifest, kind, filename, record)
                    atomic_json(run_dir / "manifest.json", manifest)
                manifest["status"] = "complete"
        except Exception as error:
            partial = getattr(getattr(error, "result", None), "record", None)
            manifest["status"] = "incomplete" if manifest["artifacts"] or partial is not None else "failed"
            manifest["error"] = {"type": type(error).__name__, "message": str(error)}
            if partial is not None:
                filename = {"main": "main.json", "ablation": "gate_ablation.json",
                            "sweep": "retention_bias_sweep.json"}.get(args.stage)
                if filename:
                    try:
                        _record_artifact(run_dir, manifest, args.stage, filename, partial)
                    except Exception as save_error:
                        manifest["error"]["result_persistence_error"] = {
                            "type": type(save_error).__name__, "message": str(save_error)}
            print(f"{args.stage} failed: {error}", file=sys.stderr)
        finally:
            manifest["completed_at"] = utc_now()
            manifest["environment"] = environment_metadata()
            atomic_json(run_dir / "manifest.json", manifest)
        print(run_dir)
        return 0 if manifest["status"] == "complete" else 1
    except (OSError, ValueError, TypeError) as error:
        print(f"Run error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
