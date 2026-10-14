"""Saved-result reporting contracts; records here are explicitly synthetic."""

import csv
import hashlib
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

import gradient_vanishing.experiments as experiments
import gradient_vanishing.reporting as reporting
from gradient_vanishing.config import load_config

BASELINE = Path(__file__).parent / "fixtures/notebook_baseline.json"


def _save(path, record):
    record["source_designation"] = "synthetic_test_fixture"
    path.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return path


def _main_record():
    config = load_config(overrides={
        "sequence_lengths": [10], "model_names": ["Vanilla RNN", "LSTM"],
        "seeds": [0, 1], "training": {"updates": 3, "tuning_updates": 2,
                                      "gradient_capture_updates": [0, 2]},
        "device": "cpu",
    })
    results = {"10": {}}
    best = {"10": {}}
    for model_index, name in enumerate(config["model_names"]):
        results["10"][name] = {}
        best["10"][name] = {"lr": 0.01, "hidden_size": 16, "init_method": "xavier"}
        for condition_index, condition in enumerate(("default", "tuned")):
            runs = []
            for seed in (0, 1):
                base = 0.2 + 0.1 * model_index + 0.05 * condition_index + 0.02 * seed
                history = {"train_loss": [base + 0.4] * 3,
                           "val_epochs": [2, 3], "val_loss": [base + 0.2, base + 0.1],
                           "final_val": base + 0.1,
                           "grad_norms": {"0": [float(t + model_index + 1) for t in range(10)],
                                          "2": [float(t + model_index + 2) for t in range(10)]}}
                runs.append({"seed": seed, "status": "complete", "test_mse": base,
                             "history": history, "metadata": {"seed": seed}})
            results["10"][name][condition] = {"runs": runs}
    return {"schema_version": 1, "status": "complete", "configuration": config,
            "sequence_length_order": [10], "model_order": config["model_names"],
            "condition_order": ["default", "tuned"], "seed_order": [0, 1],
            "results": results, "best_params": best, "gradient_heatmap": None,
            "python_version": "synthetic", "torch_version": "synthetic",
            "device": "cpu", "dtype": "torch.float32", "notebook_sha256": "synthetic"}


def _ablation_record():
    config = load_config(overrides={"ablation": {"sequence_lengths": [10]},
                                    "seeds": [0, 1], "device": "cpu"})
    conditions = [{"model": "Vanilla RNN", "label": "—", "gate_mode": "default"}]
    conditions.extend({"model": name, "label": mode, "gate_mode": mode}
                      for name in ("LSTM", "GRU") for mode in ("zero", "one", "chrono"))
    results = []
    for index, condition in enumerate(conditions):
        runs = [{"seed": seed, "status": "complete", "test_mse": 0.2 + index / 100 + seed / 100,
                 "first_threshold_update": 3 if seed == 1 else None}
                for seed in (0, 1)]
        results.append({**condition, "status": "complete", "runs": runs,
                        "decay_mean": 0.7 + index / 100, "decay_sample_std": 0.02})
    return {"schema_version": 1, "status": "complete", "configuration": config,
            "sequence_length_order": [10], "condition_order": conditions,
            "seed_order": [0, 1], "threshold": 0.01, "results": {"10": results},
            "device": "cpu", "training_dtype": "torch.float32",
            "initial_gradient_dtype": "torch.float64"}


def _sweep_record():
    config = load_config(overrides={"gradients": {"sweep_sequence_length": 10,
                                                 "sweep_seeds": 2,
                                                 "sweep_biases": [-4.0, 0.0, 4.0]},
                                    "device": "cpu"})
    results = {}
    for index, name in enumerate(("LSTM", "GRU")):
        results[name] = [{"model": name, "retention_bias": bias,
                          "theoretical_retention": float(1 / (1 + np.exp(-bias))),
                          "decay_mean": 0.2 + index / 10 + bias / 100,
                          "status": "complete"}
                         for bias in (-4.0, 0.0, 4.0)]
    return {"schema_version": 1, "status": "complete", "configuration": config,
            "sequence_length": 10, "model_order": ["LSTM", "GRU"],
            "bias_order": [-4.0, 0.0, 4.0], "results": results,
            "theoretical_reference_line": [[0.0, 0.0], [1.0, 1.0]],
            "measurement": "initial hidden-gradient decay", "device": "cpu",
            "dtype": "torch.float64"}


def _gradient_record(name):
    curves = [[0.1 + t / 10 for t in range(10)], [0.2 + 0.8 * t / 9 for t in range(10)]]
    mean = np.exp(np.mean(np.log(curves), axis=0)).tolist()
    timesteps = np.arange(5, 9)
    rates = [np.exp(-np.polyfit(timesteps, np.log(curve[5:9]), 1)[0]) for curve in curves]
    return {"kind": "initial_hidden_gradients", "model": name,
            "sequence_length": 10, "normalized_curves": curves,
            "geometric_mean": mean, "decay_mean": float(np.mean(rates)),
            "timestep": list(range(10)), "measurement": "mean batch L2 hidden-state gradient",
            "configuration": load_config(overrides={"device": "cpu"}),
            "dtype": "torch.float64", "device": "cpu", "gate_mode": "one",
            "seeds": [0, 1]}


def _study_record():
    return {"schema_version": 1, "model": "Vanilla RNN", "sequence_length": 10,
            "search_space": {"lr": {"kind": "float"}}, "best_trial_number": 0,
            "trials": [{"number": 0, "seed": 1000, "params": {"lr": 0.01},
                        "value": 0.2, "state": "complete"},
                       {"number": 1, "seed": 1001, "params": {"lr": 0.02},
                        "value": None, "state": "fail"}],
            "configuration": load_config(overrides={"device": "cpu"}),
            "device": "cpu", "dtype": "torch.float32"}


def _sources(tmp_path):
    return {
        "main": _save(tmp_path / "main.json", _main_record()),
        "ablation": _save(tmp_path / "ablation.json", _ablation_record()),
        "sweep": _save(tmp_path / "sweep.json", _sweep_record()),
        "initial_gradients": [_save(tmp_path / f"gradient_{name.replace(' ', '_')}.json",
                                    _gradient_record(name)) for name in
                              ("Vanilla RNN", "LSTM", "GRU")],
        "study": _save(tmp_path / "study.json", _study_record()),
    }


def test_numeric_tables_and_plot_data_preserve_notebook_aggregation():
    main = _main_record()
    rows = reporting.main_summary_rows(main)
    assert [(row["T"], row["Model"]) for row in rows] == [
        (10, "Vanilla RNN"), (10, "LSTM")]
    assert rows[0]["Test MSE (mặc định)"] == "0.2100 ± 0.0141"
    assert rows[0]["Test MSE (sau tuning)"] == "0.2600 ± 0.0141"
    assert rows[0]["Chênh lệch (mặc định − tuned)"] == "-0.0500"
    curves = reporting.learning_curve_data(main)
    assert curves[("default", 10, "Vanilla RNN")][0] == [2, 3]
    assert curves[("default", 10, "Vanilla RNN")][1] == pytest.approx([0.41, 0.31])
    heatmap = reporting.training_heatmap_data(main)
    assert heatmap["capture_updates"] == [0, 2]
    assert heatmap["vmin"] == -10 and heatmap["vmax"] == 0
    expected_first = np.log10(np.clip(np.arange(1, 11) / 10, 1e-10, None))
    np.testing.assert_allclose(heatmap["matrices"]["0"][0], expected_first)

    ablation_rows = reporting.ablation_summary_rows(_ablation_record())
    assert len(ablation_rows) == 7
    assert [(row["Model"], row["Gate init"]) for row in ablation_rows] == [
        ("Vanilla RNN", "—"), ("LSTM", "zero"), ("LSTM", "one"),
        ("LSTM", "chrono"), ("GRU", "zero"), ("GRU", "one"), ("GRU", "chrono")]
    assert ablation_rows[0]["Test MSE"] == "0.2050 ± 0.0071"
    assert ablation_rows[0]["Epoch đến val MSE < 0.01"] == "3 (1/2 seed đạt)"
    assert ablation_rows[0]["Hệ số suy giảm r (khởi tạo)"] == "0.700 ± 0.020"
    sweep = reporting.sweep_plot_data(_sweep_record())
    assert sweep["models"]["LSTM"]["bias"] == [-4.0, 0.0, 4.0]
    assert sweep["models"]["GRU"]["decay"] == pytest.approx([0.26, 0.3, 0.34])
    assert sweep["reference_line"] == [[0.0, 0.0], [1.0, 1.0]]
    gradient = reporting.initial_gradient_plot_data([_gradient_record("GRU"),
                                                     _gradient_record("Vanilla RNN")])
    assert list(gradient["models"]) == ["Vanilla RNN", "GRU"]
    assert gradient["models"]["GRU"]["seeds"][0] == pytest.approx(
        list(reversed(_gradient_record("GRU")["normalized_curves"][0])))
    trials = reporting.study_trial_rows(_study_record())
    assert [row["State"] for row in trials] == ["complete", "fail"]
    assert [row["Selected best"] for row in trials] == [True, False]


def test_single_seed_and_partial_results_never_fabricate_spread_or_missing_data(tmp_path):
    main = _main_record()
    main["seed_order"] = [0]
    for model in main["model_order"]:
        for condition in ("default", "tuned"):
            main["results"]["10"][model][condition]["runs"] = [
                main["results"]["10"][model][condition]["runs"][0]]
    assert reporting.main_summary_rows(main)[0]["Test MSE (mặc định)"] == "0.2000 ± NA"
    single_seed_ablation = _ablation_record()
    single_seed_ablation["seed_order"] = [0]
    for item in single_seed_ablation["results"]["10"]:
        item["runs"] = item["runs"][:1]
        item["decay_sample_std"] = None
    single_row = reporting.ablation_summary_rows(single_seed_ablation)[0]
    assert single_row["Test MSE"] == "0.2000 ± NA"
    assert single_row["Hệ số suy giảm r (khởi tạo)"] == "0.700 ± NA"
    main["status"] = "incomplete"
    main["results"]["10"]["Vanilla RNN"]["tuned"]["runs"][0]["status"] = "failed"
    rows = reporting.main_summary_rows(main)
    assert rows[0]["Status"] == "incomplete"
    assert rows[0]["Test MSE (sau tuning)"] == ""
    assert rows[0]["Chênh lệch (mặc định − tuned)"] == ""
    with pytest.raises(reporting.ReportInputError, match="Missing complete learning curve"):
        reporting.learning_curve_data(main)

    ablation = _ablation_record()
    ablation["results"]["10"].pop(3)
    assert reporting.ablation_summary_rows(ablation)[3]["Status"] == "incomplete"
    sweep = _sweep_record()
    sweep["results"]["GRU"].pop()
    with pytest.raises(reporting.ReportInputError, match="Missing bias values"):
        reporting.sweep_plot_data(sweep)

    sources = {"main": _save(tmp_path / "partial_main.json", main),
               "ablation": _save(tmp_path / "partial_ablation.json", ablation),
               "sweep": _save(tmp_path / "partial_sweep.json", sweep)}
    output = tmp_path / "reports"
    index = reporting.render_reports(sources, output)
    assert (output / "main_summary.csv").exists()
    assert (output / "gate_ablation.csv").exists()
    with (output / "retention_bias_values.csv").open(encoding="utf-8", newline="") as handle:
        sweep_rows = list(csv.DictReader(handle))
    assert sweep_rows[-1]["Status"] == "missing"
    assert sweep_rows[-1]["Measured initial decay r"] == ""
    assert {item["kind"] for item in index["skipped"]} >= {
        "main_learning_curves", "training_gradient_heatmap", "retention_bias_sweep"}


def test_regeneration_removes_stale_figures_but_preserves_unrelated_files(tmp_path):
    sources = _sources(tmp_path)
    output = tmp_path / "reports"
    reporting.render_reports(sources, output)
    stale = [output / "main_learning_curves.png", output / "main_learning_curves.svg",
             output / "retention_bias_sweep_T10.png", output / "retention_bias_sweep_T10.svg"]
    assert all(path.exists() for path in stale)
    unrelated = output / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")

    main = json.loads(sources["main"].read_text(encoding="utf-8"))
    main["status"] = "incomplete"
    main["results"]["10"]["Vanilla RNN"]["tuned"]["runs"][0]["status"] = "failed"
    sources["main"].write_text(json.dumps(main), encoding="utf-8")
    sweep = json.loads(sources["sweep"].read_text(encoding="utf-8"))
    sweep["results"]["GRU"].pop()
    sources["sweep"].write_text(json.dumps(sweep), encoding="utf-8")

    index = reporting.render_reports(sources, output)
    assert all(not path.exists() for path in stale)
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert {entry["kind"] for entry in index["skipped"]} >= {
        "main_learning_curves", "retention_bias_sweep"}
    assert not any(entry["path"] in {path.name for path in stale}
                   for entry in index["artifacts"])


def test_headless_rendering_is_read_only_and_traceable(tmp_path, monkeypatch):
    sources = _sources(tmp_path)
    source_paths = [sources["main"], sources["ablation"], sources["sweep"],
                    *sources["initial_gradients"], sources["study"]]
    before_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    torch_state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    python_state = random.getstate()

    def unexpected(*args, **kwargs):
        raise AssertionError("reporting invoked an experiment")

    for name in ("train_model", "tune_model", "run_main_comparison", "run_gate_ablation",
                 "run_retention_bias_sweep", "measure_init_gradients"):
        monkeypatch.setattr(experiments, name, unexpected)
    output = tmp_path / "reports"
    index = reporting.render_reports(sources, output)
    expected = {"main_summary.csv", "main_learning_curves.png", "main_learning_curves.svg",
                "training_gradient_heatmap_T10.png", "training_gradient_heatmap_T10.svg",
                "gate_ablation.csv", "retention_bias_values.csv",
                "retention_bias_sweep_T10.png", "retention_bias_sweep_T10.svg",
                "initial_gradient_curves_T10.png", "initial_gradient_curves_T10.svg",
                "optuna_trials.csv", "report_index.json"}
    assert {path.name for path in output.iterdir()} == expected
    assert all((output / filename).stat().st_size > 100 for filename in expected)
    assert (output / "main_learning_curves.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "<svg" in (output / "retention_bias_sweep_T10.svg").read_text(encoding="utf-8")
    assert [source["scope"] for source in index["sources"]] == ["reduced_or_modified"] * 7
    assert all(source["source_designation"] == "synthetic_test_fixture"
               for source in index["sources"])
    assert [source["sha256"] for source in index["sources"]] == list(before_hashes.values())
    assert all(source["path"] in [str(path.resolve()) for path in source_paths]
               for source in index["sources"])
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert random.getstate() == python_state
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths} == before_hashes
    with (output / "main_summary.csv").open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["Test MSE (mặc định)"] == "0.2100 ± 0.0141"
    assert json.loads((output / "report_index.json").read_text(encoding="utf-8")) == index
    repeated = tmp_path / "reports_again"
    reporting.render_reports(sources, repeated)
    assert {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in expected} == {
        name: hashlib.sha256((repeated / name).read_bytes()).hexdigest() for name in expected}


def test_figure_scales_lines_and_clipping_match_notebook(tmp_path, monkeypatch):
    captured = []

    def capture(fig, directory, stem):
        captured.append((stem, fig))
        return []

    monkeypatch.setattr(reporting, "_save_figure", capture)
    reporting._draw_learning_curves(_main_record(), tmp_path)
    _, fig = captured.pop()
    first = fig.axes[0]
    assert first.get_yscale() == "log"
    assert first.lines[0].get_xdata().tolist() == [2, 3]
    assert first.lines[0].get_ydata().tolist() == pytest.approx([0.41, 0.31])
    assert first.lines[-1].get_ydata()[0] == pytest.approx(1 / 3)
    assert not first.collections  # source learning curve has no uncertainty band
    plt.close(fig)

    reporting._draw_heatmap(_main_record(), tmp_path)
    _, fig = captured.pop()
    assert fig.axes[0].images[0].get_clim() == (-10.0, 0.0)
    np.testing.assert_allclose(fig.axes[0].images[0].get_array(),
                               reporting.training_heatmap_data(_main_record())["matrices"]["0"])
    plt.close(fig)

    reporting._draw_initial_gradients([_gradient_record("LSTM")], tmp_path)
    _, fig = captured.pop()
    assert fig.axes[0].get_yscale() == "log"
    plt.close(fig)

    reporting._draw_sweep(_sweep_record(), tmp_path)
    _, fig = captured.pop()
    assert fig.axes[0].lines[-1].get_xdata().tolist() == [0.0, 1.0]
    assert fig.axes[0].lines[-1].get_ydata().tolist() == [0.0, 1.0]
    assert "initial hidden-gradient decay" in fig.axes[0].get_ylabel()
    plt.close(fig)


def test_invalid_sources_are_rejected(tmp_path):
    source = _save(tmp_path / "invalid.json", {"status": "complete"})
    with pytest.raises(reporting.ReportInputError, match="missing main fields"):
        reporting.load_saved_result(source, "main")
    with pytest.raises(reporting.ReportInputError, match="Unsupported result kind"):
        reporting.load_saved_result(source, "unknown")
    with pytest.raises(reporting.ReportInputError, match="Unsupported source kinds"):
        reporting.render_reports({"unknown": source}, tmp_path / "output")
    one = _save(tmp_path / "one_gradient.json", _gradient_record("LSTM"))
    index = reporting.render_reports({"initial_gradients": [one]}, tmp_path / "one_report")
    assert any(item["kind"] == "initial_gradient_models" and "Vanilla RNN" in item["reason"]
               and "GRU" in item["reason"] for item in index["skipped"])


def test_heatmap_numbers_use_immutable_training_gradient_observations():
    cases = {case["id"]: case for case in json.loads(BASELINE.read_text(encoding="utf-8"))["cases"]}
    record = _main_record()
    record["model_order"] = ["Vanilla RNN", "LSTM", "GRU"]
    record["seed_order"] = [0]
    for name, case_id in (("Vanilla RNN", "vanilla_rnn_training_t10"),
                          ("LSTM", "lstm_training_t10"), ("GRU", "gru_training_t10")):
        original = cases[case_id]["expected"]["history"]["grad_norms"]
        record["results"]["10"].setdefault(name, {})["tuned"] = {
            "runs": [{"seed": 0, "status": "complete", "test_mse": 0.0,
                      "history": {"grad_norms": original}}]}
    heatmap = reporting.training_heatmap_data(record)
    for row, name in enumerate(record["model_order"]):
        case_id = {"Vanilla RNN": "vanilla_rnn_training_t10",
                   "LSTM": "lstm_training_t10", "GRU": "gru_training_t10"}[name]
        raw = np.asarray(cases[case_id]["expected"]["history"]["grad_norms"]["0"])
        expected = np.log10(np.clip(raw / (raw[-1] + 1e-30), 1e-10, None))
        np.testing.assert_allclose(heatmap["matrices"]["0"][row], expected,
                                   rtol=1e-5, atol=1e-6)
