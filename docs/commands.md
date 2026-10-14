# Research commands and run manifests (Ticket 12)

Install the pinned CPU environment as in [installation.md](installation.md), then
install the package and check the entry point:

```powershell
.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.venv\Scripts\gradient-vanishing.exe --help
```

`python -m gradient_vanishing` runs the same interface. The supported commands
are `main`, `gradients`, `ablation`, `sweep`, and `report`. `main` includes the
notebook's Optuna studies before default and tuned training; it saves each
study as a separate JSON result as well as embedding it in `main.json`.
A non-editable wheel was also built and installed into an isolated local target;
its `python -m gradient_vanishing --help` command succeeded outside the source
checkout.

## Configuration and safe execution

Each command accepts `--config PARTIAL.json`, repeatable `--set KEY=JSON`,
and `--output-dir DIRECTORY`. Precedence is package defaults, then the partial
JSON file, then `--set`, then `--output-dir`. Unknown keys and invalid values
are rejected before a run directory is created. The file must be UTF-8 JSON
without a byte-order mark. Nested objects merge by key.

The checked-in defaults describe the original notebook matrix: three lengths,
three models, five comparison seeds, 20 Optuna trials per model and length,
300 final updates, 150 tuning updates, ten initial-gradient seeds, and the
17-value sweep. Commands reject a full-size stage unless `--allow-full` is
supplied. This flag allows execution; it does not reduce any budget. A reduced
run requires an explicit partial config or `--set` overrides, and its manifest
is labeled `reduced_or_modified`. Runs matching the original stage budget are
labeled `full_configuration`; that label does not assert empirical equivalence.
Report commands are labeled `report_from_saved_results`; each source's own
configuration scope is recorded in `report_index.json`.

For a CPU check, save the following as `reduced.json`:

```json
{
  "sequence_lengths": [5], "model_names": ["Vanilla RNN"], "seeds": [0],
  "device": "cpu",
  "training": {"updates": 1, "tuning_updates": 1, "batch_size": 2,
    "evaluation_interval": 1, "evaluation_size": 4,
    "gradient_capture_updates": [0]},
  "tuning": {"trials": 1, "hidden_sizes": [2], "initializations": ["default"]},
  "ablation": {"sequence_lengths": [5], "gate_modes": ["zero"]},
  "gradients": {"initial_seeds": 2, "sweep_seeds": 2,
    "sweep_sequence_length": 5, "sweep_biases": [0.0]}
}
```

These commands were executed in the pinned local CPU environment:

```powershell
gradient-vanishing main --config reduced.json --output-dir runs
gradient-vanishing gradients --config reduced.json --output-dir runs
gradient-vanishing ablation --config reduced.json --output-dir runs
gradient-vanishing sweep --config reduced.json --output-dir runs
```

The real validation used the same JSON at `.tmp/ticket12_validation/reduced.json`.
All four commands completed. `main` ran one trial and two one-update training
runs. The ablation ran the configured vanilla, LSTM, and GRU conditions. The
sweep ran both gated models at bias zero. These are execution checks, not
research conclusions at the notebook's sequence lengths or budgets.

## Saved results and reports

Each invocation creates a unique directory under `--output-dir` (or the
validated configuration's `output_dir`). It never replaces a previous run:

```text
runs/
  main-<UTC timestamp>-<unique suffix>/
    manifest.json
    results/
      main.json
      study_vanilla_rnn_T5.json
  gradients-<UTC timestamp>-<unique suffix>/
    manifest.json
    results/initial_gradient_vanilla_rnn_T5.json
  report-<UTC timestamp>-<unique suffix>/
    manifest.json
    reports/report_index.json
    reports/*.csv
    reports/*.png
    reports/*.svg
```

Use paths from the completed run manifests to regenerate reports. `--source`
accepts `main`, `ablation`, `sweep`, and `study` once each; repeat
`initial_gradients` for each saved model/length record. For example:

```powershell
gradient-vanishing report --config reduced.json --output-dir runs `
  --source main=runs/<main-run>/results/main.json `
  --source ablation=runs/<ablation-run>/results/gate_ablation.json `
  --source sweep=runs/<sweep-run>/results/retention_bias_sweep.json `
  --source initial_gradients=runs/<gradients-run>/results/initial_gradient_vanilla_rnn_T5.json `
  --source study=runs/<main-run>/results/study_vanilla_rnn_T5.json
```

The report command only reads saved JSON and calls the Ticket 11 renderer. It
does not import the training or experiment module. Missing report sources and
incomplete conditions are listed in `report_index.json`. The sample run above
has only one initial-gradient model; the report index names the other two as
missing. See [reporting.md](reporting.md) for the plotted quantities and table
definitions. Report input files are hashed; source files are never rewritten.

## Manifest contract

Every run writes `manifest.json` with schema version 1. It records the run ID,
stage and command, `running`/`complete`/`incomplete`/`failed` status, UTC start
and completion times, effective validated configuration, CLI overrides and
optional config-file path/hash, scope, seeds, models, lengths, gate and bias
settings, trial and update budgets, selected parameters, requested and resolved
device, dtype, dependency versions, deterministic/CUDA details when available,
Git revision when available, the original notebook reference hash, saved
result paths and SHA-256 hashes, and error type/message when a run stops.
The manifest's `seeds`, `model_names`, and `sequence_lengths` describe the
selected stage: initial gradients and sweeps use their own zero-based seed
ranges; ablation and sweep enumerate their fixed notebook model sets.
Unavailable metadata is `null`, rather than guessed. Result files preserve
the established Ticket 06–10 JSON schemas and per-seed arrays. Main's partial
comparison record, ablation record, or sweep record is saved when the existing
experiment exception carries it. Files are serialized as strict JSON through
atomic replacement. An error returns a nonzero exit code and leaves a manifest
with the work completed so far.
An installed wheel outside a Git checkout has no discoverable source commit,
so `git_commit` is `null`; the notebook reference hash and package versions
remain recorded. An editable install in this checkout records the current
Git commit.

CPU was verified locally. `device=auto` and `device=cuda` set
`CUBLAS_WORKSPACE_CONFIG` before Torch imports; Torch deterministic-operation
errors are surfaced. GPU numerical equivalence has not been verified. No full
matrix was run for this ticket; Ticket 13 covers the broader end-to-end
reproduction validation.
