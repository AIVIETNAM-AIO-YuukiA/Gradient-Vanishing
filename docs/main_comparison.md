# Default and tuned main comparison (Ticket 08)

`experiments.run_main_comparison(config=None, device=None)` extracts notebook
cell 17. It returns a `ComparisonResult` with a JSON-compatible `record`;
`save_main_comparison(path, result)` writes it without running experiments.
Use `load_config(overrides=...)` to make a reduced run. The checked-in config
schedules the original nine studies and 90 final runs; this ticket does not
execute that full matrix.

## Notebook execution contract

The notebook iterates sequence lengths `10, 30, 100`, then models
`Vanilla RNN, LSTM, GRU`. For each length it obtains the cached fixed test
pair before tuning any model. For each `(length, model)` it creates a fresh
Optuna study, takes its best trial's `lr`, `hidden_size`, and `init_method`,
and then trains default runs followed by tuned runs for seeds `0..4` in that
order. The best parameters are reused across all five tuned seeds, with no
new study between final runs. Default values are `lr=0.001`,
`hidden_size=32`, and `init_method=default`. Both conditions use the
configured gate mode `one`.

Each trial trains for 150 updates by default; each final model trains for
300. Training seeds Python, NumPy, and Torch before model construction,
uses the global Torch RNG for fresh batches, validates every ten updates
and on the final update, and captures pre-clipping hidden gradient norms at
zero-based updates `0, 50, 150, 299`. The existing training module preserves
the notebook's evaluation side effect: `evaluate` returns the model to
training mode. The comparison calls `evaluate` once after training with the
fixed test pair for that length. It does not choose a best final update.

The notebook stores per-seed training history and final test MSE. It prints
the arithmetic mean and sample standard deviation (`ddof=1`) of the five
test MSE values. The extracted record keeps those values, each run's final
validation MSE, gradient history, effective parameters, seed, split seeds,
update and evaluation settings, selected study reference, dependency
versions, device, dtype, final Torch RNG digest, and execution status. The
numeric heatmap input uses the maximum length, tuned seed-zero run from each
model and the existing `training_log10_relative_norms` definition. Figure
and table rendering remain outside this ticket.

A failed condition stops at the notebook's failure point. The raised
`ComparisonInterruptedError.result` retains completed runs, the failed run
status, and available study information so it can be saved. A run with only
one seed records no sample standard deviation, since `ddof=1` is undefined.

## Verification boundary

The immutable Ticket 02 fixture contains three-update training probes but
no Optuna or main-comparison outcomes. The comparison test uses that fixture
for the default Vanilla RNN seed-zero training history. It also executes the
original notebook definitions and cell 17 directly in a separate process
for `T=10`, Vanilla RNN, seeds `0,1`, two Optuna trials of two updates each,
and three final updates. Both paths retain the original batch size 128 and
fixed-set size 2048. They compare selected parameters, each run's losses,
gradient norms, validation and test metrics, summary statistics, and final
Torch RNG digest. The test uses the same reduced settings on both paths.
Other models and lengths are covered by earlier training and study tests,
plus orchestration checks here; their full main-comparison outcomes remain
unverified until Ticket 14. Reduced results are methodological checks, not
research conclusions about which condition performs better.
