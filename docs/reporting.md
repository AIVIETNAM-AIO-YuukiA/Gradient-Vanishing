# Reports from saved results (Ticket 11)

`gradient_vanishing.reporting.render_reports` reads saved JSON records and
writes tables, figures, and `report_index.json`. It does not import or call
training, tuning, model construction, or gradient measurement. Matplotlib
uses the headless Agg backend; no interactive notebook or display is needed.
The unified command interface remains Ticket 12.

```python
from gradient_vanishing.reporting import render_reports

render_reports({
    "main": "saved/main.json",
    "ablation": "saved/ablation.json",
    "sweep": "saved/sweep.json",
    "initial_gradients": [
        "saved/gradients_vanilla.json",
        "saved/gradients_lstm.json",
        "saved/gradients_gru.json",
    ],
    "study": "saved/optuna_study.json",
}, "reports")
```

Use the JSON writers from Tickets 06–10 to create those source files. Each
key is optional; omitted report kinds are listed as skipped. The initial
gradient key accepts one or more saved `initial_hidden_gradients` records.
If fewer than the notebook's three models are supplied, the available curves
are rendered and the missing models are named in the report index.
`load_saved_result(path, kind)` checks the top-level report schema and raises
`ReportInputError` when required fields are absent. The pure table and plot
data functions (`main_summary_rows`, `learning_curve_data`,
`training_heatmap_data`, `ablation_summary_rows`, `sweep_plot_data`,
`initial_gradient_plot_data`, `study_trial_rows`) can be used for inspection
without rendering.

## Notebook cell mapping

| Notebook cell | Saved source | Output |
| --- | --- | --- |
| 19, main summary | `main` | `main_summary.csv` |
| 21, mean validation curves | `main` training histories | `main_learning_curves.png`, `.svg` |
| 23, initial hidden-gradient curves | `initial_gradients` | `initial_gradient_curves_T{T}.png`, `.svg` |
| 25, tuned seed-zero training heatmap | `main` gradient captures | `training_gradient_heatmap_T{T}.png`, `.svg` |
| 27, gate ablation table | `ablation` | `gate_ablation.csv` |
| 28, numeric retention-bias sweep | `sweep` | `retention_bias_values.csv`, `retention_bias_sweep_T{T}.png`, `.svg` |
| 15, Optuna trial observations | `study` | `optuna_trials.csv` |

All files are placed directly in the requested output directory. Filenames
are deterministic; PNG is suitable for documents and SVG for editable vector
placement. The `report_index.json` lists artifact paths, exact source paths
and SHA-256 hashes, source status, configuration scope, model/length, seeds,
hyperparameters and gate mode where available, device/dtype, dependency versions,
notebook hash, and skipped
outputs. `full_configuration` means the source configuration exactly equals
the checked-in default; it does **not** assert that notebook-level empirical
reproduction has been verified. `reduced_or_modified` labels other inputs.
Figures can be included in an academic report with the corresponding source
file and index retained for traceability.
Regenerating in the same directory removes artifacts listed in the previous
report index before writing new outputs. This prevents a skipped figure from
leaving an image that represents an earlier, more complete source. Files not
listed in the report index are preserved.

## Numeric and display semantics

The main table uses arithmetic test-MSE means and sample standard deviations
(`ddof=1`) at four decimals, and default minus tuned mean. A single-seed
spread is displayed as `NA`. Validation curves average saved validation MSE
at matching checked updates, use a logarithmic y-axis, and show the
**theoretical** memoryless `1/3` MSE line. They deliberately have no spread
band, matching the notebook's plotted behavior despite its source comment.
The finite test-set zero-predictor MSE is a different quantity.

The initial-gradient plot shows each seed's saved hidden-gradient ratio and
its geometric mean, reversed to temporal distance `T−1−t` on a semilog
axis. The metric is the batch mean of the L2 norm of the gradient of final
prediction MSE with respect to each hidden state; it is not a parameter
gradient. The trained-model heatmap uses tuned seed-zero raw gradient norms,
`log10(clip(g_t / (g_last + 1e-30), 1e-10, None))`, and the notebook's
`magma` color limits `[-10, 0]`. Its captures are float32 and pre-clipping.

The ablation table uses final test MSE mean/sample spread, the first checked
validation update strictly below the threshold, success count, and the
initial-gradient decay mean/sample spread. The sweep plot uses the original
bias order and plots measured **initial** hidden-gradient decay against
theoretical `sigmoid(b)`, with the notebook's `y=x` gate-path reference.
For the custom GRU, `W_z.bias = −b`; `W_r` is untouched. The decay is the
mean of per-seed factors obtained by fitting the log normalized gradient
at timesteps `T//2..T-2` and exponentiating the negative slope. The sweep
contains no per-bias validation or test MSE.

Partial records retain every configured table row and mark missing or failed
conditions explicitly. Figures needing a complete set of aligned numeric
inputs are skipped with a reason in the report index; no missing values or
uncertainty bands are fabricated. Optuna's trial table retains failed and
pruned trial states and does not imply any test-set measurement.

## Validation boundary

No saved reduced main, ablation, sweep, or study results were present in the
repository when Ticket 11 began. Focused tests use **synthetic saved JSON
records** marked `synthetic_test_fixture` to verify grouping, values, plot transformations, headless export,
deterministic bytes, provenance, partial-result handling, and unchanged
source files and RNG state. These synthetic illustrations are renderer
checks, not empirical results. The reports for the full experiment matrix
remain unavailable until those raw results are run and saved. The tests
check plotted data and transformations against notebook cells; they do not
claim pixel-level visual identity with notebook output.
