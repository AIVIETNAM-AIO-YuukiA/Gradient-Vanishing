# Hidden-gradient diagnostics (Ticket 06)

The source is notebook cells 9, 13, 23, and 25 in the unchanged
`Notebooks/GradientVanishingAnalysis.ipynb`. The numeric implementation is
`src/gradient_vanishing/gradients.py`; recurrent equations and hidden-state
retention stay in `models.py`, and training updates stay in `training.py`.

## Quantity and execution order

For a sequence of length `T`, `h_t` is the hidden output after input step `t`.
The loss is the mean-reduced MSE between the final-step prediction and the
first input value, with input shape `(T, B, 1)` and target shape `(B, 1)`.
The notebook retains each intermediate `h_t` gradient and records

`g_t = mean_{b=1..B} || dL / d h_t[b, :] ||_2`, for `t = 0, ..., T-1`.

The L2 norm runs across hidden units for each batch member; the arithmetic
mean then runs across batch members. It is not the norm of a batch-mean
gradient, a parameter-gradient norm, or a measurement of every LSTM cell-state
path. LSTM's saved `c_states` are detached inspection copies; its live cell
state remains connected in the recurrence. Vanilla RNN, LSTM, and the custom
GRU use full BPTT from the final loss through all `T` steps. `retain_grad()`
on hidden states makes their non-leaf gradients readable after `backward()`.

Initial measurements reseed Python, NumPy, and Torch for each seed, construct
and initialize the model on CPU, transfer it to the selected device as
float64, draw one fresh batch from the global Torch RNG, and backpropagate
without clipping or an optimizer step. The default is ten seeds, batch size
128, gate mode `one`, and the checked-in default hyperparameters. The model
remains in training mode. Training captures use the existing float32 training
loop at zero-based update indices `0, 50, 150, 299` by default. The loop
reads `g_t` after backward and before parameter-gradient clipping at norm 5
and before Adam steps. Reduced runs must supply valid capture indices.

## Transformations and indices

For each initial seed, the notebook divides `g_t` by `g_{T-1}` with no
denominator guard. The geometric mean curve is
`exp(mean_seed(log(clip(ratio, 1e-300, None))))`. Zero ratios are counted;
nonfinite ratios are counted separately and remain nonfinite in the numeric
result. A zero final norm therefore keeps the notebook's `NaN`/infinity
behavior. Strict JSON output spells nonfinite values `"NaN"`, `"Infinity"`,
or `"-Infinity"` while retaining their positions and counts.

The decay factor fits a line to each seed's log ratios at time indices
`T//2, ..., T-2`, computes `exp(-slope)`, then reports the arithmetic mean
and sample standard deviation (`ddof=1`) of those factors. Arrays stay in
forward-time order `t=0..T-1`; `temporal_distance = T-1-t` is supplied for
plots that reverse the curves. For training heatmaps, each captured profile
uses `log10(clip(g_t / (g_{T-1} + 1e-30), 1e-10, None))`, preserving the
notebook's different denominator rule and display floor.

`save_diagnostic_result(path, result)` writes the numeric arrays and
provenance as JSON. The initial record includes seeds, configuration,
hyperparameters, Python and PyTorch versions, dtype, device, fit window,
RNG digest, and notebook hash.
The training record includes update indices, raw and transformed profiles,
history, and the single-run metadata. The function does not train or plot.

## Reference coverage and limits

The immutable Ticket 02 fixture supplies hidden-gradient norms, MSE,
initialization RNG digest, and mode observations for all three models at
`T=1,10,30` in float32 and `T=30` in float64. It also supplies pre-clipping
training profiles at updates 0 and 2 for three-update `T=10` runs. It does
not store full hidden-gradient tensors or initial-curve aggregates. Focused
tests compare those tensors' shapes/dtypes and independently recompute the
batch-mean L2 norms. For initial curves, tests execute the audited notebook
definitions directly at `T=10`, two seeds, float64, and compare curves,
geometric means, decay factors, and final Torch RNG state. These reduced
checks establish numerical equivalence on the pinned CPU environment; they
do not establish a universal vanishing-gradient claim or verify the full
`T=100` empirical profile.
