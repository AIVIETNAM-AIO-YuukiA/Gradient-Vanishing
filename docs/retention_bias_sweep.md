# Numeric retention-bias sweep (Ticket 10)

`experiments.run_retention_bias_sweep(config=None, device=None)` extracts
notebook cell 28's numeric calculation. It returns a `SweepResult` containing
JSON-compatible plot inputs and provenance; `save_retention_bias_sweep`
writes the record without rerunning models. The default sweep uses
`T=100`, models `LSTM` then `GRU`, and `np.linspace(-4, 4, 17)` in ascending
order, with five initial-gradient seeds (`0..4`) at every bias value.

## What the notebook measures

For each `(model, bias)`, the notebook calls `measure_init_gradients` with
that numeric bias and five seeds, then uses `decay_factor_from_curves` and
plots only the mean fitted decay. The extracted record also retains the
per-seed curves and sample standard deviation already computed by the same
gradient function. The models are newly constructed and seeded inside each
measurement; there is no training loop, optimizer update, validation set,
test set, or task-performance metric in cell 28. The reported decay is an
**initial hidden-state gradient** statistic, not a parameter gradient or a
trained-model score.

Each measurement uses float64 model computation and the notebook's final
prediction mean MSE. At timestep `t`, the hidden-gradient norm is the batch
mean of the hidden-unit L2 norms. Each seed's norm curve is divided by its
last-timestep norm. A line is fitted to the log of that normalized curve at
timesteps `T//2..T-2`; the exponent of the negated slope is that seed's decay
factor. The condition's measured decay is the arithmetic mean of those
five factors. The record retains the sample standard deviation (`ddof=1`),
fit timesteps, raw norms, normalized curves, geometric mean curve, seeds,
initialization, device, dtype, versions, notebook hash, and status.

## Retention-bias convention

The numeric value `b` enters LSTM as `W_f.bias = b`. The custom GRU blends
`h_t = (1-z_t)h_prev + z_t*h_tilde` and uses **`W_z.bias = -b`**. `W_z` is
the Update Gate; `W_r` is the Reset Gate and is unchanged. With other
preactivation contributions held fixed, the theoretical coefficient on
the previous hidden state is `sigmoid(b)` for either gate path. The numeric
plot uses `sigmoid(b)` on the horizontal axis and measured mean decay on
the vertical axis, with the notebook's `y=x` reference line. That line is a
reference for the gate path, not a claim that the measured decay must equal
the theoretical coefficient or that larger retention improves learning.

## Verification boundary

The immutable baseline includes the numeric GRU bias `b=2.0` gate case;
the existing model regression test checks its parameters and Torch RNG
exactly. A focused reduced comparison evaluates **all 17 original bias
values** for both models at `T=5`, with two seeds and batch size 8. It
executes the notebook's original gradient and decay definitions directly
in a separate process, with the same reductions as the modular call. The
test compares model order, bias order, complete initialized parameter digests,
gate vectors, GRU reset bias, predictions, initial MSE, normalized gradient curves, geometric
means, fitted decay mean and sample standard deviation, and final Torch
RNG digest. It does not execute the full `T=100`, five-seed sweep. No
per-bias validation or test MSE exists to compare in the notebook sweep.
