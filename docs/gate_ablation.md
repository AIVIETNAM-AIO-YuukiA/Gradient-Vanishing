# Gate initialization ablation (Ticket 09)

`experiments.run_gate_ablation(config=None, device=None)` extracts notebook
cell 27. `save_gate_ablation(path, result)` writes its JSON-compatible raw
record without rerunning the experiment. A reduced validated configuration
can limit lengths, seeds, and update counts for CPU verification. The default
configuration schedules 35 training runs at `T=100`; it is not executed in
this ticket.

## Notebook experiment contract

At each ablation length, the notebook obtains one fixed test pair before
running conditions. It runs the Vanilla RNN reference with gate mode
`default`, followed by LSTM and GRU, each in mode `zero`, `one`, then
`chrono`. Each condition trains default hyperparameters (`lr=0.001`, hidden
size 32, default weight initialization) for 300 updates at seeds `0..4`.
After each run it evaluates the final model on the fixed test pair and scans
the recorded validation checks for the **first** MSE strictly below `0.01`.
The validation schedule is every ten updates and the final update. Misses
are retained as null in JSON. The per-condition summary contains raw test
MSE, arithmetic mean, sample standard deviation (`ddof=1`), success count,
and mean update among successes. Display formatting remains Ticket 11.

After all training seeds of one condition, cell 27 measures ten float64
initial-gradient curves with seeds `0..9`. It uses the same gate mode and
default hyperparameters, with independent reseeding per gradient seed. The
existing gradient module computes the per-seed decay fit over timesteps
`T//2..T-2`, then arithmetic mean and sample standard deviation of fitted
rates. It records normalized hidden gradient norms and their geometric
mean. No training-time gradient capture is requested in this ablation.
The result retains training histories, final validation/test MSE, gradient
curves, decay values, the target gate name and retention-bias sign, each
run's initialized target bias vector (and chrono LSTM input-bias vector),
seeds, split seeds, effective settings, versions,
device/dtype, notebook hash, and status. The existing `evaluate` restores
training mode, preserving the notebook side effect. A failure raises
`AblationInterruptedError` with the partial record on `result`.

## Gate semantics

The custom GRU blends `h_t = (1-z_t)h_prev + z_t*h_tilde`. Its `W_z` is the
Update Gate. The retention bias `b` enters **`W_z.bias = -b`**; `W_r` is the
Reset Gate and remains untouched. Thus a larger positive `b` reduces `z`
and raises the coefficient on the previous hidden state. LSTM uses
`W_f.bias = b`; chrono also sets `W_i.bias = -b`. Chrono draws one value per
hidden unit as `log U(1, max(T-1, 2))` after the weight initialization and
before device transfer. These assignments and their RNG order remain in the
existing models module.

## Verification boundary

The immutable baseline contains GRU default, zero, one, chrono, and numeric
bias cases plus LSTM chrono, which the model regression suite compares
exactly, including RNG state. A focused test executes original notebook
cell 27 and its definitions in a separate process at `T=5`, seeds `0,1`,
two training updates, batch size 8, fixed-set size 32, and two initial
gradient seeds. The modular path uses the same settings. It compares all
seven condition labels and their initial gate biases, predictions, training
losses, validation/test MSE, normalized gradient curves, decay rates, and
final Torch RNG digest. This reduced check does not establish relative
performance at the full `T=100` budget.
