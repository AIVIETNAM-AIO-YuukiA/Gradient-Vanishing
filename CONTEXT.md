# Gradient Vanishing Experiment

This project studies how recurrent models retain the first value of a synthetic sequence and how gate initialization affects long-range gradient flow.

## Language

**First-element memorization task**: A sequence regression task in which only the first time step contains a sampled value and the target is that value after the full sequence.

**Memoryless baseline**: The expected MSE of predicting zero without retaining the first value; `1/3` for the task's `Uniform[-1, 1]` target distribution.

**Retention bias**: A comparison value `b` chosen so that larger values initially favor keeping previous state. Its concrete parameter is `W_f.bias = b` in this LSTM and `W_z.bias = -b` in this custom GRU.

**Update Gate**: The GRU gate `z` that blends previous state with a candidate state in this project's equation `h_t = (1-z)h_{t-1} + z h_tilde`. `W_z` is the linear layer that computes its preactivation; `W_z.bias` is that layer's bias parameter.
_Avoid_: Reset Gate when referring to `W_z` or `W_z.bias`.

**Reset Gate**: The GRU gate `r` used when forming the candidate state. `W_r` is its preactivation layer.

**Default configuration**: The fixed, untuned learning rate, hidden size, and weight-initialization setting. In the main comparison, it still uses retention-bias mode `one`.

**Tuned configuration**: The learning rate, hidden size, and weight initialization selected by a model-and-sequence-length Optuna study; its gate mode is also `one` in the main comparison.
