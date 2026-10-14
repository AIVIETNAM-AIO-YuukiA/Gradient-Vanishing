# Follow-up 01: Correct GRU gate terminology

Status: **after behavior-preserving extraction**. The notebook introduction and bias-ablation prose refer to the GRU Reset Gate, while the code modifies `model.W_z.bias`. `W_z` is an `nn.Linear` layer (weights **and** bias) whose output forms the Update Gate `z`; `W_r` forms the Reset Gate `r`.

Update project-facing explanations and labels to identify the intervention as initialization of the **Update Gate bias parameter** `W_z.bias`. Retain the custom equation `h=(1-z)h_prev+z*h_tilde` and the assignment `W_z.bias=-b`. Leave the source notebook unchanged; note its historical wording in the audit.

Regression test: inspect a freshly initialized GRU before/after `apply_gate_init` and assert that `W_z.bias` changes with the expected sign, while `W_z.weight` and both `W_r` parameters remain unchanged. A one-step forward calculation must still match the baseline equation.
