# First-element memorization data

The data module reproduces the synthetic task in original notebook cell 7. It samples one value uniformly from `[-1, 1]` at the first step of a sequence and sets all later steps to zero. The target is a clone of that first value. Inputs have shape `(T, B, 1)` and targets `(B, 1)`. The population MSE for predicting zero is `MEMORYLESS_BASELINE == 1/3`.

```python
import torch

from gradient_vanishing.config import load_config
from gradient_vanishing.data import (
    MEMORYLESS_BASELINE,
    generate_memorization_task,
    get_eval_sets,
)

config = load_config()
device = torch.device("cpu")
torch.manual_seed(7)
x_train, y_train = generate_memorization_task(
    config["training"]["batch_size"], 10, device=device
)
fixed = get_eval_sets(10, config, device=device)
x_val, y_val = fixed["val"]
x_test, y_test = fixed["test"]
```

Training batches use Torch's global RNG after the caller seeds the run. `generate_memorization_task` performs the same CPU `zeros`, `rand`, assignment, and target clone as the notebook before transferring both tensors to `device`. Calling it again consumes the next global draw. Passing a CPU `torch.Generator` uses that generator instead.

Validation and test sets use separate CPU generators seeded from the validated configuration (defaults `10000` and `20000`), so creating them does not advance the training RNG. Their default size is `2048` each. Repeated requests return the cached tensors for the same sequence length and effective fixed-set settings; changed size, seeds, or device get separate cache entries. As in the notebook, callers receive the cached tensor objects directly.

The [baseline fixture](../tests/fixtures/notebook_baseline.json) is the immutable regression oracle for the default `T=10` data case. [Ticket 03 tests](../tests/test_data.py) compare the generated tensors, fixed-set checksums, and RNG state with that original-notebook capture. The source notebook and fixture remain unchanged.
