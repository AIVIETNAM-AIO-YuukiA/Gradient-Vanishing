"""Synthetic first-element memorization data from notebook cell 7."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


MEMORYLESS_BASELINE = (1 - (-1)) ** 2 / 12

_EVAL_SETS: dict[
    tuple[int, int, int, int, torch.device],
    dict[str, tuple[torch.Tensor, torch.Tensor]],
] = {}


def generate_memorization_task(
    batch_size: int,
    seq_len: int,
    generator: torch.Generator | None = None,
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample on CPU, then transfer the sequence and its first-step target."""
    x = torch.zeros(seq_len, batch_size, 1)
    x[0, :, 0] = torch.rand(batch_size, generator=generator) * 2 - 1
    y = x[0].clone()
    return x.to(device), y.to(device)


def make_fixed_set(
    size: int,
    seq_len: int,
    seed: int,
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use a private CPU generator, leaving the training RNG untouched."""
    generator = torch.Generator().manual_seed(seed)
    return generate_memorization_task(size, seq_len, generator=generator, device=device)


def get_eval_sets(
    seq_len: int,
    config: Mapping[str, Any],
    *,
    device: torch.device | str,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Return cached fixed validation/test sets for one configuration and length."""
    size = config["training"]["evaluation_size"]
    validation_seed = config["split_seeds"]["validation"]
    test_seed = config["split_seeds"]["test"]
    selected_device = torch.device(device)
    key = (seq_len, size, validation_seed, test_seed, selected_device)
    if key not in _EVAL_SETS:
        _EVAL_SETS[key] = {
            "val": make_fixed_set(size, seq_len, validation_seed, device=selected_device),
            "test": make_fixed_set(size, seq_len, test_seed, device=selected_device),
        }
    return _EVAL_SETS[key]
