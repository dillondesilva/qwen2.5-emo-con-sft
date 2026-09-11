"""Pick device and dtypes for train/chat."""

from __future__ import annotations

import torch


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def use_bf16(device: str | None = None) -> bool:
    device = device or pick_device()
    return device == "cuda" and torch.cuda.is_bf16_supported()


def weight_dtype(device: str | None = None) -> torch.dtype:
    return torch.bfloat16 if use_bf16(device) else torch.float32
