"""The fastest torch device on this machine: CUDA, then Apple's MPS, then the CPU."""

import torch


def best_device() -> torch.device:
    """Return CUDA if available, else MPS (Apple GPU), else the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
