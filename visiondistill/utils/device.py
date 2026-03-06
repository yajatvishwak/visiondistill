from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> str:
    """Return a validated PyTorch device string.

    Accepted values:
        ``"auto"`` -- pick the best available (cuda > mps > cpu).
        ``"cuda"`` / ``"cuda:0"`` etc. -- use NVIDIA GPU, fall back to cpu.
        ``"mps"`` -- use Apple Silicon GPU, fall back to cpu.
        ``"cpu"`` -- force CPU.
    """
    req = requested.strip().lower()

    if req == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logger.info("Auto-detected device: %s", device)
        return device

    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return req
        logger.warning("CUDA requested but not available, falling back to cpu")
        return "cpu"

    if req == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        logger.warning("MPS requested but not available, falling back to cpu")
        return "cpu"

    if req == "cpu":
        return "cpu"

    logger.warning("Unknown device '%s', falling back to cpu", requested)
    return "cpu"


def safe_dtype(device: str, requested_dtype: str = "float16") -> torch.dtype:
    """Return a dtype that is safe for the given device.

    ``float16`` is only used on CUDA. For MPS and CPU we promote to
    ``float32`` unless the caller explicitly asked for ``bfloat16``
    (which MPS may support on newer PyTorch builds).
    """
    dtype_map: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    req = dtype_map.get(requested_dtype, torch.float32)

    if device.startswith("cuda"):
        return req

    if req == torch.float16:
        logger.info("float16 not optimal on %s, promoting to float32", device)
        return torch.float32

    return req
