from __future__ import annotations

import logging
from contextlib import nullcontext

import torch

logger = logging.getLogger(__name__)


def get_amp_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=True)


def build_grad_scaler(device: torch.device, enabled: bool):
    use_scaler = bool(enabled) and device.type == "cuda"
    try:
        return torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, RuntimeError):
        try:
            return torch.cuda.amp.GradScaler(enabled=use_scaler)
        except Exception:
            logger.warning("GradScaler creation failed, using disabled CUDA scaler")
            return torch.cuda.amp.GradScaler(enabled=False)
