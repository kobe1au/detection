from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


def move_masks_to_device(masks, device: torch.device):
    """Move a list of attention masks to the specified device."""
    if masks is None:
        return None
    return [m.to(device, non_blocking=True) for m in masks]


# ─────────────────────────────────────────────────────────────────────────────
# AMP (Automatic Mixed Precision) utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_amp_context(device: torch.device, enabled: bool):
    """Get the appropriate autocast context for mixed precision training."""
    if not enabled or device.type != "cuda":
        return nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    except AttributeError:
        return torch.cuda.amp.autocast(enabled=True)


def _safe_to_device(tensor, device, fallback_size=None, dtype=torch.float32):
    if tensor is not None:
        return tensor.to(device, non_blocking=True)
    if fallback_size is not None:
        return torch.zeros(fallback_size, device=device, dtype=dtype)
    return None


def build_grad_scaler(device: torch.device, enabled: bool):
    """Build a gradient scaler for mixed precision training."""
    use_scaler = enabled and (device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=use_scaler)
    except (AttributeError, RuntimeError):
        try:
            return torch.cuda.amp.GradScaler(enabled=use_scaler)
        except Exception:
            logger.warning("GradScaler creation failed, using no-op scaler")
            return torch.cuda.amp.GradScaler(enabled=False)


# ─────────────────────────────────────────────────────────────────────────────
# Batch preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_batch(
    batch: Dict[str, Any],
    device: torch.device,
    skip_graph: bool = False,
    skip_masks: bool = False,
) -> tuple:
    """
    Unified API + graph batch preparation for training and evaluation.

    Returns:
        (graph, masks, y, sids, explicit_info, num_failed)
        where explicit_info = (q_api, q_graph, q_align, pert_api, pert_graph, time_ids)
        or (None, None, None, None, None, num_failed) if invalid
    """
    graph = batch.get("graph_batch")
    masks = batch.get("masks")
    y = batch.get("labels")
    sids = batch.get("sids")
    time_ids = batch.get("time_ids", None)
    num_failed = batch.get("num_failed", 0)

    if y is None:
        return None, None, None, None, None, num_failed

    quality = batch.get("quality", None)
    if quality is None:
        return None, None, None, None, None, num_failed

    q_apis = quality.get("q_api")
    q_graphs = quality.get("q_graph")
    q_aligns = quality.get("q_align")
    pert_apis = quality.get("pert_api")
    pert_graphs = quality.get("pert_graph")

    if not skip_graph and graph is None:
        return None, None, None, None, None, num_failed

    if skip_graph:
        graph = None
    else:
        graph = graph.to(device, non_blocking=True)

    y = y.to(device, non_blocking=True)

    masks = None if (skip_graph or skip_masks) else move_masks_to_device(masks, device)

    _batch_size = y.size(0)
    if time_ids is not None:
        time_ids = time_ids.to(device, non_blocking=True).long().view(-1)
        if time_ids.size(0) != _batch_size:
            logger.warning(f"time_ids size {time_ids.size(0)} != batch size {_batch_size}, using zeros")
            time_ids = torch.zeros((_batch_size,), device=device, dtype=torch.long)
    else:
        time_ids = torch.zeros((_batch_size,), device=device, dtype=torch.long)
    
    if graph is not None:
        api_aug_s = batch.get("api_aug_strength")
        graph_aug_s = batch.get("graph_aug_strength")
        overall_aug_s = batch.get("overall_aug_strength")
        if api_aug_s is not None:
            graph.api_aug_strength = api_aug_s.to(device, non_blocking=True)
        if graph_aug_s is not None:
            graph.graph_aug_strength = graph_aug_s.to(device, non_blocking=True)
        if overall_aug_s is not None:
            graph.overall_aug_strength = overall_aug_s.to(device, non_blocking=True)
    
    explicit_info = (
        _safe_to_device(q_apis, device),
        _safe_to_device(q_graphs, device),
        _safe_to_device(q_aligns, device),
        _safe_to_device(pert_apis, device),
        _safe_to_device(pert_graphs, device),
        time_ids,
    )

    return graph, masks, y, sids, explicit_info, num_failed
