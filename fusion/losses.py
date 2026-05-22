#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Loss computation utilities for the main training objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _semantic_alignment_loss(
    api_features: torch.Tensor | None,
    graph_features: torch.Tensor | None,
    quality_weights: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    time_ids: torch.Tensor | None = None,
    temperature: float = 0.2,
    same_class_positive_weight: float = 0.0,
) -> torch.Tensor:
    """Quality-weighted class-aware batch contrastive API-Graph alignment."""
    if api_features is None or graph_features is None:
        device = (
            api_features.device
            if isinstance(api_features, torch.Tensor)
            else graph_features.device
            if isinstance(graph_features, torch.Tensor)
            else torch.device("cpu")
        )
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    if api_features.numel() == 0 or graph_features.numel() == 0:
        return api_features.new_tensor(0.0)
    if api_features.shape != graph_features.shape:
        return api_features.new_tensor(0.0)

    api_z = F.normalize(api_features.float(), dim=-1)
    graph_z = F.normalize(graph_features.float(), dim=-1)
    if api_z.size(0) <= 1:
        loss = 1.0 - F.cosine_similarity(api_z, graph_z, dim=-1)
    else:
        temp = max(float(temperature), 1e-4)
        sim = api_z @ graph_z.t() / temp
        logits = sim - sim.max(dim=1, keepdim=True).values.detach()
        sim_t = sim.t()
        logits_t = sim_t - sim_t.max(dim=1, keepdim=True).values.detach()
        targets = torch.arange(api_z.size(0), device=api_z.device)
        same_class_positive_weight = max(float(same_class_positive_weight), 0.0)

        if labels is not None:
            labels = labels.long().view(-1).to(api_z.device)
        if time_ids is not None:
            time_ids = time_ids.long().view(-1).to(api_z.device)

        if same_class_positive_weight > 0.0 and labels is not None and labels.numel() == api_z.size(0):
            eye = torch.eye(api_z.size(0), device=api_z.device, dtype=torch.bool)
            same_class = labels[:, None].eq(labels[None, :])
            pos = eye.float() + same_class_positive_weight * (same_class & (~eye)).float()
            pos = pos / pos.sum(dim=1, keepdim=True).clamp_min(1e-8)
            pos_t = pos.t()
            logp = F.log_softmax(logits, dim=1)
            logp_t = F.log_softmax(logits_t, dim=1)
            loss = -0.5 * (
                (pos * logp).sum(dim=1)
                + (pos_t * logp_t).sum(dim=1)
            )
        else:
            if labels is not None and time_ids is not None:
                if labels.numel() == api_z.size(0) and time_ids.numel() == api_z.size(0):
                    eye = torch.eye(api_z.size(0), device=api_z.device, dtype=torch.bool)
                    same_context = (
                        labels[:, None].eq(labels[None, :])
                        & time_ids[:, None].eq(time_ids[None, :])
                        & (~eye)
                    )
                    logits = logits.masked_fill(same_context, -1e4)
                    logits_t = logits_t.masked_fill(same_context.t(), -1e4)
            loss = 0.5 * (
                F.cross_entropy(logits, targets, reduction="none")
                + F.cross_entropy(logits_t, targets, reduction="none")
            )

    if quality_weights is not None:
        q = quality_weights.float().view(-1).to(loss.device).clamp(0.0, 1.0)
        if q.numel() == loss.numel():
            q = q / q.mean().clamp_min(1e-8)
            loss = loss * q.detach()

    return loss.mean()


def _limit_local_alignment_scope(
    node_features: torch.Tensor,
    api_features: torch.Tensor,
    weights: torch.Tensor,
    node_valid: torch.Tensor | None,
    api_valid: torch.Tensor | None,
    max_nodes: int = 0,
    max_tokens: int = 0,
):
    max_nodes = int(max_nodes or 0)
    max_tokens = int(max_tokens or 0)
    batch_size, num_nodes, node_dim = node_features.shape
    _, num_tokens, api_dim = api_features.shape
    keep_nodes = min(num_nodes, max_nodes) if max_nodes > 0 else num_nodes
    keep_tokens = min(num_tokens, max_tokens) if max_tokens > 0 else num_tokens

    if keep_nodes == num_nodes and keep_tokens == num_tokens:
        return node_features, api_features, weights, node_valid, api_valid
    if keep_nodes <= 0 or keep_tokens <= 0:
        return (
            node_features[:, :0],
            api_features[:, :0],
            weights[:, :0, :0],
            node_valid[:, :0] if isinstance(node_valid, torch.Tensor) else node_valid,
            api_valid[:, :0] if isinstance(api_valid, torch.Tensor) else api_valid,
        )

    device = node_features.device
    limited_nodes = node_features.new_zeros((batch_size, keep_nodes, node_dim))
    limited_api = api_features.new_zeros((batch_size, keep_tokens, api_dim))
    limited_weights = weights.new_zeros((batch_size, keep_nodes, keep_tokens))
    limited_node_valid = (
        node_valid.new_zeros((batch_size, keep_nodes))
        if isinstance(node_valid, torch.Tensor)
        else None
    )
    limited_api_valid = (
        api_valid.new_zeros((batch_size, keep_tokens))
        if isinstance(api_valid, torch.Tensor)
        else None
    )

    node_arange = torch.arange(num_nodes, device=device)
    api_arange = torch.arange(num_tokens, device=device)

    for b in range(batch_size):
        w = weights[b]
        node_score = w.max(dim=1).values
        api_score = w.max(dim=0).values
        if isinstance(node_valid, torch.Tensor):
            node_score = node_score + node_valid[b].to(device=device, dtype=node_score.dtype) * 1e-6
        if isinstance(api_valid, torch.Tensor):
            api_score = api_score + api_valid[b].to(device=device, dtype=api_score.dtype) * 1e-6

        node_idx = (
            torch.argsort(node_score, descending=True)[:keep_nodes]
            if keep_nodes < num_nodes
            else node_arange
        )
        api_idx = (
            torch.argsort(api_score, descending=True)[:keep_tokens]
            if keep_tokens < num_tokens
            else api_arange
        )

        limited_nodes[b, : node_idx.numel()] = node_features[b].index_select(0, node_idx)
        limited_api[b, : api_idx.numel()] = api_features[b].index_select(0, api_idx)
        limited_weights[b, : node_idx.numel(), : api_idx.numel()] = (
            w.index_select(0, node_idx).index_select(1, api_idx)
        )
        if limited_node_valid is not None:
            limited_node_valid[b, : node_idx.numel()] = node_valid[b].index_select(0, node_idx)
        if limited_api_valid is not None:
            limited_api_valid[b, : api_idx.numel()] = api_valid[b].index_select(0, api_idx)

    return limited_nodes, limited_api, limited_weights, limited_node_valid, limited_api_valid


def _local_alignment_loss(
    node_features: torch.Tensor | None,
    api_features: torch.Tensor | None,
    alignment_masks: torch.Tensor | None,
    node_valid: torch.Tensor | None = None,
    api_valid: torch.Tensor | None = None,
    quality_weights: torch.Tensor | None = None,
    time_weights: torch.Tensor | None = None,
    margin: float = 0.35,
    max_nodes: int = 0,
    max_tokens: int = 0,
) -> torch.Tensor:
    if (
        node_features is None
        or api_features is None
        or alignment_masks is None
        or node_features.ndim != 3
        or api_features.ndim != 3
        or alignment_masks.ndim != 3
    ):
        device = (
            node_features.device
            if isinstance(node_features, torch.Tensor)
            else api_features.device
            if isinstance(api_features, torch.Tensor)
            else torch.device("cpu")
        )
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    if node_features.size(0) == 0 or api_features.size(0) == 0:
        return node_features.new_tensor(0.0)
    if node_features.size(0) != api_features.size(0) or node_features.size(0) != alignment_masks.size(0):
        return node_features.new_tensor(0.0)

    weights = alignment_masks.float().to(node_features.device).clamp(0.0, 1.0)
    if node_valid is not None and isinstance(node_valid, torch.Tensor):
        node_valid = node_valid.to(node_features.device)
        nv = node_valid.float().view(weights.size(0), weights.size(1), 1)
        weights = weights * nv
    if api_valid is not None and isinstance(api_valid, torch.Tensor):
        api_valid = api_valid.to(node_features.device)
        av = api_valid.float().view(weights.size(0), 1, weights.size(2))
        weights = weights * av

    node_features, api_features, weights, node_valid, api_valid = _limit_local_alignment_scope(
        node_features,
        api_features,
        weights,
        node_valid,
        api_valid,
        max_nodes=max_nodes,
        max_tokens=max_tokens,
    )
    if node_features.size(1) == 0 or api_features.size(1) == 0:
        return weights.new_tensor(0.0)

    node_z = F.normalize(node_features.float(), dim=-1)
    api_z = F.normalize(api_features.float(), dim=-1)
    sim = torch.matmul(node_z, api_z.transpose(1, 2)).clamp(-1.0, 1.0)

    pos_mass = weights.sum(dim=(1, 2))
    valid_samples = pos_mass > 0
    if not valid_samples.any():
        return sim.new_tensor(0.0)

    pos_sim = (sim * weights).sum(dim=(1, 2)) / pos_mass.clamp_min(1e-8)

    neg_mask = 1.0 - (weights > 0).float()
    if node_valid is not None and isinstance(node_valid, torch.Tensor):
        neg_mask = neg_mask * node_valid.to(sim.device).float().view(weights.size(0), weights.size(1), 1)
    if api_valid is not None and isinstance(api_valid, torch.Tensor):
        neg_mask = neg_mask * api_valid.to(sim.device).float().view(weights.size(0), 1, weights.size(2))

    neg_scores = sim.masked_fill(neg_mask <= 0, -1.0)
    hardest_neg = neg_scores.amax(dim=(1, 2))
    hardest_neg = torch.where(hardest_neg > -1.0, hardest_neg, pos_sim.detach().new_zeros(pos_sim.shape))

    per_sample = F.relu(margin + hardest_neg - pos_sim) + (1.0 - pos_sim)

    sample_weight = torch.ones_like(per_sample)
    if quality_weights is not None:
        q = quality_weights.float().view(-1).to(per_sample.device).clamp(0.0, 1.0)
        if q.numel() == per_sample.numel():
            sample_weight = sample_weight * q.detach()
    if time_weights is not None:
        tw = time_weights.float().view(-1).to(per_sample.device).clamp(0.0, 1.0)
        if tw.numel() == per_sample.numel():
            sample_weight = sample_weight * tw.detach()

    per_sample = per_sample[valid_samples]
    sample_weight = sample_weight[valid_samples]
    sample_weight = sample_weight / sample_weight.mean().clamp_min(1e-8)
    return (per_sample * sample_weight).mean()


def compute_total_loss(logits, extra, y, criterion, loss_cfg, epoch=0, total_epochs=1):
    """Aggregate classification, semantic alignment, and branch auxiliary losses."""
    semantic_align_weight = float(loss_cfg["semantic_alignment_weight"])
    local_alignment_weight = float(loss_cfg.get("local_alignment_weight", 0.0))
    branch_aux_weight = float(loss_cfg["branch_aux_weight"])

    loss_cls = criterion(logits, y)
    if not torch.isfinite(loss_cls).all():
        loss_cls = logits.new_tensor(0.0, requires_grad=True)

    zero = loss_cls.detach().new_tensor(0.0)

    def _safe(t):
        if t is None or not torch.isfinite(t).all():
            return zero
        return t

    time_ids = extra.get("time_ids")

    loss_semantic_align = zero
    if (
        semantic_align_weight != 0.0
        and extra.get("semantic_alignment_api") is not None
        and extra.get("semantic_alignment_graph") is not None
    ):
        loss_semantic_align = _safe(_semantic_alignment_loss(
            extra.get("semantic_alignment_api"),
            extra.get("semantic_alignment_graph"),
            extra.get("semantic_alignment_quality"),
            y,
            time_ids,
            temperature=float(loss_cfg.get("class_aware_alignment_temperature", 0.2)),
            same_class_positive_weight=float(loss_cfg.get("class_aware_alignment_same_class_weight", 0.0)),
        ))

    loss_local_align = zero
    if (
        local_alignment_weight != 0.0
        and extra.get("local_alignment_node") is not None
        and extra.get("local_alignment_api") is not None
        and extra.get("local_alignment_masks") is not None
    ):
        loss_local_align = _safe(_local_alignment_loss(
            extra.get("local_alignment_node"),
            extra.get("local_alignment_api"),
            extra.get("local_alignment_masks"),
            extra.get("local_alignment_node_valid"),
            extra.get("local_alignment_api_valid"),
            extra.get("local_alignment_quality"),
            extra.get("local_alignment_time_weight"),
            max_nodes=int(loss_cfg.get("max_local_align_nodes", 0) or 0),
            max_tokens=int(loss_cfg.get("max_local_align_tokens", 0) or 0),
        ))

    loss_branch_aux = zero
    if branch_aux_weight != 0.0:
        aux_losses = []
        for key in ("api_logits_aux", "graph_logits_aux", "joint_logits_aux"):
            aux_logits = extra.get(key)
            if aux_logits is None:
                continue
            if aux_logits.shape != logits.shape:
                continue
            aux_losses.append(_safe(criterion(aux_logits, y)))
        if aux_losses:
            loss_branch_aux = torch.stack(aux_losses).mean()

    total = (
        loss_cls
        + semantic_align_weight * loss_semantic_align
        + local_alignment_weight * loss_local_align
        + branch_aux_weight * loss_branch_aux
    )

    if not torch.isfinite(total).all():
        total = loss_cls if torch.isfinite(loss_cls).all() else logits.new_tensor(0.0, requires_grad=True)

    extra["loss_components"] = {
        "semantic_align": loss_semantic_align.detach(),
        "local_align": loss_local_align.detach(),
        "branch_aux": loss_branch_aux.detach(),
        "weighted_semantic_align": (semantic_align_weight * loss_semantic_align).detach(),
        "weighted_local_align": (local_alignment_weight * loss_local_align).detach(),
        "weighted_branch_aux": (branch_aux_weight * loss_branch_aux).detach(),
    }

    return (
        total,
        loss_cls.detach(),
        (loss_semantic_align + loss_local_align).detach(),
    )
