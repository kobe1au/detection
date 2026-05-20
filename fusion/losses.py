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


def compute_total_loss(logits, extra, y, criterion, loss_cfg, epoch=0, total_epochs=1):
    """Aggregate classification, alignment, branch auxiliary, and gate oracle losses."""
    semantic_align_weight = float(loss_cfg["semantic_alignment_weight"])
    branch_aux_weight = float(loss_cfg["branch_aux_weight"])
    gate_oracle_weight = float(loss_cfg.get("gate_oracle_weight", 0.0))
    gate_oracle_start_epoch = int(loss_cfg.get("gate_oracle_start_epoch", 0))
    if epoch < gate_oracle_start_epoch:
        gate_oracle_weight = 0.0
    gate_oracle_start_phase = str(loss_cfg.get("gate_oracle_start_phase", "") or "").lower()
    if gate_oracle_start_phase:
        if str(loss_cfg.get("_continual_phase", "historical")).lower() != gate_oracle_start_phase:
            gate_oracle_weight = 0.0
    if bool(loss_cfg.get("gate_oracle_adaptation_only", False)):
        if str(loss_cfg.get("_continual_phase", "historical")).lower() != "adaptation":
            gate_oracle_weight = 0.0

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

    loss_gate_oracle = zero
    if gate_oracle_weight != 0.0:
        gate_weights = extra.get("gate_weights_train", extra.get("gate_weights"))
        branch_logits = [
            extra.get("api_logits_aux"),
            extra.get("graph_logits_aux"),
            extra.get("joint_logits_aux"),
        ]
        if (
            isinstance(gate_weights, torch.Tensor)
            and gate_weights.ndim == 2
            and gate_weights.size(0) == y.numel()
            and gate_weights.size(1) == 3
            and all(isinstance(v, torch.Tensor) and v.shape == logits.shape for v in branch_logits)
        ):
            branch_losses = torch.stack(
                [
                    F.cross_entropy(v.float(), y, reduction="none")
                    for v in branch_logits
                ],
                dim=1,
            )
            temp = max(float(loss_cfg.get("gate_oracle_temperature", 0.5)), 1e-4)
            oracle = torch.softmax(-branch_losses.detach() / temp, dim=1)
            smoothing = float(loss_cfg.get("gate_oracle_smoothing", 0.0))
            if smoothing > 0.0:
                smoothing = min(max(smoothing, 0.0), 1.0)
                oracle = (1.0 - smoothing) * oracle + smoothing / oracle.size(1)
            log_gate = torch.log(gate_weights.float().clamp_min(1e-8))
            loss_gate_oracle = _safe(F.kl_div(log_gate, oracle, reduction="batchmean"))

    total = (
        loss_cls
        + semantic_align_weight * loss_semantic_align
        + branch_aux_weight * loss_branch_aux
        + gate_oracle_weight * loss_gate_oracle
    )

    if not torch.isfinite(total).all():
        total = loss_cls if torch.isfinite(loss_cls).all() else logits.new_tensor(0.0, requires_grad=True)

    extra["loss_components"] = {
        "semantic_align": loss_semantic_align.detach(),
        "branch_aux": loss_branch_aux.detach(),
        "gate_oracle": loss_gate_oracle.detach(),
        "weighted_semantic_align": (semantic_align_weight * loss_semantic_align).detach(),
        "weighted_branch_aux": (branch_aux_weight * loss_branch_aux).detach(),
        "weighted_gate_oracle": (gate_oracle_weight * loss_gate_oracle).detach(),
    }

    return (
        total,
        loss_cls.detach(),
        loss_semantic_align.detach(),
    )
