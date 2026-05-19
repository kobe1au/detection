#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Loss computation utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fusion.constants import ArchitectureConstants


def _semantic_alignment_loss(
    api_features: torch.Tensor | None,
    graph_features: torch.Tensor | None,
    quality_weights: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    time_ids: torch.Tensor | None = None,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Drift/quality weighted batch contrastive API-Graph alignment."""
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

        if labels is not None and time_ids is not None:
            labels = labels.long().view(-1).to(api_z.device)
            time_ids = time_ids.long().view(-1).to(api_z.device)
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


def _per_year_risks(
    logits: torch.Tensor,
    labels: torch.Tensor,
    time_ids: torch.Tensor | None,
    criterion,
) -> list[torch.Tensor]:
    if time_ids is None or logits.size(0) <= 1:
        return []

    time_ids = time_ids.long().view(-1)
    labels = labels.long().view(-1)
    weight = getattr(criterion, "weight", None)
    label_smoothing = float(getattr(criterion, "label_smoothing", 0.0))

    risks = []
    for year in time_ids.unique():
        mask = time_ids == year
        if not mask.any():
            continue
        risks.append(F.cross_entropy(
            logits[mask].float(),
            labels[mask],
            weight=weight,
            label_smoothing=label_smoothing,
        ))
    return risks


def _groupdro_loss(logits, labels, time_ids, criterion, temperature: float = 0.2) -> torch.Tensor:
    risks = _per_year_risks(logits, labels, time_ids, criterion)
    if len(risks) <= 1:
        return logits.new_tensor(0.0)
    risk_t = torch.stack(risks)
    tau = float(temperature)
    if tau <= 0.0:
        return risk_t.max()
    return tau * torch.logsumexp(risk_t / tau, dim=0)


def _vrex_loss(logits, labels, time_ids, criterion) -> torch.Tensor:
    risks = _per_year_risks(logits, labels, time_ids, criterion)
    if len(risks) <= 1:
        return logits.new_tensor(0.0)
    return torch.stack(risks).var(unbiased=False)


def _irm_loss(logits, labels, time_ids, criterion) -> torch.Tensor:
    if time_ids is None or logits.size(0) <= 1:
        return logits.new_tensor(0.0)

    time_ids = time_ids.long().view(-1)
    labels = labels.long().view(-1)
    weight = getattr(criterion, "weight", None)
    label_smoothing = float(getattr(criterion, "label_smoothing", 0.0))
    scale = logits.new_tensor(1.0, requires_grad=True)

    penalties = []
    for year in time_ids.unique():
        mask = time_ids == year
        if mask.sum() < 2:
            continue
        risk = F.cross_entropy(
            logits[mask].float() * scale,
            labels[mask],
            weight=weight,
            label_smoothing=label_smoothing,
        )
        grad = torch.autograd.grad(risk, [scale], create_graph=True, retain_graph=True)[0]
        penalties.append(grad.pow(2))
    return torch.stack(penalties).mean() if penalties else logits.new_tensor(0.0)


def _coral_loss(features: torch.Tensor | None, time_ids: torch.Tensor | None) -> torch.Tensor:
    if features is None or time_ids is None or features.numel() == 0:
        device = features.device if isinstance(features, torch.Tensor) else torch.device("cpu")
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    z = features.float()
    time_ids = time_ids.long().view(-1)
    groups = []
    for year in time_ids.unique():
        part = z[time_ids == year]
        if part.size(0) < 2:
            continue
        centered = part - part.mean(dim=0, keepdim=True)
        cov = centered.t().matmul(centered) / max(part.size(0) - 1, 1)
        groups.append((part.mean(dim=0), cov))
    if len(groups) <= 1:
        return z.new_tensor(0.0)

    losses = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            mean_i, cov_i = groups[i]
            mean_j, cov_j = groups[j]
            losses.append((mean_i - mean_j).pow(2).mean() + (cov_i - cov_j).pow(2).mean())
    return torch.stack(losses).mean() if losses else z.new_tensor(0.0)


def _temporal_supervised_contrastive_loss(
    features: torch.Tensor | None,
    labels: torch.Tensor,
    time_ids: torch.Tensor | None,
    temperature: float = 0.2,
    same_year_positive_weight: float = 0.25,
) -> torch.Tensor:
    if features is None or time_ids is None or features.numel() == 0:
        return torch.tensor(0.0, device=labels.device, dtype=torch.float32)
    if features.size(0) <= 1:
        return features.new_tensor(0.0)

    z = F.normalize(features.float(), dim=-1)
    labels = labels.long().view(-1)
    time_ids = time_ids.long().view(-1)
    temp = max(float(temperature), 1e-4)

    logits = z @ z.t() / temp
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(z.size(0), device=z.device, dtype=torch.bool)
    logits = logits.masked_fill(self_mask, -1e4)

    same_cls = labels[:, None].eq(labels[None, :]) & (~self_mask)
    if not same_cls.any():
        return z.new_tensor(0.0)

    diff_year = time_ids[:, None].ne(time_ids[None, :])
    pos_weight = torch.where(
        diff_year,
        logits.new_ones(logits.shape),
        logits.new_full(logits.shape, float(same_year_positive_weight)),
    ) * same_cls.to(logits.dtype)

    exp_logits = torch.exp(logits)
    denom = exp_logits.masked_fill(self_mask, 0.0).sum(dim=1).clamp_min(1e-8)
    pos_sum = (exp_logits * pos_weight).sum(dim=1)
    valid = pos_sum > 0
    if not valid.any():
        return z.new_tensor(0.0)
    return (-torch.log(pos_sum[valid].clamp_min(1e-8) / denom[valid])).mean()


def compute_total_loss(logits, extra, y, criterion, loss_cfg, epoch=0, total_epochs=1):
    """Aggregate classification, temporal drift, and semantic alignment losses."""
    proto_current_weight = float(loss_cfg["temporal_proto_current_weight"])
    proto_future_weight = float(loss_cfg["temporal_proto_future_weight"])
    temporal_risk_calib_weight = float(loss_cfg["temporal_risk_calibration_weight"])
    semantic_align_weight = float(loss_cfg["semantic_alignment_weight"])
    branch_aux_weight = float(loss_cfg["branch_aux_weight"])
    groupdro_weight = float(loss_cfg["groupdro_weight"])
    irm_weight = float(loss_cfg["irm_weight"])
    vrex_weight = float(loss_cfg["temporal_risk_var_weight"])
    coral_weight = float(loss_cfg["coral_weight"])
    supcon_weight = float(loss_cfg["temporal_contrastive_weight"])

    loss_cls = criterion(logits, y)
    if not torch.isfinite(loss_cls).all():
        loss_cls = logits.new_tensor(0.0, requires_grad=True)

    zero = loss_cls.detach().new_tensor(0.0)

    def _safe(t):
        if t is None or not torch.isfinite(t).all():
            return zero
        return t

    features = extra.get("temporal_features", None)
    time_ids = extra.get("time_ids")
    temporal_quality = extra.get("temporal_quality")

    temporal_memory = extra.get("temporal_prototype_memory")
    loss_proto_current = zero
    loss_proto_future = zero
    if temporal_memory is not None and features is not None and time_ids is not None:
        if proto_current_weight != 0.0:
            loss_proto_current = _safe(temporal_memory.get_loss_quality_gated(
                features,
                y,
                time_ids,
                temporal_quality,
            ))
        if proto_future_weight != 0.0:
            loss_proto_future = _safe(temporal_memory.get_future_forecast_loss(
                features,
                y,
                time_ids,
                temporal_quality,
                temperature=float(loss_cfg["temporal_proto_temperature"]),
                velocity_scale=float(loss_cfg["temporal_proto_velocity_scale"]),
                min_history=int(loss_cfg["temporal_proto_min_history"]),
            ))
    loss_groupdro = zero
    if groupdro_weight != 0.0:
        loss_groupdro = _safe(_groupdro_loss(
            logits,
            y,
            time_ids,
            criterion,
            temperature=float(loss_cfg["groupdro_temperature"]),
        ))

    loss_irm = zero
    if irm_weight != 0.0:
        loss_irm = _safe(_irm_loss(logits, y, time_ids, criterion))

    loss_vrex = zero
    if vrex_weight != 0.0:
        loss_vrex = _safe(_vrex_loss(logits, y, time_ids, criterion))

    loss_coral = zero
    if coral_weight != 0.0 and features is not None:
        loss_coral = _safe(_coral_loss(features, time_ids))

    loss_supcon = zero
    if supcon_weight != 0.0:
        loss_supcon = _safe(_temporal_supervised_contrastive_loss(
            features,
            y,
            time_ids,
            temperature=float(loss_cfg["temporal_temperature"]),
            same_year_positive_weight=float(loss_cfg["temporal_same_year_positive_weight"]),
        ))

    loss_temporal_risk_calib = zero
    loss_temporal_risk_bce = zero
    loss_temporal_risk_rank = zero
    temporal_risk_pos_rate = zero
    risk_logit = extra.get("temporal_risk_logit")
    if temporal_risk_calib_weight != 0.0 and risk_logit is not None:
        risk_logit = risk_logit.float().view(-1)
        if risk_logit.numel() == y.numel():
            with torch.no_grad():
                probs = torch.softmax(logits.detach().float(), dim=-1)
                row = torch.arange(y.numel(), device=y.device)
                safe_y = y.long().clamp(0, probs.size(-1) - 1)
                pred = probs.argmax(dim=-1)
                wrong = pred.ne(safe_y).float()
                soft_error = (1.0 - probs[row, safe_y]).clamp(0.0, 1.0)
                # Put the risk head on the correct side of the ranking first:
                # wrong predictions must score higher than merely uncertain ones.
                risk_target = (0.7 * wrong + 0.3 * soft_error).clamp(0.0, 1.0)
                wrong_mask = wrong > 0.5
                correct_mask = ~wrong_mask
                pos = wrong_mask.float().sum()
                neg = correct_mask.float().sum()
                temporal_risk_pos_rate = wrong.mean()

            bce = F.binary_cross_entropy_with_logits(
                risk_logit,
                risk_target,
                reduction="none",
            )
            if float(pos.item()) > 0.0 and float(neg.item()) > 0.0:
                max_pos_weight = float(ArchitectureConstants.TEMPORAL_RISK_POS_WEIGHT_MAX)
                pos_weight = (neg / pos).clamp(1.0, max_pos_weight)
                sample_weight = torch.where(
                    wrong_mask,
                    torch.ones_like(risk_logit) * pos_weight,
                    torch.ones_like(risk_logit),
                )
                loss_temporal_risk_bce = (bce * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)
            else:
                loss_temporal_risk_bce = bce.mean()

            if bool(wrong_mask.any().item()) and bool(correct_mask.any().item()):
                wrong_logits = risk_logit[wrong_mask]
                correct_logits = risk_logit[correct_mask]
                margin = float(ArchitectureConstants.TEMPORAL_RISK_RANK_MARGIN)
                diff = wrong_logits.view(-1, 1) - correct_logits.view(1, -1)
                loss_temporal_risk_rank = F.softplus(margin - diff).mean()

            rank_weight = float(ArchitectureConstants.TEMPORAL_RISK_RANK_WEIGHT)
            loss_temporal_risk_calib = _safe(
                loss_temporal_risk_bce + rank_weight * loss_temporal_risk_rank
            )

    loss_temporal = (
        loss_proto_current
        + loss_proto_future
        + loss_temporal_risk_calib
        + loss_groupdro
        + loss_irm
        + loss_vrex
        + loss_coral
        + loss_supcon
    )

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
            temperature=float(loss_cfg["temporal_temperature"]),
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
        + proto_current_weight * loss_proto_current
        + proto_future_weight * loss_proto_future
        + groupdro_weight * loss_groupdro
        + irm_weight * loss_irm
        + vrex_weight * loss_vrex
        + coral_weight * loss_coral
        + supcon_weight * loss_supcon
        + temporal_risk_calib_weight * loss_temporal_risk_calib
        + semantic_align_weight * loss_semantic_align
        + branch_aux_weight * loss_branch_aux
    )

    if not torch.isfinite(total).all():
        total = loss_cls if torch.isfinite(loss_cls).all() else logits.new_tensor(0.0, requires_grad=True)

    extra["loss_components"] = {
        "proto_current": loss_proto_current.detach(),
        "proto_future": loss_proto_future.detach(),
        "temporal_risk_calibration": loss_temporal_risk_calib.detach(),
        "temporal_risk_bce": loss_temporal_risk_bce.detach(),
        "temporal_risk_rank": loss_temporal_risk_rank.detach(),
        "temporal_risk_pos_rate": temporal_risk_pos_rate.detach(),
        "groupdro": loss_groupdro.detach(),
        "irm": loss_irm.detach(),
        "vrex": loss_vrex.detach(),
        "coral": loss_coral.detach(),
        "supcon": loss_supcon.detach(),
        "semantic_align": loss_semantic_align.detach(),
        "branch_aux": loss_branch_aux.detach(),
        "weighted_proto_current": (proto_current_weight * loss_proto_current).detach(),
        "weighted_proto_future": (proto_future_weight * loss_proto_future).detach(),
        "weighted_temporal_risk_calibration": (
            temporal_risk_calib_weight * loss_temporal_risk_calib
        ).detach(),
        "weighted_semantic_align": (semantic_align_weight * loss_semantic_align).detach(),
        "weighted_branch_aux": (branch_aux_weight * loss_branch_aux).detach(),
    }

    return (
        total,
        loss_cls.detach(),
        loss_temporal.detach(),
        loss_semantic_align.detach(),
    )
