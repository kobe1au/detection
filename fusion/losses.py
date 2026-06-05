from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    if a.size(0) <= 1:
        return a.new_tensor(0.0)
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits_ab = a @ b.t() / max(float(temperature), 1e-4)
    logits_ba = b @ a.t() / max(float(temperature), 1e-4)
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits_ab, labels) + F.cross_entropy(logits_ba, labels))


def _weighted_symmetric_kl(clean_logits: torch.Tensor, aug_logits: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if clean_logits.size(0) == 0:
        return clean_logits.new_tensor(0.0)
    p = F.softmax(clean_logits.detach(), dim=-1)
    q = F.softmax(aug_logits, dim=-1)
    kl_1 = F.kl_div(q.clamp_min(1e-8).log(), p, reduction="none").sum(dim=-1)
    p_train = F.softmax(clean_logits, dim=-1)
    q_detach = F.softmax(aug_logits.detach(), dim=-1)
    kl_2 = F.kl_div(p_train.clamp_min(1e-8).log(), q_detach, reduction="none").sum(dim=-1)
    weight = weight.float().view(-1).to(clean_logits.device).clamp(0.0, 1.0)
    if weight.numel() != clean_logits.size(0):
        weight = torch.ones((clean_logits.size(0),), dtype=clean_logits.dtype, device=clean_logits.device)
    return ((kl_1 + kl_2) * 0.5 * weight).sum() / weight.sum().clamp_min(1.0)


def _reliability_weight(extra: dict[str, torch.Tensor]) -> torch.Tensor:
    code_rel = extra.get("code_reliability")
    manifest_rel = extra.get("manifest_reliability")
    conflict = extra.get("code_manifest_conflict")
    if not isinstance(code_rel, torch.Tensor) or not isinstance(manifest_rel, torch.Tensor):
        return torch.tensor(1.0)
    weight = (code_rel.float().view(-1) * manifest_rel.float().view(-1)).sqrt()
    if isinstance(conflict, torch.Tensor):
        weight = weight * (1.0 - conflict.float().view(-1).clamp(0.0, 1.0))
    return weight.clamp(0.0, 1.0)


def compute_aeg_loss(
    clean_logits: torch.Tensor,
    labels: torch.Tensor,
    clean_extra: dict[str, torch.Tensor],
    *,
    aug_logits: torch.Tensor | None = None,
    aug_extra: dict[str, torch.Tensor] | None = None,
    loss_cfg: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    cfg = loss_cfg or {}
    ce_weight = float(cfg.get("ce_weight", 1.0))
    clean_aug_weight = float(cfg.get("clean_degraded_contrast_weight", 0.1))
    cross_source_weight = float(cfg.get("cross_source_contrast_weight", 0.03))
    cf_kl_weight = float(cfg.get("counterfactual_kl_weight", 0.05))
    temperature = float(cfg.get("temperature", 0.2))

    labels = labels.view(-1).long()
    ce = F.cross_entropy(clean_logits, labels)
    clean_aug = clean_logits.new_tensor(0.0)
    cf_kl = clean_logits.new_tensor(0.0)
    if aug_logits is not None and aug_extra is not None:
        clean_aug = _info_nce(clean_extra["fused_emb"], aug_extra["fused_emb"], temperature)
        cf_weight = aug_extra.get("cf_weight", clean_logits.new_ones((clean_logits.size(0),)))
        cf_kl = _weighted_symmetric_kl(clean_logits, aug_logits, cf_weight)

    cross_source = _info_nce(clean_extra["code_emb"], clean_extra["manifest_emb"], temperature)
    rel_weight = _reliability_weight(clean_extra).to(device=clean_logits.device, dtype=clean_logits.dtype)
    if rel_weight.ndim > 0 and rel_weight.numel() == clean_logits.size(0):
        cross_source = cross_source * rel_weight.mean()

    total = ce_weight * ce + clean_aug_weight * clean_aug + cross_source_weight * cross_source + cf_kl_weight * cf_kl
    parts = {
        "loss": float(total.detach().item()),
        "ce": float(ce.detach().item()),
        "clean_degraded_contrast": float(clean_aug.detach().item()),
        "cross_source_contrast": float(cross_source.detach().item()),
        "counterfactual_kl": float(cf_kl.detach().item()),
        "ce_weight": ce_weight,
        "clean_degraded_contrast_weight": clean_aug_weight,
        "cross_source_contrast_weight": cross_source_weight,
        "counterfactual_kl_weight": cf_kl_weight,
    }
    return total, parts
