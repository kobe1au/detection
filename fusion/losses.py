from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fusion.constants import VIEW_TYPES


def _info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float, weights: torch.Tensor | None = None) -> torch.Tensor:
    if a.size(0) <= 1:
        return a.new_tensor(0.0)
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits_ab = a @ b.t() / max(float(temperature), 1e-4)
    logits_ba = b @ a.t() / max(float(temperature), 1e-4)
    labels = torch.arange(a.size(0), device=a.device)
    loss = 0.5 * (
        F.cross_entropy(logits_ab, labels, reduction="none")
        + F.cross_entropy(logits_ba, labels, reduction="none")
    )
    if weights is None:
        return loss.mean()
    weights = weights.to(device=a.device, dtype=a.dtype).view(-1).clamp(0.0, 1.0)
    if weights.numel() != loss.numel():
        return loss.mean()
    return (loss * weights).sum() / weights.sum().clamp_min(1e-8)


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


def _conditional_cf_weight(clean_extra: dict[str, torch.Tensor], aug_extra: dict[str, torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    base = aug_extra.get("cf_weight", ref.new_ones((ref.size(0),)))
    if not isinstance(base, torch.Tensor):
        base = ref.new_ones((ref.size(0),))
    base = base.to(device=ref.device, dtype=ref.dtype).view(-1).clamp(0.0, 1.0)
    view = aug_extra.get("view_type_id")
    if not isinstance(view, torch.Tensor):
        return base
    view = view.to(device=ref.device).view(-1).long()
    code_rel = clean_extra.get("code_reliability", ref.new_zeros((ref.size(0),))).to(ref.device).float().view(-1)
    manifest_rel = clean_extra.get("manifest_reliability", ref.new_zeros((ref.size(0),))).to(ref.device).float().view(-1)
    conditional = torch.ones_like(base)
    manifest_views = torch.tensor(
        [
            VIEW_TYPES["manifest_degraded"],
            VIEW_TYPES["manifest_zeroed"],
            VIEW_TYPES["manifest_noisy"],
            VIEW_TYPES["manifest_shuffled"],
            VIEW_TYPES["manifest_noisy_blind"],
            VIEW_TYPES["manifest_shuffled_blind"],
            VIEW_TYPES["manifest_missing"],
        ],
        device=ref.device,
    )
    code_views = torch.tensor(
        [
            VIEW_TYPES["api_degraded"],
            VIEW_TYPES["graph_degraded"],
            VIEW_TYPES["api_graph_degraded"],
            VIEW_TYPES["api_missing"],
            VIEW_TYPES["graph_missing"],
        ],
        device=ref.device,
    )
    all_view = view == VIEW_TYPES["all_degraded"]
    manifest_mask = torch.isin(view, manifest_views)
    code_mask = torch.isin(view, code_views)
    conditional = torch.where(manifest_mask, code_rel, conditional)
    conditional = torch.where(code_mask, manifest_rel, conditional)
    conditional = torch.where(all_view, torch.minimum(code_rel, manifest_rel), conditional)
    return (base * conditional.clamp(0.0, 1.0)).clamp(0.0, 1.0)


def _extra_vector(extra: dict[str, torch.Tensor], key: str, ref: torch.Tensor, default: float) -> torch.Tensor:
    value = extra.get(key)
    if isinstance(value, torch.Tensor):
        out = value.to(device=ref.device, dtype=ref.dtype).view(-1)
        if out.numel() == ref.size(0):
            return out
    return ref.new_full((ref.size(0),), float(default))


def _source_contrast_weights(
    clean_extra: dict[str, torch.Tensor],
    aug_extra: dict[str, torch.Tensor],
    ref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    clean_code = _extra_vector(clean_extra, "code_reliability", ref, 1.0)
    aug_code = _extra_vector(aug_extra, "code_reliability", ref, 1.0)
    clean_manifest = _extra_vector(clean_extra, "manifest_reliability", ref, 1.0)
    aug_manifest = _extra_vector(aug_extra, "manifest_reliability", ref, 1.0)
    code_weight = torch.minimum(clean_code, aug_code).clamp(0.0, 1.0)
    manifest_weight = torch.minimum(clean_manifest, aug_manifest).clamp(0.0, 1.0)

    view = aug_extra.get("view_type_id")
    if isinstance(view, torch.Tensor):
        view = view.to(device=ref.device).view(-1).long()
        manifest_unalignable = torch.tensor(
            [
                VIEW_TYPES["manifest_zeroed"],
                VIEW_TYPES["manifest_noisy"],
                VIEW_TYPES["manifest_shuffled"],
                VIEW_TYPES["manifest_noisy_blind"],
                VIEW_TYPES["manifest_shuffled_blind"],
                VIEW_TYPES["manifest_missing"],
            ],
            device=ref.device,
        )
        code_unalignable = torch.tensor(
            [
                VIEW_TYPES["api_missing"],
                VIEW_TYPES["graph_missing"],
            ],
            device=ref.device,
        )
        manifest_weight = torch.where(torch.isin(view, manifest_unalignable), torch.zeros_like(manifest_weight), manifest_weight)
        code_weight = torch.where(torch.isin(view, code_unalignable), torch.zeros_like(code_weight), code_weight)
    risk_weight = torch.maximum(code_weight, manifest_weight).clamp(0.0, 1.0)
    return code_weight, manifest_weight, risk_weight


def _fused_contrast_weight(
    clean_extra: dict[str, torch.Tensor],
    aug_extra: dict[str, torch.Tensor],
    ref: torch.Tensor,
) -> torch.Tensor:
    clean_code = _extra_vector(clean_extra, "code_reliability", ref, 1.0)
    aug_code = _extra_vector(aug_extra, "code_reliability", ref, 1.0)
    clean_manifest = _extra_vector(clean_extra, "manifest_reliability", ref, 1.0)
    aug_manifest = _extra_vector(aug_extra, "manifest_reliability", ref, 1.0)
    clean_available = torch.maximum(clean_code, clean_manifest)
    aug_available = torch.maximum(aug_code, aug_manifest)
    return torch.minimum(clean_available, aug_available).clamp(0.0, 1.0)


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
    source_aug_weight = float(cfg.get("source_degraded_contrast_weight", 0.05))
    cross_source_weight = float(cfg.get("cross_source_contrast_weight", 0.03))
    cf_kl_weight = float(cfg.get("counterfactual_kl_weight", 0.05))
    aug_ce_weight = float(cfg.get("aug_ce_weight", 0.0))
    reliability_weighted = bool(cfg.get("reliability_weighted_contrast", True))
    temperature = float(cfg.get("temperature", 0.2))

    labels = labels.view(-1).long()
    ce = F.cross_entropy(clean_logits, labels)
    aug_ce = clean_logits.new_tensor(0.0)
    clean_aug = clean_logits.new_tensor(0.0)
    source_aug = clean_logits.new_tensor(0.0)
    cf_kl = clean_logits.new_tensor(0.0)
    if aug_logits is not None and aug_extra is not None:
        aug_ce = F.cross_entropy(aug_logits, labels)
        fused_weight = _fused_contrast_weight(clean_extra, aug_extra, clean_logits) if reliability_weighted else None
        clean_aug = _info_nce(
            clean_extra["fused_emb"],
            aug_extra["fused_emb"],
            temperature,
            fused_weight,
        )
        if reliability_weighted:
            code_weight, manifest_weight, risk_weight = _source_contrast_weights(clean_extra, aug_extra, clean_logits)
        else:
            code_weight = manifest_weight = risk_weight = None
        source_terms = [
            _info_nce(clean_extra["method_emb"], aug_extra["method_emb"], temperature, code_weight),
            _info_nce(clean_extra["api_family_emb"], aug_extra["api_family_emb"], temperature, code_weight),
            _info_nce(clean_extra["permission_emb"], aug_extra["permission_emb"], temperature, manifest_weight),
            _info_nce(clean_extra["component_emb"], aug_extra["component_emb"], temperature, manifest_weight),
            _info_nce(clean_extra["risk_emb"], aug_extra["risk_emb"], temperature, risk_weight),
        ]
        source_aug = torch.stack(source_terms).mean()
        cf_weight = _conditional_cf_weight(clean_extra, aug_extra, clean_logits)
        cf_kl = _weighted_symmetric_kl(clean_logits, aug_logits, cf_weight)

    rel_weight = (
        _reliability_weight(clean_extra).to(device=clean_logits.device, dtype=clean_logits.dtype)
        if reliability_weighted
        else None
    )
    cross_source = _info_nce(clean_extra["code_emb"], clean_extra["manifest_emb"], temperature, rel_weight)

    total = (
        ce_weight * ce
        + aug_ce_weight * aug_ce
        + clean_aug_weight * clean_aug
        + source_aug_weight * source_aug
        + cross_source_weight * cross_source
        + cf_kl_weight * cf_kl
    )
    parts = {
        "loss": float(total.detach().item()),
        "ce": float(ce.detach().item()),
        "aug_ce": float(aug_ce.detach().item()),
        "clean_degraded_contrast": float(clean_aug.detach().item()),
        "source_degraded_contrast": float(source_aug.detach().item()),
        "cross_source_contrast": float(cross_source.detach().item()),
        "counterfactual_kl": float(cf_kl.detach().item()),
        "ce_weight": ce_weight,
        "clean_degraded_contrast_weight": clean_aug_weight,
        "source_degraded_contrast_weight": source_aug_weight,
        "cross_source_contrast_weight": cross_source_weight,
        "counterfactual_kl_weight": cf_kl_weight,
        "aug_ce_weight": aug_ce_weight,
        "reliability_weighted_contrast": float(reliability_weighted),
    }
    return total, parts
