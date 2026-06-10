from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fusion.constants import VIEW_TYPES


def _weighted_symmetric_kl(
    clean_logits: torch.Tensor,
    aug_logits: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Symmetric KL consistency with optional per-sample reliability weights.

    clean_logits: logits from the clean/original view.
    aug_logits: logits from the degraded/augmented view.
    weight: per-sample weight in [0, 1].
    """
    if clean_logits.size(0) == 0:
        return clean_logits.new_tensor(0.0)

    # KL(clean || aug), where clean distribution is treated as teacher.
    p_clean_detached = F.softmax(clean_logits.detach(), dim=-1)
    p_aug = F.softmax(aug_logits, dim=-1)
    kl_clean_to_aug = F.kl_div(
        p_aug.clamp_min(1e-8).log(),
        p_clean_detached,
        reduction="none",
    ).sum(dim=-1)

    # KL(aug || clean), where aug distribution is treated as teacher.
    p_clean = F.softmax(clean_logits, dim=-1)
    p_aug_detached = F.softmax(aug_logits.detach(), dim=-1)
    kl_aug_to_clean = F.kl_div(
        p_clean.clamp_min(1e-8).log(),
        p_aug_detached,
        reduction="none",
    ).sum(dim=-1)

    per_sample = 0.5 * (kl_clean_to_aug + kl_aug_to_clean)

    weight = weight.to(device=clean_logits.device, dtype=clean_logits.dtype).view(-1)
    if weight.numel() != clean_logits.size(0):
        weight = torch.ones(
            clean_logits.size(0),
            device=clean_logits.device,
            dtype=clean_logits.dtype,
        )

    weight = weight.clamp(0.0, 1.0)

    return (per_sample * weight).sum() / weight.sum().clamp_min(1.0)


def _extra_vector(
    extra: dict[str, torch.Tensor],
    key: str,
    ref: torch.Tensor,
    default: float,
) -> torch.Tensor:
    """Read a batch-size vector from model extra outputs.

    If the key is missing or has a wrong shape, use a default vector.
    """
    value = extra.get(key)
    if isinstance(value, torch.Tensor):
        out = value.to(device=ref.device, dtype=ref.dtype).view(-1)
        if out.numel() == ref.size(0):
            return out

    return ref.new_full((ref.size(0),), float(default))


def _view_tensor(names: list[str], ref: torch.Tensor) -> torch.Tensor:
    """Convert view names to a tensor of view ids, ignoring missing names."""
    values = [VIEW_TYPES[name] for name in names if name in VIEW_TYPES]
    if not values:
        return torch.empty((0,), device=ref.device, dtype=torch.long)

    return torch.tensor(values, device=ref.device, dtype=torch.long)


def _consistency_weight(
    clean_extra: dict[str, torch.Tensor],
    aug_extra: dict[str, torch.Tensor],
    ref: torch.Tensor,
) -> torch.Tensor:
    """Reliability-aware consistency weight for compact_kl.

    Intuition:
    - Manifest-corrupted views should mainly trust code reliability.
    - Code/API/graph-corrupted views should mainly trust manifest reliability.
    - All-degraded views require both sides to be reliable.
    - cf_weight from the augmented view is kept as a base gate.
    """
    base = aug_extra.get("cf_weight")
    if not isinstance(base, torch.Tensor):
        base = ref.new_ones((ref.size(0),))

    base = base.to(device=ref.device, dtype=ref.dtype).view(-1)
    if base.numel() != ref.size(0):
        base = ref.new_ones((ref.size(0),))
    base = base.clamp(0.0, 1.0)

    view = aug_extra.get("view_type_id")
    if not isinstance(view, torch.Tensor):
        return base

    view = view.to(device=ref.device).view(-1).long()
    if view.numel() != ref.size(0):
        return base

    code_rel = _extra_vector(clean_extra, "code_reliability", ref, 1.0).clamp(0.0, 1.0)
    manifest_rel = _extra_vector(clean_extra, "manifest_reliability", ref, 1.0).clamp(0.0, 1.0)

    conditional = torch.ones_like(base)

    manifest_views = _view_tensor(
        [
            "manifest_degraded",
            "manifest_zeroed",
            "manifest_noisy",
            "manifest_shuffled",
            "manifest_noisy_blind",
            "manifest_shuffled_blind",
            "manifest_missing",
        ],
        ref,
    )

    code_views = _view_tensor(
        [
            "api_degraded",
            "graph_degraded",
            "api_graph_degraded",
            "api_missing",
            "graph_missing",
        ],
        ref,
    )

    if manifest_views.numel() > 0:
        manifest_mask = torch.isin(view, manifest_views)
        # Manifest 被扰动时，一致性约束主要依赖 clean view 的 code reliability。
        conditional = torch.where(manifest_mask, code_rel, conditional)

    if code_views.numel() > 0:
        code_mask = torch.isin(view, code_views)
        # Code/API/graph 被扰动时，一致性约束主要依赖 clean view 的 manifest reliability。
        conditional = torch.where(code_mask, manifest_rel, conditional)

    if "all_degraded" in VIEW_TYPES:
        all_mask = view == VIEW_TYPES["all_degraded"]
        # 全部退化时，两侧都可靠才强约束。
        conditional = torch.where(all_mask, torch.minimum(code_rel, manifest_rel), conditional)

    return (base * conditional).clamp(0.0, 1.0)


def compute_aeg_loss(
    clean_logits: torch.Tensor,
    labels: torch.Tensor,
    clean_extra: dict[str, torch.Tensor],
    *,
    aug_logits: torch.Tensor | None = None,
    aug_extra: dict[str, torch.Tensor] | None = None,
    loss_cfg: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute AEG training loss.

    Supported modes:
    - ce_only:
        L = CE(clean)

    - plain_kl:
        L = CE(clean) + lambda * KL(clean, aug)

    - compact_kl:
        L = CE(clean) + lambda * reliability_weight * KL(clean, aug)

    This intentionally removes the old multi-contrastive objectives to keep the
    method aligned with reliability-aware multi-view consistency learning.
    """
    cfg = loss_cfg or {}
    mode = str(cfg.get("mode", "compact_kl")).lower()

    if mode not in {"ce_only", "plain_kl", "compact_kl"}:
        raise ValueError(
            f"Unsupported loss.mode={mode!r}. "
            "Expected one of: ce_only, plain_kl, compact_kl."
        )

    ce_weight = float(cfg.get("ce_weight", 1.0))
    consistency_weight = float(cfg.get("consistency_weight", 0.05))
    aug_ce_weight = float(cfg.get("aug_ce_weight", 0.0))

    labels = labels.view(-1).long()
    ce = F.cross_entropy(clean_logits, labels)

    aug_ce = clean_logits.new_tensor(0.0)
    consistency = clean_logits.new_tensor(0.0)

    if mode in {"plain_kl", "compact_kl"} and (aug_logits is None or aug_extra is None):
        raise ValueError(
            f"loss.mode={mode} requires augmented logits/extra. "
            "Set robust.train_aug=true or use loss.mode=ce_only."
        )

    if aug_logits is not None and aug_extra is not None:
        if aug_ce_weight > 0.0:
            aug_ce = F.cross_entropy(aug_logits, labels)

        if mode == "plain_kl":
            weight = clean_logits.new_ones((clean_logits.size(0),))
            consistency = _weighted_symmetric_kl(clean_logits, aug_logits, weight)

        elif mode == "compact_kl":
            weight = _consistency_weight(clean_extra, aug_extra, clean_logits)
            consistency = _weighted_symmetric_kl(clean_logits, aug_logits, weight)

    if mode == "ce_only":
        consistency_weight = 0.0

    total = ce_weight * ce + aug_ce_weight * aug_ce + consistency_weight * consistency

    weighted_ce = ce_weight * ce
    weighted_aug_ce = aug_ce_weight * aug_ce
    weighted_consistency = consistency_weight * consistency

    aux_to_ce_ratio = float(
        (weighted_consistency.detach() / weighted_ce.detach().clamp_min(1e-8)).item()
    )

    parts = {
        "loss": float(total.detach().item()),
        "loss_mode": mode,

        "ce": float(ce.detach().item()),
        "aug_ce": float(aug_ce.detach().item()),
        "consistency": float(consistency.detach().item()),

        "weighted_ce": float(weighted_ce.detach().item()),
        "weighted_aug_ce": float(weighted_aug_ce.detach().item()),
        "weighted_consistency": float(weighted_consistency.detach().item()),
        "aux_to_ce_ratio": aux_to_ce_ratio,

        "ce_weight": ce_weight,
        "aug_ce_weight": aug_ce_weight,
        "consistency_weight": consistency_weight,
    }

    return total, parts