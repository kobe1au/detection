from __future__ import annotations

import torch
import torch.nn.functional as F


BRANCH_AUX_KEYS = (
    "api_logits_aux",
    "graph_logits_aux",
    "manifest_logits_aux",
    "joint_logits_aux",
)

BRANCH_AUX_NAMES = {
    "api_logits_aux": "api",
    "graph_logits_aux": "graph",
    "manifest_logits_aux": "manifest",
    "joint_logits_aux": "joint",
}


def _matrix(extra: dict, key: str, ref: torch.Tensor) -> torch.Tensor | None:
    value = extra.get(key)
    if not isinstance(value, torch.Tensor):
        return None
    out = value.to(device=ref.device, dtype=ref.dtype)
    if out.ndim == 1:
        out = out.view(1, -1).expand(ref.size(0), -1)
    else:
        out = out.view(ref.size(0), -1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _reliability(extra: dict, key: str, ref: torch.Tensor, default: float) -> torch.Tensor:
    value = extra.get(key)
    if not isinstance(value, torch.Tensor):
        return torch.full((ref.size(0),), float(default), device=ref.device, dtype=ref.dtype)
    out = value.to(device=ref.device, dtype=ref.dtype).view(ref.size(0), -1)[:, 0]
    return torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _weighted_cosine_direction_loss(
    pred_logits: torch.Tensor | None,
    target_counts: torch.Tensor | None,
    weights: torch.Tensor,
    active_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred_logits is None or target_counts is None:
        zero = weights.new_tensor(0.0)
        return zero, zero
    pred = F.softplus(pred_logits.float())
    target = target_counts.float().clamp_min(0.0)
    if pred.size(1) != target.size(1):
        zero = weights.new_tensor(0.0)
        return zero, zero
    if active_only:
        active = target > 0
        pred = pred * active.to(dtype=pred.dtype)
        target = target * active.to(dtype=target.dtype)
    valid = (target.abs().sum(dim=-1) > 0) & (weights > 0)
    if not valid.any():
        zero = weights.new_tensor(0.0)
        return zero, zero
    sim = F.cosine_similarity(pred[valid], target[valid], dim=-1).clamp(-1.0, 1.0)
    loss = 1.0 - sim
    w = weights[valid].float()
    return (loss * w).sum() / w.sum().clamp_min(1e-8), w.sum()


def _pair_weight(
    base_weight: torch.Tensor,
    consistency: torch.Tensor | None,
    min_reliability: float,
    min_consistency: float,
) -> torch.Tensor:
    weight = base_weight
    if min_reliability > 0.0:
        weight = torch.where(weight >= min_reliability, weight, torch.zeros_like(weight))
    if consistency is not None and min_consistency > 0.0:
        weight = torch.where(consistency >= min_consistency, weight, torch.zeros_like(weight))
    return weight


def _semantic_losses(
    extra: dict,
    ref_logits: torch.Tensor,
    loss_cfg: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    loss_cfg = loss_cfg or {}
    api_pred = _matrix(extra, "api_semantic_logits", ref_logits)
    graph_pred = _matrix(extra, "graph_semantic_logits", ref_logits)
    manifest_pred = _matrix(extra, "manifest_semantic_logits", ref_logits)
    api_counts = _matrix(extra, "api_semantic_category_counts", ref_logits)
    graph_counts = _matrix(extra, "graph_semantic_category_counts", ref_logits)
    manifest_counts = _matrix(extra, "manifest_category_counts", ref_logits)

    r_api = _reliability(extra, "r_api", ref_logits, 1.0)
    r_graph = _reliability(extra, "r_graph", ref_logits, 1.0)
    r_manifest = _reliability(extra, "r_manifest", ref_logits, 0.0)
    api_manifest = _reliability(extra, "api_manifest_consistency", ref_logits, 0.0)
    graph_manifest = _reliability(extra, "graph_manifest_consistency", ref_logits, 0.0)
    min_reliability = float(loss_cfg.get("cross_source_min_reliability", 0.0))
    min_consistency = float(loss_cfg.get("cross_source_min_consistency", 0.0))
    active_only = bool(loss_cfg.get("semantic_active_only", False))

    reconstruction_terms: list[torch.Tensor] = []
    cross_source_terms: list[torch.Tensor] = []
    for pred, target, weight, consistency in (
        (api_pred, api_counts, r_api, None),
        (graph_pred, graph_counts, r_graph, None),
        (manifest_pred, manifest_counts, r_manifest, None),
    ):
        filtered_weight = _pair_weight(weight, None, min_reliability, 0.0)
        term, used_weight = _weighted_cosine_direction_loss(pred, target, filtered_weight, active_only=active_only)
        if float(used_weight.detach().item()) > 0.0:
            reconstruction_terms.append(term)

    for pred, target, weight, consistency in (
        (api_pred, manifest_counts, r_api * r_manifest, api_manifest),
        (graph_pred, manifest_counts, r_graph * r_manifest, graph_manifest),
        (manifest_pred, api_counts, r_manifest * r_api, api_manifest),
        (manifest_pred, graph_counts, r_manifest * r_graph, graph_manifest),
    ):
        filtered_weight = _pair_weight(weight, consistency, min_reliability, min_consistency)
        term, used_weight = _weighted_cosine_direction_loss(pred, target, filtered_weight, active_only=active_only)
        if float(used_weight.detach().item()) > 0.0:
            cross_source_terms.append(term)

    reconstruction = (
        torch.stack(reconstruction_terms).mean().to(dtype=ref_logits.dtype)
        if reconstruction_terms
        else ref_logits.new_tensor(0.0)
    )
    cross_source = (
        torch.stack(cross_source_terms).mean().to(dtype=ref_logits.dtype)
        if cross_source_terms
        else ref_logits.new_tensor(0.0)
    )
    return reconstruction, cross_source


def _gate_prior_target(extra: dict, ref_logits: torch.Tensor) -> torch.Tensor:
    r_api = _reliability(extra, "r_api", ref_logits, 1.0).view(-1, 1)
    r_graph = _reliability(extra, "r_graph", ref_logits, 1.0).view(-1, 1)
    r_manifest = _reliability(extra, "r_manifest", ref_logits, 0.0).view(-1, 1)
    api_manifest = _reliability(extra, "api_manifest_consistency", ref_logits, 0.0).view(-1, 1)
    graph_manifest = _reliability(extra, "graph_manifest_consistency", ref_logits, 0.0).view(-1, 1)
    api_graph = _reliability(extra, "api_graph_consistency", ref_logits, 0.0).view(-1, 1)
    api_alive = _reliability(extra, "api_alive", ref_logits, 1.0).view(-1, 1)
    graph_alive = _reliability(extra, "graph_alive", ref_logits, 1.0).view(-1, 1)
    manifest_alive = _reliability(extra, "manifest_alive", ref_logits, 0.0).view(-1, 1)

    alive_sum = (api_alive + graph_alive + manifest_alive).clamp_min(1.0)
    reliability_support = (
        r_api * api_alive + r_graph * graph_alive + r_manifest * manifest_alive
    ) / alive_sum
    pair_support = api_alive * graph_alive + api_alive * manifest_alive + graph_alive * manifest_alive
    pair_consistency = (
        api_graph * api_alive * graph_alive
        + api_manifest * api_alive * manifest_alive
        + graph_manifest * graph_alive * manifest_alive
    ) / pair_support.clamp_min(1.0)
    joint_score = (
        reliability_support.square() * (0.5 + 0.5 * pair_consistency)
    ).clamp(0.0, 1.0)
    joint_availability = (alive_sum / 3.0).clamp(0.0, 1.0)
    scores = torch.cat(
        [
            r_api * api_alive,
            r_graph * graph_alive,
            r_manifest * manifest_alive,
            joint_score * joint_availability,
        ],
        dim=-1,
    )
    denom = scores.sum(dim=-1, keepdim=True)
    target = scores / denom.clamp_min(1e-8)
    fallback = torch.full_like(scores, 0.25)
    return torch.where(denom > 1e-8, target, fallback).detach()


def _gate_prior_loss(extra: dict, ref_logits: torch.Tensor) -> torch.Tensor:
    if not bool(extra.get("gate_prior_enabled", False)):
        return ref_logits.new_tensor(0.0)
    gate_weights = extra.get("gate_weights_train")
    if not isinstance(gate_weights, torch.Tensor):
        return ref_logits.new_tensor(0.0)
    gate_weights = gate_weights.to(device=ref_logits.device, dtype=ref_logits.dtype)
    if gate_weights.ndim != 2 or gate_weights.size(0) != ref_logits.size(0) or gate_weights.size(1) != 4:
        return ref_logits.new_tensor(0.0)
    target = _gate_prior_target(extra, ref_logits).to(device=ref_logits.device, dtype=ref_logits.dtype)
    return F.kl_div(
        gate_weights.clamp_min(1e-8).log(),
        target,
        reduction="batchmean",
    ).to(dtype=ref_logits.dtype)


def compute_robust_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    extra: dict | None = None,
    loss_cfg: dict | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Robust objective with independently attributable auxiliary terms."""
    extra = extra or {}
    loss_cfg = loss_cfg or {}
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    branch_aux_weight = float(loss_cfg.get("branch_aux_weight", 0.05))
    semantic_reconstruction_weight = float(loss_cfg.get("semantic_reconstruction_weight", 0.0))
    cross_source_consistency_weight = float(loss_cfg.get("cross_source_consistency_weight", 0.0))
    gate_prior_weight = float(loss_cfg.get("gate_prior_weight", 0.0))
    named_weights = {
        "branch_aux_weight": branch_aux_weight,
        "semantic_reconstruction_weight": semantic_reconstruction_weight,
        "cross_source_consistency_weight": cross_source_consistency_weight,
        "gate_prior_weight": gate_prior_weight,
    }
    for name, value in named_weights.items():
        if not torch.isfinite(logits.new_tensor(value)).item() or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {value}")
    for name in ("cross_source_min_reliability", "cross_source_min_consistency"):
        value = float(loss_cfg.get(name, 0.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be within [0, 1], got {value}")

    ce = F.cross_entropy(logits, labels.long(), label_smoothing=label_smoothing)
    branch_loss = logits.new_tensor(0.0)
    branch_weight_sum = 0.0
    branch_weights = loss_cfg.get("branch_aux_weights") or {}
    if not isinstance(branch_weights, dict):
        branch_weights = {}
    for key in BRANCH_AUX_KEYS:
        aux_logits = extra.get(key)
        if isinstance(aux_logits, torch.Tensor) and aux_logits.shape == logits.shape:
            branch_name = BRANCH_AUX_NAMES.get(key, key)
            weight = float(branch_weights.get(branch_name, branch_weights.get(key, 1.0)))
            if weight <= 0.0:
                continue
            branch_loss = branch_loss + weight * F.cross_entropy(
                aux_logits,
                labels.long(),
                label_smoothing=label_smoothing,
            )
            branch_weight_sum += weight
    if branch_weight_sum > 0.0:
        branch_loss = branch_loss / branch_loss.new_tensor(branch_weight_sum)

    if semantic_reconstruction_weight > 0.0 or cross_source_consistency_weight > 0.0:
        semantic_reconstruction, cross_source_consistency = _semantic_losses(extra, logits, loss_cfg)
    else:
        semantic_reconstruction = logits.new_tensor(0.0)
        cross_source_consistency = logits.new_tensor(0.0)
    gate_prior = (
        _gate_prior_loss(extra, logits)
        if gate_prior_weight > 0.0
        else logits.new_tensor(0.0)
    )

    total = (
        ce
        + branch_aux_weight * branch_loss
        + semantic_reconstruction_weight * semantic_reconstruction
        + cross_source_consistency_weight * cross_source_consistency
        + gate_prior_weight * gate_prior
    )
    return total, {
        "loss": float(total.detach().item()),
        "ce": float(ce.detach().item()),
        "branch_aux": float(branch_loss.detach().item()),
        "branch_aux_weight": branch_aux_weight,
        "semantic_reconstruction": float(semantic_reconstruction.detach().item()),
        "semantic_reconstruction_weight": semantic_reconstruction_weight,
        "cross_source_consistency": float(cross_source_consistency.detach().item()),
        "cross_source_consistency_weight": cross_source_consistency_weight,
        "cross_source_min_reliability": float(loss_cfg.get("cross_source_min_reliability", 0.0)),
        "cross_source_min_consistency": float(loss_cfg.get("cross_source_min_consistency", 0.0)),
        "gate_prior": float(gate_prior.detach().item()),
        "gate_prior_weight": gate_prior_weight,
    }
