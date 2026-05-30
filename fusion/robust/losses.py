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
) -> tuple[torch.Tensor, torch.Tensor]:
    if pred_logits is None or target_counts is None:
        zero = weights.new_tensor(0.0)
        return zero, zero
    pred = F.softplus(pred_logits.float())
    target = target_counts.float().clamp_min(0.0)
    if pred.size(1) != target.size(1):
        zero = weights.new_tensor(0.0)
        return zero, zero
    valid = (target.abs().sum(dim=-1) > 0) & (weights > 0)
    if not valid.any():
        zero = weights.new_tensor(0.0)
        return zero, zero
    sim = F.cosine_similarity(pred[valid], target[valid], dim=-1).clamp(-1.0, 1.0)
    loss = 1.0 - sim
    w = weights[valid].float()
    return (loss * w).sum() / w.sum().clamp_min(1e-8), w.sum()


def _soft_consistency_loss(extra: dict, ref_logits: torch.Tensor) -> torch.Tensor:
    api_pred = _matrix(extra, "api_semantic_logits", ref_logits)
    graph_pred = _matrix(extra, "graph_semantic_logits", ref_logits)
    manifest_pred = _matrix(extra, "manifest_semantic_logits", ref_logits)
    api_counts = _matrix(extra, "api_semantic_category_counts", ref_logits)
    graph_counts = _matrix(extra, "graph_semantic_category_counts", ref_logits)
    manifest_counts = _matrix(extra, "manifest_category_counts", ref_logits)

    r_api = _reliability(extra, "r_api", ref_logits, 1.0)
    r_graph = _reliability(extra, "r_graph", ref_logits, 1.0)
    r_manifest = _reliability(extra, "r_manifest", ref_logits, 0.0)

    terms: list[torch.Tensor] = []
    for pred, target, weight in (
        (api_pred, api_counts, r_api),
        (graph_pred, graph_counts, r_graph),
        (manifest_pred, manifest_counts, r_manifest),
        (api_pred, manifest_counts, r_api * r_manifest),
        (graph_pred, manifest_counts, r_graph * r_manifest),
        (manifest_pred, api_counts, r_manifest * r_api),
        (manifest_pred, graph_counts, r_manifest * r_graph),
        (api_pred, graph_counts, r_api * r_graph),
        (graph_pred, api_counts, r_graph * r_api),
    ):
        term, used_weight = _weighted_cosine_direction_loss(pred, target, weight)
        if float(used_weight.detach().item()) > 0.0:
            terms.append(term)

    if not terms:
        return ref_logits.new_tensor(0.0)
    return torch.stack(terms).mean().to(dtype=ref_logits.dtype)


def _gate_prior_target(extra: dict, ref_logits: torch.Tensor) -> torch.Tensor:
    r_api = _reliability(extra, "r_api", ref_logits, 1.0).view(-1, 1)
    r_graph = _reliability(extra, "r_graph", ref_logits, 1.0).view(-1, 1)
    r_manifest = _reliability(extra, "r_manifest", ref_logits, 0.0).view(-1, 1)
    api_manifest = _reliability(extra, "api_manifest_consistency", ref_logits, 0.0).view(-1, 1)
    graph_manifest = _reliability(extra, "graph_manifest_consistency", ref_logits, 0.0).view(-1, 1)
    api_alive = _reliability(extra, "api_alive", ref_logits, 1.0).view(-1, 1)
    graph_alive = _reliability(extra, "graph_alive", ref_logits, 1.0).view(-1, 1)
    manifest_alive = _reliability(extra, "manifest_alive", ref_logits, 0.0).view(-1, 1)

    joint_score = (
        (r_api * r_graph).sqrt()
        * (0.5 + 0.25 * api_manifest + 0.25 * graph_manifest)
        * (0.5 + 0.5 * r_manifest)
    ).clamp(0.0, 1.0)
    scores = torch.cat(
        [
            r_api * api_alive,
            r_graph * graph_alive,
            r_manifest * manifest_alive,
            joint_score * api_alive.clamp_min(0.25) * graph_alive.clamp_min(0.25),
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
    """Robust objective: CE(final) + branch CE + optional soft consistency/gate prior."""
    extra = extra or {}
    loss_cfg = loss_cfg or {}
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    branch_aux_weight = float(loss_cfg.get("branch_aux_weight", 0.05))
    soft_consistency_weight = float(loss_cfg.get("soft_consistency_weight", 0.0))
    gate_prior_weight = float(loss_cfg.get("gate_prior_weight", 0.0))

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

    soft_loss = (
        _soft_consistency_loss(extra, logits)
        if soft_consistency_weight > 0.0
        else logits.new_tensor(0.0)
    )
    gate_prior = (
        _gate_prior_loss(extra, logits)
        if gate_prior_weight > 0.0
        else logits.new_tensor(0.0)
    )

    total = (
        ce
        + branch_aux_weight * branch_loss
        + soft_consistency_weight * soft_loss
        + gate_prior_weight * gate_prior
    )
    return total, {
        "loss": float(total.detach().item()),
        "ce": float(ce.detach().item()),
        "branch_aux": float(branch_loss.detach().item()),
        "branch_aux_weight": branch_aux_weight,
        "soft_consistency": float(soft_loss.detach().item()),
        "soft_consistency_weight": soft_consistency_weight,
        "gate_prior": float(gate_prior.detach().item()),
        "gate_prior_weight": gate_prior_weight,
    }
