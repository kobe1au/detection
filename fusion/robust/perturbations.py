from __future__ import annotations

import random
from typing import Iterable

import torch

from fusion.robust.semantic_categories import (
    SEMANTIC_CATEGORY_DIM,
    api_semantic_counts_from_type_ids,
    graph_semantic_counts_from_method_api_edges,
)


API_PERTURBATIONS = {
    "api_event_dropout",
    "api_sensitive_event_dropout",
    "api_category_dropout",
    "api_feature_noise",
    "modality_dropout_api",
    "api_degraded",
    "api_missing",
}

GRAPH_PERTURBATIONS = {
    "graph_sparsify",
    "graph_local_break",
    "graph_feature_obfuscation",
    "graph_node_feature_mask",
    "modality_dropout_graph",
    "graph_degraded",
    "graph_missing",
}

MANIFEST_PERTURBATIONS = {
    "manifest_permission_mask",
    "manifest_permission_injection",
    "manifest_intent_mask",
    "manifest_component_mask",
    "manifest_feature_noise",
    "modality_dropout_manifest",
    "manifest_degraded",
    "manifest_missing",
}

COMBINED_PERTURBATIONS = {
    "clean",
    "api_graph_degraded",
    "api_manifest_degraded",
    "graph_manifest_degraded",
    "all_degraded",
}

EVAL_PERTURB_TYPES = (
    {None}
    | API_PERTURBATIONS
    | GRAPH_PERTURBATIONS
    | MANIFEST_PERTURBATIONS
    | COMBINED_PERTURBATIONS
)


def _clamp_strength(strength: float) -> float:
    return max(0.0, min(1.0, float(strength)))


def _num_to_perturb(total: int, strength: float) -> int:
    total = max(0, int(total))
    strength = _clamp_strength(strength)
    if total <= 0 or strength <= 0.0:
        return 0
    return min(total, max(1, int(round(total * strength))))


def _scalar_float(value, default: float = 0.0) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float(default)
        return float(value.detach().float().view(-1)[0].item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _set_min_pert(data: dict, key: str, strength: float) -> None:
    current = _scalar_float(data.get(key), 0.0)
    data[key] = max(current, _clamp_strength(strength))


def _degrade_quality(data: dict, key: str, strength: float) -> None:
    q = _scalar_float(data.get(key), 1.0)
    data[key] = max(0.0, min(1.0, q * (1.0 - _clamp_strength(strength))))


def refresh_align_quality_after_code_perturb(data: dict) -> None:
    q_api = _scalar_float(data.get("q_api"), 0.0)
    q_graph = _scalar_float(data.get("q_graph"), 0.0)
    pert_api = _scalar_float(data.get("pert_api"), 0.0)
    pert_graph = _scalar_float(data.get("pert_graph"), 0.0)
    old = _scalar_float(data.get("q_align"), 0.0)
    code_alive = min(max(q_api, 0.0), max(q_graph, 0.0))
    code_pert = max(_clamp_strength(pert_api), _clamp_strength(pert_graph))
    data["q_align"] = max(0.0, min(old, code_alive)) * (1.0 - code_pert)


def _should_degrade_category_counts(data: dict) -> bool:
    return bool(data.get("degrade_category_counts", True))


def degrade_category_counts(data: dict, key: str, strength: float, mode: str) -> None:
    if not _should_degrade_category_counts(data):
        return
    counts = data.get(key)
    if not isinstance(counts, torch.Tensor) or counts.numel() == 0:
        return
    strength = _clamp_strength(strength)
    if strength <= 0.0:
        return
    out = counts.clone()
    flat = out.view(-1)
    if mode == "mask":
        candidates = torch.where(flat > 0)[0]
        if candidates.numel() == 0:
            return
        n = _num_to_perturb(candidates.numel(), strength)
        if n <= 0:
            return
        idx = candidates[torch.randperm(candidates.numel(), device=flat.device)[:n]]
        flat[idx] = 0.0
    elif mode == "inject":
        n = _num_to_perturb(flat.numel(), strength)
        if n <= 0:
            return
        idx = torch.randperm(flat.numel(), device=flat.device)[:n]
        flat[idx] = flat[idx] + 1.0
    elif mode == "noise":
        out = (out.float() + torch.randn_like(out.float()) * strength).clamp_min(0.0).to(dtype=counts.dtype)
    elif mode == "scale":
        out = (out.float() * (1.0 - strength)).clamp_min(0.0).to(dtype=counts.dtype)
    else:
        raise ValueError(f"Unsupported category count degradation mode: {mode}")
    data[key] = out


def degrade_manifest_counts(data: dict, strength: float, mode: str) -> None:
    degrade_category_counts(data, "manifest_category_counts", strength, mode)


def degrade_graph_counts(data: dict, strength: float) -> None:
    if not _should_degrade_category_counts(data):
        return
    strength = _clamp_strength(strength)
    if strength <= 0.0:
        return
    counts = data.get("graph_semantic_category_counts")
    if not isinstance(counts, torch.Tensor) or counts.numel() == 0:
        counts = data.get("graph_category_counts")
    if not isinstance(counts, torch.Tensor) or counts.numel() == 0:
        return

    out = counts.clone()
    if strength >= 1.0:
        out = torch.zeros_like(out)
        data["graph_semantic_category_counts"] = out
        data["graph_category_counts"] = out.clone()
        return
    flat = out.view(-1)
    candidates = torch.where(flat > 0)[0]
    if candidates.numel() == 0:
        return
    n = _num_to_perturb(candidates.numel(), strength)
    if n <= 0:
        return
    idx = candidates[torch.randperm(candidates.numel(), device=flat.device)[:n]]
    flat[idx] = 0.0
    if strength >= 0.5:
        noise = torch.randn_like(out.float()) * (0.05 * strength)
        out = (out.float() + noise).clamp_min(0.0).to(dtype=counts.dtype)
    data["graph_semantic_category_counts"] = out
    data["graph_category_counts"] = out.clone()


def _zero_tensor_like(value):
    if isinstance(value, torch.Tensor):
        return torch.zeros_like(value)
    return value


def recompute_api_category_counts(data: dict) -> None:
    ids = data.get("api_type_ids")
    counts = api_semantic_counts_from_type_ids(ids)
    device = ids.device if isinstance(ids, torch.Tensor) else None
    counts = counts.to(device=device) if device is not None else counts
    data["api_semantic_category_counts"] = counts
    data["api_category_counts"] = counts


def recompute_graph_category_counts_from_api_edges(data: dict) -> None:
    ids = data.get("api_type_ids")
    edge = data.get("method_api_edge_index")
    counts = graph_semantic_counts_from_method_api_edges(ids, edge)
    device = None
    if isinstance(ids, torch.Tensor):
        device = ids.device
    elif isinstance(edge, torch.Tensor):
        device = edge.device
    counts = counts.to(device=device) if device is not None else counts
    data["graph_semantic_category_counts"] = counts
    data["graph_category_counts"] = counts


def _select_api_events(data: dict, keep: torch.Tensor) -> None:
    keep = keep.bool().view(-1)
    n_api = int(keep.numel())
    if n_api <= 0:
        return
    keep_idx = torch.where(keep)[0]

    for key in (
        "api_ids",
        "api_type_ids",
        "api_sensitive_mask",
        "api_method_index",
        "api_in_graph_mask",
    ):
        value = data.get(key)
        if isinstance(value, torch.Tensor) and value.numel() == n_api:
            data[key] = value[keep_idx.to(value.device)].clone()

    mask = data.get("mask")
    if isinstance(mask, torch.Tensor) and mask.ndim == 2 and mask.size(1) == n_api:
        data["mask"] = mask[:, keep_idx.to(mask.device)].clone()

    edge = data.get("method_api_edge_index")
    if not (
        isinstance(edge, torch.Tensor)
        and edge.ndim == 2
        and edge.size(0) == 2
        and edge.numel() > 0
    ):
        return

    mapping = torch.full((n_api,), -1, dtype=torch.long, device=edge.device)
    keep_edge = keep_idx.to(edge.device)
    mapping[keep_edge] = torch.arange(keep_edge.numel(), dtype=torch.long, device=edge.device)
    dst = edge[1].long()
    valid = (dst >= 0) & (dst < n_api)
    if valid.any():
        valid = valid & (mapping[dst.clamp(0, max(n_api - 1, 0))] >= 0)
    edge = edge[:, valid].clone()
    if edge.numel() > 0:
        edge[1] = mapping[edge[1].long()]
    data["method_api_edge_index"] = edge


def apply_api_event_dropout(data: dict, strength: float, sensitive_only: bool = False) -> dict:
    strength = _clamp_strength(strength)
    api_ids = data.get("api_ids")
    if not isinstance(api_ids, torch.Tensor) or api_ids.numel() == 0 or strength <= 0.0:
        return data

    n_api = int(api_ids.numel())
    device = api_ids.device
    if sensitive_only and isinstance(data.get("api_sensitive_mask"), torch.Tensor):
        candidate = torch.where(data["api_sensitive_mask"].to(device).float() > 0.5)[0]
        if candidate.numel() == 0:
            candidate = torch.arange(n_api, device=device)
    else:
        candidate = torch.arange(n_api, device=device)

    n_drop = _num_to_perturb(candidate.numel(), strength)
    if n_drop <= 0:
        return data
    drop_idx = candidate[torch.randperm(candidate.numel(), device=device)[:n_drop]]
    keep = torch.ones((n_api,), dtype=torch.bool, device=device)
    keep[drop_idx] = False

    actual = float(drop_idx.numel()) / max(n_api, 1)
    _select_api_events(data, keep)
    _degrade_quality(data, "q_api", actual)
    _set_min_pert(data, "pert_api", actual)
    data["api_aug_type"] = "api_sensitive_event_dropout" if sensitive_only else "api_event_dropout"
    recompute_api_category_counts(data)
    recompute_graph_category_counts_from_api_edges(data)
    refresh_align_quality_after_code_perturb(data)
    return data


def apply_api_category_dropout(data: dict, strength: float) -> dict:
    strength = _clamp_strength(strength)
    if strength <= 0.0:
        return data
    ids = data.get("api_type_ids")
    if not isinstance(ids, torch.Tensor) or ids.numel() == 0:
        return data
    non_other = torch.where(ids.long().view(-1) > 0)[0]
    if non_other.numel() == 0:
        return apply_api_event_dropout(data, strength, sensitive_only=False)
    n_drop = _num_to_perturb(non_other.numel(), strength)
    if n_drop <= 0:
        return data
    chosen = non_other[torch.randperm(non_other.numel(), device=ids.device)[:n_drop]]
    ids_new = ids.clone()
    ids_new[chosen] = 0
    data["api_type_ids"] = ids_new
    actual = float(n_drop) / max(int(ids.numel()), 1)
    _degrade_quality(data, "q_api", actual)
    _set_min_pert(data, "pert_api", actual)
    data["api_aug_type"] = "api_category_dropout"
    recompute_api_category_counts(data)
    recompute_graph_category_counts_from_api_edges(data)
    refresh_align_quality_after_code_perturb(data)
    return data


def apply_api_feature_noise(data: dict, strength: float) -> dict:
    strength = _clamp_strength(strength)
    if strength <= 0.0:
        return data
    ids = data.get("api_ids")
    if not isinstance(ids, torch.Tensor) or ids.numel() == 0:
        return data
    n = int(ids.numel())
    n_noise = _num_to_perturb(n, strength)
    if n_noise <= 0:
        return data
    idx = torch.randperm(n, device=ids.device)[:n_noise]
    ids_new = ids.clone()
    ids_new[idx] = torch.randint(1, int(ids.max().item()) + 2, (n_noise,), device=ids.device)
    data["api_ids"] = ids_new
    _degrade_quality(data, "q_api", 0.5 * strength)
    _set_min_pert(data, "pert_api", strength)
    data["api_aug_type"] = "api_feature_noise"
    refresh_align_quality_after_code_perturb(data)
    return data


def apply_api_missing(data: dict) -> dict:
    for key in (
        "api_ids",
        "api_type_ids",
        "api_sensitive_mask",
        "api_method_index",
        "api_in_graph_mask",
    ):
        value = data.get(key)
        if isinstance(value, torch.Tensor):
            data[key] = value[:0].clone()
    edge = data.get("method_api_edge_index")
    if isinstance(edge, torch.Tensor):
        data["method_api_edge_index"] = edge.new_empty((2, 0), dtype=torch.long)
    mask = data.get("mask")
    if isinstance(mask, torch.Tensor) and mask.ndim == 2:
        data["mask"] = mask.new_empty((mask.size(0), 0), dtype=torch.float32)
    template = data.get("api_semantic_category_counts", data.get("api_category_counts"))
    if isinstance(template, torch.Tensor) and template.numel() == SEMANTIC_CATEGORY_DIM:
        counts = torch.zeros_like(template)
    else:
        device = template.device if isinstance(template, torch.Tensor) else None
        counts = torch.zeros((SEMANTIC_CATEGORY_DIM,), dtype=torch.float32, device=device)
    data["api_semantic_category_counts"] = counts
    data["api_category_counts"] = counts
    data["graph_semantic_category_counts"] = counts.clone()
    data["graph_category_counts"] = counts.clone()
    data["q_api"] = 0.0
    data["pert_api"] = 1.0
    data["api_aug_type"] = "modality_dropout_api"
    refresh_align_quality_after_code_perturb(data)
    return data


def apply_graph_sparsify(data: dict, strength: float) -> dict:
    strength = _clamp_strength(strength)
    if strength <= 0.0:
        return data
    edge = data.get("edge_index")
    if not isinstance(edge, torch.Tensor) or edge.ndim != 2 or edge.size(1) == 0:
        _degrade_quality(data, "q_graph", 1.0)
        _set_min_pert(data, "pert_graph", 1.0)
        degrade_graph_counts(data, 1.0)
        data["graph_aug_type"] = "graph_sparsify"
        refresh_align_quality_after_code_perturb(data)
        return data
    keep = torch.rand(edge.size(1), device=edge.device) > strength
    if keep.sum() == 0:
        keep[random.randrange(edge.size(1))] = True
    data["edge_index"] = edge[:, keep]
    actual = 1.0 - float(keep.float().mean().item())
    _degrade_quality(data, "q_graph", actual)
    _set_min_pert(data, "pert_graph", actual)
    data["graph_aug_type"] = "graph_sparsify"
    degrade_graph_counts(data, actual)
    refresh_align_quality_after_code_perturb(data)
    return data


def apply_graph_local_break(data: dict, strength: float) -> dict:
    strength = _clamp_strength(strength)
    if strength <= 0.0:
        return data
    edge = data.get("edge_index")
    if not isinstance(edge, torch.Tensor) or edge.ndim != 2 or edge.size(1) == 0:
        return apply_graph_sparsify(data, 1.0)
    num_nodes = int(data.get("x").size(0)) if isinstance(data.get("x"), torch.Tensor) else int(edge.max().item()) + 1
    sensitive = data.get("sensitive_mask")
    if isinstance(sensitive, torch.Tensor) and sensitive.numel() > 0 and sensitive.bool().any():
        candidates = torch.where(sensitive.to(edge.device).bool())[0]
        center = int(candidates[random.randrange(candidates.numel())].item())
    else:
        center = random.randrange(max(num_nodes, 1))
    local = (edge[0] == center) | (edge[1] == center)
    drop = local & (torch.rand(edge.size(1), device=edge.device) < strength)
    keep = ~drop
    if keep.sum() == 0:
        keep[random.randrange(edge.size(1))] = True
    data["edge_index"] = edge[:, keep]
    actual = 1.0 - float(keep.float().mean().item())
    _degrade_quality(data, "q_graph", actual)
    _set_min_pert(data, "pert_graph", actual)
    data["graph_aug_type"] = "graph_local_break"
    degrade_graph_counts(data, actual)
    refresh_align_quality_after_code_perturb(data)
    return data


def apply_graph_feature_obfuscation(data: dict, strength: float) -> dict:
    strength = _clamp_strength(strength)
    if strength <= 0.0:
        return data
    x = data.get("x")
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.size(0) == 0:
        return data
    n = int(x.size(0))
    n_mask = _num_to_perturb(n, strength)
    if n_mask <= 0:
        return data
    idx = torch.randperm(n, device=x.device)[:n_mask]
    x_new = x.clone()
    x_new[idx] = 0.15 * x_new[idx] + 0.03 * torch.randn_like(x_new[idx])
    data["x"] = x_new
    actual = float(n_mask) / max(n, 1)
    _degrade_quality(data, "q_graph", actual)
    _set_min_pert(data, "pert_graph", actual)
    data["graph_aug_type"] = "graph_feature_obfuscation"
    degrade_graph_counts(data, actual)
    refresh_align_quality_after_code_perturb(data)
    return data


def apply_graph_node_feature_mask(data: dict, strength: float) -> dict:
    strength = _clamp_strength(strength)
    if strength <= 0.0:
        return data
    x = data.get("x")
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.size(0) == 0:
        return data
    n = int(x.size(0))
    n_mask = _num_to_perturb(n, strength)
    if n_mask <= 0:
        return data
    idx = torch.randperm(n, device=x.device)[:n_mask]
    x_new = x.clone()
    x_new[idx] = 0.0
    data["x"] = x_new
    actual = float(n_mask) / max(n, 1)
    _degrade_quality(data, "q_graph", actual)
    _set_min_pert(data, "pert_graph", actual)
    data["graph_aug_type"] = "graph_node_feature_mask"
    degrade_graph_counts(data, actual)
    refresh_align_quality_after_code_perturb(data)
    return data


def apply_graph_missing(data: dict) -> dict:
    x = data.get("x")
    if isinstance(x, torch.Tensor):
        data["x"] = torch.zeros_like(x)
    edge = data.get("edge_index")
    if isinstance(edge, torch.Tensor):
        data["edge_index"] = edge.new_empty((2, 0), dtype=torch.long)
    sensitive = data.get("sensitive_mask")
    if isinstance(sensitive, torch.Tensor):
        data["sensitive_mask"] = torch.zeros_like(sensitive)
    graph_counts = data.get("graph_category_counts")
    if isinstance(graph_counts, torch.Tensor):
        data["graph_category_counts"] = torch.zeros_like(graph_counts)
    graph_semantic_counts = data.get("graph_semantic_category_counts")
    if isinstance(graph_semantic_counts, torch.Tensor):
        data["graph_semantic_category_counts"] = torch.zeros_like(graph_semantic_counts)
    data["q_graph"] = 0.0
    data["pert_graph"] = 1.0
    data["graph_aug_type"] = "modality_dropout_graph"
    refresh_align_quality_after_code_perturb(data)
    return data


def _mask_vector_positions(vec: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    out = vec.clone()
    if out.ndim == 1:
        out[positions] = 0.0
    elif out.ndim == 2:
        out[:, positions] = 0.0
    return out


def _inject_vector_positions(vec: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    out = vec.clone()
    if out.ndim == 1:
        out[positions] = 1.0
    elif out.ndim == 2:
        out[:, positions] = 1.0
    return out


def _choose_vector_positions(vec: torch.Tensor, strength: float, start: int = 0, end: int | None = None) -> torch.Tensor:
    width = int(vec.size(-1))
    start = max(0, min(int(start), width))
    end = width if end is None else max(start, min(int(end), width))
    span = end - start
    strength = _clamp_strength(strength)
    if span <= 0 or strength <= 0.0:
        return torch.empty((0,), dtype=torch.long, device=vec.device)
    n = _num_to_perturb(span, strength)
    if n <= 0:
        return torch.empty((0,), dtype=torch.long, device=vec.device)
    return start + torch.randperm(span, device=vec.device)[:n]


def apply_manifest_permission_mask(data: dict, strength: float) -> dict:
    strength = _clamp_strength(strength)
    if strength <= 0.0:
        return data
    vec = data.get("manifest_x")
    if isinstance(vec, torch.Tensor) and vec.numel() > 0:
        perm_dim = int(data.get("manifest_permission_dim", max(1, vec.size(-1) // 2)))
        pos = _choose_vector_positions(vec, strength, 0, perm_dim)
        if pos.numel() > 0:
            data["manifest_x"] = _mask_vector_positions(vec, pos)
    ids = data.get("manifest_permission_ids")
    if isinstance(ids, torch.Tensor) and ids.numel() > 0:
        n = _num_to_perturb(ids.numel(), strength)
        if n > 0:
            idx = torch.randperm(ids.numel(), device=ids.device)[:n]
            ids_new = ids.clone()
            ids_new[idx] = 0
            data["manifest_permission_ids"] = ids_new
    degrade_manifest_counts(data, strength, "mask")
    _degrade_quality(data, "q_manifest", strength)
    _set_min_pert(data, "pert_manifest", strength)
    data["manifest_aug_type"] = "manifest_permission_mask"
    return data


def apply_manifest_permission_injection(data: dict, strength: float) -> dict:
    strength = _clamp_strength(strength)
    if strength <= 0.0:
        return data
    vec = data.get("manifest_x")
    if isinstance(vec, torch.Tensor) and vec.numel() > 0:
        perm_dim = int(data.get("manifest_permission_dim", max(1, vec.size(-1) // 2)))
        pos = _choose_vector_positions(vec, strength, 0, perm_dim)
        if pos.numel() > 0:
            data["manifest_x"] = _inject_vector_positions(vec, pos)
    degrade_manifest_counts(data, strength, "inject")
    _degrade_quality(data, "q_manifest", 0.5 * _clamp_strength(strength))
    _set_min_pert(data, "pert_manifest", strength)
    data["manifest_aug_type"] = "manifest_permission_injection"
    return data


def apply_manifest_intent_mask(data: dict, strength: float) -> dict:
    strength = _clamp_strength(strength)
    if strength <= 0.0:
        return data
    vec = data.get("manifest_x")
    if isinstance(vec, torch.Tensor) and vec.numel() > 0:
        perm_dim = int(data.get("manifest_permission_dim", max(1, vec.size(-1) // 2)))
        intent_dim = int(data.get("manifest_intent_dim", max(1, vec.size(-1) // 4)))
        pos = _choose_vector_positions(vec, strength, perm_dim, perm_dim + intent_dim)
        if pos.numel() > 0:
            data["manifest_x"] = _mask_vector_positions(vec, pos)
    ids = data.get("manifest_intent_ids")
    if isinstance(ids, torch.Tensor) and ids.numel() > 0:
        n = _num_to_perturb(ids.numel(), strength)
        if n > 0:
            idx = torch.randperm(ids.numel(), device=ids.device)[:n]
            ids_new = ids.clone()
            ids_new[idx] = 0
            data["manifest_intent_ids"] = ids_new
    degrade_manifest_counts(data, strength, "mask")
    _degrade_quality(data, "q_manifest", strength)
    _set_min_pert(data, "pert_manifest", strength)
    data["manifest_aug_type"] = "manifest_intent_mask"
    return data


def apply_manifest_component_mask(data: dict, strength: float) -> dict:
    strength = _clamp_strength(strength)
    if strength <= 0.0:
        return data
    stats = data.get("manifest_stats")
    if isinstance(stats, torch.Tensor) and stats.numel() > 0:
        n = _num_to_perturb(stats.numel(), strength)
        if n > 0:
            stats_new = stats.clone()
            idx = torch.randperm(stats.numel(), device=stats.device)[:n]
            stats_new.view(-1)[idx] = 0.0
            data["manifest_stats"] = stats_new
    vec = data.get("manifest_x")
    if isinstance(vec, torch.Tensor) and vec.numel() > 0:
        perm_dim = int(data.get("manifest_permission_dim", 0))
        intent_dim = int(data.get("manifest_intent_dim", 0))
        feature_dim = int(data.get("manifest_feature_dim", 0))
        stats_dim = int(stats.numel()) if isinstance(stats, torch.Tensor) else max(1, vec.size(-1) // 8)
        start = perm_dim + intent_dim + feature_dim + SEMANTIC_CATEGORY_DIM
        if start >= vec.size(-1):
            start = max(0, vec.size(-1) - stats_dim)
        end = min(vec.size(-1), start + stats_dim)
        pos = _choose_vector_positions(vec, strength, start, end)
        if pos.numel() > 0:
            data["manifest_x"] = _mask_vector_positions(vec, pos)
    degrade_manifest_counts(data, strength, "mask")
    _degrade_quality(data, "q_manifest", strength)
    _set_min_pert(data, "pert_manifest", strength)
    data["manifest_aug_type"] = "manifest_component_mask"
    return data


def apply_manifest_feature_noise(data: dict, strength: float) -> dict:
    strength = _clamp_strength(strength)
    if strength <= 0.0:
        return data
    vec = data.get("manifest_x")
    if isinstance(vec, torch.Tensor) and vec.numel() > 0:
        noise = torch.randn_like(vec.float()) * (0.05 + 0.20 * _clamp_strength(strength))
        data["manifest_x"] = (vec.float() + noise).clamp(0.0, 1.0)
    degrade_manifest_counts(data, strength, "noise")
    _degrade_quality(data, "q_manifest", 0.5 * _clamp_strength(strength))
    _set_min_pert(data, "pert_manifest", strength)
    data["manifest_aug_type"] = "manifest_feature_noise"
    return data


def apply_manifest_missing(data: dict) -> dict:
    for key in (
        "manifest_x",
        "manifest_permission_ids",
        "manifest_intent_ids",
        "manifest_category_counts",
        "manifest_stats",
    ):
        value = data.get(key)
        if isinstance(value, torch.Tensor):
            data[key] = torch.zeros_like(value)
    data["q_manifest"] = 0.0
    data["pert_manifest"] = 1.0
    data["manifest_aug_type"] = "modality_dropout_manifest"
    return data


def apply_manifest_degraded(data: dict, strength: float) -> dict:
    if _clamp_strength(strength) <= 0.0:
        return data
    op = random.choice(
        [
            apply_manifest_permission_mask,
            apply_manifest_permission_injection,
            apply_manifest_intent_mask,
            apply_manifest_component_mask,
            apply_manifest_feature_noise,
        ]
    )
    return op(data, strength)


def _apply_one(data: dict, perturb_type: str, strength: float) -> dict:
    if perturb_type in {None, "clean"}:
        return data
    if perturb_type in {"api_degraded"}:
        return random.choice(
            [apply_api_event_dropout, apply_api_category_dropout, apply_api_feature_noise]
        )(data, strength)
    if perturb_type in {"api_missing", "modality_dropout_api"}:
        return apply_api_missing(data)
    if perturb_type == "api_event_dropout":
        return apply_api_event_dropout(data, strength, sensitive_only=False)
    if perturb_type == "api_sensitive_event_dropout":
        return apply_api_event_dropout(data, strength, sensitive_only=True)
    if perturb_type == "api_category_dropout":
        return apply_api_category_dropout(data, strength)
    if perturb_type == "api_feature_noise":
        return apply_api_feature_noise(data, strength)

    if perturb_type in {"graph_degraded"}:
        return random.choice(
            [apply_graph_sparsify, apply_graph_local_break, apply_graph_feature_obfuscation, apply_graph_node_feature_mask]
        )(data, strength)
    if perturb_type in {"graph_missing", "modality_dropout_graph"}:
        return apply_graph_missing(data)
    if perturb_type == "graph_sparsify":
        return apply_graph_sparsify(data, strength)
    if perturb_type == "graph_local_break":
        return apply_graph_local_break(data, strength)
    if perturb_type == "graph_feature_obfuscation":
        return apply_graph_feature_obfuscation(data, strength)
    if perturb_type == "graph_node_feature_mask":
        return apply_graph_node_feature_mask(data, strength)

    if perturb_type in {"manifest_degraded"}:
        return apply_manifest_degraded(data, strength)
    if perturb_type in {"manifest_missing", "modality_dropout_manifest"}:
        return apply_manifest_missing(data)
    if perturb_type == "manifest_permission_mask":
        return apply_manifest_permission_mask(data, strength)
    if perturb_type == "manifest_permission_injection":
        return apply_manifest_permission_injection(data, strength)
    if perturb_type == "manifest_intent_mask":
        return apply_manifest_intent_mask(data, strength)
    if perturb_type == "manifest_component_mask":
        return apply_manifest_component_mask(data, strength)
    if perturb_type == "manifest_feature_noise":
        return apply_manifest_feature_noise(data, strength)

    raise ValueError(f"Unsupported perturb_type: {perturb_type}")


def apply_perturbation(data: dict, perturb_type: str | None, strength: float) -> dict:
    strength = _clamp_strength(strength)
    if perturb_type in {None, "clean"}:
        return data
    if perturb_type == "api_graph_degraded":
        data = _apply_one(data, "api_degraded", strength)
        return _apply_one(data, "graph_degraded", strength)
    if perturb_type == "api_manifest_degraded":
        data = _apply_one(data, "api_degraded", strength)
        return _apply_one(data, "manifest_degraded", strength)
    if perturb_type == "graph_manifest_degraded":
        data = _apply_one(data, "graph_degraded", strength)
        return _apply_one(data, "manifest_degraded", strength)
    if perturb_type == "all_degraded":
        data = _apply_one(data, "api_degraded", strength)
        data = _apply_one(data, "graph_degraded", strength)
        return _apply_one(data, "manifest_degraded", strength)
    return _apply_one(data, perturb_type, strength)


def sample_training_perturbation(
    perturb_prob: float,
    perturb_strengths: Iterable[float],
) -> tuple[str | None, float]:
    if random.random() >= max(0.0, min(1.0, float(perturb_prob))):
        return None, 0.0
    strengths = list(perturb_strengths) or [0.3]
    perturb_type = random.choice(
        [
            "api_degraded",
            "graph_degraded",
            "manifest_degraded",
            "api_graph_degraded",
            "api_manifest_degraded",
            "graph_manifest_degraded",
            "all_degraded",
        ]
    )
    return perturb_type, float(random.choice(strengths))
