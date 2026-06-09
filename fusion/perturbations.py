from __future__ import annotations

import torch
from torch_geometric.data import Data

from fusion.constants import EDGE_TYPES, NODE_TYPES, SOURCE_TYPES, VIEW_TYPES


API_EDGE_TYPES = {
    EDGE_TYPES["METHOD_INVOKES_API_FAMILY"],
    EDGE_TYPES["API_FAMILY_INVOKED_BY_METHOD"],
    EDGE_TYPES["PERMISSION_RELATED_TO_API_FAMILY"],
    EDGE_TYPES["API_FAMILY_RELATED_TO_PERMISSION"],
    EDGE_TYPES["METHOD_HAS_RISK"],
    EDGE_TYPES["RISK_OBSERVED_IN_METHOD"],
    EDGE_TYPES["METHOD_HAS_STRING_HINT"],
    EDGE_TYPES["STRING_HINT_IN_METHOD"],
}

GRAPH_EDGE_TYPES = {
    EDGE_TYPES["APK_HAS_METHOD"],
    EDGE_TYPES["METHOD_IN_APK"],
    EDGE_TYPES["METHOD_CALLS_METHOD"],
    EDGE_TYPES["METHOD_CALLED_BY_METHOD"],
    EDGE_TYPES["COMPONENT_MATCHES_METHOD"],
    EDGE_TYPES["METHOD_MATCHES_COMPONENT"],
    EDGE_TYPES["METHOD_HAS_RISK"],
    EDGE_TYPES["RISK_OBSERVED_IN_METHOD"],
    EDGE_TYPES["METHOD_HAS_STRING_HINT"],
    EDGE_TYPES["STRING_HINT_IN_METHOD"],
}

MANIFEST_EDGE_TYPES = {
    EDGE_TYPES["APK_REQUESTS_PERMISSION"],
    EDGE_TYPES["PERMISSION_REQUESTED_BY_APK"],
    EDGE_TYPES["APK_HAS_COMPONENT"],
    EDGE_TYPES["COMPONENT_IN_APK"],
    EDGE_TYPES["COMPONENT_DECLARES_INTENT"],
    EDGE_TYPES["INTENT_DECLARED_BY_COMPONENT"],
    EDGE_TYPES["PERMISSION_RELATED_TO_API_FAMILY"],
    EDGE_TYPES["API_FAMILY_RELATED_TO_PERMISSION"],
    EDGE_TYPES["COMPONENT_MATCHES_METHOD"],
    EDGE_TYPES["METHOD_MATCHES_COMPONENT"],
    EDGE_TYPES["MANIFEST_HAS_RISK"],
    EDGE_TYPES["RISK_DECLARED_BY_MANIFEST"],
}


def _clamp_strength(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _edge_mask(data: Data, *, edge_types: set[int] | None = None, sources: set[int] | None = None) -> torch.Tensor:
    mask = torch.zeros((int(data.edge_index.size(1)),), dtype=torch.bool)
    if mask.numel() == 0:
        return mask
    if edge_types is not None and hasattr(data, "edge_type"):
        mask |= torch.isin(data.edge_type.cpu(), torch.tensor(sorted(edge_types), dtype=torch.long))
    if sources is not None and hasattr(data, "edge_source"):
        mask |= torch.isin(data.edge_source.cpu(), torch.tensor(sorted(sources), dtype=torch.long))
    return mask.to(data.edge_index.device)


def _node_mask(data: Data, *, node_types: set[int] | None = None, sources: set[int] | None = None) -> torch.Tensor:
    mask = torch.zeros((int(data.x.size(0)),), dtype=torch.bool, device=data.x.device)
    if node_types is not None and hasattr(data, "node_type"):
        mask |= torch.isin(data.node_type, torch.tensor(sorted(node_types), dtype=torch.long, device=data.x.device))
    if sources is not None and hasattr(data, "node_source"):
        mask |= torch.isin(data.node_source, torch.tensor(sorted(sources), dtype=torch.long, device=data.x.device))
    return mask


def _soft_degrade_nodes(data: Data, mask: torch.Tensor, strength: float, *, zero: bool = False, noise: bool = False) -> None:
    if mask.numel() == 0 or not bool(mask.any()):
        return
    strength = _clamp_strength(strength)
    if zero:
        data.x[mask] = 0.0
        data.node_quality[mask] = 0.0
        if hasattr(data, "node_semantic"):
            data.node_semantic[mask] = 0.0
        return
    data.node_quality[mask] = data.node_quality[mask] * (1.0 - strength)
    if hasattr(data, "node_semantic"):
        data.node_semantic[mask] = data.node_semantic[mask] * (1.0 - strength)
    data.x[mask] = data.x[mask] * (1.0 - 0.5 * strength)
    if noise:
        data.x[mask] = data.x[mask] + torch.randn_like(data.x[mask]) * strength * 0.1


def _soft_degrade_edges(data: Data, mask: torch.Tensor, strength: float, *, zero: bool = False) -> None:
    if mask.numel() == 0 or not bool(mask.any()) or not hasattr(data, "edge_quality"):
        return
    strength = _clamp_strength(strength)
    if zero:
        data.edge_quality[mask] = 0.0
    else:
        data.edge_quality[mask] = data.edge_quality[mask] * (1.0 - strength)


def _set_scalar(data: Data, name: str, value: float) -> None:
    setattr(data, name, torch.tensor([float(value)], dtype=torch.float32, device=data.x.device))


def _set_view_tracking(
    data: Data,
    *,
    requested_view: str,
    effective_view: str | None = None,
    fallback: bool = False,
) -> None:
    requested_name = str(requested_view or "clean")
    effective_name = str(effective_view or requested_name)
    requested_id = VIEW_TYPES.get(requested_name, VIEW_TYPES["clean"])
    effective_id = VIEW_TYPES.get(effective_name, VIEW_TYPES["clean"])
    device = data.x.device
    data.view_type_id = torch.tensor([requested_id], dtype=torch.long, device=device)
    data.requested_view_type_id = torch.tensor([requested_id], dtype=torch.long, device=device)
    data.effective_view_type_id = torch.tensor([effective_id], dtype=torch.long, device=device)
    data.manifest_shuffle_fallback = torch.tensor([1 if fallback else 0], dtype=torch.long, device=device)


def _refresh_align_after_code_perturb(data: Data) -> None:
    q_api = float(getattr(data, "q_api", torch.tensor([0.0])).view(-1)[0].item())
    q_graph = float(getattr(data, "q_graph", torch.tensor([0.0])).view(-1)[0].item())
    q_align = float(getattr(data, "q_align", torch.tensor([0.0])).view(-1)[0].item())
    pert_api = float(getattr(data, "pert_api", torch.tensor([0.0])).view(-1)[0].item())
    pert_graph = float(getattr(data, "pert_graph", torch.tensor([0.0])).view(-1)[0].item())
    r_api = q_api * (1.0 - _clamp_strength(pert_api))
    r_graph = q_graph * (1.0 - _clamp_strength(pert_graph))
    _set_scalar(data, "q_align", min(q_align, r_api, r_graph))


def clear_aggregate_apk_semantic(data: Data) -> None:
    if not hasattr(data, "node_semantic"):
        return
    apk_mask = _node_mask(data, node_types={NODE_TYPES["APK"]})
    if bool(apk_mask.any()):
        data.node_semantic[apk_mask] = 0.0


def refresh_apk_node_quality(data: Data) -> None:
    if not hasattr(data, "node_quality") or not hasattr(data, "x"):
        return
    apk_mask = _node_mask(data, node_types={NODE_TYPES["APK"]})
    if not bool(apk_mask.any()):
        return
    device = data.x.device
    q_api = float(getattr(data, "q_api", torch.tensor([0.0], device=device)).view(-1)[0].item())
    q_graph = float(getattr(data, "q_graph", torch.tensor([0.0], device=device)).view(-1)[0].item())
    q_manifest = float(getattr(data, "q_manifest", torch.tensor([0.0], device=device)).view(-1)[0].item())
    pert_api = float(getattr(data, "pert_api", torch.tensor([0.0], device=device)).view(-1)[0].item())
    pert_graph = float(getattr(data, "pert_graph", torch.tensor([0.0], device=device)).view(-1)[0].item())
    pert_manifest = float(getattr(data, "pert_manifest", torch.tensor([0.0], device=device)).view(-1)[0].item())
    data.node_quality[apk_mask] = float(
        max(
            q_api * (1.0 - _clamp_strength(pert_api)),
            q_graph * (1.0 - _clamp_strength(pert_graph)),
            q_manifest * (1.0 - _clamp_strength(pert_manifest)),
        )
    )


def _degrade_api_derived_method_semantic(data: Data, strength: float, *, missing: bool) -> None:
    if not hasattr(data, "node_semantic"):
        return
    method_mask = _node_mask(data, node_types={NODE_TYPES["METHOD"]})
    if not bool(method_mask.any()):
        return
    if missing:
        data.node_semantic[method_mask] = 0.0
    else:
        data.node_semantic[method_mask] = data.node_semantic[method_mask] * (1.0 - _clamp_strength(strength))


def _degrade_api_derived_method_features(data: Data, strength: float, *, missing: bool) -> None:
    if not hasattr(data, "x"):
        return
    hint_dim_tensor = getattr(data, "graph_behavior_hint_dim", None)
    hint_start_tensor = getattr(data, "graph_behavior_hint_start", None)
    if not isinstance(hint_dim_tensor, torch.Tensor) or not isinstance(hint_start_tensor, torch.Tensor):
        return
    hint_dim = int(hint_dim_tensor.view(-1)[0].item()) if hint_dim_tensor.numel() else 0
    hint_start = int(hint_start_tensor.view(-1)[0].item()) if hint_start_tensor.numel() else 0
    if hint_dim <= 0 or hint_start < 0 or hint_start >= data.x.size(1):
        return
    end = min(data.x.size(1), hint_start + hint_dim)
    method_mask = _node_mask(data, node_types={NODE_TYPES["METHOD"]})
    if not bool(method_mask.any()) or end <= hint_start:
        return
    if missing:
        data.x[method_mask, hint_start:end] = 0.0
    else:
        data.x[method_mask, hint_start:end] = data.x[method_mask, hint_start:end] * (1.0 - _clamp_strength(strength))


def refresh_risk_node_quality(data: Data) -> None:
    if not hasattr(data, "node_quality") or not hasattr(data, "node_semantic") or not hasattr(data, "edge_type"):
        return
    risk_nodes = torch.where(_node_mask(data, node_types={NODE_TYPES["RISK_SEMANTIC"]}))[0]
    if risk_nodes.numel() == 0:
        return
    if not hasattr(data, "edge_quality") or data.edge_index.numel() == 0:
        data.node_quality[risk_nodes] = 0.0
        data.node_semantic[risk_nodes] = 0.0
        return

    device = data.node_quality.device
    edge_type = data.edge_type.to(device)
    edge_quality = data.edge_quality.to(device).float().view(-1).clamp_min(0.0)
    src, dst = data.edge_index.to(device).long()
    code_edge_types = torch.tensor(
        [EDGE_TYPES["METHOD_HAS_RISK"], EDGE_TYPES["RISK_OBSERVED_IN_METHOD"]],
        dtype=torch.long,
        device=device,
    )
    manifest_edge_types = torch.tensor(
        [EDGE_TYPES["MANIFEST_HAS_RISK"], EDGE_TYPES["RISK_DECLARED_BY_MANIFEST"]],
        dtype=torch.long,
        device=device,
    )
    q_api = float(getattr(data, "q_api", torch.tensor([0.0], device=device)).view(-1)[0].item())
    q_graph = float(getattr(data, "q_graph", torch.tensor([0.0], device=device)).view(-1)[0].item())
    q_manifest = float(getattr(data, "q_manifest", torch.tensor([0.0], device=device)).view(-1)[0].item())
    code_reliability = max(q_api, 0.0) ** 0.5 * max(q_graph, 0.0) ** 0.5

    code_type_mask = torch.isin(edge_type, code_edge_types)
    manifest_type_mask = torch.isin(edge_type, manifest_edge_types)
    for risk_idx in risk_nodes.tolist():
        incident = (src == risk_idx) | (dst == risk_idx)
        code_strength = edge_quality[incident & code_type_mask].max().item() if bool((incident & code_type_mask).any()) else 0.0
        manifest_strength = edge_quality[incident & manifest_type_mask].max().item() if bool((incident & manifest_type_mask).any()) else 0.0
        quality = max(code_reliability * code_strength, q_manifest * manifest_strength)
        data.node_quality[risk_idx] = float(max(0.0, min(1.0, quality)))
        if quality <= 0.0:
            data.node_semantic[risk_idx] = 0.0


def _degrade_api(data: Data, strength: float, *, missing: bool = False) -> None:
    # STRING_HINT evidence is derived from both method names and API tokens.
    # Until provenance is separable, degrade it with either code modality to
    # prevent a missing API view from retaining API-derived behavior hints.
    api_nodes = _node_mask(data, node_types={NODE_TYPES["API_FAMILY"], NODE_TYPES["STRING_HINT"]})
    api_edges = _edge_mask(data, edge_types=API_EDGE_TYPES)
    _soft_degrade_nodes(data, api_nodes, strength, zero=missing)
    _soft_degrade_edges(data, api_edges, strength, zero=missing)
    _degrade_api_derived_method_semantic(data, strength, missing=missing)
    _degrade_api_derived_method_features(data, strength, missing=missing)
    clear_aggregate_apk_semantic(data)
    _set_scalar(data, "pert_api", 1.0 if missing else max(float(data.pert_api.view(-1)[0].item()), strength))
    _refresh_align_after_code_perturb(data)
    refresh_apk_node_quality(data)
    refresh_risk_node_quality(data)


def _degrade_graph(data: Data, strength: float, *, missing: bool = False) -> None:
    graph_nodes = _node_mask(data, node_types={NODE_TYPES["METHOD"], NODE_TYPES["STRING_HINT"]})
    graph_edges = _edge_mask(data, edge_types=GRAPH_EDGE_TYPES)
    _soft_degrade_nodes(data, graph_nodes, strength, zero=missing, noise=not missing)
    _soft_degrade_edges(data, graph_edges, strength, zero=missing)
    _set_scalar(data, "pert_graph", 1.0 if missing else max(float(data.pert_graph.view(-1)[0].item()), strength))
    _refresh_align_after_code_perturb(data)
    refresh_apk_node_quality(data)
    refresh_risk_node_quality(data)


def _degrade_manifest(data: Data, strength: float, *, missing: bool = False, noisy: bool = False, blind: bool = False) -> None:
    manifest_nodes = _node_mask(data, sources={SOURCE_TYPES["manifest"]})
    manifest_edges = _edge_mask(data, edge_types=MANIFEST_EDGE_TYPES, sources={SOURCE_TYPES["manifest"]})

    if blind and not missing:
        # Blind mode: degrade manifest evidence but do NOT update pert_manifest
        # This tests if the model can autonomously detect code-manifest conflict
        # without being explicitly told that manifest is corrupted
        if bool(manifest_nodes.any()):
            strength = _clamp_strength(strength)
            if hasattr(data, "node_semantic"):
                data.node_semantic[manifest_nodes] = data.node_semantic[manifest_nodes] * (1.0 - strength)
            data.x[manifest_nodes] = data.x[manifest_nodes] * (1.0 - 0.5 * strength)
            if noisy:
                data.x[manifest_nodes] = data.x[manifest_nodes] + torch.randn_like(data.x[manifest_nodes]) * strength * 0.1
        clear_aggregate_apk_semantic(data)
        refresh_risk_node_quality(data)
        # ✅ Key: Do NOT set pert_manifest in blind mode
        return

    # Non-blind mode: normal degradation with pert_manifest update
    _soft_degrade_nodes(data, manifest_nodes, strength, zero=missing, noise=noisy)
    _soft_degrade_edges(data, manifest_edges, strength, zero=missing)
    clear_aggregate_apk_semantic(data)
    # ⚠️ Only set pert_manifest in non-blind mode
    _set_scalar(data, "pert_manifest", 1.0 if missing else max(float(data.pert_manifest.view(-1)[0].item()), strength))
    refresh_apk_node_quality(data)
    refresh_risk_node_quality(data)


def apply_aeg_view(data: Data, *, view: str, strength: float = 0.5) -> Data:
    """Return a perturbed AEG view.

    Perturbations update both observable evidence and reliability scalars. This
    is deliberately graph-level, not field-level legacy dropout.
    """

    out = data.clone()
    strength = _clamp_strength(strength)
    view = str(view or "clean")
    _set_view_tracking(out, requested_view=view)
    out.cf_weight = torch.tensor([0.0], dtype=torch.float32)

    if view == "clean" or strength <= 0:
        return out
    if view == "api_degraded":
        _degrade_api(out, strength)
        out.cf_weight = torch.tensor([0.4 * strength], dtype=torch.float32)
    elif view == "graph_degraded":
        _degrade_graph(out, strength)
        out.cf_weight = torch.tensor([0.4 * strength], dtype=torch.float32)
    elif view == "api_graph_degraded":
        _degrade_api(out, strength)
        _degrade_graph(out, strength)
        out.cf_weight = torch.tensor([0.8 * strength], dtype=torch.float32)
    elif view in {"manifest_degraded", "manifest_zeroed"}:
        _degrade_manifest(out, strength, missing=(view == "manifest_zeroed"))
        out.cf_weight = torch.tensor([0.6 * strength], dtype=torch.float32)
    elif view == "manifest_noisy":
        _degrade_manifest(out, strength, noisy=True)
        out.cf_weight = torch.tensor([0.8 * strength], dtype=torch.float32)
    elif view == "manifest_shuffled":
        out.cf_weight = torch.tensor([0.9], dtype=torch.float32)
    elif view == "manifest_noisy_blind":
        _degrade_manifest(out, strength, noisy=True, blind=True)
        out.cf_weight = torch.tensor([0.8 * strength], dtype=torch.float32)
    elif view == "manifest_shuffled_blind":
        out.cf_weight = torch.tensor([0.9], dtype=torch.float32)
    elif view == "all_degraded":
        _degrade_api(out, strength)
        _degrade_graph(out, strength)
        _degrade_manifest(out, strength)
        out.cf_weight = torch.tensor([1.0 * strength], dtype=torch.float32)
    elif view == "api_missing":
        _degrade_api(out, 1.0, missing=True)
        out.cf_weight = torch.tensor([0.8], dtype=torch.float32)
    elif view == "graph_missing":
        _degrade_graph(out, 1.0, missing=True)
        out.cf_weight = torch.tensor([0.8], dtype=torch.float32)
    elif view == "manifest_missing":
        _degrade_manifest(out, 1.0, missing=True)
        out.cf_weight = torch.tensor([0.8], dtype=torch.float32)
    else:
        raise ValueError(f"Unknown AEG perturbation view: {view}")
    return out
