from __future__ import annotations

import torch
from torch_geometric.data import Data

from fusion.constants import EDGE_TYPES, NODE_TYPES, SOURCE_TYPES, VIEW_TYPES


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


def _refresh_align_after_code_perturb(data: Data) -> None:
    q_api = float(getattr(data, "q_api", torch.tensor([0.0])).view(-1)[0].item())
    q_graph = float(getattr(data, "q_graph", torch.tensor([0.0])).view(-1)[0].item())
    q_align = float(getattr(data, "q_align", torch.tensor([0.0])).view(-1)[0].item())
    _set_scalar(data, "q_align", min(q_align, q_api, q_graph))


def _degrade_api(data: Data, strength: float, *, missing: bool = False) -> None:
    api_nodes = _node_mask(data, node_types={NODE_TYPES["API_FAMILY"]})
    api_edges = _edge_mask(data, edge_types={EDGE_TYPES["METHOD_INVOKES_API_FAMILY"], EDGE_TYPES["PERMISSION_RELATED_TO_API_FAMILY"]})
    _soft_degrade_nodes(data, api_nodes, strength, zero=missing)
    _soft_degrade_edges(data, api_edges, strength, zero=missing)
    _set_scalar(data, "q_api", 0.0 if missing else float(data.q_api.view(-1)[0].item()) * (1.0 - _clamp_strength(strength)))
    _refresh_align_after_code_perturb(data)


def _degrade_graph(data: Data, strength: float, *, missing: bool = False) -> None:
    graph_nodes = _node_mask(data, node_types={NODE_TYPES["METHOD"], NODE_TYPES["STRING_HINT"]})
    graph_edges = _edge_mask(
        data,
        edge_types={
            EDGE_TYPES["APK_HAS_METHOD"],
            EDGE_TYPES["METHOD_CALLS_METHOD"],
            EDGE_TYPES["COMPONENT_MATCHES_METHOD"],
            EDGE_TYPES["METHOD_HAS_RISK"],
            EDGE_TYPES["METHOD_HAS_STRING_HINT"],
        },
    )
    _soft_degrade_nodes(data, graph_nodes, strength, zero=missing, noise=not missing)
    _soft_degrade_edges(data, graph_edges, strength, zero=missing)
    _set_scalar(data, "q_graph", 0.0 if missing else float(data.q_graph.view(-1)[0].item()) * (1.0 - _clamp_strength(strength)))
    _refresh_align_after_code_perturb(data)


def _degrade_manifest(data: Data, strength: float, *, missing: bool = False, noisy: bool = False) -> None:
    manifest_nodes = _node_mask(data, sources={SOURCE_TYPES["manifest"]})
    manifest_edges = _edge_mask(data, sources={SOURCE_TYPES["manifest"]})
    _soft_degrade_nodes(data, manifest_nodes, strength, zero=missing, noise=noisy)
    _soft_degrade_edges(data, manifest_edges, strength, zero=missing)
    _set_scalar(data, "q_manifest", 0.0 if missing else float(data.q_manifest.view(-1)[0].item()) * (1.0 - _clamp_strength(strength)))


def apply_aeg_view(data: Data, *, view: str, strength: float = 0.5) -> Data:
    """Return a perturbed AEG view.

    Perturbations update both observable evidence and reliability scalars. This
    is deliberately graph-level, not field-level legacy dropout.
    """

    out = data.clone()
    strength = _clamp_strength(strength)
    view = str(view or "clean")
    out.view_type_id = torch.tensor([VIEW_TYPES.get(view, 0)], dtype=torch.long)
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
