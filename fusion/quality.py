from __future__ import annotations

import torch

from fusion.constants import QualityConstants
from fusion.utils import clamp01, scalar_float


def compute_api_quality(
    api_ids,
    api_type_ids=None,
    api_in_graph_mask=None,
) -> float:
    if not isinstance(api_ids, torch.Tensor):
        return 0.0

    api_ids = api_ids.view(-1)
    n = int(api_ids.numel())
    if n <= 0:
        return 0.0

    # Quality must describe extraction integrity, not behavior richness. Counts,
    # diversity, graph coverage, and taxonomy coverage are evidence/alignment
    # signals and can be class-correlated, so they must not enter reliability.
    return 1.0


def compute_graph_quality(edge_index, num_nodes: int, node_features=None, real_node_mask=None) -> float:
    """Estimate observable graph extraction quality.

    ``num_nodes`` must count real extracted nodes, not synthetic ghost nodes
    inserted solely to keep PyG batching valid.
    """
    num_nodes = int(num_nodes)
    if num_nodes <= 0:
        return 0.0

    structure_score = 0.0
    if isinstance(edge_index, torch.Tensor) and edge_index.ndim == 2 and edge_index.size(0) == 2:
        if edge_index.numel() == 0:
            structure_score = 1.0
        else:
            endpoints = edge_index.long().view(-1)
            structure_score = float(((endpoints >= 0) & (endpoints < num_nodes)).float().mean().item())

    feature_score = 1.0
    if isinstance(node_features, torch.Tensor) and node_features.ndim == 2:
        if isinstance(real_node_mask, torch.Tensor) and real_node_mask.numel() == node_features.size(0):
            real = node_features[real_node_mask.view(-1).bool()].float()
        else:
            real = node_features[:num_nodes].float()
        if real.numel() <= 0:
            feature_score = 0.0
        else:
            finite_rows = torch.isfinite(real).all(dim=1)
            feature_score = float(finite_rows.float().mean().item())

    return clamp01(
        QualityConstants.GRAPH_STRUCTURE_WEIGHT * structure_score
        + QualityConstants.GRAPH_FEATURE_WEIGHT * feature_score
    )


def compute_align_quality(
    q_api: float,
    q_graph: float,
    method_api_edge_index,
    num_nodes: int,
    num_api: int,
) -> float:
    num_nodes = int(num_nodes)
    num_api = int(num_api)

    if (
        num_nodes <= 0
        or num_api <= 0
        or not isinstance(method_api_edge_index, torch.Tensor)
        or method_api_edge_index.ndim != 2
        or method_api_edge_index.size(0) != 2
        or method_api_edge_index.numel() == 0
    ):
        return 0.0

    node_cover = float(method_api_edge_index[0].unique().numel()) / max(num_nodes, 1)
    api_cover = float(method_api_edge_index[1].unique().numel()) / max(num_api, 1)
    edge_cover = (
        QualityConstants.ALIGN_NODE_COVER_WEIGHT * node_cover
        + QualityConstants.ALIGN_API_COVER_WEIGHT * api_cover
    )

    code_quality = (clamp01(q_api) * clamp01(q_graph)) ** 0.5
    return clamp01(edge_cover * code_quality)


def refresh_api_quality(data: dict) -> None:
    data["q_api"] = compute_api_quality(
        data.get("api_ids"),
        data.get("api_type_ids"),
        data.get("api_in_graph_mask"),
    )


def refresh_graph_quality(data: dict) -> None:
    x = data.get("x")
    fallback_nodes = int(x.size(0)) if isinstance(x, torch.Tensor) and x.ndim == 2 else 0
    num_nodes = int(data.get("real_num_nodes", fallback_nodes))
    data["q_graph"] = compute_graph_quality(
        data.get("edge_index"),
        num_nodes,
        x,
        data.get("real_node_mask"),
    )


def refresh_align_quality(data: dict) -> None:
    api_ids = data.get("api_ids")
    x = data.get("x")
    num_api = int(api_ids.numel()) if isinstance(api_ids, torch.Tensor) else 0
    fallback_nodes = int(x.size(0)) if isinstance(x, torch.Tensor) and x.ndim == 2 else 0
    num_nodes = int(data.get("real_num_nodes", fallback_nodes))

    data["q_align"] = compute_align_quality(
        scalar_float(data.get("q_api", 0.0)),
        scalar_float(data.get("q_graph", 0.0)),
        data.get("method_api_edge_index"),
        num_nodes,
        num_api,
    )


def compute_manifest_quality(
    manifest_x,
    manifest_category_counts=None,
    manifest_stats=None,
    manifest_meta=None,
) -> float:
    """Estimate extraction integrity without rewarding feature richness."""
    meta = manifest_meta if isinstance(manifest_meta, dict) else {}
    if meta.get("parse_error"):
        return 0.0
    stored = meta.get("quality_score")
    if stored is not None:
        return clamp01(scalar_float(stored, 0.0))

    has_vector = isinstance(manifest_x, torch.Tensor) and manifest_x.numel() > 0
    has_counts = isinstance(manifest_category_counts, torch.Tensor) and manifest_category_counts.numel() > 0
    has_stats = isinstance(manifest_stats, torch.Tensor) and manifest_stats.numel() > 0
    return 1.0 if has_vector or has_counts or has_stats else 0.0


def refresh_code_quality(data: dict) -> None:
    refresh_api_quality(data)
    refresh_graph_quality(data)
    refresh_align_quality(data)
