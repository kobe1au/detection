from __future__ import annotations

import torch

from fusion.constants import QualityConstants


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


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

    count_score = min(1.0, n / QualityConstants.API_COUNT_NORM)
    diversity_score = min(
        1.0,
        float(api_ids.unique().numel()) / max(n, 1) * QualityConstants.API_DIVERSITY_SCALE,
    )

    if isinstance(api_in_graph_mask, torch.Tensor) and api_in_graph_mask.numel() == n:
        coverage_score = float(api_in_graph_mask.float().view(-1).mean().item())
    else:
        coverage_score = 0.0

    if isinstance(api_type_ids, torch.Tensor) and api_type_ids.numel() == n:
        type_score = float((api_type_ids.long().view(-1) > 0).float().mean().item())
    else:
        type_score = 0.0

    return _clamp01(
        QualityConstants.API_COUNT_WEIGHT * count_score
        + QualityConstants.API_DIVERSITY_WEIGHT * diversity_score
        + QualityConstants.API_COVERAGE_WEIGHT * coverage_score
        + QualityConstants.API_TYPE_WEIGHT * type_score
    )


def compute_graph_quality(edge_index, num_nodes: int) -> float:
    num_nodes = int(num_nodes)
    if num_nodes <= 0:
        return 0.0

    if isinstance(edge_index, torch.Tensor) and edge_index.ndim == 2 and edge_index.size(0) == 2:
        num_edges = int(edge_index.size(1))
    else:
        num_edges = 0

    node_score = min(1.0, num_nodes / QualityConstants.GRAPH_NODE_NORM)
    edge_score = min(1.0, num_edges / max(num_nodes, 1))

    return _clamp01(
        QualityConstants.GRAPH_NODE_WEIGHT * node_score
        + QualityConstants.GRAPH_EDGE_WEIGHT * edge_score
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

    code_quality = (_clamp01(q_api) * _clamp01(q_graph)) ** 0.5
    return _clamp01(edge_cover * code_quality)


def refresh_api_quality(data: dict) -> None:
    data["q_api"] = compute_api_quality(
        data.get("api_ids"),
        data.get("api_type_ids"),
        data.get("api_in_graph_mask"),
    )


def refresh_graph_quality(data: dict) -> None:
    x = data.get("x")
    num_nodes = int(x.size(0)) if isinstance(x, torch.Tensor) and x.ndim == 2 else 0
    data["q_graph"] = compute_graph_quality(data.get("edge_index"), num_nodes)


def refresh_align_quality(data: dict) -> None:
    api_ids = data.get("api_ids")
    x = data.get("x")
    num_api = int(api_ids.numel()) if isinstance(api_ids, torch.Tensor) else 0
    num_nodes = int(x.size(0)) if isinstance(x, torch.Tensor) and x.ndim == 2 else 0

    data["q_align"] = compute_align_quality(
        data.get("q_api", 0.0),
        data.get("q_graph", 0.0),
        data.get("method_api_edge_index"),
        num_nodes,
        num_api,
    )


def refresh_code_quality(data: dict) -> None:
    refresh_api_quality(data)
    refresh_graph_quality(data)
    refresh_align_quality(data)