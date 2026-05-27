from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from fusion.robust.manifest_features import DEFAULT_CATEGORIES


SEMANTIC_CATEGORIES = tuple(DEFAULT_CATEGORIES)
SEMANTIC_CATEGORY_DIM = len(SEMANTIC_CATEGORIES)
CATEGORY_TO_INDEX = {name: idx for idx, name in enumerate(SEMANTIC_CATEGORIES)}

# The API extractor stores compact type ids rather than category names. This
# mapping makes the reliability/consistency space explicit and keeps id 0 as
# unknown/background. If the extractor changes its taxonomy, update this table
# rather than comparing raw type-id histograms to Manifest categories.
DEFAULT_API_TYPE_ID_TO_CATEGORY: dict[int, str] = {
    1: "network",
    2: "sms",
    3: "location",
    4: "contacts",
    5: "storage",
    6: "telephony",
    7: "camera_media",
    8: "receiver",
    9: "component_exposure",
    10: "dynamic_loading",
    11: "crypto",
    12: "system_settings",
    13: "dynamic_loading",
    14: "crypto",
    15: "system_settings",
}


def sanitize_semantic_counts(value: Any, *, require_exact: bool = False) -> torch.Tensor:
    """Return a clean 12-D semantic count vector.

    `require_exact=True` is used for graph counts loaded from existing .pt
    files. If a source vector is not already in the shared 12-D taxonomy, it is
    treated as unavailable instead of being silently trimmed or padded.
    """
    if isinstance(value, torch.Tensor):
        out = value.detach().float().view(-1)
    elif value is None:
        out = torch.empty((0,), dtype=torch.float32)
    else:
        out = torch.as_tensor(value, dtype=torch.float32).view(-1)

    if require_exact and out.numel() not in {0, SEMANTIC_CATEGORY_DIM}:
        out = torch.empty((0,), dtype=torch.float32)

    if out.numel() < SEMANTIC_CATEGORY_DIM:
        pad = torch.zeros((SEMANTIC_CATEGORY_DIM - out.numel(),), dtype=torch.float32)
        out = torch.cat([out, pad], dim=0)
    elif out.numel() > SEMANTIC_CATEGORY_DIM:
        out = out[:SEMANTIC_CATEGORY_DIM]
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def api_semantic_counts_from_type_ids(
    api_type_ids: torch.Tensor | None,
    mapping: Mapping[int, str] | None = None,
) -> torch.Tensor:
    counts = torch.zeros((SEMANTIC_CATEGORY_DIM,), dtype=torch.float32)
    if not isinstance(api_type_ids, torch.Tensor) or api_type_ids.numel() == 0:
        return counts

    mapping = mapping or DEFAULT_API_TYPE_ID_TO_CATEGORY
    flat = api_type_ids.detach().long().view(-1).cpu()
    for type_id in flat.tolist():
        category = mapping.get(int(type_id))
        category_idx = CATEGORY_TO_INDEX.get(category or "")
        if category_idx is not None:
            counts[category_idx] += 1.0
    return counts


def graph_semantic_counts_from_method_api_edges(
    api_type_ids: torch.Tensor | None,
    method_api_edge_index: torch.Tensor | None,
    *,
    mapping: Mapping[int, str] | None = None,
) -> torch.Tensor:
    """Aggregate API semantic categories carried by graph-aligned methods.

    The graph branch is structural, so its semantic category distribution is
    derived only from API events that are anchored to a graph method through
    `method_api_edge_index`. This gives Graph-Manifest consistency a real
    structural-context basis instead of reusing the full API histogram.
    """
    counts = torch.zeros((SEMANTIC_CATEGORY_DIM,), dtype=torch.float32)
    if (
        not isinstance(api_type_ids, torch.Tensor)
        or not isinstance(method_api_edge_index, torch.Tensor)
        or method_api_edge_index.ndim != 2
        or method_api_edge_index.size(0) != 2
        or api_type_ids.numel() == 0
        or method_api_edge_index.numel() == 0
    ):
        return counts

    api_idx = method_api_edge_index[1].detach().long().view(-1).cpu()
    valid = (api_idx >= 0) & (api_idx < int(api_type_ids.numel()))
    if not valid.any():
        return counts
    aligned_types = api_type_ids.detach().long().view(-1).cpu()[api_idx[valid]]
    return api_semantic_counts_from_type_ids(aligned_types, mapping=mapping)
