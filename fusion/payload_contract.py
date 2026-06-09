from __future__ import annotations

import re
from typing import Any

import torch

from fusion.constants import (
    AEG_PAYLOAD_CONTRACT_FINGERPRINT,
    AEG_PAYLOAD_CONTRACT_VERSION,
    AEG_REQUIRED_PAYLOAD_FIELDS,
    AEG_SCHEMA_TABLE_FINGERPRINT,
    AEG_SCHEMA_TABLES,
    AEG_SCHEMA_VERSION,
    NUM_EDGE_TYPES,
    NUM_NODE_TYPES,
    NUM_SOURCE_TYPES,
)
from fusion.semantic_categories import SEMANTIC_CATEGORY_DIM


class AEGPayloadContractError(ValueError):
    pass


def _require_tensor(payload: dict[str, Any], key: str) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise AEGPayloadContractError(f"AEG payload field {key!r} must be a tensor")
    return value


def _check_finite(key: str, value: torch.Tensor) -> None:
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise AEGPayloadContractError(f"AEG payload contains non-finite values in {key}")


def _check_unit_interval(key: str, value: torch.Tensor) -> None:
    _check_finite(key, value)
    if value.numel() and (float(value.min()) < 0.0 or float(value.max()) > 1.0):
        raise AEGPayloadContractError(f"AEG payload contains out-of-range values in {key}")


def _check_type_ids(key: str, value: torch.Tensor, upper: int) -> None:
    if value.numel() and (int(value.min()) < 0 or int(value.max()) >= upper):
        raise AEGPayloadContractError(f"AEG payload contains invalid ids in {key}")


def validate_aeg_payload(
    payload: dict[str, Any],
    *,
    expected_node_feature_dim: int | None = None,
) -> None:
    if not isinstance(payload, dict):
        raise AEGPayloadContractError("AEG payload must be a dictionary")
    missing = [key for key in AEG_REQUIRED_PAYLOAD_FIELDS if key not in payload]
    if missing:
        raise AEGPayloadContractError(f"AEG payload is missing required fields: {missing}")

    if int(payload["schema_version"]) != AEG_SCHEMA_VERSION:
        raise AEGPayloadContractError(
            f"AEG schema version mismatch: got={payload['schema_version']} expected={AEG_SCHEMA_VERSION}"
        )
    if payload["aeg_schema_fingerprint"] != AEG_SCHEMA_TABLE_FINGERPRINT:
        raise AEGPayloadContractError("AEG schema table fingerprint mismatch")
    if int(payload["aeg_payload_contract_version"]) != AEG_PAYLOAD_CONTRACT_VERSION:
        raise AEGPayloadContractError("AEG payload contract version mismatch")
    if payload["aeg_payload_contract_fingerprint"] != AEG_PAYLOAD_CONTRACT_FINGERPRINT:
        raise AEGPayloadContractError("AEG payload contract fingerprint mismatch")
    if str(payload["sid"]).lower() != str(payload["sha256"]).lower():
        raise AEGPayloadContractError("AEG sid and sha256 do not match")
    if re.fullmatch(r"[0-9a-f]{64}", str(payload["sha256"]).lower()) is None:
        raise AEGPayloadContractError("AEG sha256 must be a 64-character hexadecimal digest")

    meta = payload["aeg_meta"]
    if not isinstance(meta, dict) or meta.get("schema_fingerprint") != AEG_SCHEMA_TABLE_FINGERPRINT:
        raise AEGPayloadContractError("AEG payload has invalid aeg_meta schema fingerprint")
    for key, expected in AEG_SCHEMA_TABLES.items():
        if dict(meta.get(key) or {}) != expected:
            raise AEGPayloadContractError(f"AEG payload has invalid aeg_meta {key}")

    node_x = _require_tensor(payload, "node_x")
    node_semantic = _require_tensor(payload, "node_semantic")
    edge_index = _require_tensor(payload, "edge_index")
    if node_x.ndim != 2 or node_x.size(0) <= 0 or node_x.size(1) <= 0:
        raise AEGPayloadContractError("AEG payload has invalid or empty node_x")
    if expected_node_feature_dim is not None and node_x.size(1) != int(expected_node_feature_dim):
        raise AEGPayloadContractError(
            f"AEG node feature width mismatch: got={node_x.size(1)} expected={expected_node_feature_dim}"
        )
    num_nodes = int(node_x.size(0))
    if node_semantic.shape != (num_nodes, SEMANTIC_CATEGORY_DIM):
        raise AEGPayloadContractError("AEG payload has invalid node_semantic")
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise AEGPayloadContractError("AEG payload has invalid edge_index")
    num_edges = int(edge_index.size(1))
    if int(meta.get("num_nodes", -1)) != num_nodes:
        raise AEGPayloadContractError("AEG aeg_meta num_nodes does not match node_x")
    if int(meta.get("num_edges", -1)) != num_edges:
        raise AEGPayloadContractError("AEG aeg_meta num_edges does not match edge_index")
    if int(meta.get("node_feature_dim", -1)) != int(node_x.size(1)):
        raise AEGPayloadContractError("AEG aeg_meta node_feature_dim does not match node_x")
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes):
        raise AEGPayloadContractError("AEG payload contains out-of-range edge indices")

    node_type = _require_tensor(payload, "node_type").view(-1)
    node_source = _require_tensor(payload, "node_source").view(-1)
    node_quality = _require_tensor(payload, "node_quality").view(-1)
    edge_type = _require_tensor(payload, "edge_type").view(-1)
    edge_source = _require_tensor(payload, "edge_source").view(-1)
    edge_quality = _require_tensor(payload, "edge_quality").view(-1)
    for key, value in (
        ("node_type", node_type),
        ("node_source", node_source),
        ("node_quality", node_quality),
    ):
        if value.numel() != num_nodes:
            raise AEGPayloadContractError(f"AEG payload has invalid {key} length")
    for key, value in (
        ("edge_type", edge_type),
        ("edge_source", edge_source),
        ("edge_quality", edge_quality),
    ):
        if value.numel() != num_edges:
            raise AEGPayloadContractError(f"AEG payload has invalid {key} length")
    _check_type_ids("node_type", node_type, NUM_NODE_TYPES)
    _check_type_ids("node_source", node_source, NUM_SOURCE_TYPES)
    _check_type_ids("edge_type", edge_type, NUM_EDGE_TYPES)
    _check_type_ids("edge_source", edge_source, NUM_SOURCE_TYPES)

    for key in ("node_x", "node_semantic"):
        _check_finite(key, _require_tensor(payload, key))
    for key in ("node_quality", "edge_quality", "q_api", "q_graph", "q_manifest", "q_align"):
        value = _require_tensor(payload, key).view(-1)
        if key.startswith("q_") and value.numel() != 1:
            raise AEGPayloadContractError(f"AEG payload field {key!r} must contain one scalar")
        _check_unit_interval(key, value)
    for key in ("pert_api", "pert_graph", "pert_manifest"):
        value = _require_tensor(payload, key).view(-1)
        if value.numel() != 1:
            raise AEGPayloadContractError(f"AEG payload field {key!r} must contain one scalar")
        _check_unit_interval(key, value)

    for key in (
        "api_semantic_category_counts",
        "graph_semantic_category_counts",
        "manifest_category_counts",
        "manifest_component_category_counts",
    ):
        value = _require_tensor(payload, key).view(-1)
        if value.numel() != SEMANTIC_CATEGORY_DIM:
            raise AEGPayloadContractError(f"AEG payload has invalid semantic category vector {key}")
        _check_finite(key, value)
        if value.numel() and float(value.min()) < 0.0:
            raise AEGPayloadContractError(f"AEG payload has negative semantic counts in {key}")
    stats = _require_tensor(payload, "manifest_stats").view(-1)
    if stats.numel() == 0:
        raise AEGPayloadContractError("AEG payload has empty manifest_stats")
    _check_unit_interval("manifest_stats", stats)

    for key in ("package_name", "apk_name", "split", "manifest_parse_error"):
        if not isinstance(payload[key], str):
            raise AEGPayloadContractError(f"AEG payload field {key!r} must be a string")
    if not isinstance(payload["sample_meta"], dict):
        raise AEGPayloadContractError("AEG payload sample_meta must be a dictionary")
    if payload["storage_dtype"] not in {"float16", "float32"}:
        raise AEGPayloadContractError("AEG payload storage_dtype must be float16 or float32")
    for key in ("year", "multi_dex_total", "multi_dex_success", "graph_behavior_hint_start", "graph_behavior_hint_dim"):
        if not isinstance(payload[key], int) or isinstance(payload[key], bool):
            raise AEGPayloadContractError(f"AEG payload field {key!r} must be an integer")
    if payload["multi_dex_total"] < 0 or payload["multi_dex_success"] < 0:
        raise AEGPayloadContractError("AEG multi-DEX counts must be non-negative")
    if payload["multi_dex_success"] > payload["multi_dex_total"]:
        raise AEGPayloadContractError("AEG multi_dex_success cannot exceed multi_dex_total")
    if not isinstance(payload["dex_success_ratio"], (int, float)) or isinstance(payload["dex_success_ratio"], bool):
        raise AEGPayloadContractError("AEG dex_success_ratio must be numeric")
    if not 0.0 <= float(payload["dex_success_ratio"]) <= 1.0:
        raise AEGPayloadContractError("AEG dex_success_ratio must be within [0, 1]")
    expected_ratio = payload["multi_dex_success"] / max(1, payload["multi_dex_total"])
    if abs(float(payload["dex_success_ratio"]) - expected_ratio) > 1e-6:
        raise AEGPayloadContractError("AEG dex_success_ratio does not match multi-DEX counts")
    for key in (
        "manifest_parse_ok",
        "graph_behavior_hints",
        "has_reflection",
        "has_dynamic_loading",
        "has_native",
        "has_string_encryption_hint",
    ):
        if not isinstance(payload[key], bool):
            raise AEGPayloadContractError(f"AEG payload field {key!r} must be boolean")
    if payload["manifest_parse_ok"] == bool(payload["manifest_parse_error"]):
        raise AEGPayloadContractError("AEG Manifest parse status and parse error are inconsistent")
